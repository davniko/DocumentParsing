"""Evaluation and plotting helpers for the isolated generator-method probe."""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from document_ocr.synthesis.transport_identity import SourceTransportIdentityGuard
from document_ocr.synthesis.vessel_lexical import (
    CharacterNGramVesselRenderer,
    lexical_discriminator_auc,
    lexical_realism_metrics,
)

_SPACE = re.compile(r"\s+")


def _surface(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return _SPACE.sub(" ", normalized.strip()).casefold()


def _js_similarity(left: Sequence[int], right: Sequence[int]) -> float:
    left_counts, right_counts = Counter(left), Counter(right)
    keys = sorted(set(left_counts) | set(right_counts))
    left_total, right_total = sum(left_counts.values()), sum(right_counts.values())
    if not keys or not left_total or not right_total:
        raise ValueError("distribution similarity requires non-empty values")
    divergence = 0.0
    for key in keys:
        p = left_counts[key] / left_total
        q = right_counts[key] / right_total
        midpoint = (p + q) / 2
        if p:
            divergence += 0.5 * p * math.log2(p / midpoint)
        if q:
            divergence += 0.5 * q * math.log2(q / midpoint)
    return 1.0 - math.sqrt(max(0.0, min(1.0, divergence)))


def evaluate_vessel_names(
    *,
    reference_names: Sequence[str],
    generated_names: Sequence[str],
    evaluator: CharacterNGramVesselRenderer,
    guard: SourceTransportIdentityGuard,
    seed: int,
) -> dict[str, Any]:
    if min(len(reference_names), len(generated_names)) < 6:
        raise ValueError("vessel evaluation requires at least six names per side")
    lexical = lexical_realism_metrics(
        reference_names=reference_names,
        generated_names=generated_names,
    )
    discriminator = lexical_discriminator_auc(
        reference_names=reference_names,
        generated_names=generated_names,
        seed=seed,
    )
    distances = [guard.vessel_distance_diagnostics(value) for value in generated_names]
    return {
        "names": len(generated_names),
        "uniqueFraction": len(set(generated_names)) / len(generated_names),
        "lexicalRealism": lexical,
        "discriminator": discriminator,
        "bitsPerCharacter": {
            "reference": evaluator.bits_per_character(reference_names),
            "generated": evaluator.bits_per_character(generated_names),
            "absoluteGap": abs(
                evaluator.bits_per_character(reference_names)
                - evaluator.bits_per_character(generated_names)
            ),
        },
        "lengthDistributionSimilarity": _js_similarity(
            [len(value) for value in reference_names],
            [len(value) for value in generated_names],
        ),
        "wordCountDistributionSimilarity": _js_similarity(
            [len(value.split()) for value in reference_names],
            [len(value.split()) for value in generated_names],
        ),
        "nearestSourceNormalizedEditDistance": {
            "minimum": min(float(value["normalized"]) for value in distances),
            "median": statistics.median(float(value["normalized"]) for value in distances),
        },
        "maximumSourceSubstringFraction": {
            "maximum": max(float(value["substringFraction"]) for value in distances),
            "median": statistics.median(float(value["substringFraction"]) for value in distances),
        },
    }


def evaluate_character_fields(
    *,
    reference_rows: Sequence[Mapping[str, Any]],
    generated_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in fields:
        reference = tuple(filter(None, (_surface(row.get(field)) for row in reference_rows)))
        generated = tuple(filter(None, (_surface(row.get(field)) for row in generated_rows)))
        source = frozenset(filter(None, (_surface(row.get(field)) for row in source_rows)))
        collisions = sum(value in source for value in generated)
        result[field] = {
            "referencePresent": len(reference),
            "generatedPresent": len(generated),
            "generatedPresentFraction": len(generated) / max(len(generated_rows), 1),
            "uniqueFraction": len(set(generated)) / max(len(generated), 1),
            "exactFullSourceCollisionFraction": collisions / max(len(generated), 1),
            "lengthDistributionSimilarity": (
                _js_similarity(
                    [len(value) for value in reference],
                    [len(value) for value in generated],
                )
                if reference and generated
                else None
            ),
        }
    return result


def evaluate_hs_suffixes(
    *, generated_rows: Sequence[Mapping[str, Any]], source_codes: frozenset[str]
) -> dict[str, Any]:
    violations: Counter[str] = Counter()
    valid: list[str] = []
    for row in generated_rows:
        digits = row.get("output_digits")
        hs6 = row.get("hs6")
        suffix = row.get("suffix")
        if type(digits) is not int or not 7 <= digits <= 18:
            violations["invalid_output_digits"] += 1
            continue
        if not isinstance(hs6, str) or len(hs6) != 6 or not hs6.isdigit():
            violations["invalid_hs6"] += 1
            continue
        if not isinstance(suffix, str) or not suffix.isdigit():
            violations["suffix_not_digits"] += 1
            continue
        if len(suffix) != digits - 6:
            violations["suffix_length_mismatch"] += 1
            continue
        valid.append(hs6 + suffix)
    return {
        "rows": len(generated_rows),
        "validRows": len(valid),
        "validFraction": len(valid) / max(len(generated_rows), 1),
        "uniqueFraction": len(set(valid)) / max(len(valid), 1),
        "exactSourceReplayFraction": sum(value in source_codes for value in valid)
        / max(len(valid), 1),
        "violationCounts": dict(sorted(violations.items())),
    }


def plot_generator_probe(
    *,
    vessel_runs: Sequence[Mapping[str, Any]],
    real_baselines: Sequence[Mapping[str, Any]],
    method_results: Mapping[str, Any],
    output: Path,
) -> None:
    matplotlib = __import__("matplotlib")
    matplotlib.use("Agg")
    pyplot = __import__("matplotlib.pyplot", fromlist=["pyplot"])
    seaborn = __import__("seaborn")
    pandas = __import__("pandas")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in vessel_runs:
        if run.get("status") != "complete":
            continue
        metrics = run["metrics"]
        rows.append(
            {
                "candidate": run["candidate"],
                "fold": run["foldIndex"],
                "lexical realism": metrics["lexicalRealism"]["lexicalRealismScore"],
                "BPC gap": metrics["bitsPerCharacter"]["absoluteGap"],
                "discriminator excess": metrics["discriminator"]["excessOverChance"],
                "acceptance": run["acceptanceFraction"],
                "fit seconds": run["fitSeconds"],
                "sample seconds": run["sampleSeconds"],
                "total seconds": run["totalSeconds"],
            }
        )
    for baseline in real_baselines:
        rows.append(
            {
                "candidate": "real_vs_real",
                "fold": baseline["foldIndex"],
                "lexical realism": baseline["lexicalRealism"]["lexicalRealismScore"],
                "BPC gap": baseline["bitsPerCharacterGap"],
                "discriminator excess": baseline["discriminator"]["excessOverChance"],
                "acceptance": None,
                "fit seconds": None,
                "sample seconds": None,
                "total seconds": None,
            }
        )
    frame = pandas.DataFrame(rows)
    if not frame.empty:
        for index, metric in enumerate(
            ("lexical realism", "BPC gap", "discriminator excess", "acceptance"), start=1
        ):
            figure, axis = pyplot.subplots(figsize=(12, 6))
            metric_frame = frame.dropna(subset=[metric])
            seaborn.barplot(
                data=metric_frame,
                x="candidate",
                y=metric,
                errorbar="sd",
                ax=axis,
            )
            axis.tick_params(axis="x", rotation=35)
            axis.set_title(f"Grouped vessel benchmark: {metric}")
            figure.tight_layout()
            figure.savefig(output / f"{index:02d}_vessel_{metric.replace(' ', '_')}.png", dpi=180)
            pyplot.close(figure)
        for index, metric in enumerate(("fit seconds", "sample seconds", "total seconds"), start=5):
            metric_frame = frame.dropna(subset=[metric])
            figure, axis = pyplot.subplots(figsize=(12, 6))
            seaborn.barplot(
                data=metric_frame,
                x="candidate",
                y=metric,
                errorbar="sd",
                ax=axis,
            )
            axis.set_yscale("log")
            axis.tick_params(axis="x", rotation=35)
            axis.set_title(f"Grouped vessel benchmark {metric} (log scale)")
            figure.tight_layout()
            figure.savefig(
                output / f"{index:02d}_vessel_{metric.replace(' ', '_')}.png",
                dpi=180,
            )
            pyplot.close(figure)

    method_rows = []
    party = method_results.get("party", {})
    for field, values in (party.get("fieldMetrics") or {}).items():
        component = f"party:{field}"
        method_rows.extend(
            (
                {
                    "component": component,
                    "metric": "source non-replay",
                    "fraction": 1 - float(values["exactFullSourceCollisionFraction"]),
                },
                {
                    "component": component,
                    "metric": "uniqueness",
                    "fraction": float(values["uniqueFraction"]),
                },
            )
        )
    hs = method_results.get("hsSuffix", {})
    if hs:
        method_rows.extend(
            (
                {
                    "component": "hs_suffix",
                    "metric": "structural validity",
                    "fraction": float(hs.get("validFraction", 0)),
                },
                {
                    "component": "hs_suffix",
                    "metric": "uniqueness",
                    "fraction": float(hs.get("uniqueFraction", 0)),
                },
            )
        )
    if method_rows:
        method_frame = pandas.DataFrame(method_rows)
        figure, axis = pyplot.subplots(figsize=(10, 6))
        seaborn.barplot(data=method_frame, x="component", y="fraction", hue="metric", ax=axis)
        axis.set_ylim(0, 1)
        axis.set_title("MostlyAI mixed-field diagnostic")
        figure.tight_layout()
        figure.savefig(output / "08_mostlyai_mixed_field_diagnostics.png", dpi=180)
        pyplot.close(figure)
