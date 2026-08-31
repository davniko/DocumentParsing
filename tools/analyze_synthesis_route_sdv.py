# ruff: noqa: E501
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]

DEFAULT_ROUTE_RUN = Path("artifacts/kie-synthesis/mpci-bl-combined1157-route-scenario-pilot50-v10")
DEFAULT_BENCHMARK_RUN = Path(
    "artifacts/kie-synthesis/mpci-bl-combined1157-party-structure-sdv-gpu-benchmark-v8"
)
DEFAULT_OUTPUT = Path("artifacts/kie-synthesis/mpci-bl-route-party-sdv-audit-v5")
METHOD_ORDER = ("empirical", "gaussian_copula", "ctgan", "tvae")
METHOD_LABELS = {
    "empirical": "Empirical",
    "gaussian_copula": "Gaussian copula",
    "ctgan": "CTGAN",
    "tvae": "TVAE",
}
METHOD_COLORS = {
    "empirical": "#4C78A8",
    "gaussian_copula": "#F58518",
    "ctgan": "#E45756",
    "tvae": "#54A24B",
}


Json = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and visualize the MPCI B/L route pilot and SDV benchmark."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--route-run", type=Path, default=DEFAULT_ROUTE_RUN)
    parser.add_argument("--benchmark-run", type=Path, default=DEFAULT_BENCHMARK_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> Json:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[Json]:
    rows: list[Json] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise RuntimeError(f"blank JSONL row at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def resolve_under(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def verify_listed_files(
    root: Path,
    entries: Iterable[Mapping[str, Any]],
    *,
    path_key: str,
    size_key: str,
) -> list[Json]:
    verified: list[Json] = []
    for entry in entries:
        relative_path = str(entry[path_key])
        path = root / relative_path
        if not path.is_file():
            raise RuntimeError(f"manifest-listed file is absent: {path}")
        actual_size = path.stat().st_size
        expected_size = int(entry[size_key])
        if actual_size != expected_size:
            raise RuntimeError(
                f"manifest size mismatch for {path}: expected {expected_size}, got {actual_size}"
            )
        actual_sha = sha256_file(path)
        expected_sha = str(entry["sha256"])
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"manifest SHA-256 mismatch for {path}: expected {expected_sha}, got {actual_sha}"
            )
        verified.append({"path": relative_path, "sizeBytes": actual_size, "sha256": actual_sha})
    return verified


def counter_dict(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def assert_equal(actual: Any, expected: Any, description: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{description} mismatch: expected {expected!r}, got {actual!r}")


def assert_close(actual: float, expected: float, description: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError(f"{description} mismatch: expected {expected}, got {actual}")


def audit_route_run(route_root: Path) -> tuple[Json, list[Json], Json]:
    commit = read_json(route_root / "_COMMIT.json")
    manifest = read_json(route_root / "manifest.json")
    distribution = read_json(route_root / "generation/distribution-summary.json")
    validation = read_json(route_root / "generation/validation-summary.json")
    runtime = read_json(route_root / "runtime.json")
    scenario_support_audit = read_json(route_root / "modeling/scenario-support-audit.json")
    trade_flow_support_audit = read_json(route_root / "modeling/trade-flow-support-audit.json")
    scenarios = read_jsonl(route_root / "generation/shipment-scenarios.jsonl")
    projected_targets = read_jsonl(route_root / "generation/projected-targets.jsonl")

    verified_files = verify_listed_files(
        route_root,
        commit["artifacts"],
        path_key="relative_path",
        size_key="bytes",
    )
    assert_equal(
        manifest["status"],
        "route_scenario_pilot_complete_pending_identity_vessel_text",
        "route status",
    )
    assert_equal(manifest["trainingEligible"], False, "route manifest training eligibility")
    assert_equal(manifest["modelOrApiCalls"], 0, "route model/API calls")
    requested_documents = int(validation["requestedDocuments"])
    assert_equal(len(scenarios), int(manifest["scenarioDocuments"]), "route scenario row count")
    assert_equal(len(projected_targets), len(scenarios), "route projected-target row count")
    assert_equal(len(scenarios), requested_documents, "requested route scenario count")
    assert_equal({row["trainingEligible"] for row in scenarios}, {False}, "scenario eligibility")
    assert_equal(
        len({row["syntheticDocumentId"] for row in scenarios}),
        len(scenarios),
        "scenario IDs",
    )
    assert_equal(
        len({row["templateId"] for row in scenarios}),
        int(validation["distinctTemplates"]),
        "scenario templates",
    )

    origin = counter_dict(str(row["commercialOriginCountryCode"]) for row in scenarios)
    destination = counter_dict(str(row["commercialDestinationCountryCode"]) for row in scenarios)
    loading = counter_dict(str(row["loadingPort"]["countryCode"]) for row in scenarios)
    discharge = counter_dict(str(row["dischargePort"]["countryCode"]) for row in scenarios)
    freight = counter_dict(str(row["freight"]["arrangement"] or "missing") for row in scenarios)
    vessel = counter_dict(str(row["vesselStatus"]) for row in scenarios)
    port_sources = counter_dict(
        str(port["source"])
        for row in scenarios
        for port in (row["loadingPort"], row["dischargePort"])
    )
    party_relations = counter_dict(
        f"{party['role']}:{party['relation']}"
        for row in scenarios
        for party in row["partyLocalities"]
    )
    party_modes = counter_dict(
        f"{party['role']}:{party['localityMode']}"
        for row in scenarios
        for party in row["partyLocalities"]
    )

    expected_distribution_keys = {
        "commercialOriginCountries": origin,
        "commercialDestinationCountries": destination,
        "loadingCountries": loading,
        "dischargeCountries": discharge,
        "freightArrangementCounts": freight,
        "vesselStatusCounts": vessel,
        "portSourceCounts": port_sources,
        "partyRelationCounts": party_relations,
        "partyLocalityModeCounts": party_modes,
    }
    for key, expected in expected_distribution_keys.items():
        assert_equal(distribution[key], expected, f"route distribution {key}")

    assert_equal(validation["generatedScenarios"], len(scenarios), "route generated scenarios")
    assert_equal(validation["strictSchemaValid"], len(scenarios), "route schema-valid scenarios")
    assert_equal(
        validation["relationalInverseValid"],
        len(scenarios),
        "route inverse-valid scenarios",
    )
    assert_equal(validation["trainingRecordsPublished"], 0, "route training records")

    scenario_classified = sum(
        int(scenario_support_audit[key])
        for key in (
            "usableExportCountryDocuments",
            "missingExportCountryDocuments",
            "exportCountryWithoutMaritimePortDocuments",
            "exportCountryWithoutLocalityDepthDocuments",
            "exportCountryWithoutLoadingCountrySupportDocuments",
            "exportCountryWithoutTradeFlowDocuments",
            "excludedTransshipmentDocuments",
        )
    )
    assert_equal(
        int(scenario_support_audit["countryCleanDocuments"]),
        int(scenario_support_audit["inputDocuments"])
        - int(scenario_support_audit["excludedUnresolvedCountryDocuments"]),
        "scenario support country resolution balance",
    )
    assert_equal(
        scenario_classified,
        int(scenario_support_audit["countryCleanDocuments"]),
        "scenario support disposition balance",
    )
    trade_flow_classified = sum(
        int(trade_flow_support_audit[key])
        for key in (
            "eligibleRecords",
            "excludedNonpositiveRecords",
            "excludedDomesticRecords",
            "excludedOriginWithoutSourceSupportRecords",
            "excludedOriginWithoutMaritimePortRecords",
            "excludedDestinationWithoutMaritimePortRecords",
            "excludedDestinationWithoutLocalityDepthRecords",
            "excludedDestinationWithoutDistinctPhysicalEndpointRecords",
        )
    )
    assert_equal(
        trade_flow_classified,
        int(trade_flow_support_audit["inputRecords"]),
        "trade-flow support disposition balance",
    )

    party_rows = [party for row in scenarios for party in row["partyLocalities"]]
    locality_source_counts = counter_dict(
        str(party["locality"]["source"]) for party in party_rows if party["locality"] is not None
    )
    locality_mode_totals = counter_dict(str(party["localityMode"]) for party in party_rows)
    issue_place_counts = {
        "present": sum(row["placeOfIssue"] is not None for row in scenarios),
        "missing": sum(row["placeOfIssue"] is None for row in scenarios),
    }
    payment_place_counts = {
        "present": sum(row["freight"]["paymentPlace"] is not None for row in scenarios),
        "missing": sum(row["freight"]["paymentPlace"] is None for row in scenarios),
    }
    transshipment_counts = counter_dict(str(row["transshipmentStatus"]) for row in scenarios)
    trade_flow_evidence_count = sum(row["tradeFlowEvidence"] is not None for row in scenarios)
    code_to_names: dict[str, set[str]] = defaultdict(set)
    for row in scenarios:
        for location in (
            row["loadingPort"],
            row["dischargePort"],
            row["placeOfIssue"],
            row["freight"]["paymentPlace"],
        ):
            if location is not None:
                code_to_names[str(location["countryCode"])].add(str(location["countryName"]))
    for party in party_rows:
        if party["countryCode"] is not None:
            code_to_names[str(party["countryCode"])].add(str(party["countryName"]))
        code_to_names[str(party["conditioningCountryCode"])].add(
            str(party["conditioningCountryName"])
        )
    noncanonical_surfaces = {
        code: sorted(names) for code, names in code_to_names.items() if len(names) != 1
    }
    assert_equal(noncanonical_surfaces, {}, "one canonical country surface per ISO code")

    supported_port_endpoints = port_sources.get("observed_port", 0) + port_sources.get(
        "registry_port_exploration", 0
    )
    exploration_fraction = (
        port_sources.get("registry_port_exploration", 0) / supported_port_endpoints
        if supported_port_endpoints
        else 0.0
    )

    route_summary: Json = {
        "runId": manifest["runId"],
        "status": manifest["status"],
        "sourceRecords": manifest["sourceRecords"],
        "fitDocuments": manifest["fitDocuments"],
        "scenarioDocuments": len(scenarios),
        "requestedDocuments": requested_documents,
        "distinctTemplates": len({row["templateId"] for row in scenarios}),
        "trainingEligible": False,
        "trainingRecordsPublished": validation["trainingRecordsPublished"],
        "modelOrApiCalls": manifest["modelOrApiCalls"],
        "validation": validation,
        "runtime": runtime,
        "commercialOriginCountries": origin,
        "commercialDestinationCountries": destination,
        "loadingCountries": loading,
        "dischargeCountries": discharge,
        "portSourceCounts": port_sources,
        "observedSupportedPortEndpoints": supported_port_endpoints,
        "registryExplorationFractionWithinObservedSupport": exploration_fraction,
        "partyRows": len(party_rows),
        "partyRelationCounts": party_relations,
        "partyLocalityModeCounts": party_modes,
        "partyLocalityModeTotals": locality_mode_totals,
        "localitySourceCounts": locality_source_counts,
        "freightArrangementCounts": freight,
        "freightPaymentPlaceCounts": payment_place_counts,
        "placeOfIssueCounts": issue_place_counts,
        "vesselStatusCounts": vessel,
        "transshipmentStatusCounts": transshipment_counts,
        "tradeFlowEvidenceDocuments": trade_flow_evidence_count,
        "canonicalCountrySurfaceAudit": {
            "oneSurfacePerIsoCode": True,
            "distinctCountryCodes": len(code_to_names),
            "sampledAliases": False,
        },
        "registryPins": manifest["registryPins"],
        "generationMethods": manifest["generationMethods"],
        "scenarioSupportAudit": scenario_support_audit,
        "tradeFlowSupportAudit": trade_flow_support_audit,
        "destinationTradeFlowExclusions": {
            "excludedDestinationWithoutLocalityDepthRecords": trade_flow_support_audit[
                "excludedDestinationWithoutLocalityDepthRecords"
            ],
            "excludedDestinationWithoutDistinctPhysicalEndpointRecords": (
                trade_flow_support_audit[
                    "excludedDestinationWithoutDistinctPhysicalEndpointRecords"
                ]
            ),
        },
    }
    integrity: Json = {
        "verified": True,
        "commitContentSha256": commit["content_sha256"],
        "verifiedFiles": verified_files,
    }
    return route_summary, scenarios, integrity


def receipt_to_row(receipt: Json) -> Json:
    fold = receipt["fold"]
    fit = receipt["fit"]
    sample = receipt["sample"]
    evaluation = receipt["evaluation"]
    return {
        "candidate": receipt["candidate"],
        "fold": int(fold["index"]),
        "trainRows": int(fold["train_rows"]),
        "validationRows": int(fold["validation_rows"]),
        "quality": float(evaluation["quality"]["score"]),
        "columnShapes": float(evaluation["quality"]["properties"]["Column Shapes"]),
        "columnPairTrends": float(evaluation["quality"]["properties"]["Column Pair Trends"]),
        "diagnostic": float(evaluation["diagnostic"]["score"]),
        "rawValidFraction": float(receipt["validity"]["raw_valid_fraction"]),
        "outputValidFraction": float(receipt["validity"]["output_valid_fraction"]),
        "novelFraction": float(receipt["novelty"]["novel_fraction"]),
        "uniqueFraction": float(receipt["novelty"]["unique_fraction"]),
        "fitSeconds": float(fit["elapsed_seconds"]),
        "sampleSeconds": float(sample["elapsed_seconds"]),
        "evaluationSeconds": float(receipt["evaluation_elapsed_seconds"]),
        "totalSeconds": float(receipt["total_elapsed_seconds"]),
        "device": str(fit.get("accelerator", {}).get("device", "cpu")),
        "cudaPeakAllocatedBytes": fit.get("cuda_peak_allocated_bytes"),
        "cudaPeakReservedBytes": fit.get("cuda_peak_reserved_bytes"),
        "rawProposals": int(receipt["proposal"]["raw_proposals"]),
        "acceptedProposals": int(receipt["proposal"]["accepted_proposals_before_truncation"]),
        "outputRows": int(receipt["proposal"]["output_rows"]),
    }


def audit_benchmark_run(benchmark_root: Path) -> tuple[Json, list[Json], Json]:
    manifest = read_json(benchmark_root / "manifest.json")
    summary = read_json(benchmark_root / "summary.json")
    contract = read_json(benchmark_root / "party-structure/benchmark-contract.json")
    preflight = read_json(benchmark_root / "party-structure/gpu-preflight.json")

    verified_files = verify_listed_files(
        benchmark_root,
        manifest["files"],
        path_key="path",
        size_key="sizeBytes",
    )
    assert_equal(manifest["status"], "complete_no_production_selection", "benchmark status")
    assert_equal(summary["benchmarkStatus"], "complete", "benchmark completion")
    assert_equal(summary["productionSelectionPerformed"], False, "production selection")
    assert_equal(summary["failedCandidateRuns"], 0, "failed benchmark runs")
    assert_equal(summary["candidateFailures"], [], "candidate failures")
    assert_equal(summary["requestedCandidates"], list(METHOD_ORDER), "requested candidates")
    assert_equal(summary["completeCandidates"], list(METHOD_ORDER), "complete candidates")

    receipt_rows: list[Json] = []
    for candidate in METHOD_ORDER:
        paths = sorted((benchmark_root / "party-structure/run-receipts" / candidate).glob("*.json"))
        assert_equal(len(paths), 3, f"{candidate} receipt count")
        for path in paths:
            receipt_rows.append(receipt_to_row(read_json(path)))

    assert_equal(len(receipt_rows), 12, "benchmark receipt count")
    candidate_summaries = {row["candidate"]: row for row in summary["candidateSummaries"]}
    for candidate in METHOD_ORDER:
        rows = [row for row in receipt_rows if row["candidate"] == candidate]
        assert_equal(sorted(row["fold"] for row in rows), [0, 1, 2], f"{candidate} folds")
        candidate_summary = candidate_summaries[candidate]
        assert_close(
            fmean(row["quality"] for row in rows),
            float(candidate_summary["mean_quality"]),
            f"{candidate} mean quality",
        )
        for receipt_key, summary_key in (
            ("rawValidFraction", "raw_valid_fraction"),
            ("novelFraction", "novel_fraction"),
            ("uniqueFraction", "unique_fraction"),
        ):
            assert_close(
                fmean(row[receipt_key] for row in rows),
                float(candidate_summary[summary_key]),
                f"{candidate} {summary_key}",
            )
        assert_close(
            sum(row["fitSeconds"] for row in rows),
            float(candidate_summary["total_fit_seconds"]),
            f"{candidate} total fit seconds",
        )
        assert_equal(
            {row["outputValidFraction"] for row in rows},
            {1.0},
            f"{candidate} post-validation output validity",
        )

    benchmark_summary: Json = {
        "runId": summary["runId"],
        "status": manifest["status"],
        "rows": summary["rows"],
        "templates": summary["templates"],
        "folds": 3,
        "trainRowsPerFold": sorted({row["trainRows"] for row in receipt_rows}),
        "validationRowsPerFold": sorted({row["validationRows"] for row in receipt_rows}),
        "completedCandidateRuns": summary["completedCandidateRuns"],
        "failedCandidateRuns": summary["failedCandidateRuns"],
        "elapsedSeconds": summary["elapsedSeconds"],
        "candidateSummaries": summary["candidateSummaries"],
        "productionSelectionPerformed": False,
        "syntheticRowsPublished": 0,
        "containsRawPartyPii": contract["contains_raw_party_pii"],
        "modeledColumns": contract["columns"],
        "environment": contract["environment"],
        "gpuPreflight": preflight,
        "foldMetrics": receipt_rows,
    }
    integrity: Json = {
        "verified": True,
        "verifiedFiles": verified_files,
        "manifestSha256": sha256_file(benchmark_root / "manifest.json"),
    }
    return benchmark_summary, receipt_rows, integrity


def apply_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 180,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#303030",
            "text.color": "#303030",
            "font.size": 10,
        }
    )


def annotate_barh(axis: Any) -> None:
    for patch in axis.patches:
        width = patch.get_width()
        axis.text(
            width + max(axis.get_xlim()[1] * 0.012, 0.05),
            patch.get_y() + patch.get_height() / 2,
            f"{width:.0f}",
            va="center",
            fontsize=8,
        )


def annotate_bars(axis: Any, *, decimals: int = 0) -> None:
    for patch in axis.patches:
        height = patch.get_height()
        if math.isnan(height):
            continue
        label = f"{height:.{decimals}f}"
        axis.text(
            patch.get_x() + patch.get_width() / 2,
            height + max(axis.get_ylim()[1] * 0.015, 0.001),
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_country_distributions(route: Json, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 10), constrained_layout=True)
    for axis, key, title, color in (
        (
            axes[0],
            "commercialOriginCountries",
            f"Commercial origin countries (all {len(route['commercialOriginCountries'])})",
            "#4C78A8",
        ),
        (
            axes[1],
            "commercialDestinationCountries",
            (
                "Commercial destination countries "
                f"(all {len(route['commercialDestinationCountries'])})"
            ),
            "#F58518",
        ),
    ):
        values = sorted(route[key].items(), key=lambda item: (item[1], item[0]))
        labels = [item[0] for item in values]
        counts = [item[1] for item in values]
        axis.barh(labels, counts, color=color, alpha=0.88)
        axis.set_xlabel("Scenarios")
        axis.set_title(title)
        axis.set_xlim(0, max(counts) * 1.15)
        annotate_barh(axis)
    figure.suptitle("Route pilot: country coverage", fontsize=15, fontweight="bold")
    figure.savefig(output, bbox_inches="tight", metadata={"Software": "matplotlib"})
    plt.close(figure)


def heatmap_from_compound_counts(
    axis: Any,
    counts: Mapping[str, int],
    *,
    title: str,
    cmap: str,
) -> None:
    records = []
    for compound, count in counts.items():
        role, value = compound.split(":", maxsplit=1)
        records.append({"role": role, "value": value, "count": count})
    table = (
        pd.DataFrame(records)
        .pivot_table(index="role", columns="value", values="count", fill_value=0)
        .astype(int)
    )
    sns.heatmap(
        table,
        annot=True,
        fmt="d",
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Party occurrences"},
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.tick_params(axis="x", rotation=30)
    axis.tick_params(axis="y", rotation=0)


def plot_party_conditioning(route: Json, output: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    heatmap_from_compound_counts(
        axes[0],
        route["partyRelationCounts"],
        title="Party role vs. route relation",
        cmap="Blues",
    )
    heatmap_from_compound_counts(
        axes[1],
        route["partyLocalityModeCounts"],
        title="Party role vs. locality sampling mode",
        cmap="Greens",
    )
    figure.suptitle(
        f"Route-conditioned party structure ({route['partyRows']} party occurrences)",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output, bbox_inches="tight", metadata={"Software": "matplotlib"})
    plt.close(figure)


def plot_route_status(route: Json, output: Path) -> None:
    panels = (
        ("portSourceCounts", "Port source modes", "#4C78A8"),
        ("partyLocalityModeTotals", "Party locality modes", "#54A24B"),
        ("freightArrangementCounts", "Freight arrangement", "#F58518"),
        ("vesselStatusCounts", "Vessel synthesis status", "#E45756"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for axis, (key, title, color) in zip(axes.flat, panels, strict=True):
        items = sorted(route[key].items(), key=lambda item: (-item[1], item[0]))
        labels = [item[0].replace("_", "\n") for item in items]
        counts = [item[1] for item in items]
        axis.bar(labels, counts, color=color, alpha=0.88)
        axis.set_title(title)
        axis.set_ylabel("Count")
        axis.set_ylim(0, max(counts) * 1.18)
        axis.tick_params(axis="x", rotation=10)
        annotate_bars(axis)
    figure.suptitle("Route pilot: support and pending components", fontsize=15, fontweight="bold")
    figure.savefig(output, bbox_inches="tight", metadata={"Software": "matplotlib"})
    plt.close(figure)


def metric_frame(receipts: Sequence[Json], metrics: Sequence[str]) -> pd.DataFrame:
    records: list[Json] = []
    for row in receipts:
        for metric in metrics:
            records.append(
                {
                    "candidate": row["candidate"],
                    "method": METHOD_LABELS[row["candidate"]],
                    "fold": row["fold"],
                    "metric": metric,
                    "value": row[metric],
                }
            )
    return pd.DataFrame(records)


def plot_benchmark_core_metrics(receipts: Sequence[Json], output: Path) -> None:
    metric_specs = (
        ("quality", "SDMetrics quality"),
        ("rawValidFraction", "Raw proposal validity"),
        ("novelFraction", "Novelty vs. training fold"),
        ("uniqueFraction", "Within-output uniqueness"),
    )
    frame = metric_frame(receipts, [item[0] for item in metric_specs])
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    method_labels = [METHOD_LABELS[name] for name in METHOD_ORDER]
    palette = [METHOD_COLORS[name] for name in METHOD_ORDER]
    for axis, (metric, title) in zip(axes.flat, metric_specs, strict=True):
        subset = frame[frame["metric"] == metric]
        means = subset.groupby("method", sort=False)["value"].mean().reindex(method_labels)
        axis.bar(method_labels, means.values, color=palette, alpha=0.86)
        for method_index, method in enumerate(method_labels):
            values = subset.loc[subset["method"] == method, "value"].tolist()
            x_offsets = (-0.09, 0.0, 0.09)
            axis.scatter(
                [method_index + x_offsets[index] for index in range(len(values))],
                values,
                color="#222222",
                s=25,
                zorder=3,
            )
        axis.set_ylim(0, 1.08)
        axis.set_title(title)
        axis.set_ylabel("Fraction / score")
        axis.tick_params(axis="x", rotation=12)
        annotate_bars(axis, decimals=3)
    figure.suptitle(
        "Three-fold party-structure benchmark (bars = mean, dots = folds)",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output, bbox_inches="tight", metadata={"Software": "matplotlib"})
    plt.close(figure)


def plot_benchmark_quality_components(receipts: Sequence[Json], output: Path) -> None:
    frame = metric_frame(receipts, ["columnShapes", "columnPairTrends"])
    labels = {"columnShapes": "Column shapes", "columnPairTrends": "Column pair trends"}
    frame["component"] = frame["metric"].map(labels)
    figure, axis = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    sns.barplot(
        data=frame,
        x="method",
        y="value",
        hue="component",
        order=[METHOD_LABELS[name] for name in METHOD_ORDER],
        errorbar=None,
        palette=["#72B7B2", "#B279A2"],
        ax=axis,
    )
    sns.stripplot(
        data=frame,
        x="method",
        y="value",
        hue="component",
        order=[METHOD_LABELS[name] for name in METHOD_ORDER],
        dodge=True,
        jitter=False,
        palette={"Column shapes": "#222222", "Column pair trends": "#222222"},
        size=4,
        legend=False,
        ax=axis,
    )
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("")
    axis.set_ylabel("SDMetrics component score")
    axis.set_title("Quality decomposition: marginal shapes vs. pairwise relationships")
    axis.legend(title="")
    figure.savefig(output, bbox_inches="tight", metadata={"Software": "matplotlib"})
    plt.close(figure)


def plot_benchmark_operations(receipts: Sequence[Json], output: Path) -> None:
    rows = []
    for candidate in METHOD_ORDER:
        candidate_rows = [row for row in receipts if row["candidate"] == candidate]
        allocated = [
            float(row["cudaPeakAllocatedBytes"]) / (1024**2)
            for row in candidate_rows
            if row["cudaPeakAllocatedBytes"] is not None
        ]
        reserved = [
            float(row["cudaPeakReservedBytes"]) / (1024**2)
            for row in candidate_rows
            if row["cudaPeakReservedBytes"] is not None
        ]
        rows.append(
            {
                "candidate": candidate,
                "method": METHOD_LABELS[candidate],
                "totalFitSeconds": sum(float(row["fitSeconds"]) for row in candidate_rows),
                "meanSampleSeconds": fmean(float(row["sampleSeconds"]) for row in candidate_rows),
                "maxCudaAllocatedMiB": max(allocated) if allocated else 0.0,
                "maxCudaReservedMiB": max(reserved) if reserved else 0.0,
            }
        )
    frame = pd.DataFrame(rows)
    colors = [METHOD_COLORS[name] for name in METHOD_ORDER]
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    axes[0].bar(frame["method"], frame["totalFitSeconds"], color=colors, alpha=0.88)
    axes[0].set_yscale("symlog", linthresh=0.01)
    axes[0].set_ylabel("Total fit time across 3 folds (seconds, symlog)")
    axes[0].set_title("Fit cost")
    axes[0].tick_params(axis="x", rotation=12)
    for patch, value in zip(axes[0].patches, frame["totalFitSeconds"], strict=True):
        axes[0].text(
            patch.get_x() + patch.get_width() / 2,
            value * 1.15 if value >= 0.01 else value + 0.001,
            f"{value:.3g}s",
            ha="center",
            fontsize=8,
        )

    x_positions = list(range(len(frame)))
    width = 0.34
    axes[1].bar(
        [value - width / 2 for value in x_positions],
        frame["maxCudaAllocatedMiB"],
        width,
        label="Peak allocated",
        color="#E45756",
    )
    axes[1].bar(
        [value + width / 2 for value in x_positions],
        frame["maxCudaReservedMiB"],
        width,
        label="Peak reserved",
        color="#B279A2",
    )
    axes[1].set_xticks(x_positions, frame["method"], rotation=12)
    axes[1].set_ylabel("CUDA memory (MiB)")
    axes[1].set_title("Measured fit-stage CUDA peaks")
    axes[1].legend()
    axes[1].set_ylim(0, max(frame["maxCudaReservedMiB"]) * 1.22)
    for patch in axes[1].patches:
        if patch.get_height() > 0:
            axes[1].text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height() + 1.0,
                f"{patch.get_height():.1f}",
                ha="center",
                fontsize=8,
            )
    figure.suptitle("Benchmark runtime and GPU footprint", fontsize=15, fontweight="bold")
    figure.savefig(output, bbox_inches="tight", metadata={"Software": "matplotlib"})
    plt.close(figure)


def markdown_count_table(counts: Mapping[str, int], total: int) -> str:
    lines = ["| Value | Count | Share |", "|---|---:|---:|"]
    for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{key}` | {count} | {count / total:.1%} |")
    return "\n".join(lines)


def benchmark_table(candidate_summaries: Sequence[Mapping[str, Any]]) -> str:
    by_candidate = {str(row["candidate"]): row for row in candidate_summaries}
    lines = [
        "| Method | Quality | Raw valid | Output valid | Novel | Unique | Fit time (3 folds) | Device |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for candidate in METHOD_ORDER:
        row = by_candidate[candidate]
        lines.append(
            "| {label} | {quality:.4f} ± {std:.4f} | {valid:.1%} | 100.0% | "
            "{novel:.1%} | {unique:.1%} | {fit:.3f}s | `{devices}` |".format(
                label=METHOD_LABELS[candidate],
                quality=float(row["mean_quality"]),
                std=float(row["quality_stddev"]),
                valid=float(row["raw_valid_fraction"]),
                novel=float(row["novel_fraction"]),
                unique=float(row["unique_fraction"]),
                fit=float(row["total_fit_seconds"]),
                devices=", ".join(str(value) for value in row["devices"]),
            )
        )
    return "\n".join(lines)


def build_report(summary: Json) -> str:
    route = summary["routePilot"]
    support_audit = route["scenarioSupportAudit"]
    benchmark = summary["partyStructureBenchmark"]
    candidate_summaries = benchmark["candidateSummaries"]
    by_candidate = {row["candidate"]: row for row in candidate_summaries}
    empirical = by_candidate["empirical"]
    gaussian = by_candidate["gaussian_copula"]
    ctgan = by_candidate["ctgan"]
    tvae = by_candidate["tvae"]
    top_origin_code, top_origin_count = max(
        route["commercialOriginCountries"].items(), key=lambda item: (item[1], item[0])
    )
    top_destination_code, top_destination_count = max(
        route["commercialDestinationCountries"].items(), key=lambda item: (item[1], item[0])
    )
    endpoint_count = route["scenarioDocuments"] * 2
    train_rows = ", ".join(str(value) for value in benchmark["trainRowsPerFold"])
    validation_rows = ", ".join(str(value) for value in benchmark["validationRowsPerFold"])
    return f"""# MPCI B/L route and SDV synthesis audit

## Outcome

This artifact audits two completed, immutable experimental runs: the {route["scenarioDocuments"]}-scenario route pilot and the three-fold party-structure SDV benchmark. Source manifests and every manifest-listed file passed SHA-256 and byte-size verification before any metric or plot was produced.

**No production synthesizer was selected. No training-ready synthetic rows were published.** The route scenarios all carry `trainingEligible=false`; the route run published 0 training records. The benchmark evaluated synthetic rows in memory and retained receipts and hashes, not a production dataset.

## Scope and reproducibility

- Route input: `artifacts/kie-synthesis/{route["runId"]}`
- Benchmark input: `artifacts/kie-synthesis/{benchmark["runId"]}`
- Reproduction script: `tools/analyze_synthesis_route_sdv.py`
- Command: `TMPDIR=/tmp UV_CACHE_DIR=/tmp/documentparsing-uv-cache uv run --frozen --group analysis python tools/analyze_synthesis_route_sdv.py --project-root /mnt/d/projects/documentparsing`
- Machine-readable results: [summary.json](summary.json)
- Per-fold metrics: [benchmark-fold-metrics.csv](benchmark-fold-metrics.csv)
- Integrity receipt: [source-integrity.json](source-integrity.json)

## Route pilot

The route pilot generated {route["scenarioDocuments"]} scenarios from {route["distinctTemplates"]} distinct templates in {route["runtime"]["elapsedSeconds"]:.3f} seconds, with {route["runtime"]["peakRssMiB"]:.1f} MiB peak RSS. All {route["validation"]["strictSchemaValid"]} scenarios passed strict schema validation and all {route["validation"]["relationalInverseValid"]} passed relational inverse validation. It made {route["modelOrApiCalls"]} model/API calls.

The pilot is route-first: commercial origin, bilateral destination, physical loading/discharge endpoints, party route relations, party localities, freight, and issue/payment localities are sampled coherently. Country output used one canonical name per ISO code in the observed scenarios; no country aliases were sampled. Registry identities are pinned in `summary.json`.

![Country distributions](plots/route-country-distributions.png)

Origins span {len(route["commercialOriginCountries"])} countries and destinations span {len(route["commercialDestinationCountries"])}. The largest origin is `{top_origin_code}` ({top_origin_count}/{route["scenarioDocuments"]}); the largest destination is `{top_destination_code}` ({top_destination_count}/{route["scenarioDocuments"]}).

Country fitting is exact-identity and fail-closed. Of {support_audit["inputDocuments"]} train/template-isolated documents, {support_audit["countryCleanDocuments"]} were country-clean; {support_audit["excludedUnresolvedCountryDocuments"]} complete documents were excluded because {support_audit["unresolvedCountryCells"]} printed country cells did not resolve directly through the pinned ISO registry. Those source surfaces remain listed in the machine-readable audit and were never coerced through an alias table.

The support audit separately balances source-document eligibility and bilateral-flow eligibility. In particular, `{route["destinationTradeFlowExclusions"]["excludedDestinationWithoutLocalityDepthRecords"]}` trade-flow rows were explicitly excluded because the destination lacked populated-locality depth, and `{route["destinationTradeFlowExclusions"]["excludedDestinationWithoutDistinctPhysicalEndpointRecords"]}` were excluded because no distinct physical endpoint could be sampled. Neither destination counter is conflated with the source-side `exportCountryWithoutLocalityDepthDocuments` disposition.

### Port source modes

{markdown_count_table(route["portSourceCounts"], endpoint_count)}

There are {endpoint_count} port endpoints because every scenario has loading and discharge. Of the {route["observedSupportedPortEndpoints"]} endpoints backed by observed source-port support, {route["portSourceCounts"].get("registry_port_exploration", 0)} used registry exploration ({route["registryExplorationFractionWithinObservedSupport"]:.1%}). The remaining {route["portSourceCounts"].get("registry_port_no_observed_support", 0)} endpoints explicitly disclose that the source corpus had no matched observed-port support; they are not mislabeled as observed.

### Parties and localities

![Party conditioning](plots/route-party-conditioning.png)

The {route["scenarioDocuments"]} scenarios contain {route["partyRows"]} party occurrences. Locality modes are:

{markdown_count_table(route["partyLocalityModeTotals"], route["partyRows"])}

Of {sum(route["localitySourceCounts"].values())} non-missing party localities, {route["localitySourceCounts"].get("geonames_population_weighted", 0)} were sampled from the pinned GeoNames population-weighted registry and {route["localitySourceCounts"].get("endpoint", 0)} reuse a selected route endpoint. Missing locality remains explicit rather than receiving a hidden default. Party roles remain conditioned on route relation, as the heatmaps show.

### Freight, vessel, and route support

![Route statuses](plots/route-status-dashboard.png)

Freight arrangements are {route["freightArrangementCounts"].get("prepaid", 0)} prepaid, {route["freightArrangementCounts"].get("collect", 0)} collect, and {route["freightArrangementCounts"].get("missing", 0)} missing. A payment place is present in {route["freightPaymentPlaceCounts"]["present"]} scenarios. Place of issue is present in {route["placeOfIssueCounts"]["present"]} scenarios. Bilateral trade-flow evidence is attached to all {route["tradeFlowEvidenceDocuments"]} scenarios.

Vessel generation is intentionally unfinished: {route["vesselStatusCounts"].get("pending_pinned_vessel_registry", 0)} scenarios are marked `pending_pinned_vessel_registry`, while {route["vesselStatusCounts"].get("not_present", 0)} preserve vessel absence. The pilot contains {route["validation"]["directRoutes"]} direct and {route["validation"]["transshipmentRoutes"]} transshipment routes. These statuses are why the scenarios are not training eligible.

## Party-structure SDV benchmark

The benchmark used {benchmark["rows"]:,} PII-free party-structure rows from {benchmark["templates"]} template groups. Each candidate ran on three template-group-isolated folds; observed train-row counts were [{train_rows}] and validation-row counts were [{validation_rows}]. All {benchmark["completedCandidateRuns"]} requested runs completed, all diagnostic scores were 1.0, and every post-validation output row was valid. The environment used SDV {benchmark["environment"]["sdv"]} and SDMetrics {benchmark["environment"]["sdmetrics"]}.

{benchmark_table(candidate_summaries)}

![Core benchmark metrics](plots/sdv-core-metrics.png)

![Quality components](plots/sdv-quality-components.png)

![Runtime and GPU](plots/sdv-runtime-gpu.png)

### What the comparison demonstrates

- Empirical resampling has the highest mean quality ({empirical["mean_quality"]:.4f}) and perfect raw validity, but zero novelty. It is a fidelity control, not a way to expand support.
- TVAE is the strongest neural candidate on this limited party-structure view: mean quality {tvae["mean_quality"]:.4f}, raw proposal validity {tvae["raw_valid_fraction"]:.1%}, novelty {tvae["novel_fraction"]:.1%}, and uniqueness {tvae["unique_fraction"]:.1%}. Its three-fold fit cost was {tvae["total_fit_seconds"]:.2f}s.
- CTGAN trades quality ({ctgan["mean_quality"]:.4f}) and raw validity ({ctgan["raw_valid_fraction"]:.1%}) for higher novelty ({ctgan["novel_fraction"]:.1%}); it cost {ctgan["total_fit_seconds"]:.2f}s to fit across the three folds.
- Gaussian copula produces very high novelty ({gaussian["novel_fraction"]:.1%}) but only {gaussian["raw_valid_fraction"]:.1%} of raw proposals satisfy the structural constraints and its mean quality is {gaussian["mean_quality"]:.4f}. Deterministic rejection makes the published evaluation rows valid, but the low proposal yield is an efficiency and modeling-quality warning.
- Neural fitting ran on `{benchmark["gpuPreflight"]["device"]}` ({benchmark["gpuPreflight"]["device_name"]}); CPU candidates remained on CPU. The measured CUDA peaks are small because the benchmark view has only {len(benchmark["modeledColumns"])} structural columns and the observed fit-row counts shown above—not because full-document synthesis has been proven cheap.

### Decision boundary

These results do **not** justify selecting TVAE, CTGAN, or another method for production. They cover only the PII-free party-structure driver view; they do not validate party identities, text rendering, vessel entities, transshipment, complete label generation, or cross-table end-to-end fidelity. Selection requires a task-level benchmark after those components exist, with cohort coverage, privacy, downstream extraction benefit, and end-to-end inverse/rendering validation.

## Limitations and next gates

1. This route pilot is only {route["scenarioDocuments"]} scenarios ({route["validation"]["directRoutes"]} direct); its country and party distributions are a wiring/conditioning check, not a target production distribution.
2. Party names, addresses, contact strings, auxiliary/flavor PII, vessel entities, and raw-OCR text patching are absent. No PDF generation is in scope.
3. Transshipment routes generated: {route["validation"]["transshipmentRoutes"]}. A connectivity-backed transshipment source is still required before that branch can become training eligible.
4. The SDV benchmark models structure and presence/count profiles only. It neither emits party PII nor proves full-label utility.
5. Country aliases are not a synthesis feature. The pilot emits canonical registry-backed country surfaces; any observed-alias inventory remains audit-only and outside sampling.

## Integrity result

- Route source: {len(summary["sourceIntegrity"]["route"]["verifiedFiles"])} manifest-listed files verified.
- Benchmark source: {len(summary["sourceIntegrity"]["benchmark"]["verifiedFiles"])} manifest-listed files verified.
- Generated plots: all PNGs decoded successfully through Matplotlib and have non-zero dimensions.
- Artifact manifest: [manifest.json](manifest.json) records SHA-256 and byte size for every published file except itself.
"""


def write_fold_csv(path: Path, receipts: Sequence[Json]) -> None:
    fieldnames = [
        "candidate",
        "fold",
        "trainRows",
        "validationRows",
        "quality",
        "columnShapes",
        "columnPairTrends",
        "diagnostic",
        "rawValidFraction",
        "outputValidFraction",
        "novelFraction",
        "uniqueFraction",
        "fitSeconds",
        "sampleSeconds",
        "evaluationSeconds",
        "totalSeconds",
        "device",
        "cudaPeakAllocatedBytes",
        "cudaPeakReservedBytes",
        "rawProposals",
        "acceptedProposals",
        "outputRows",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(receipts)


def validate_plots(plot_paths: Sequence[Path]) -> list[Json]:
    validated = []
    for path in plot_paths:
        image = plt.imread(path)
        if image.ndim not in (2, 3) or image.shape[0] <= 0 or image.shape[1] <= 0:
            raise RuntimeError(f"invalid generated plot dimensions: {path}: {image.shape}")
        validated.append(
            {
                "path": path.name,
                "heightPixels": int(image.shape[0]),
                "widthPixels": int(image.shape[1]),
            }
        )
    return validated


def publish_manifest(root: Path) -> Json:
    files = []
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        if path.name == "manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sizeBytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest: Json = {
        "schemaVersion": 1,
        "status": "complete_no_production_selection_no_training_rows",
        "files": files,
    }
    (root / "manifest.json").write_text(stable_json(manifest), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    route_root = resolve_under(project_root, args.route_run).resolve()
    benchmark_root = resolve_under(project_root, args.benchmark_run).resolve()
    output_root = resolve_under(project_root, args.output).resolve()

    if output_root.exists():
        raise RuntimeError(f"immutable output already exists: {output_root}")
    staging_root = output_root.with_name(f".{output_root.name}.staging-{os.getpid()}")
    if staging_root.exists():
        raise RuntimeError(f"staging path already exists: {staging_root}")
    staging_root.mkdir(parents=True)
    (staging_root / "plots").mkdir()

    try:
        route_summary, _, route_integrity = audit_route_run(route_root)
        benchmark_summary, receipts, benchmark_integrity = audit_benchmark_run(benchmark_root)
        summary: Json = {
            "schemaVersion": 1,
            "status": "complete_no_production_selection_no_training_rows",
            "productionSelectionPerformed": False,
            "trainingReadySyntheticRowsPublished": 0,
            "routePilot": route_summary,
            "partyStructureBenchmark": benchmark_summary,
            "sourceIntegrity": {
                "route": route_integrity,
                "benchmark": benchmark_integrity,
            },
        }

        apply_plot_style()
        plot_paths = [
            staging_root / "plots/route-country-distributions.png",
            staging_root / "plots/route-party-conditioning.png",
            staging_root / "plots/route-status-dashboard.png",
            staging_root / "plots/sdv-core-metrics.png",
            staging_root / "plots/sdv-quality-components.png",
            staging_root / "plots/sdv-runtime-gpu.png",
        ]
        plot_country_distributions(route_summary, plot_paths[0])
        plot_party_conditioning(route_summary, plot_paths[1])
        plot_route_status(route_summary, plot_paths[2])
        plot_benchmark_core_metrics(receipts, plot_paths[3])
        plot_benchmark_quality_components(receipts, plot_paths[4])
        plot_benchmark_operations(receipts, plot_paths[5])
        summary["plotValidation"] = validate_plots(plot_paths)

        (staging_root / "summary.json").write_text(stable_json(summary), encoding="utf-8")
        (staging_root / "source-integrity.json").write_text(
            stable_json(summary["sourceIntegrity"]), encoding="utf-8"
        )
        write_fold_csv(staging_root / "benchmark-fold-metrics.csv", receipts)
        (staging_root / "report.md").write_text(build_report(summary), encoding="utf-8")
        publish_manifest(staging_root)
        staging_root.rename(output_root)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    print(
        stable_json(
            {
                "status": "complete",
                "output": str(output_root),
                "productionSelectionPerformed": False,
                "trainingReadySyntheticRowsPublished": 0,
            }
        ),
        end="",
    )


if __name__ == "__main__":
    main()
