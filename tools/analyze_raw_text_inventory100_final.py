#!/usr/bin/env python3
"""Publish a detailed, reproducible audit of a committed 100-document rewrite run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{number}")
        output.append(value)
    return output


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_committed_run(root: Path) -> dict[str, Any]:
    commit_path = root / "_COMMIT.json"
    if not commit_path.is_file():
        raise ValueError(f"run is not committed: {root}")
    commit = _json(commit_path)
    artifacts = commit.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"run commit has no artifact inventory: {commit_path}")
    for row in artifacts:
        if not isinstance(row, dict) or not isinstance(row.get("relative_path"), str):
            raise ValueError(f"invalid run commit row: {commit_path}")
        path = root / row["relative_path"]
        if not path.is_file():
            raise ValueError(f"committed artifact is missing: {path}")
        if path.stat().st_size != row.get("bytes") or _sha256(path) != row.get("sha256"):
            raise ValueError(f"committed artifact differs from receipt: {path}")
    return _json(root / "summary.json")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cost(usage: dict[str, Any]) -> float:
    reported = usage.get("providerReportedCostUsd")
    return float(reported if reported is not None else usage["estimatedCostUsd"])


def _target_features(target: dict[str, Any]) -> dict[str, int]:
    patch = target.get("documentPatch") or {}
    parties = patch.get("parties") or {}
    containers = patch.get("containers") or []
    cargo_groups = patch.get("cargoGroups") or []
    packages = patch.get("cargoPackages") or []
    allocation_groups = patch.get("cargoAllocationGroups") or []
    allocations = sum(len(row.get("allocations") or []) for row in allocation_groups)
    dg_rows = sum(len(row.get("dangerousGoods") or []) for row in cargo_groups)
    hs_codes = sum(len(row.get("hsCodes") or []) for row in cargo_groups)
    info_rows = sum(len(row.get("additionalInformation") or []) for row in cargo_groups)
    marks = sum(len(row.get("marksAndNumbers") or []) for row in cargo_groups)
    notify = parties.get("notifyParties") or []
    named_party_roles = sum(
        isinstance(value, dict) and bool(value.get("name"))
        for key, value in parties.items()
        if key != "notifyParties"
    ) + sum(isinstance(value, dict) and bool(value.get("name")) for value in notify)
    return {
        "containers": len(containers),
        "cargo_groups": len(cargo_groups),
        "packages": len(packages),
        "allocation_groups": len(allocation_groups),
        "allocations": allocations,
        "dangerous_goods": dg_rows,
        "hs_codes": hs_codes,
        "additional_information": info_rows,
        "marks": marks,
        "notify_parties": len(notify),
        "named_party_roles": named_party_roles,
        "temperature_containers": sum(
            isinstance(row, dict) and row.get("temperatureSetpoint") is not None
            for row in containers
        ),
    }


def _semantic_value_rows(
    *, document_id: str, variant: str, label: dict[str, Any]
) -> list[dict[str, str]]:
    """Flatten task-facing categorical values for source/target distribution checks."""

    patch = label.get("documentPatch") or {}
    output: list[dict[str, str]] = []

    def add(dimension: str, value: Any, role: str = "") -> None:
        if isinstance(value, str) and value.strip():
            output.append(
                {
                    "document_id": document_id,
                    "variant": variant,
                    "dimension": dimension,
                    "role": role,
                    "value": value.strip(),
                }
            )

    for row in patch.get("cargoPackages") or ():
        if isinstance(row, dict):
            add("package_type", row.get("typeCategory"), str(row.get("groupId") or ""))
    for row in patch.get("containers") or ():
        if not isinstance(row, dict):
            continue
        add("container_type", row.get("typeCategory"))
        add("container_size", row.get("sizeCategory"))
        if row.get("temperatureSetpoint") is not None:
            add("thermal_container", "temperature_setpoint_present")
    parties = patch.get("parties") or {}
    if isinstance(parties, dict):
        for role, raw_parties in parties.items():
            values = (
                raw_parties
                if role == "notifyParties" and isinstance(raw_parties, list)
                else [raw_parties]
            )
            for row in values:
                if isinstance(row, dict):
                    add("party_country", row.get("country"), role)
    route = patch.get("route") or {}
    if isinstance(route, dict):
        for role, row in route.items():
            if isinstance(row, dict):
                add("route_country", row.get("country"), role)
    for group in patch.get("cargoGroups") or ():
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("groupId") or "")
        for code in group.get("hsCodes") or ():
            if isinstance(code, str) and len(code) >= 2 and code[:2].isdigit():
                add("hs_chapter", code[:2], group_id)
        for row in group.get("dangerousGoods") or ():
            if isinstance(row, dict):
                add("hazard_category", row.get("hazardCategory"), group_id)
    return output


def _edit_metrics(source: str, final: str) -> dict[str, Any]:
    source_lines = source.splitlines()
    final_lines = final.splitlines()
    matcher = SequenceMatcher(a=source_lines, b=final_lines, autojunk=False)
    equal = changed_source = changed_final = blocks = 0
    for tag, first_start, first_end, second_start, second_end in matcher.get_opcodes():
        if tag == "equal":
            equal += first_end - first_start
            continue
        blocks += 1
        changed_source += first_end - first_start
        changed_final += second_end - second_start
    source_pages = tuple(
        line for line in source_lines if re.fullmatch(r"--- PAGE [0-9]+ ---", line)
    )
    final_pages = tuple(
        line for line in final_lines if re.fullmatch(r"--- PAGE [0-9]+ ---", line)
    )
    denominator = max(len(source_lines), len(final_lines), 1)
    return {
        "source_chars": len(source),
        "final_chars": len(final),
        "source_physical_lines": len(source_lines),
        "final_physical_lines": len(final_lines),
        "line_count_delta": len(final_lines) - len(source_lines),
        "blank_line_delta": sum(not line.strip() for line in final_lines)
        - sum(not line.strip() for line in source_lines),
        "changed_source_lines": changed_source,
        "changed_final_lines": changed_final,
        "changed_blocks": blocks,
        "unchanged_line_fraction": equal / denominator,
        "page_markers_preserved": source_pages == final_pages,
        "terminal_newline_preserved": source.endswith("\n") == final.endswith("\n"),
    }


def _select_case_roots(
    run_root: Path,
    recovery_roots: list[Path],
) -> tuple[list[tuple[str, Path, str, bool]], list[dict[str, Any]]]:
    base_summary = _verify_committed_run(run_root)
    base_results = _jsonl(run_root / "generation/results.jsonl")
    ordered_ids = [row["documentId"] for row in base_results]
    if len(ordered_ids) != 100 or len(set(ordered_ids)) != 100:
        raise ValueError("base run must contain the fixed 100-document cohort")
    selected: dict[str, tuple[Path, str, bool]] = {
        document_id: (run_root / "cases" / document_id, base_summary["runId"], False)
        for document_id in ordered_ids
    }
    base_status = {row["documentId"]: row["status"] for row in base_results}
    lineage = [base_summary]
    recovered: set[str] = set()
    for recovery_root in recovery_roots:
        summary = _verify_committed_run(recovery_root)
        lineage.append(summary)
        for row in _jsonl(recovery_root / "generation/results.jsonl"):
            document_id = row["documentId"]
            if document_id not in selected:
                raise ValueError(f"recovery contains a document outside the cohort: {document_id}")
            if row["status"] != "training_ready":
                continue
            if base_status[document_id] == "training_ready":
                raise ValueError(f"recovery redundantly replaces a base success: {document_id}")
            if document_id in recovered:
                raise ValueError(f"multiple accepted recoveries exist for: {document_id}")
            base_case = run_root / "cases" / document_id
            recovery_case = recovery_root / "cases" / document_id
            for name in ("source.txt", "source-label.json", "target-label.json"):
                if _sha256(base_case / name) != _sha256(recovery_case / name):
                    raise ValueError(
                        f"recovery changes the pinned source or target for {document_id}: {name}"
                    )
            selected[document_id] = (recovery_case, summary["runId"], True)
            recovered.add(document_id)
    output = [
        (document_id, *selected[document_id])
        for document_id in ordered_ids
    ]
    unresolved = [
        document_id
        for document_id, case_root, _run_id, _recovered in output
        if _json(case_root / "result.json")["status"] != "training_ready"
    ]
    if unresolved:
        raise ValueError(f"composite still has unresolved documents: {unresolved}")
    return output, lineage


def _case_rows(
    selections: list[tuple[str, Path, str, bool]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, str]],
]:
    cases: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []
    semantic_values: list[dict[str, str]] = []
    for document_id, case_root, source_run_id, recovered in selections:
        result = _json(case_root / "result.json")
        source = (case_root / "source.txt").read_text(encoding="utf-8")
        final = (case_root / "final.txt").read_text(encoding="utf-8")
        source_label = _json(case_root / "source-label.json")
        target = _json(case_root / "target-label.json")
        semantic_values.extend(
            _semantic_value_rows(
                document_id=document_id, variant="source", label=source_label
            )
        )
        semantic_values.extend(
            _semantic_value_rows(
                document_id=document_id, variant="synthetic_target", label=target
            )
        )
        contract_path = case_root / "contract.json"
        contract = _json(contract_path) if contract_path.is_file() else {}
        stage_values = json.loads((case_root / "stages.json").read_text(encoding="utf-8"))
        if not isinstance(stage_values, list):
            raise ValueError(f"expected stage array: {case_root / 'stages.json'}")
        usage = result["usage"]
        features = _target_features(target)
        edits = _edit_metrics(source, final)
        semantic_changes = (contract.get("targetIntegrity") or {}).get("semantic_changes") or []
        projection_reasons = Counter(
            row.get("reason") for row in semantic_changes if isinstance(row, dict)
        )
        cases.append(
            {
                "document_id": document_id,
                "source_run_id": source_run_id,
                "recovered": recovered,
                "status": result["status"],
                "reason": result["reason"],
                "source_lines": result["sourceLines"],
                "model_lines": result["modelLines"],
                "compiler_work_items": result["compilerWorkItems"],
                "inventory_candidates": result["inventoryCandidates"],
                "deterministic_inventory_edits": result["deterministicInventoryEdits"],
                "legacy_residual_candidates": result["legacyResidualCandidates"],
                "requests": usage["requests"],
                "input_tokens": usage["inputTokens"],
                "reasoning_tokens": usage["reasoningTokens"],
                "visible_tokens": usage["visibleOutputTokens"],
                "output_tokens": usage["outputTokens"],
                "cost_usd": _cost(usage),
                "host_rewrite_passed": result["hostRewriteAuditPassed"],
                "full_document_passed": bool(
                    result.get("fullDocumentAudit") and result["fullDocumentAudit"]["passed"]
                ),
                "target_integrity_changes": len(semantic_changes),
                "equipment_topology_projections": projection_reasons[
                    "unprinted_equipment_topology"
                ],
                **features,
                **edits,
            }
        )
        for row in semantic_changes:
            if not isinstance(row, dict):
                continue
            projections.append(
                {
                    "document_id": document_id,
                    "source_run_id": source_run_id,
                    "recovered": recovered,
                    "path": row.get("path"),
                    "reason": row.get("reason"),
                    "before": row.get("before"),
                    "after": row.get("after"),
                }
            )
        for stage in stage_values:
            if not isinstance(stage, dict):
                continue
            stage_usage = stage["usage"]
            stages.append(
                {
                    "document_id": document_id,
                    "source_run_id": source_run_id,
                    "recovered": recovered,
                    "semantic_attempt": stage["semanticAttempt"],
                    "route_round": stage["routeRound"],
                    "route_attempt": stage["routeAttempt"],
                    "route_provider": stage["routeProvider"],
                    "transport_outcome": "error" if stage["errorType"] else "response",
                    "error_type": stage["errorType"],
                    "downstream_providers": ";".join(stage_usage["downstreamProviders"]),
                    "latency_seconds": float(stage["completedAtUnixSeconds"])
                    - float(stage["startedAtUnixSeconds"]),
                    "input_tokens": stage_usage["inputTokens"],
                    "reasoning_tokens": stage_usage["reasoningTokens"],
                    "visible_tokens": stage_usage["visibleOutputTokens"],
                    "cost_usd": _cost(stage_usage),
                }
            )
    return cases, stages, projections, semantic_values


def _comparison_rows(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        summary = _verify_committed_run(root)
        documents = int(summary["liveDocuments"])
        reported = summary.get("providerReportedCostUsd")
        cost = float(reported if reported is not None else summary["estimatedCostUsd"])
        rows.append(
            {
                "run_id": summary["runId"],
                "run_label": re.search(r"v[0-9]+[^/]*", summary["runId"]).group(0),
                "documents": documents,
                "training_ready": summary["trainingReadyDocuments"],
                "needs_review": summary["needsReviewDocuments"],
                "call_failed": summary["callFailedDocuments"],
                "compiler_blocked": summary["compilerBlockedDocuments"],
                "pass_fraction": summary["trainingReadyDocuments"] / documents,
                "requests": summary["requests"],
                "requests_per_document": summary["requests"] / documents,
                "input_tokens_per_document": summary["inputTokens"] / documents,
                "reasoning_tokens_per_document": summary["reasoningTokens"] / documents,
                "visible_tokens_per_document": summary["visibleOutputTokens"] / documents,
                "cost_usd": cost,
                "cost_per_1000_usd": cost / documents * 1000,
                "provider_attempts": summary["providerAttempts"],
                "failed_provider_attempts": summary["failedProviderAttempts"],
                "throughput_documents_per_hour": summary["throughputDocumentsPerHour"],
                "wall_seconds": summary["wallSeconds"],
                "cost_basis": "provider_reported" if reported is not None else "pinned_estimate",
            }
        )
    return rows


def _composite_comparison_row(
    *,
    composite_id: str,
    cases: list[dict[str, Any]],
    stages: list[dict[str, Any]],
) -> dict[str, Any]:
    documents = len(cases)
    cost = sum(float(row["cost_usd"]) for row in cases)
    return {
        "run_id": composite_id,
        "run_label": "final composite",
        "documents": documents,
        "training_ready": sum(row["status"] == "training_ready" for row in cases),
        "needs_review": sum(row["status"] == "needs_review" for row in cases),
        "call_failed": sum(row["status"] == "call_failed" for row in cases),
        "compiler_blocked": sum(row["status"] == "compiler_blocked" for row in cases),
        "pass_fraction": sum(row["status"] == "training_ready" for row in cases) / documents,
        "requests": sum(int(row["requests"]) for row in cases),
        "requests_per_document": sum(int(row["requests"]) for row in cases) / documents,
        "input_tokens_per_document": sum(int(row["input_tokens"]) for row in cases)
        / documents,
        "reasoning_tokens_per_document": sum(int(row["reasoning_tokens"]) for row in cases)
        / documents,
        "visible_tokens_per_document": sum(int(row["visible_tokens"]) for row in cases)
        / documents,
        "cost_usd": cost,
        "cost_per_1000_usd": cost / documents * 1000,
        "provider_attempts": len(stages),
        "failed_provider_attempts": sum(
            row["transport_outcome"] == "error" for row in stages
        ),
        # The selected cases came from several concurrent immutable runs.  Combining their wall
        # clocks would not estimate a clean batch's throughput, so this is deliberately absent.
        "throughput_documents_per_hour": math.nan,
        "wall_seconds": math.nan,
        "cost_basis": "provider_reported_selected_accepted_outputs",
    }


def _save(path: Path) -> None:
    plt.savefig(path, dpi=190, bbox_inches="tight")
    plt.close()


def _plots(
    cases: pd.DataFrame,
    stages: pd.DataFrame,
    projections: pd.DataFrame,
    semantic_values: pd.DataFrame,
    comparisons: pd.DataFrame,
    manual: list[dict[str, Any]],
    output: Path,
) -> None:
    root = output / "plots"
    root.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    outcome = comparisons.melt(
        id_vars=["run_label"],
        value_vars=["training_ready", "needs_review", "call_failed", "compiler_blocked"],
        var_name="outcome",
        value_name="documents",
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=outcome, x="run_label", y="documents", hue="outcome")
    plt.title("Rewrite outcomes on the same 100-document cohort")
    plt.xlabel("")
    plt.ylabel("documents")
    plt.xticks(rotation=18, ha="right")
    _save(root / "01_outcome_comparison.png")

    _figure, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    sns.barplot(data=comparisons, x="run_label", y="cost_per_1000_usd", ax=axes[0])
    sns.barplot(data=comparisons, x="run_label", y="requests_per_document", ax=axes[1])
    sns.barplot(
        data=comparisons, x="run_label", y="throughput_documents_per_hour", ax=axes[2]
    )
    axes[0].set_ylabel("USD / 1,000 attempted")
    axes[1].set_ylabel("requests / document")
    axes[2].set_ylabel("documents / hour")
    for axis, title in zip(axes, ("Cost", "Model calls", "Throughput"), strict=True):
        axis.set_title(title)
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=25)
    _save(root / "02_efficiency_comparison.png")

    token_frame = comparisons.melt(
        id_vars="run_label",
        value_vars=[
            "input_tokens_per_document",
            "reasoning_tokens_per_document",
            "visible_tokens_per_document",
        ],
        var_name="token_class",
        value_name="tokens_per_document",
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=token_frame, x="run_label", y="tokens_per_document", hue="token_class")
    plt.title("Token load per attempted document")
    plt.xlabel("")
    plt.ylabel("tokens")
    plt.xticks(rotation=18, ha="right")
    _save(root / "03_token_comparison.png")

    _figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.countplot(data=cases, x="requests", discrete=True, ax=axes[0])
    sns.histplot(data=cases, x="cost_usd", bins=18, kde=True, ax=axes[1])
    axes[0].set_title("Successful model responses per document")
    axes[0].set_ylabel("documents")
    axes[1].set_title("Per-document provider cost")
    axes[1].set_xlabel("USD")
    _save(root / "04_request_and_cost_distributions.png")

    ordered = cases.sort_values("cost_usd", ascending=False).reset_index(drop=True)
    ordered["documents"] = range(1, len(ordered) + 1)
    ordered["cumulative_cost_fraction"] = ordered["cost_usd"].cumsum() / ordered[
        "cost_usd"
    ].sum()
    plt.figure(figsize=(10, 5.5))
    sns.lineplot(data=ordered, x="documents", y="cumulative_cost_fraction", marker="o")
    plt.axhline(0.5, color="gray", linestyle="--")
    plt.title("Cost concentration (documents ordered most expensive first)")
    plt.ylabel("cumulative cost fraction")
    _save(root / "05_cost_pareto.png")

    _figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.scatterplot(data=cases, x="source_lines", y="cost_usd", hue="requests", ax=axes[0])
    sns.scatterplot(data=cases, x="model_lines", y="cost_usd", hue="requests", ax=axes[1])
    axes[0].set_title("Source length and cost")
    axes[1].set_title("Model-owned line scope and cost")
    for axis in axes:
        axis.set_ylabel("USD")
    _save(root / "06_scope_vs_cost.png")

    _figure, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    sns.histplot(data=cases, x="unchanged_line_fraction", bins=18, ax=axes[0])
    sns.histplot(data=cases, x="changed_source_lines", bins=18, ax=axes[1])
    sns.histplot(data=cases, x="changed_blocks", bins=18, ax=axes[2])
    axes[0].set_title("Unchanged-line fraction")
    axes[1].set_title("Changed source lines")
    axes[2].set_title("Disjoint edit blocks")
    _save(root / "07_edit_footprint.png")

    topology = cases.melt(
        id_vars="document_id",
        value_vars=["containers", "cargo_groups", "packages", "allocations", "named_party_roles"],
        var_name="feature",
        value_name="count",
    )
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=topology, x="feature", y="count", showfliers=False)
    sns.stripplot(data=topology, x="feature", y="count", color="black", alpha=0.25, size=3)
    plt.title("Synthetic target topology")
    plt.xlabel("")
    plt.ylabel("count per document")
    _save(root / "08_target_topology.png")

    prevalence = pd.DataFrame(
        [
            {"feature": label, "documents": int((cases[column] > 0).sum())}
            for column, label in (
                ("dangerous_goods", "dangerous goods"),
                ("temperature_containers", "temperature"),
                ("hs_codes", "HS code"),
                ("additional_information", "additional information"),
                ("marks", "marks and numbers"),
                ("equipment_topology_projections", "equipment projection"),
            )
        ]
    )
    plt.figure(figsize=(11, 5.5))
    sns.barplot(data=prevalence, y="feature", x="documents", hue="feature", legend=False)
    plt.title("Special-feature coverage in the 100 synthetic targets")
    plt.xlabel("documents")
    plt.ylabel("")
    _save(root / "09_feature_prevalence.png")

    numeric = cases[
        [
            "cost_usd",
            "requests",
            "input_tokens",
            "reasoning_tokens",
            "source_lines",
            "model_lines",
            "inventory_candidates",
            "changed_source_lines",
            "containers",
            "cargo_groups",
            "packages",
            "named_party_roles",
        ]
    ].corr(numeric_only=True)
    plt.figure(figsize=(12, 9))
    sns.heatmap(numeric, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f")
    plt.title("Complexity, token, edit, and cost correlations")
    _save(root / "10_correlation_heatmap.png")

    provider = stages.groupby(
        ["route_provider", "transport_outcome"], dropna=False
    ).size().reset_index(name="attempts")
    plt.figure(figsize=(11, 5.5))
    sns.barplot(data=provider, x="route_provider", y="attempts", hue="transport_outcome")
    plt.title("Configured provider-route attempts")
    plt.xlabel("")
    plt.ylabel("attempts")
    _save(root / "11_provider_routing.png")

    responses = stages[stages["transport_outcome"] == "response"]
    plt.figure(figsize=(11, 5.5))
    sns.boxplot(data=responses, x="route_provider", y="latency_seconds")
    sns.stripplot(data=responses, x="route_provider", y="latency_seconds", color="black", alpha=0.3)
    plt.title("Successful provider latency")
    plt.xlabel("")
    plt.ylabel("seconds")
    _save(root / "12_provider_latency.png")

    if not projections.empty:
        counts = projections.groupby("reason").size().reset_index(name="changes")
        plt.figure(figsize=(11, 5.5))
        sns.barplot(data=counts, y="reason", x="changes", hue="reason", legend=False)
        plt.title("Audited target-integrity projections")
        plt.xlabel("changed fields")
        plt.ylabel("")
        _save(root / "13_target_integrity_changes.png")

    if manual:
        audit_frame = pd.DataFrame(manual)
        measures = pd.DataFrame(
            [
                {"check": field, "passed": int(audit_frame[field].sum())}
                for field in ("semanticConsistency", "templateFidelity", "privacyRewritten")
            ]
        )
        plt.figure(figsize=(10, 5.5))
        sns.barplot(data=measures, x="check", y="passed", hue="check", legend=False)
        plt.axhline(len(audit_frame), color="gray", linestyle="--")
        plt.title(f"Stratified manual audit (n={len(audit_frame)})")
        plt.ylabel("documents passing")
        plt.xlabel("")
        _save(root / "14_manual_audit.png")

    def paired_distribution(
        *, dimension: str, filename: str, title: str, maximum_values: int = 24
    ) -> None:
        values = semantic_values[semantic_values["dimension"] == dimension]
        if values.empty:
            return
        ordered = values.groupby("value").size().sort_values(ascending=False)
        retained = set(ordered.head(maximum_values).index)
        values = values[values["value"].isin(retained)]
        counts = (
            values.groupby(["value", "variant"])
            .size()
            .reset_index(name="occurrences")
        )
        order = list(ordered.loc[list(retained)].sort_values(ascending=True).index)
        height = max(6.0, min(13.0, 0.4 * len(order) + 2.0))
        plt.figure(figsize=(13, height))
        sns.barplot(
            data=counts,
            y="value",
            x="occurrences",
            hue="variant",
            order=order,
        )
        plt.title(title)
        plt.xlabel("field occurrences")
        plt.ylabel("")
        _save(root / filename)

    paired_distribution(
        dimension="package_type",
        filename="15_package_type_distribution.png",
        title="Source vs synthetic package categories",
    )
    paired_distribution(
        dimension="container_type",
        filename="16_container_type_distribution.png",
        title="Source vs synthetic container categories",
    )
    paired_distribution(
        dimension="route_country",
        filename="17_route_country_distribution.png",
        title="Source vs synthetic route-country surfaces",
        maximum_values=20,
    )
    paired_distribution(
        dimension="hs_chapter",
        filename="18_hs_chapter_distribution.png",
        title="Source vs synthetic HS chapters",
        maximum_values=20,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, action="append", default=[])
    parser.add_argument("--comparison-root", type=Path, action="append", default=[])
    parser.add_argument("--composite-id", required=True)
    parser.add_argument("--manual-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve(strict=True)
    recovery_roots = [root.resolve(strict=True) for root in args.recovery_root]
    summary = _verify_committed_run(run_root)
    if summary["liveDocuments"] != 100:
        raise ValueError("analysis requires the fixed 100-document cohort")
    selections, lineage = _select_case_roots(run_root, recovery_roots)
    manual_payload = _json(args.manual_audit.resolve(strict=True))
    if manual_payload.get("compositeId") != args.composite_id:
        raise ValueError("manual-audit composite ID differs from the requested analysis")
    manual = manual_payload.get("cases")
    if not isinstance(manual, list) or not manual:
        raise ValueError("manual audit must contain at least one reviewed case")

    cases, stages, projections, semantic_values = _case_rows(selections)
    if len(cases) != 100 or len({row["document_id"] for row in cases}) != 100:
        raise ValueError("run does not contain 100 unique case results")
    unknown_manual = {row.get("documentId") for row in manual} - {
        row["document_id"] for row in cases
    }
    if unknown_manual:
        raise ValueError(f"manual audit refers to unknown documents: {sorted(unknown_manual)}")
    for row in manual:
        required = {"semanticConsistency", "templateFidelity", "privacyRewritten"}
        if any(not isinstance(row.get(field), bool) for field in required):
            raise ValueError(f"manual audit has incomplete booleans: {row.get('documentId')}")

    comparison_roots = [root.resolve(strict=True) for root in args.comparison_root]
    if run_root not in comparison_roots:
        comparison_roots.append(run_root)
    comparison_rows = _comparison_rows(comparison_roots)
    comparison_rows.append(
        _composite_comparison_row(
            composite_id=args.composite_id,
            cases=cases,
            stages=stages,
        )
    )
    case_frame = pd.DataFrame(cases)
    stage_frame = pd.DataFrame(stages)
    projection_frame = pd.DataFrame(projections)
    semantic_value_frame = pd.DataFrame(semantic_values)
    comparison_frame = pd.DataFrame(comparison_rows)

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"analysis output already exists: {output}")
    output.mkdir(parents=True)
    _write_csv(output / "cases.csv", cases)
    _write_csv(output / "provider-attempts.csv", stages)
    _write_csv(output / "target-integrity-changes.csv", projections)
    _write_csv(output / "semantic-value-distributions.csv", semantic_values)
    _write_csv(output / "run-comparison.csv", comparison_rows)
    lineage_rows = [
        {
            "run_id": row["runId"],
            "documents": row["liveDocuments"],
            "training_ready": row["trainingReadyDocuments"],
            "call_failed": row["callFailedDocuments"],
            "compiler_blocked": row["compilerBlockedDocuments"],
            "requests": row["requests"],
            "provider_attempts": row["providerAttempts"],
            "failed_provider_attempts": row["failedProviderAttempts"],
            "provider_cost_usd": row.get("providerReportedCostUsd")
            or row["estimatedCostUsd"],
            "wall_seconds": row["wallSeconds"],
        }
        for row in lineage
    ]
    _write_csv(output / "execution-lineage.csv", lineage_rows)
    _write_csv(
        output / "manual-audit.csv",
        [
            {
                "document_id": row["documentId"],
                "semantic_consistency": row["semanticConsistency"],
                "template_fidelity": row["templateFidelity"],
                "privacy_rewritten": row["privacyRewritten"],
                "selection_basis": ";".join(row["selectionBasis"]),
                "notes": row["notes"],
            }
            for row in manual
        ],
    )

    cost_total = float(case_frame["cost_usd"].sum())
    lineage_cost_total = sum(float(row["provider_cost_usd"]) for row in lineage_rows)
    status_counts = Counter(case_frame["status"])
    projection_counts = Counter(row["reason"] for row in projections)
    provider_counts = Counter(
        row["downstream_providers"]
        for row in stages
        if row["transport_outcome"] == "response" and row["downstream_providers"]
    )
    failed_attempts = sum(row["transport_outcome"] == "error" for row in stages)
    manual_passes = sum(
        row["semanticConsistency"] and row["templateFidelity"] and row["privacyRewritten"]
        for row in manual
    )
    top_cost_count = max(1, math.ceil(len(cases) * 0.1))
    top_cost_share = float(
        case_frame.nlargest(top_cost_count, "cost_usd")["cost_usd"].sum() / cost_total
    )
    analysis = {
        "schemaVersion": 1,
        "compositeId": args.composite_id,
        "baseRunId": summary["runId"],
        "runCommitSha256": _sha256(run_root / "_COMMIT.json"),
        "recoveryRunIds": [row["runId"] for row in lineage[1:]],
        "manualAuditSha256": _sha256(args.manual_audit.resolve(strict=True)),
        "documents": len(cases),
        "statusCounts": dict(status_counts),
        "hostAndFullDocumentPasses": int(
            (case_frame["host_rewrite_passed"] & case_frame["full_document_passed"]).sum()
        ),
        "pageMarkersPreserved": int(case_frame["page_markers_preserved"].sum()),
        "terminalNewlinePreserved": int(case_frame["terminal_newline_preserved"].sum()),
        "zeroPhysicalLineCountDelta": int((case_frame["line_count_delta"] == 0).sum()),
        "zeroBlankLineCountDelta": int((case_frame["blank_line_delta"] == 0).sum()),
        "meanUnchangedLineFraction": float(case_frame["unchanged_line_fraction"].mean()),
        "medianUnchangedLineFraction": float(case_frame["unchanged_line_fraction"].median()),
        "providerReportedOrPinnedCostUsd": cost_total,
        "costPerThousandAttemptedUsd": cost_total / len(cases) * 1000,
        "actualExecutionLineageCostUsd": lineage_cost_total,
        "actualExecutionLineageCostPerThousandBaseDocumentsUsd": (
            lineage_cost_total / len(cases) * 1000
        ),
        "recoveredDocuments": int(case_frame["recovered"].sum()),
        "costPerThousandTrainingReadyUsd": (
            cost_total / status_counts["training_ready"] * 1000
            if status_counts["training_ready"]
            else None
        ),
        "topDecileCostShare": top_cost_share,
        "requests": int(case_frame["requests"].sum()),
        "documentsRequiringRepairResponse": int((case_frame["requests"] > 1).sum()),
        "inputTokens": int(case_frame["input_tokens"].sum()),
        "reasoningTokens": int(case_frame["reasoning_tokens"].sum()),
        "visibleOutputTokens": int(case_frame["visible_tokens"].sum()),
        "failedProviderAttempts": failed_attempts,
        "successfulDownstreamProviders": dict(provider_counts),
        "targetIntegrityChangeCounts": dict(projection_counts),
        "manualAuditDocuments": len(manual),
        "manualAuditPasses": manual_passes,
        "trainingRecordsPublished": summary["trainingRecordsPublished"],
    }
    (output / "summary.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plots(
        case_frame,
        stage_frame,
        projection_frame,
        semantic_value_frame,
        comparison_frame,
        manual,
        output,
    )

    report = [
        "# Synthetic raw-OCR rewrite: final 100-document validation",
        "",
        "## Result",
        "",
        f"- Outcomes: **{dict(status_counts)}**.",
        f"- Composite provenance: **{100 - analysis['recoveredDocuments']} base-run** outputs and "
        f"**{analysis['recoveredDocuments']} accepted targeted recoveries**.",
        f"- Both deterministic rewrite and independent full-document audits passed for "
        f"**{analysis['hostAndFullDocumentPasses']}/100** documents.",
        f"- Provider cost: **${cost_total:.8f}**, or "
        f"**${analysis['costPerThousandAttemptedUsd']:.3f}/1,000 selected accepted outputs**.",
        f"- Actual base-plus-recovery execution spend: **${lineage_cost_total:.8f}**, or "
        f"**${analysis['actualExecutionLineageCostPerThousandBaseDocumentsUsd']:.3f}/1,000 "
        "base documents**. This includes failed diagnostic recoveries and is not the expected "
        "clean-run rate.",
        f"- Calls: **{analysis['requests']}** successful responses; "
        f"**{analysis['documentsRequiringRepairResponse']}** documents needed a correction "
        "response.",
        f"- Tokens: **{analysis['inputTokens']:,} input**, "
        f"**{analysis['reasoningTokens']:,} reasoning**, and "
        f"**{analysis['visibleOutputTokens']:,} visible output**.",
        f"- Manual stratified audit: **{manual_passes}/{len(manual)}** passed semantic, "
        "template-fidelity, and privacy checks together.",
        f"- Frozen historical false-pass rejection: **{summary['glmFalsePassesRejected']}/12 GLM** "
        f"and **{summary['lunaFalsePassesRejected']}/12 Luna**.",
        "- Training records published: **0**.",
        "",
        "## Template fidelity",
        "",
        f"- Page marker sequence preserved: **{analysis['pageMarkersPreserved']}/100**.",
        f"- Terminal newline convention preserved: **{analysis['terminalNewlinePreserved']}/100**.",
        f"- Physical line count unchanged: **{analysis['zeroPhysicalLineCountDelta']}/100**; "
        f"blank-line count unchanged: **{analysis['zeroBlankLineCountDelta']}/100**.",
        f"- Median unchanged-line fraction: **{analysis['medianUnchangedLineFraction']:.1%}**.",
        "",
        "A changed line is expected whenever a source value is replaced. The fidelity measures "
        "above test document structure, not lexical identity. Semantic acceptance additionally "
        "requires every target value, role, repeated relationship, source-only private value, "
        "and frozen negative fixture to satisfy the host contract.",
        "",
        "## Cost and provider behavior",
        "",
        f"- Failed provider-route attempts: **{failed_attempts}**; successful downstream routes: "
        f"**{dict(provider_counts)}**.",
        f"- The most expensive 10 documents contributed **{top_cost_share:.1%}** of total cost.",
        "- Per-document distributions, routing, latency, edit scope, semantic topology, and "
        "correlations are published under `plots/` and the corresponding CSV files.",
        "",
        "## Audited target projection",
        "",
        f"Target-integrity changes: **{dict(projection_counts)}**. Equipment categories are only "
        "projected out when the OCR template has no printable slot for them; temperature-bearing "
        "equipment remains fail-closed. Each projection is retained with path/before/after "
        "provenance in `target-integrity-changes.csv`.",
        "",
        "## Manual stratified review",
        "",
        "The sample deliberately combines prior failures, highest-cost cases, longest source/model "
        "scopes, repair cases, and deterministic random coverage. It is a risk-weighted audit, not "
        "an IID confidence interval.",
        "",
        "| Document | Semantic | Template | Privacy | Selection | Notes |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in manual:
        report.append(
            f"| `{row['documentId']}` | {'yes' if row['semanticConsistency'] else 'no'} | "
            f"{'yes' if row['templateFidelity'] else 'no'} | "
            f"{'yes' if row['privacyRewritten'] else 'no'} | "
            f"{', '.join(row['selectionBasis'])} | {row['notes']} |"
        )
    report.extend(
        [
            "",
        "## Interpretation",
            "",
        "This artifact is a composite evaluation publication. It proves the current contract on "
        "the pinned 100-document cohort, verifies byte-identical source/target inputs across every "
        "recovery, and does not publish training records. Production promotion "
            "should continue to require the same fail-closed checks and immutable receipts.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    manifest = {
        str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
