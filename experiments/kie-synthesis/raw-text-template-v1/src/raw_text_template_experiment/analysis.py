from __future__ import annotations

import csv
import io
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd  # type: ignore[import-untyped]
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.run_safety import StagedArtifactRun
from matplotlib.figure import Figure


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _full_corpus_document_count(config: Mapping[str, Any]) -> int:
    inputs = config.get("inputs")
    source = inputs.get("source_corpus") if isinstance(inputs, Mapping) else None
    records = source.get("records") if isinstance(source, Mapping) else None
    if isinstance(records, bool) or not isinstance(records, int) or records <= 0:
        raise ValueError("committed run config lacks a positive source-corpus record count")
    return records


def _verify_committed_run(run_dir: Path) -> dict[str, Any]:
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError(f"analysis input is not a regular run directory: {run_dir}")
    commit = _load_json(run_dir / "_COMMIT.json")
    artifacts = commit.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("input run commit has no artifact ledger")
    for row in artifacts:
        relative = row["relative_path"]
        path = (run_dir / relative).resolve(strict=True)
        if run_dir.resolve() not in path.parents or path.is_symlink() or not path.is_file():
            raise ValueError(f"committed artifact is unsafe or missing: {relative}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"committed artifact hash differs: {relative}")
    return commit


def _jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            output.append(value)
    return output


def _failure_category(reasons: Sequence[str]) -> str:
    message = " ".join(reasons)
    if "credit_balance_exhausted" in message or "openrouter_credits" in message:
        return "provider_quota_exhausted"
    if "Exceeded maximum output retries" in message:
        return "provider_schema_retry_exhausted"
    if "additional-call launch threshold reached" in message:
        return "per_document_cost_guard"
    if (
        "source texts cannot be resolved" in message
        or "exact quote resolution failed" in message
        or "exact source text is absent" in message
    ):
        return "invalid_exact_occurrence"
    if "compact equipment locality" in message:
        return "compact_equipment_locality"
    if "multiple logical owners" in message:
        return "duplicate_target_ownership"
    if "template spans overlap" in message:
        return "span_overlap"
    if "binding realization contract violations" in message:
        return "binding_realization_contract"
    if "compiler retained unresolved items" in message:
        return "compiler_declared_unresolved"
    if "critic made no progress" in message:
        return "critic_stagnation"
    if message.startswith("Independent critic required"):
        return "critic_required_semantic_revision"
    return "other"


def _csv_bytes(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _figure_bytes(figure: Figure) -> bytes:
    stream = io.BytesIO()
    figure.savefig(stream, format="png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return stream.getvalue()


def _failure_figure(categories: Counter[str], *, title: str) -> bytes:
    figure, axis = plt.subplots(figsize=(10, 5.5))
    if not categories:
        axis.text(0.5, 0.5, "No terminal failures", ha="center", va="center")
        axis.set_axis_off()
        axis.set_title(title)
        return _figure_bytes(figure)
    labels, values = zip(
        *sorted(categories.items(), key=lambda row: (-row[1], row[0])), strict=True
    )
    bars = axis.barh(labels, values, color="#3465a4")
    axis.invert_yaxis()
    axis.set_xlabel("Documents")
    axis.set_title(title)
    axis.bar_label(bars, padding=3)
    axis.spines[["top", "right"]].set_visible(False)
    return _figure_bytes(figure)


def _cost_figure(frame: pd.DataFrame) -> bytes:
    figure, axis = plt.subplots(figsize=(10, 6))
    categories = sorted(frame["failure_category"].unique())
    palette = plt.get_cmap("tab10")
    for index, category in enumerate(categories):
        rows = frame[frame["failure_category"] == category]
        axis.scatter(
            rows["compiler_request_bytes"],
            rows["estimated_cost_usd"],
            label=category,
            color=palette(index),
            alpha=0.85,
        )
    axis.set_xlabel("Preflight compiler request bytes")
    axis.set_ylabel("All-attempt estimated cost (USD)")
    axis.set_title("Cost versus request size")
    axis.legend(fontsize=8, loc="upper left")
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    return _figure_bytes(figure)


def analyze_failed_extraction(*, run_dir: Path, output_parent: Path, run_name: str) -> Path:
    run_dir = run_dir.resolve(strict=True)
    commit = _verify_committed_run(run_dir)
    config = _load_json(run_dir / "config.json")
    summary = _load_json(run_dir / "summary.json")
    selection = _load_json(run_dir / "selection-manifest.json")
    preflight = _load_json(run_dir / "preflight.json")
    results = _jsonl(run_dir / "results.jsonl")
    if len(results) != selection["rows"].__len__() or len(results) != summary["documents"]:
        raise ValueError("input run aggregate counts disagree")

    selected = {row["document_id"]: row for row in selection["rows"]}
    preflight_by_id = {row["documentId"]: row for row in preflight["cases"]}
    case_rows: list[dict[str, Any]] = []
    finding_rows: list[dict[str, Any]] = []
    stage_status: Counter[tuple[str, str]] = Counter()
    categories: Counter[str] = Counter()
    underlying_categories: Counter[str] = Counter()
    total_cost = Decimal(0)
    marginal_total_cost = Decimal(0)
    for result in results:
        document_id = cast(str, result["document_id"])
        stages = [*result["compiler_stages"], *result["critic_stages"]]
        for stage in stages:
            stage_status[(stage["role"], stage["status"])] += 1
            output = stage.get("output") or {}
            for finding in output.get("findings") or ():
                finding_rows.append(
                    {
                        "document_id": document_id,
                        "critic_pass": stage["pass_number"],
                        "finding_kind": finding["finding_kind"],
                        "line_ids": "|".join(finding["line_ids"]),
                        "evidence": finding["evidence"],
                        "explanation": finding["explanation"],
                    }
                )
        cost = sum((Decimal(stage["usage"]["estimatedCostUsd"]) for stage in stages), Decimal(0))
        total_cost += cost
        category = (
            "certified"
            if result["status"] == "certified"
            else _failure_category(result["rejection_reasons"])
        )
        last_stage_error = next(
            (
                str(stage["error_message"])
                for stage in reversed(stages)
                if stage.get("error_message")
            ),
            "",
        )
        underlying_category = (
            "certified"
            if result["status"] == "certified"
            else (
                _failure_category((last_stage_error,))
                if category == "per_document_cost_guard" and last_stage_error
                else category
            )
        )
        if result["status"] != "certified":
            categories[category] += 1
            underlying_categories[underlying_category] += 1
        resumed_cost = Decimal(str(result.get("resumed_prior_estimated_cost_usd", "0")))
        marginal_cost = cost - resumed_cost
        marginal_total_cost += marginal_cost
        source = selected[document_id]
        request = preflight_by_id[document_id]
        case_rows.append(
            {
                "ordinal": source["ordinal"],
                "document_id": document_id,
                "carrier_name": source["carrier_name"],
                "document_type": source["document_type"],
                "pages": source["page_count"],
                "source_bytes": request["sourceBytes"],
                "compiler_request_bytes": request["compilerRequestBytes"],
                "accepted_anchor_bindings": request["acceptedAnchorBindings"],
                "risk_candidates": request["riskCandidates"],
                "containers": source["container_count"],
                "cargo_groups": source["cargo_group_count"],
                "dangerous_goods": source["dangerous_goods_count"],
                "temperature": source["temperature_count"],
                "status": result["status"],
                "failure_category": category,
                "underlying_failure_category": underlying_category,
                "compiler_stages": len(result["compiler_stages"]),
                "critic_stages": len(result["critic_stages"]),
                "provider_requests": sum(stage["usage"]["requests"] for stage in stages),
                "input_tokens": sum(stage["usage"]["inputTokens"] for stage in stages),
                "output_tokens": sum(stage["usage"]["outputTokens"] for stage in stages),
                "reasoning_tokens": sum(stage["usage"]["reasoningTokens"] for stage in stages),
                "estimated_cost_usd": float(cost),
                "marginal_estimated_cost_usd": float(marginal_cost),
                "elapsed_seconds": result["elapsed_seconds"],
                "terminal_reason": " | ".join(result["rejection_reasons"]),
            }
        )
    case_rows.sort(key=lambda row: row["ordinal"])
    finding_counts = Counter(row["finding_kind"] for row in finding_rows)
    frame = pd.DataFrame(case_rows)
    correlation_value = frame["compiler_request_bytes"].corr(frame["estimated_cost_usd"])
    correlation = None if pd.isna(correlation_value) else float(correlation_value)
    mean_cost = total_cost / Decimal(len(case_rows))
    marginal_mean_cost = marginal_total_cost / Decimal(len(case_rows))
    full_corpus_documents = _full_corpus_document_count(config)
    projection = mean_cost * Decimal(full_corpus_documents)
    marginal_projection = marginal_mean_cost * Decimal(full_corpus_documents)
    status_counts = Counter(row["status"] for row in case_rows)
    quality_gate_passed = bool(summary.get("acceptanceGatePassed", False))
    analysis_summary = {
        "schemaVersion": 2,
        "inputRun": run_dir.name,
        "inputCommitSha256": sha256_file(run_dir / "_COMMIT.json"),
        "documents": len(case_rows),
        "phase": summary.get("phase"),
        "statusCounts": dict(status_counts),
        "terminalFailureCounts": dict(sorted(categories.items())),
        "underlyingFailureCounts": dict(sorted(underlying_categories.items())),
        "criticFindingCounts": dict(sorted(finding_counts.items())),
        "stageStatusCounts": {
            f"{role}:{status}": count for (role, status), count in sorted(stage_status.items())
        },
        "cumulativeLineageEstimatedCostUsd": str(total_cost),
        "marginalEstimatedCostUsd": str(marginal_total_cost),
        "cumulativeMeanCostPerDocumentUsd": str(mean_cost),
        "marginalMeanCostPerDocumentUsd": str(marginal_mean_cost),
        "fullCorpusDocuments": full_corpus_documents,
        "cumulativeFullCorpusProjectionUsd": str(projection),
        "marginalFullCorpusProjectionUsd": str(marginal_projection),
        "requestBytesCostPearsonCorrelation": correlation,
        "providerQuotaExhausted": categories["provider_quota_exhausted"] > 0,
        "qualityGatePassed": quality_gate_passed,
    }
    phase_label = str(summary.get("phase") or "template extraction")
    report = "\n".join(
        [
            f"# {phase_label} extraction analysis",
            "",
            "## Outcome",
            "",
            f"- Certified: **{status_counts['certified']} / {len(case_rows)}**.",
            f"- Rejected: **{status_counts['rejected']} / {len(case_rows)}**.",
            f"- Cumulative lineage estimated cost: **${total_cost}**.",
            f"- Marginal estimated cost in this run: **${marginal_total_cost}**.",
            f"- Cumulative mean cost per selected document: **${mean_cost}**.",
            f"- Cumulative {full_corpus_documents:,}-document projection: **${projection}**.",
            f"- Provider quota-exhausted cases: **{categories['provider_quota_exhausted']}**.",
            f"- Configured quality gate: **{'passed' if quality_gate_passed else 'failed'}**.",
            "",
            "## Terminal failure modes",
            "",
            *[
                f"- `{name}`: **{count}** documents."
                for name, count in sorted(categories.items(), key=lambda row: (-row[1], row[0]))
            ],
            "",
            "## Underlying failure modes",
            "",
            *[
                f"- `{name}`: **{count}** documents."
                for name, count in sorted(
                    underlying_categories.items(), key=lambda row: (-row[1], row[0])
                )
            ],
            "",
            "## Diagnosis",
            "",
            "Terminal categories are mutually exclusive classifications of each rejected case's "
            "final recorded reason. Certified cases are retained in the cost and complexity data "
            "but excluded from the terminal-failure chart. Projections are descriptive linear "
            "extrapolations from this selected run, not a launch authorization.",
            "",
            "## Artifact map",
            "",
            "- `cases.csv`: per-document selection, complexity, stage, cost, and terminal outcome.",
            "- `critic-findings.csv`: every independent critic finding with line evidence.",
            "- `stage-status.csv`: compiler/critic stage-status counts.",
            "- `terminal-failure-counts.png`: mutually exclusive terminal failure distribution.",
            "- `cost-vs-request-size.png`: all-attempt cost against pinned request size.",
            "",
        ]
    )
    transaction = sha256_bytes(
        canonical_json_bytes(
            {
                "inputCommitSha256": analysis_summary["inputCommitSha256"],
                "analysisSourceSha256": sha256_file(Path(__file__)),
                "runName": run_name,
            }
        )
    )
    staged = StagedArtifactRun(
        output_parent=output_parent.resolve(),
        run_name=run_name,
        transaction_sha256=transaction,
    )
    if staged.completed:
        return staged.final_root
    staged.recover_interrupted_temporary_files()
    staged.publish_json("input-commit.json", commit)
    staged.publish_json("summary.json", analysis_summary)
    staged.publish_bytes("REPORT.md", report.encode("utf-8"))
    staged.publish_bytes("cases.csv", _csv_bytes(case_rows, tuple(case_rows[0])))
    finding_columns = (
        "document_id",
        "critic_pass",
        "finding_kind",
        "line_ids",
        "evidence",
        "explanation",
    )
    staged.publish_bytes("critic-findings.csv", _csv_bytes(finding_rows, finding_columns))
    status_rows = [
        {"role": role, "status": status, "stages": count}
        for (role, status), count in sorted(stage_status.items())
    ]
    staged.publish_bytes("stage-status.csv", _csv_bytes(status_rows, ("role", "status", "stages")))
    staged.publish_bytes(
        "terminal-failure-counts.png",
        _failure_figure(categories, title=f"{phase_label} terminal failure modes"),
    )
    staged.publish_bytes("cost-vs-request-size.png", _cost_figure(frame))
    expected = (
        "REPORT.md",
        "cases.csv",
        "cost-vs-request-size.png",
        "critic-findings.csv",
        "input-commit.json",
        "stage-status.csv",
        "summary.json",
        "terminal-failure-counts.png",
    )
    staged.commit(
        expected_artifacts=expected,
        metadata={
            "schemaVersion": 2,
            "inputRun": run_dir.name,
            "documents": len(case_rows),
            "qualityGatePassed": quality_gate_passed,
        },
    )
    return staged.final_root
