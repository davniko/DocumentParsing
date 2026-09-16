from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import stat
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.run_safety import StagedArtifactRun
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

plt.switch_backend("Agg")


_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)
_MILLION = Decimal(1_000_000)
_OUTCOME_COLORS = {
    "first_pass_certified": "#2E7D32",
    "other_certified": "#1976D2",
    "not_certified": "#C62828",
}


@dataclass(frozen=True)
class RunMetrics:
    root: Path
    commit: dict[str, Any]
    commit_sha256: str
    config: dict[str, Any]
    summary: dict[str, Any]
    full_corpus_documents: int
    documents: tuple[dict[str, Any], ...]
    stages: tuple[dict[str, Any], ...]
    mode_counts: Mapping[tuple[str, str], int]
    quality_gates: tuple[dict[str, Any], ...]
    critic_findings: tuple[dict[str, Any], ...]
    observed_cost_usd: Decimal
    cold_cost_usd: Decimal


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(read_regular_file_bytes(path))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _safe_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str):
        raise ValueError("committed artifact path is not a string")
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe committed artifact path: {value!r}")
    return path


def _verify_committed_run(run_dir: Path) -> tuple[Path, dict[str, Any], str]:
    """Validate a complete immutable run without materializing large JSONL files."""

    if run_dir.is_symlink():
        raise ValueError(f"analysis input is a symbolic link: {run_dir}")
    root = run_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"analysis input is not a directory: {run_dir}")
    commit_path = root / "_COMMIT.json"
    commit = _load_object(commit_path)
    content_sha256 = commit.get("content_sha256")
    body = {key: value for key, value in commit.items() if key != "content_sha256"}
    if (
        not isinstance(content_sha256, str)
        or sha256_bytes(canonical_json_bytes(body)) != content_sha256
    ):
        raise ValueError("input commit receipt content SHA-256 is invalid")
    if commit.get("run_name") != root.name:
        raise ValueError("input commit receipt names a different run")
    marker_path = root / "_TRANSACTION.json"
    if marker_path.is_symlink() or not stat.S_ISREG(marker_path.lstat().st_mode):
        raise ValueError("input transaction marker is not a regular file")
    marker_sha256 = commit.get("transaction_marker_sha256")
    if not isinstance(marker_sha256, str) or sha256_file(marker_path) != marker_sha256:
        raise ValueError("input transaction marker differs from its commit receipt")

    ledger = commit.get("artifacts")
    if not isinstance(ledger, list) or not ledger:
        raise ValueError("input commit has no artifact ledger")
    expected: dict[str, Mapping[str, Any]] = {}
    for raw in ledger:
        if not isinstance(raw, Mapping):
            raise ValueError("input commit artifact receipt is not an object")
        relative = _safe_relative_path(raw.get("relative_path")).as_posix()
        if relative in expected:
            raise ValueError(f"input commit repeats an artifact path: {relative}")
        expected[relative] = raw

    actual: set[str] = set()
    for directory, directory_names, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            if (directory_path / name).is_symlink():
                raise ValueError("input run contains a symbolic-link directory")
        for name in filenames:
            path = directory_path / name
            if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError(f"input run artifact is not a regular file: {path}")
            relative = path.relative_to(root).as_posix()
            if relative not in {"_COMMIT.json", "_TRANSACTION.json"}:
                actual.add(relative)
    if actual != set(expected):
        raise ValueError(
            "input committed artifact inventory differs; "
            f"missing={sorted(set(expected) - actual)}, extra={sorted(actual - set(expected))}"
        )
    for relative, receipt in expected.items():
        path = root / relative
        expected_bytes = receipt.get("bytes")
        expected_sha256 = receipt.get("sha256")
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
            raise ValueError(f"committed artifact has invalid byte count: {relative}")
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"committed artifact byte count differs: {relative}")
        if not isinstance(expected_sha256, str) or sha256_file(path) != expected_sha256:
            raise ValueError(f"committed artifact hash differs: {relative}")
    return root, commit, sha256_file(commit_path)


def _stream_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot safely open JSONL artifact: {path}") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"JSONL artifact is not a regular file: {path}")
        with os.fdopen(descriptor, encoding="utf-8", closefd=False) as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ValueError(f"blank JSONL row at {path}:{line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object JSONL row at {path}:{line_number}")
                yield value
    finally:
        os.close(descriptor)


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not an object")
    return cast(Mapping[str, Any], value)


def _require_sequence(value: object, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} is not an array")
    return value


def _require_str(mapping: Mapping[str, Any], key: str, *, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.{key} is not a non-empty string")
    return value


def _require_int(mapping: Mapping[str, Any], key: str, *, label: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label}.{key} is not an integer")
    return value


def _require_number(mapping: Mapping[str, Any], key: str, *, label: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}.{key} is not a finite number")
    return float(value)


def _full_corpus_documents(config: Mapping[str, Any]) -> int:
    inputs = _require_mapping(config.get("inputs"), label="config.inputs")
    source = _require_mapping(inputs.get("source_corpus"), label="config.inputs.source_corpus")
    records = _require_int(source, "records", label="config.inputs.source_corpus")
    if records <= 0:
        raise ValueError("source-corpus record count must be positive")
    return records


def _pricing(config: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    provider = _require_mapping(config.get(f"{role}_provider"), label=f"config.{role}_provider")
    return _require_mapping(provider.get("pricing"), label=f"config.{role}_provider.pricing")


def _decimal(mapping: Mapping[str, Any], key: str, *, label: str) -> Decimal:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(f"{label}.{key} is not decimal-compatible")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError(f"{label}.{key} is not finite")
    return result


def _priced_cost(
    usage: Mapping[str, Any], pricing: Mapping[str, Any], *, cold_normalized: bool
) -> Decimal:
    input_tokens = Decimal(_require_int(usage, "inputTokens", label="stage.usage"))
    cache_read = Decimal(_require_int(usage, "cacheReadTokens", label="stage.usage"))
    cache_write = Decimal(_require_int(usage, "cacheWriteTokens", label="stage.usage"))
    output_tokens = Decimal(_require_int(usage, "outputTokens", label="stage.usage"))
    ordinary = input_tokens - cache_read - cache_write
    if ordinary < 0:
        raise ValueError("stage cache-token accounting exceeds input tokens")
    input_rate = _decimal(pricing, "input_usd_per_million", label="pricing")
    cached_input_rate = _decimal(pricing, "cached_input_usd_per_million", label="pricing")
    output_rate = _decimal(pricing, "output_usd_per_million", label="pricing")
    write_multiplier = _decimal(pricing, "cache_write_multiplier", label="pricing")
    cache_read_rate = input_rate * write_multiplier if cold_normalized else cached_input_rate
    return (
        ordinary * input_rate
        + cache_read * cache_read_rate
        + cache_write * input_rate * write_multiplier
        + output_tokens * output_rate
    ) / _MILLION


def _stage_metrics(
    *,
    document_id: str,
    raw_stage: Mapping[str, Any],
    expected_role: str,
    pricing: Mapping[str, Any],
) -> dict[str, Any]:
    role = _require_str(raw_stage, "role", label="stage")
    if role != expected_role:
        raise ValueError(f"stage role mismatch for {document_id}: {role} != {expected_role}")
    usage = _require_mapping(raw_stage.get("usage"), label="stage.usage")
    output = raw_stage.get("output")
    output_mapping = output if isinstance(output, Mapping) else {}
    verdict = output_mapping.get("verdict")
    if verdict is not None and not isinstance(verdict, str):
        raise ValueError(f"critic verdict is not a string for {document_id}")
    findings = output_mapping.get("findings", ())
    finding_rows = _require_sequence(findings, label="stage.output.findings")
    requests = _require_int(usage, "requests", label="stage.usage")
    observed = _decimal(usage, "estimatedCostUsd", label="stage.usage")
    recomputed_observed = _priced_cost(usage, pricing, cold_normalized=False)
    if observed != recomputed_observed:
        raise ValueError(f"stage estimated cost disagrees with committed pricing for {document_id}")
    row: dict[str, Any] = {
        "document_id": document_id,
        "role": role,
        "pass_number": _require_int(raw_stage, "pass_number", label="stage"),
        "status": _require_str(raw_stage, "status", label="stage"),
        "verdict": verdict or "",
        "duration_seconds": _require_number(raw_stage, "duration_seconds", label="stage"),
        "requests": requests,
        "structured_output_retry_requests": max(0, requests - 1),
        "input_tokens": _require_int(usage, "inputTokens", label="stage.usage"),
        "cache_read_tokens": _require_int(usage, "cacheReadTokens", label="stage.usage"),
        "cache_write_tokens": _require_int(usage, "cacheWriteTokens", label="stage.usage"),
        "output_tokens": _require_int(usage, "outputTokens", label="stage.usage"),
        "reasoning_tokens": _require_int(usage, "reasoningTokens", label="stage.usage"),
        "visible_output_tokens": _require_int(usage, "visibleOutputTokens", label="stage.usage"),
        "observed_cost_usd": observed,
        "cold_cost_usd": _priced_cost(usage, pricing, cold_normalized=True),
        "critic_findings": len(finding_rows),
    }
    if row["reasoning_tokens"] + row["visible_output_tokens"] != row["output_tokens"]:
        raise ValueError(f"output token accounting disagrees for {document_id}")
    return row


def _ids_by_key(rows: Sequence[Mapping[str, Any]], key: str, *, label: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for row in rows:
        value = _require_str(row, key, label=label)
        if value in output:
            raise ValueError(f"{label} repeats document ID: {value}")
        output[value] = row
    return output


def _sum_counter_values(rows: Sequence[Mapping[str, Any]], key: str) -> Counter[str]:
    result: Counter[str] = Counter()
    for row in rows:
        values = _require_mapping(row.get(key), label=f"catalog.{key}")
        for name, count in values.items():
            if not isinstance(name, str) or isinstance(count, bool) or not isinstance(count, int):
                raise ValueError(f"catalog.{key} contains an invalid counter")
            result[name] += count
    return result


def _validated_counter(value: object, *, label: str) -> dict[str, int]:
    mapping = _require_mapping(value, label=label)
    output: dict[str, int] = {}
    for key, count in mapping.items():
        if not isinstance(key, str) or isinstance(count, bool) or not isinstance(count, int):
            raise ValueError(f"{label} contains an invalid counter")
        if count < 0:
            raise ValueError(f"{label} contains a negative counter")
        output[key] = count
    return output


def _validate_summary_usage(
    summary: Mapping[str, Any], stages: Sequence[Mapping[str, Any]]
) -> None:
    usage = _require_mapping(summary.get("usage"), label="summary.usage")
    all_usage = _require_mapping(usage.get("all"), label="summary.usage.all")
    expected_stages = _require_int(all_usage, "stages", label="summary.usage.all")
    if expected_stages != len(stages):
        raise ValueError("summary stage count disagrees with result artifacts")
    stage_to_summary = {
        "requests": "requests",
        "input_tokens": "inputTokens",
        "cache_read_tokens": "cacheReadTokens",
        "cache_write_tokens": "cacheWriteTokens",
        "output_tokens": "outputTokens",
        "reasoning_tokens": "reasoningTokens",
        "visible_output_tokens": "visibleOutputTokens",
    }
    for stage_key, summary_key in stage_to_summary.items():
        actual = sum(int(row[stage_key]) for row in stages)
        if summary_key == "cacheWriteTokens" and summary_key not in all_usage:
            # Extraction summary schema v1 omits this aggregate even though each immutable
            # stage receipt records it. The EDA retains and validates the stage-level values.
            continue
        expected = _require_int(all_usage, summary_key, label="summary.usage.all")
        if actual != expected:
            raise ValueError(f"summary usage disagrees for {summary_key}")
    cost = sum((cast(Decimal, row["observed_cost_usd"]) for row in stages), Decimal(0))
    if cost != _decimal(all_usage, "estimatedCostUsd", label="summary.usage.all"):
        raise ValueError("summary estimated cost disagrees with result artifacts")


def _extract_run(
    run_dir: Path,
    *,
    expected_documents: int,
    inspect_templates: bool,
) -> RunMetrics:
    root, commit, commit_sha256 = _verify_committed_run(run_dir)
    config = _load_object(root / "config.json")
    summary = _load_object(root / "summary.json")
    selection = _load_object(root / "selection-manifest.json")
    preflight = _load_object(root / "preflight.json")
    selection_rows = tuple(
        _require_mapping(row, label="selection row")
        for row in _require_sequence(selection.get("rows"), label="selection.rows")
    )
    preflight_rows = tuple(
        _require_mapping(row, label="preflight case")
        for row in _require_sequence(preflight.get("cases"), label="preflight.cases")
    )
    if len(selection_rows) != expected_documents or len(preflight_rows) != expected_documents:
        raise ValueError(
            f"EDA requires {expected_documents} selected/preflight documents; "
            f"found {len(selection_rows)}/{len(preflight_rows)}"
        )
    selected_by_id = _ids_by_key(selection_rows, "document_id", label="selection")
    preflight_by_id = _ids_by_key(preflight_rows, "documentId", label="preflight")
    if set(selected_by_id) != set(preflight_by_id):
        raise ValueError("selection and preflight document sets disagree")

    catalog_rows = tuple(_stream_jsonl(root / "catalog.jsonl"))
    catalog_by_id = _ids_by_key(catalog_rows, "documentId", label="catalog")
    compiler_pricing = _pricing(config, "compiler")
    critic_pricing = _pricing(config, "critic")
    result_metrics: dict[str, dict[str, Any]] = {}
    stages: list[dict[str, Any]] = []
    critic_findings: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for raw_result in _stream_jsonl(root / "results.jsonl"):
        document_id = _require_str(raw_result, "document_id", label="result")
        if document_id in result_metrics:
            raise ValueError(f"results repeat document ID: {document_id}")
        compiler_stages = tuple(
            _require_mapping(row, label="compiler stage")
            for row in _require_sequence(
                raw_result.get("compiler_stages"), label="result.compiler_stages"
            )
        )
        critic_stages = tuple(
            _require_mapping(row, label="critic stage")
            for row in _require_sequence(
                raw_result.get("critic_stages"), label="result.critic_stages"
            )
        )
        current_stages: list[dict[str, Any]] = []
        for raw_stage in compiler_stages:
            current_stages.append(
                _stage_metrics(
                    document_id=document_id,
                    raw_stage=raw_stage,
                    expected_role="compiler",
                    pricing=compiler_pricing,
                )
            )
        for raw_stage in critic_stages:
            stage = _stage_metrics(
                document_id=document_id,
                raw_stage=raw_stage,
                expected_role="critic",
                pricing=critic_pricing,
            )
            current_stages.append(stage)
            output = raw_stage.get("output")
            if isinstance(output, Mapping):
                for finding in _require_sequence(
                    output.get("findings", ()), label="critic findings"
                ):
                    finding_mapping = _require_mapping(finding, label="critic finding")
                    critic_findings.append(
                        {
                            "document_id": document_id,
                            "critic_pass": stage["pass_number"],
                            "finding_kind": str(finding_mapping.get("finding_kind", "")),
                            "line_count": len(finding_mapping.get("line_ids", ())),
                        }
                    )
        stages.extend(current_stages)
        status = _require_str(raw_result, "status", label="result")
        status_counts[status] += 1
        compiler_rows = [row for row in current_stages if row["role"] == "compiler"]
        critic_rows = [row for row in current_stages if row["role"] == "critic"]
        total_requests = sum(int(row["requests"]) for row in current_stages)
        raw_resume_mode = raw_result.get("resume_mode")
        if raw_resume_mode is None:
            resume_mode = "fresh_execution"
        elif isinstance(raw_resume_mode, str) and raw_resume_mode:
            resume_mode = raw_resume_mode
        else:
            raise ValueError(f"result resume mode is invalid for {document_id}")
        fresh_execution = resume_mode == "fresh_execution"
        first_pass = bool(
            fresh_execution
            and status == "certified"
            and len(compiler_rows) == 1
            and len(critic_rows) == 1
            and compiler_rows[0]["pass_number"] == 1
            and critic_rows[0]["pass_number"] == 1
            and compiler_rows[0]["status"] == "success"
            and critic_rows[0]["status"] == "success"
            and critic_rows[0]["verdict"] == "pass"
            and total_requests == 2
        )
        result_metrics[document_id] = {
            "status": status,
            "resume_mode": resume_mode,
            "fresh_execution": fresh_execution,
            "first_pass_certified": first_pass,
            "compiler_stages": len(compiler_rows),
            "critic_stages": len(critic_rows),
            "provider_requests": total_requests,
            "additional_requests_beyond_two": max(0, total_requests - 2),
            "structured_output_retry_requests": sum(
                int(row["structured_output_retry_requests"]) for row in current_stages
            ),
            "additional_stage_count": max(0, len(current_stages) - 2),
            "compiler_repair_stages": sum(row["pass_number"] > 1 for row in compiler_rows),
            "critic_revision_verdicts": sum(row["verdict"] == "revise" for row in critic_rows),
            "host_rejected_stages": sum(row["status"] == "host_rejected" for row in current_stages),
            "input_tokens": sum(int(row["input_tokens"]) for row in current_stages),
            "cache_read_tokens": sum(int(row["cache_read_tokens"]) for row in current_stages),
            "cache_write_tokens": sum(int(row["cache_write_tokens"]) for row in current_stages),
            "output_tokens": sum(int(row["output_tokens"]) for row in current_stages),
            "reasoning_tokens": sum(int(row["reasoning_tokens"]) for row in current_stages),
            "visible_output_tokens": sum(
                int(row["visible_output_tokens"]) for row in current_stages
            ),
            "observed_cost_usd": sum(
                (cast(Decimal, row["observed_cost_usd"]) for row in current_stages),
                Decimal(0),
            ),
            "cold_cost_usd": sum(
                (cast(Decimal, row["cold_cost_usd"]) for row in current_stages),
                Decimal(0),
            ),
            "elapsed_seconds": _require_number(raw_result, "elapsed_seconds", label="result"),
            "critic_findings": sum(int(row["critic_findings"]) for row in critic_rows),
        }
    if set(result_metrics) != set(selected_by_id):
        raise ValueError("results and selection document sets disagree")
    certified_ids = {
        document_id
        for document_id, metrics in result_metrics.items()
        if metrics["status"] == "certified"
    }
    if set(catalog_by_id) != certified_ids:
        raise ValueError("catalog rows do not exactly match certified results")
    if _require_int(summary, "documents", label="summary") != expected_documents:
        raise ValueError("summary document count disagrees with the requested EDA cohort")
    recorded_status = _validated_counter(summary.get("statusCounts"), label="summary.statusCounts")
    if recorded_status != dict(status_counts):
        raise ValueError("summary status counts disagree with result artifacts")
    _validate_summary_usage(summary, stages)

    scalar_catalog_fields = {
        "bindings": "bindings",
        "occurrences": "occurrences",
        "deterministicBindings": "deterministicBindings",
        "agentAssistedBindings": "agentAssistedBindings",
        "agentResidualBindings": "agentResidualBindings",
        "semanticOnlyTargetFacts": "semanticOnlyTargetFacts",
    }
    for catalog_key, summary_key in scalar_catalog_fields.items():
        actual = sum(_require_int(row, catalog_key, label="catalog") for row in catalog_rows)
        expected = _require_int(summary, summary_key, label="summary")
        if actual != expected:
            raise ValueError(f"catalog aggregate disagrees with summary.{summary_key}")
    mode_counts: Counter[tuple[str, str]] = Counter()
    for dimension, catalog_key, summary_key in (
        ("render_mode", "renderModes", "renderModeCounts"),
        ("realization_mode", "realizationModes", "realizationModeCounts"),
    ):
        aggregate = _sum_counter_values(catalog_rows, catalog_key)
        recorded = _validated_counter(summary.get(summary_key), label=f"summary.{summary_key}")
        if dict(aggregate) != recorded:
            raise ValueError(f"catalog aggregate disagrees with summary.{summary_key}")
        for name, count in aggregate.items():
            mode_counts[(dimension, name)] = count
    for name, count in _sum_counter_values(catalog_rows, "valueKinds").items():
        mode_counts[("value_kind", name)] = count
    coherence_presence = ["coherenceContracts" in row for row in catalog_rows]
    if coherence_presence and any(coherence_presence) != all(coherence_presence):
        raise ValueError("catalog mixes rows with and without coherence contract summaries")
    if coherence_presence and all(coherence_presence):
        for row in catalog_rows:
            counts = _validated_counter(
                row.get("coherenceContracts"), label="catalog.coherenceContracts"
            )
            declared = _require_int(row, "coherenceBindings", label="catalog")
            if (sum(counts.values()) == 0) != (declared == 0):
                raise ValueError(
                    "catalog coherence contracts and participating bindings disagree on "
                    "whether a coherence topology exists"
                )
            for name, count in counts.items():
                mode_counts[("coherence_contract", name)] += count
    deterministic_total = sum(
        _require_int(row, "deterministicBindings", label="catalog") for row in catalog_rows
    )
    agent_total = sum(
        _require_int(row, "agentAssistedBindings", label="catalog") for row in catalog_rows
    )
    binding_total = sum(_require_int(row, "bindings", label="catalog") for row in catalog_rows)
    if deterministic_total + agent_total != binding_total:
        raise ValueError("deterministic plus agent-assisted binding counts do not equal bindings")
    mode_counts[("binding_route", "deterministic")] = deterministic_total
    mode_counts[("binding_route", "agent_assisted")] = agent_total

    documents: list[dict[str, Any]] = []
    quality_gate_counts: Counter[tuple[str, str]] = Counter()
    for document_id, selected in selected_by_id.items():
        request = cast(Mapping[str, Any], preflight_by_id[document_id])
        if _require_int(selected, "ordinal", label="selection") != _require_int(
            request, "ordinal", label="preflight"
        ):
            raise ValueError(f"selection/preflight ordinal mismatch for {document_id}")
        if _require_str(selected, "source_sha256", label="selection") != _require_str(
            request, "sourceSha256", label="preflight"
        ):
            raise ValueError(f"selection/preflight source hash mismatch for {document_id}")
        if _require_int(selected, "ocr_lines", label="selection") != _require_int(
            request, "sourceLines", label="preflight"
        ):
            raise ValueError(f"selection/preflight line count mismatch for {document_id}")
        metrics = result_metrics[document_id]
        catalog = catalog_by_id.get(document_id)
        if catalog is not None:
            if not catalog.get("certified"):
                raise ValueError(f"catalog row is not certified: {document_id}")
            if _require_str(catalog, "sourceSha256", label="catalog") != selected["source_sha256"]:
                raise ValueError(f"catalog source hash mismatch for {document_id}")
            catalog_identity = (
                ("pages", "page_count"),
                ("lines", "ocr_lines"),
                ("characters", "ocr_characters"),
            )
            for catalog_key, selection_key in catalog_identity:
                if _require_int(catalog, catalog_key, label="catalog") != _require_int(
                    selected, selection_key, label="selection"
                ):
                    raise ValueError(
                        f"catalog {catalog_key} disagrees with selection for {document_id}"
                    )
            if _require_str(catalog, "documentType", label="catalog") != _require_str(
                selected, "document_type", label="selection"
            ):
                raise ValueError(f"catalog document type mismatch for {document_id}")
            if _require_int(catalog, "bindings", label="catalog") != (
                _require_int(catalog, "deterministicBindings", label="catalog")
                + _require_int(catalog, "agentAssistedBindings", label="catalog")
            ):
                raise ValueError(f"catalog binding route counts disagree for {document_id}")
        bindings = _require_int(catalog, "bindings", label="catalog") if catalog else 0
        agent_bindings = (
            _require_int(catalog, "agentAssistedBindings", label="catalog") if catalog else 0
        )
        row = {
            "ordinal": _require_int(selected, "ordinal", label="selection"),
            "document_id": document_id,
            "carrier_name": _require_str(selected, "carrier_name", label="selection"),
            "carrier_family": _require_str(selected, "carrier_family", label="selection"),
            "document_type": _require_str(selected, "document_type", label="selection"),
            "template_proxy_id": _require_str(selected, "template_proxy_id", label="selection"),
            "pages": _require_int(selected, "page_count", label="selection"),
            "source_lines": _require_int(selected, "ocr_lines", label="selection"),
            "source_characters": _require_int(selected, "ocr_characters", label="selection"),
            "source_bytes": _require_int(request, "sourceBytes", label="preflight"),
            "compiler_request_bytes": _require_int(
                request, "compilerRequestBytes", label="preflight"
            ),
            "accepted_anchor_bindings": _require_int(
                request, "acceptedAnchorBindings", label="preflight"
            ),
            "risk_candidates": _require_int(request, "riskCandidates", label="preflight"),
            "required_target_cobindings": _require_int(
                request, "requiredTargetCoBindings", label="preflight"
            ),
            "container_count": _require_int(selected, "container_count", label="selection"),
            "cargo_group_count": _require_int(selected, "cargo_group_count", label="selection"),
            "package_count": _require_int(selected, "package_count", label="selection"),
            "dangerous_goods_count": _require_int(
                selected, "dangerous_goods_count", label="selection"
            ),
            "temperature_count": _require_int(selected, "temperature_count", label="selection"),
            **metrics,
            "bindings": bindings,
            "occurrences": _require_int(catalog, "occurrences", label="catalog") if catalog else 0,
            "deterministic_bindings": _require_int(
                catalog, "deterministicBindings", label="catalog"
            )
            if catalog
            else 0,
            "agent_assisted_bindings": agent_bindings,
            "agent_residual_bindings": _require_int(
                catalog, "agentResidualBindings", label="catalog"
            )
            if catalog
            else 0,
            "semantic_only_target_facts": _require_int(
                catalog, "semanticOnlyTargetFacts", label="catalog"
            )
            if catalog
            else 0,
            "coherence_bindings": _require_int(catalog, "coherenceBindings", label="catalog")
            if catalog and coherence_presence and all(coherence_presence)
            else 0,
            "deterministic_binding_fraction": (
                _require_int(catalog, "deterministicBindings", label="catalog") / bindings
                if catalog is not None and bindings
                else 0.0
            ),
            "agent_binding_fraction": agent_bindings / bindings if bindings else 0.0,
        }
        documents.append(row)
        if inspect_templates and catalog is not None:
            template = _load_object(root / "cases" / document_id / "template.json")
            if _require_str(template, "document_id", label="template") != document_id:
                raise ValueError(f"template document ID mismatch for {document_id}")
            constraints = _require_sequence(
                template.get("coherence_constraints"),
                label="template.coherence_constraints",
            )
            constraint_kinds: Counter[str] = Counter()
            constraint_members: set[str] = set()
            for index, raw_constraint in enumerate(constraints):
                constraint = _require_mapping(
                    raw_constraint,
                    label=f"template.coherence_constraints[{index}]",
                )
                constraint_kinds[
                    _require_str(
                        constraint,
                        "kind",
                        label=f"template.coherence_constraints[{index}]",
                    )
                ] += 1
                members = _require_sequence(
                    constraint.get("member_logical_keys"),
                    label=f"template.coherence_constraints[{index}].member_logical_keys",
                )
                if any(not isinstance(member, str) or not member for member in members):
                    raise ValueError("template coherence member key is not a non-empty string")
                constraint_members.update(cast(Sequence[str], members))
            catalog_kinds = _validated_counter(
                catalog.get("coherenceContracts"), label="catalog.coherenceContracts"
            )
            if dict(constraint_kinds) != catalog_kinds:
                raise ValueError(
                    f"template coherence contracts disagree with catalog for {document_id}"
                )
            if len(constraint_members) != _require_int(
                catalog, "coherenceBindings", label="catalog"
            ):
                raise ValueError(
                    f"template coherence binding union disagrees with catalog for {document_id}"
                )
            certification = _require_mapping(
                template.get("certification"), label="template.certification"
            )
            for gate, passed in certification.items():
                if not isinstance(gate, str) or (
                    not isinstance(passed, bool) and passed is not None
                ):
                    raise ValueError(f"template certification gate is invalid for {document_id}")
                disposition = (
                    "not_applicable" if passed is None else "passed" if passed else "failed"
                )
                quality_gate_counts[(f"certification:{gate}", disposition)] += 1
            literal = _require_mapping(
                template.get("literal_certification"), label="template.literal_certification"
            )
            final_critic = literal.get("final_critic_pass")
            remaining_risks = literal.get("remaining_unowned_risk_candidates")
            if (
                not isinstance(final_critic, bool)
                or isinstance(remaining_risks, bool)
                or not isinstance(remaining_risks, int)
            ):
                raise ValueError(f"template literal certification is invalid for {document_id}")
            quality_gate_counts[
                ("literal:final_critic_pass", "passed" if final_critic else "failed")
            ] += 1
            quality_gate_counts[
                (
                    "literal:no_unowned_risk_candidates",
                    "passed" if remaining_risks == 0 else "failed",
                )
            ] += 1
    documents.sort(key=lambda row: int(row["ordinal"]))
    quality_gates = tuple(
        {
            "gate": gate,
            "passed": quality_gate_counts[(gate, "passed")],
            "failed": quality_gate_counts[(gate, "failed")],
            "not_applicable": quality_gate_counts[(gate, "not_applicable")],
            "documents_evaluated": sum(
                quality_gate_counts[(gate, disposition)]
                for disposition in ("passed", "failed", "not_applicable")
            ),
        }
        for gate in sorted({gate for gate, _ in quality_gate_counts})
    )
    observed = sum((cast(Decimal, row["observed_cost_usd"]) for row in documents), Decimal(0))
    cold = sum((cast(Decimal, row["cold_cost_usd"]) for row in documents), Decimal(0))
    return RunMetrics(
        root=root,
        commit=commit,
        commit_sha256=commit_sha256,
        config=config,
        summary=summary,
        full_corpus_documents=_full_corpus_documents(config),
        documents=tuple(documents),
        stages=tuple(stages),
        mode_counts=mode_counts,
        quality_gates=quality_gates,
        critic_findings=tuple(critic_findings),
        observed_cost_usd=observed,
        cold_cost_usd=cold,
    )


def _csv_bytes(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for raw in rows:
        writer.writerow(
            {key: str(value) if isinstance(value, Decimal) else value for key, value in raw.items()}
        )
    return stream.getvalue().encode("utf-8")


def _figure_bytes(figure: Figure) -> bytes:
    stream = io.BytesIO()
    figure.savefig(stream, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return stream.getvalue()


def _clean_axis(axis: Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.18)


def _outcome(row: Mapping[str, Any]) -> str:
    if row["first_pass_certified"]:
        return "first_pass_certified"
    if row["status"] == "certified":
        return "other_certified"
    return "not_certified"


def _acceptance_figure(documents: Sequence[Mapping[str, Any]]) -> bytes:
    counts = Counter(_outcome(row) for row in documents)
    labels = ("Fresh first-pass\ncertified", "Other\ncertified", "Review or\nrejected")
    keys = ("first_pass_certified", "other_certified", "not_certified")
    values = [counts[key] for key in keys]
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    bars = axis.bar(labels, values, color=[_OUTCOME_COLORS[key] for key in keys])
    axis.bar_label(bars, padding=3)
    axis.set_ylabel("Documents")
    axis.set_title("Certification outcome (fresh and resumed executions separated)")
    axis.set_ylim(0, max([*values, 1]) * 1.18)
    _clean_axis(axis)
    return _figure_bytes(figure)


def _request_figure(documents: Sequence[Mapping[str, Any]]) -> bytes:
    request_values = sorted({int(row["provider_requests"]) for row in documents})
    outcomes = ("first_pass_certified", "other_certified", "not_certified")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    bottoms = [0] * len(request_values)
    for outcome in outcomes:
        values = [
            sum(
                _outcome(row) == outcome and int(row["provider_requests"]) == requests
                for row in documents
            )
            for requests in request_values
        ]
        axes[0].bar(
            request_values,
            values,
            bottom=bottoms,
            color=_OUTCOME_COLORS[outcome],
            label={
                "first_pass_certified": "fresh first-pass certified",
                "other_certified": "other certified",
                "not_certified": "review or rejected",
            }[outcome],
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
    axes[0].set_xlabel("Provider requests per document")
    axes[0].set_ylabel("Documents")
    axes[0].set_title("Request distribution by outcome")
    axes[0].legend(fontsize=8)
    _clean_axis(axes[0])
    repair_metrics = (
        ("Extra\nstages", "additional_stage_count"),
        ("Schema\nretries", "structured_output_retry_requests"),
        ("Critic\nrevisions", "critic_revision_verdicts"),
        ("Host\nrejections", "host_rejected_stages"),
    )
    totals = [sum(int(row[key]) for row in documents) for _, key in repair_metrics]
    bars = axes[1].bar([label for label, _ in repair_metrics], totals, color="#6A1B9A")
    axes[1].bar_label(bars, padding=3)
    axes[1].set_ylabel("Recorded events")
    axes[1].set_title("Repair and retry burden")
    _clean_axis(axes[1])
    return _figure_bytes(figure)


def _cost_distribution_figure(documents: Sequence[Mapping[str, Any]]) -> bytes:
    observed = sorted(float(row["observed_cost_usd"]) for row in documents)
    cold = sorted(float(row["cold_cost_usd"]) for row in documents)
    bins = min(20, max(5, round(math.sqrt(len(documents)))))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].hist(observed, bins=bins, alpha=0.72, color="#1976D2", label="Observed")
    axes[0].hist(cold, bins=bins, alpha=0.52, color="#EF6C00", label="Cold-normalized")
    axes[0].set_xlabel("Cost per document (USD)")
    axes[0].set_ylabel("Documents")
    axes[0].set_title("Per-document cost distribution")
    axes[0].legend()
    _clean_axis(axes[0])
    probability = [(index + 1) / len(documents) for index in range(len(documents))]
    axes[1].plot(observed, probability, color="#1976D2", label="Observed")
    axes[1].plot(cold, probability, color="#EF6C00", label="Cold-normalized")
    axes[1].set_xlabel("Cost per document (USD)")
    axes[1].set_ylabel("Empirical cumulative fraction")
    axes[1].set_title("Cost ECDF")
    axes[1].legend()
    _clean_axis(axes[1])
    return _figure_bytes(figure)


def _role_summary(stages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    total_observed = sum((cast(Decimal, row["observed_cost_usd"]) for row in stages), Decimal(0))
    for role in ("compiler", "critic"):
        rows = [row for row in stages if row["role"] == role]
        observed = sum((cast(Decimal, row["observed_cost_usd"]) for row in rows), Decimal(0))
        cold = sum((cast(Decimal, row["cold_cost_usd"]) for row in rows), Decimal(0))
        output.append(
            {
                "role": role,
                "stages": len(rows),
                "requests": sum(int(row["requests"]) for row in rows),
                "structured_output_retry_requests": sum(
                    int(row["structured_output_retry_requests"]) for row in rows
                ),
                "input_tokens": sum(int(row["input_tokens"]) for row in rows),
                "cache_read_tokens": sum(int(row["cache_read_tokens"]) for row in rows),
                "cache_write_tokens": sum(int(row["cache_write_tokens"]) for row in rows),
                "output_tokens": sum(int(row["output_tokens"]) for row in rows),
                "reasoning_tokens": sum(int(row["reasoning_tokens"]) for row in rows),
                "visible_output_tokens": sum(int(row["visible_output_tokens"]) for row in rows),
                "observed_cost_usd": observed,
                "cold_cost_usd": cold,
                "cold_uplift_usd": cold - observed,
                "observed_cost_fraction": observed / total_observed
                if total_observed
                else Decimal(0),
            }
        )
    return output


def _cost_drivers_figure(
    documents: Sequence[Mapping[str, Any]], role_summary: Sequence[Mapping[str, Any]]
) -> bytes:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    roles = [str(row["role"]).title() for row in role_summary]
    observed = [float(row["observed_cost_usd"]) for row in role_summary]
    uplift = [float(row["cold_uplift_usd"]) for row in role_summary]
    axes[0].bar(roles, observed, color="#1976D2", label="Observed")
    axes[0].bar(roles, uplift, bottom=observed, color="#EF6C00", label="Cold uplift")
    axes[0].set_ylabel("Aggregate cost (USD)")
    axes[0].set_title("Role cost and cache normalization")
    axes[0].legend()
    _clean_axis(axes[0])
    scatter = axes[1].scatter(
        [int(row["compiler_request_bytes"]) for row in documents],
        [float(row["cold_cost_usd"]) for row in documents],
        c=[int(row["provider_requests"]) for row in documents],
        cmap="viridis",
        alpha=0.78,
        edgecolors="none",
    )
    axes[1].set_xlabel("Compiler request bytes")
    axes[1].set_ylabel("Cold-normalized cost (USD)")
    axes[1].set_title("Cost versus request size")
    figure.colorbar(scatter, ax=axes[1], label="Provider requests")
    _clean_axis(axes[1])
    return _figure_bytes(figure)


def _role_token_figure(role_summary: Sequence[Mapping[str, Any]]) -> bytes:
    roles = [str(row["role"]).title() for row in role_summary]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ordinary = [
        int(row["input_tokens"]) - int(row["cache_read_tokens"]) - int(row["cache_write_tokens"])
        for row in role_summary
    ]
    cache_read = [int(row["cache_read_tokens"]) for row in role_summary]
    cache_write = [int(row["cache_write_tokens"]) for row in role_summary]
    axes[0].bar(roles, ordinary, color="#546E7A", label="Ordinary input")
    axes[0].bar(roles, cache_read, bottom=ordinary, color="#26A69A", label="Cache read")
    second_bottom = [a + b for a, b in zip(ordinary, cache_read, strict=True)]
    axes[0].bar(roles, cache_write, bottom=second_bottom, color="#9CCC65", label="Cache write")
    axes[0].set_ylabel("Tokens")
    axes[0].set_title("Input-token classes by role")
    axes[0].legend(fontsize=8)
    _clean_axis(axes[0])
    visible = [int(row["visible_output_tokens"]) for row in role_summary]
    reasoning = [int(row["reasoning_tokens"]) for row in role_summary]
    axes[1].bar(roles, visible, color="#42A5F5", label="Visible output")
    axes[1].bar(roles, reasoning, bottom=visible, color="#7E57C2", label="Reasoning")
    axes[1].set_ylabel("Tokens")
    axes[1].set_title("Output-token composition by role")
    axes[1].legend(fontsize=8)
    _clean_axis(axes[1])
    return _figure_bytes(figure)


def _coverage_figure(documents: Sequence[Mapping[str, Any]]) -> bytes:
    deterministic = sum(int(row["deterministic_bindings"]) for row in documents)
    agent = sum(int(row["agent_assisted_bindings"]) for row in documents)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    bars = axes[0].bar(
        ["Deterministic", "Agent-assisted"],
        [deterministic, agent],
        color=["#1976D2", "#EF6C00"],
    )
    axes[0].bar_label(bars, padding=3)
    axes[0].set_ylabel("Compiled bindings")
    axes[0].set_title("Global rendering coverage")
    _clean_axis(axes[0])
    fractions = [100 * float(row["agent_binding_fraction"]) for row in documents]
    axes[1].hist(fractions, bins=min(20, max(5, round(math.sqrt(len(documents))))), color="#EF6C00")
    axes[1].set_xlabel("Agent-assisted bindings per document (%)")
    axes[1].set_ylabel("Documents")
    axes[1].set_title("Residual agent surface distribution")
    _clean_axis(axes[1])
    return _figure_bytes(figure)


def _complexity_figure(documents: Sequence[Mapping[str, Any]]) -> bytes:
    pairs = (
        ("source_lines", "Source lines", "cold_cost_usd", "Cold cost (USD)"),
        ("risk_candidates", "Risk candidates", "provider_requests", "Provider requests"),
        ("bindings", "Compiled bindings", "elapsed_seconds", "Elapsed seconds"),
        (
            "compiler_request_bytes",
            "Compiler request bytes",
            "agent_binding_fraction",
            "Agent binding fraction",
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, (x_key, x_label, y_key, y_label) in zip(axes.flat, pairs, strict=True):
        axis.scatter(
            [float(row[x_key]) for row in documents],
            [float(row[y_key]) for row in documents],
            color="#3949AB",
            alpha=0.7,
            edgecolors="none",
        )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        _clean_axis(axis)
    figure.suptitle("Document complexity relationships", fontsize=14)
    figure.tight_layout()
    return _figure_bytes(figure)


def _projection_rows(current: RunMetrics, baseline: RunMetrics | None) -> list[dict[str, Any]]:
    rows = []
    for lineage, metrics in (("current", current), ("baseline", baseline)):
        if metrics is None:
            continue
        certified = sum(row["status"] == "certified" for row in metrics.documents)
        for basis, total in (
            ("observed", metrics.observed_cost_usd),
            ("cold_normalized", metrics.cold_cost_usd),
        ):
            mean = total / Decimal(len(metrics.documents))
            rows.append(
                {
                    "lineage": lineage,
                    "pricing_basis": basis,
                    "sample_documents": len(metrics.documents),
                    "certified_documents": certified,
                    "acceptance_rate": Decimal(certified) / Decimal(len(metrics.documents)),
                    "sample_total_cost_usd": total,
                    "mean_cost_per_selected_document_usd": mean,
                    "full_corpus_documents": metrics.full_corpus_documents,
                    "full_corpus_projection_usd": mean * Decimal(metrics.full_corpus_documents),
                }
            )
    return rows


def _baseline_figure(projections: Sequence[Mapping[str, Any]]) -> bytes:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    labels = [
        f"{row['lineage']}\n{str(row['pricing_basis']).replace('_', ' ')}" for row in projections
    ]
    colors = ["#1976D2" if row["lineage"] == "current" else "#78909C" for row in projections]
    means = [float(row["mean_cost_per_selected_document_usd"]) for row in projections]
    projected = [float(row["full_corpus_projection_usd"]) for row in projections]
    bars = axes[0].bar(labels, means, color=colors)
    axes[0].bar_label(bars, fmt="$%.4f", padding=3, fontsize=8)
    axes[0].set_ylabel("USD per selected document")
    axes[0].set_title("Per-document cost baseline")
    axes[0].tick_params(axis="x", labelsize=8)
    _clean_axis(axes[0])
    bars = axes[1].bar(labels, projected, color=colors)
    axes[1].bar_label(bars, fmt="$%.2f", padding=3, fontsize=8)
    axes[1].set_ylabel("Projected USD")
    axes[1].set_title("Full source-corpus projection")
    axes[1].tick_params(axis="x", labelsize=8)
    _clean_axis(axes[1])
    return _figure_bytes(figure)


def _correlations(documents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    predictors = (
        "pages",
        "source_lines",
        "source_bytes",
        "compiler_request_bytes",
        "accepted_anchor_bindings",
        "risk_candidates",
        "container_count",
        "cargo_group_count",
        "bindings",
        "occurrences",
        "agent_binding_fraction",
    )
    outcomes = ("observed_cost_usd", "cold_cost_usd", "provider_requests", "elapsed_seconds")
    frame = pd.DataFrame(documents)
    rows: list[dict[str, Any]] = []
    for predictor in predictors:
        for outcome in outcomes:
            pair = frame[[predictor, outcome]].astype(float)
            if pair[predictor].nunique() < 2 or pair[outcome].nunique() < 2:
                pearson = None
                spearman = None
            else:
                pearson = pair[predictor].corr(pair[outcome], method="pearson")
                ranked = pair.rank(method="average")
                spearman = ranked[predictor].corr(ranked[outcome], method="pearson")
            rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "documents": len(pair),
                    "pearson": "" if pearson is None or pd.isna(pearson) else float(pearson),
                    "spearman": "" if spearman is None or pd.isna(spearman) else float(spearman),
                }
            )
    return rows


def _outliers(documents: Sequence[Mapping[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    metrics = (
        "cold_cost_usd",
        "provider_requests",
        "additional_stage_count",
        "elapsed_seconds",
        "compiler_request_bytes",
        "source_lines",
        "agent_binding_fraction",
    )
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        ranked = sorted(
            documents,
            key=lambda row: (float(row[metric]), str(row["document_id"])),
            reverse=True,
        )
        if not ranked or float(ranked[0][metric]) == 0:
            continue
        for rank, row in enumerate(ranked[: min(limit, len(ranked))], start=1):
            rows.append(
                {
                    "metric": metric,
                    "rank": rank,
                    "value": row[metric],
                    "document_id": row["document_id"],
                    "ordinal": row["ordinal"],
                    "status": row["status"],
                    "carrier_family": row["carrier_family"],
                    "document_type": row["document_type"],
                    "pages": row["pages"],
                    "source_lines": row["source_lines"],
                    "provider_requests": row["provider_requests"],
                    "cold_cost_usd": row["cold_cost_usd"],
                }
            )
    return rows


def _strata(documents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for dimension in ("document_type", "carrier_family"):
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in documents:
            grouped[str(row[dimension])].append(row)
        for value, group in sorted(grouped.items()):
            count = len(group)
            rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "documents": count,
                    "certified": sum(row["status"] == "certified" for row in group),
                    "first_pass_certified": sum(bool(row["first_pass_certified"]) for row in group),
                    "mean_provider_requests": sum(int(row["provider_requests"]) for row in group)
                    / count,
                    "mean_observed_cost_usd": sum(
                        (cast(Decimal, row["observed_cost_usd"]) for row in group),
                        Decimal(0),
                    )
                    / Decimal(count),
                    "mean_cold_cost_usd": sum(
                        (cast(Decimal, row["cold_cost_usd"]) for row in group), Decimal(0)
                    )
                    / Decimal(count),
                    "mean_agent_binding_fraction": sum(
                        float(row["agent_binding_fraction"]) for row in group
                    )
                    / count,
                }
            )
    return rows


def _mode_rows(metrics: RunMetrics) -> list[dict[str, Any]]:
    totals: Counter[str] = Counter()
    for (dimension, _), count in metrics.mode_counts.items():
        totals[dimension] += count
    return [
        {
            "dimension": dimension,
            "value": value,
            "count": count,
            "fraction_within_dimension": Decimal(count) / Decimal(totals[dimension])
            if totals[dimension]
            else None,
        }
        for (dimension, value), count in sorted(metrics.mode_counts.items())
    ]


def _baseline_summary(current: RunMetrics, baseline: RunMetrics | None) -> dict[str, Any] | None:
    if baseline is None:
        return None
    if {row["document_id"] for row in current.documents} != {
        row["document_id"] for row in baseline.documents
    }:
        raise ValueError("baseline is not the exact same document cohort")
    if current.full_corpus_documents != baseline.full_corpus_documents:
        raise ValueError("current and baseline source-corpus sizes disagree")
    current_requests = sum(int(row["provider_requests"]) for row in current.documents)
    baseline_requests = sum(int(row["provider_requests"]) for row in baseline.documents)
    current_certified = sum(row["status"] == "certified" for row in current.documents)
    baseline_certified = sum(row["status"] == "certified" for row in baseline.documents)
    return {
        "run": baseline.root.name,
        "commitSha256": baseline.commit_sha256,
        "documents": len(baseline.documents),
        "certified": baseline_certified,
        "observedCostUsd": str(baseline.observed_cost_usd),
        "coldCostUsd": str(baseline.cold_cost_usd),
        "providerRequests": baseline_requests,
        "observedCostSavingsUsd": str(baseline.observed_cost_usd - current.observed_cost_usd),
        "observedCostReductionFraction": str(
            (baseline.observed_cost_usd - current.observed_cost_usd) / baseline.observed_cost_usd
        )
        if baseline.observed_cost_usd
        else None,
        "requestReduction": baseline_requests - current_requests,
        "requestReductionFraction": str(
            Decimal(baseline_requests - current_requests) / Decimal(baseline_requests)
        )
        if baseline_requests
        else None,
        "certifiedDocumentDelta": current_certified - baseline_certified,
    }


def _paired_baseline_rows(current: RunMetrics, baseline: RunMetrics | None) -> list[dict[str, Any]]:
    if baseline is None:
        return []
    baseline_by_id = {str(row["document_id"]): row for row in baseline.documents}
    return [
        {
            "ordinal": row["ordinal"],
            "document_id": row["document_id"],
            "current_status": row["status"],
            "baseline_status": baseline_by_id[str(row["document_id"])]["status"],
            "current_first_pass_certified": row["first_pass_certified"],
            "baseline_first_pass_certified": baseline_by_id[str(row["document_id"])][
                "first_pass_certified"
            ],
            "current_provider_requests": row["provider_requests"],
            "baseline_provider_requests": baseline_by_id[str(row["document_id"])][
                "provider_requests"
            ],
            "request_delta": int(row["provider_requests"])
            - int(baseline_by_id[str(row["document_id"])]["provider_requests"]),
            "current_observed_cost_usd": row["observed_cost_usd"],
            "baseline_observed_cost_usd": baseline_by_id[str(row["document_id"])][
                "observed_cost_usd"
            ],
            "observed_cost_delta_usd": cast(Decimal, row["observed_cost_usd"])
            - cast(
                Decimal,
                baseline_by_id[str(row["document_id"])]["observed_cost_usd"],
            ),
            "current_cold_cost_usd": row["cold_cost_usd"],
            "baseline_cold_cost_usd": baseline_by_id[str(row["document_id"])]["cold_cost_usd"],
            "cold_cost_delta_usd": cast(Decimal, row["cold_cost_usd"])
            - cast(Decimal, baseline_by_id[str(row["document_id"])]["cold_cost_usd"]),
        }
        for row in current.documents
    ]


def analyze_template_extraction(
    *,
    run_dir: Path,
    output_parent: Path,
    run_name: str,
    baseline_run_dir: Path | None = None,
    expected_documents: int = 200,
    outlier_limit: int = 10,
) -> Path:
    """Publish integrity-pinned, success-oriented EDA for one extraction cohort."""

    if expected_documents <= 0:
        raise ValueError("expected_documents must be positive")
    if outlier_limit <= 0:
        raise ValueError("outlier_limit must be positive")
    current = _extract_run(
        run_dir,
        expected_documents=expected_documents,
        inspect_templates=True,
    )
    baseline = (
        _extract_run(
            baseline_run_dir,
            expected_documents=expected_documents,
            inspect_templates=False,
        )
        if baseline_run_dir is not None
        else None
    )
    baseline_summary = _baseline_summary(current, baseline)
    documents = list(current.documents)
    stages = list(current.stages)
    certified = sum(row["status"] == "certified" for row in documents)
    fresh_executions = sum(bool(row["fresh_execution"]) for row in documents)
    fresh_certified = sum(
        bool(row["fresh_execution"]) and row["status"] == "certified" for row in documents
    )
    first_pass = sum(bool(row["first_pass_certified"]) for row in documents)
    requests = sum(int(row["provider_requests"]) for row in documents)
    observed_mean = current.observed_cost_usd / Decimal(expected_documents)
    cold_mean = current.cold_cost_usd / Decimal(expected_documents)
    projections = _projection_rows(current, baseline)
    paired_baseline = _paired_baseline_rows(current, baseline)
    role_summary = _role_summary(stages)
    correlations = _correlations(documents)
    outliers = _outliers(documents, limit=outlier_limit)
    strata = _strata(documents)
    modes = _mode_rows(current)
    deterministic = sum(int(row["deterministic_bindings"]) for row in documents)
    agent = sum(int(row["agent_assisted_bindings"]) for row in documents)
    bindings = deterministic + agent
    coherence_counts = {
        value: count
        for (dimension, value), count in current.mode_counts.items()
        if dimension == "coherence_contract"
    }
    coherence_contracts = sum(coherence_counts.values())
    coherence_bindings = sum(int(row["coherence_bindings"]) for row in documents)
    failed_gates = sum(int(row["failed"]) for row in current.quality_gates)
    configured_gate = current.summary.get("acceptanceGatePassed")
    if not isinstance(configured_gate, bool):
        raise ValueError("summary.acceptanceGatePassed is not a boolean")
    summary = {
        "schemaVersion": 1,
        "inputRun": current.root.name,
        "inputCommitSha256": current.commit_sha256,
        "documents": expected_documents,
        "statusCounts": dict(Counter(str(row["status"]) for row in documents)),
        "certifiedDocuments": certified,
        "acceptanceRate": str(Decimal(certified) / Decimal(expected_documents)),
        "freshExecutionDocuments": fresh_executions,
        "freshCertifiedDocuments": fresh_certified,
        "firstPassCertifiedDocuments": first_pass,
        "firstPassRate": str(Decimal(first_pass) / Decimal(fresh_executions))
        if fresh_executions
        else None,
        "firstPassRateAmongCertified": str(Decimal(first_pass) / Decimal(fresh_certified))
        if fresh_certified
        else None,
        "resumeModeCounts": dict(Counter(str(row["resume_mode"]) for row in documents)),
        "configuredAcceptanceGatePassed": configured_gate,
        "templateQualityGateFailures": failed_gates,
        "providerRequests": requests,
        "meanProviderRequestsPerDocument": str(Decimal(requests) / Decimal(expected_documents)),
        "additionalRequestsBeyondTwo": sum(
            int(row["additional_requests_beyond_two"]) for row in documents
        ),
        "structuredOutputRetryRequests": sum(
            int(row["structured_output_retry_requests"]) for row in documents
        ),
        "additionalStages": sum(int(row["additional_stage_count"]) for row in documents),
        "observedCostUsd": str(current.observed_cost_usd),
        "coldNormalizedCostUsd": str(current.cold_cost_usd),
        "meanObservedCostPerDocumentUsd": str(observed_mean),
        "meanColdNormalizedCostPerDocumentUsd": str(cold_mean),
        "cacheColdUpliftUsd": str(current.cold_cost_usd - current.observed_cost_usd),
        "coldNormalization": (
            "Each cache-read input token is repriced as a cache-write token using its role's "
            "committed provider pricing; ordinary input, existing writes, and output are unchanged."
        ),
        "deterministicBindings": deterministic,
        "agentAssistedBindings": agent,
        "coherenceContracts": coherence_contracts,
        "coherenceBindings": coherence_bindings,
        "coherenceContractCounts": dict(sorted(coherence_counts.items())),
        "reviewRequiredCoherenceBindings": coherence_counts.get("review_required", 0),
        "deterministicBindingFraction": str(Decimal(deterministic) / Decimal(bindings))
        if bindings
        else None,
        "agentAssistedBindingFraction": str(Decimal(agent) / Decimal(bindings))
        if bindings
        else None,
        "fullCorpusDocuments": current.full_corpus_documents,
        "observedFullCorpusProjectionUsd": str(
            observed_mean * Decimal(current.full_corpus_documents)
        ),
        "coldFullCorpusProjectionUsd": str(cold_mean * Decimal(current.full_corpus_documents)),
        "baseline": baseline_summary,
        "integrityChecks": {
            "commitReceiptAndArtifactLedgerVerified": True,
            "selectionPreflightResultsAligned": True,
            "catalogExactlyMatchesCertifiedResults": True,
            "summaryUsageReconciledToStages": True,
            "catalogCountsReconciledToSummary": True,
            "certifiedTemplatesInspected": certified,
        },
    }
    report_lines = [
        "# Template compilation EDA",
        "",
        "## Outcome",
        "",
        f"- Certified: **{certified}/{expected_documents}** "
        f"(**{certified / expected_documents:.2%}**).",
        (
            f"- Fresh strict first-pass certified: **{first_pass}/{fresh_executions}** "
            f"(**{first_pass / fresh_executions:.2%}**)."
            if fresh_executions
            else "- Fresh strict first-pass rate: **not estimable**; all documents were "
            "resumed, recertified, or routed to source-integrity review."
        ),
        f"- Provider requests: **{requests}** "
        f"(**{requests / expected_documents:.3f} per document**).",
        f"- Observed cost: **${current.observed_cost_usd}** (**${observed_mean} per document**).",
        f"- Cold-normalized cost: **${current.cold_cost_usd}** (**${cold_mean} per document**).",
        f"- Deterministic binding coverage: **{deterministic}/{bindings}** "
        f"(**{deterministic / bindings:.2%}**)."
        if bindings
        else "- No compiled bindings.",
        f"- Explicit semantic-coherence contracts: **{coherence_contracts}** across "
        f"**{coherence_bindings}** participating bindings; "
        f"review-required contracts in certified templates: "
        f"**{coherence_counts.get('review_required', 0)}**.",
        f"- Certified-template quality-gate failures: **{failed_gates}**.",
        "",
        "## Projection",
        "",
        f"- Source corpus: **{current.full_corpus_documents:,} documents**.",
        "- Observed-rate projection: "
        f"**${observed_mean * Decimal(current.full_corpus_documents)}**.",
        f"- Cold-normalized projection: **${cold_mean * Decimal(current.full_corpus_documents)}**.",
        "",
        "Cold normalization is deliberately conservative: every recorded cache-read input token "
        "is repriced as a cache write using the committed role-specific pricing. It does not "
        "constrain reasoning or output tokens.",
    ]
    if baseline_summary is not None:
        report_lines.extend(
            [
                "",
                "## Exact-cohort baseline",
                "",
                f"- Baseline run: `{baseline_summary['run']}`.",
                "- Observed cost reduction: "
                f"**${baseline_summary['observedCostSavingsUsd']}** "
                "(**"
                f"{Decimal(cast(str, baseline_summary['observedCostReductionFraction'])):.2%}"
                "**).",
                f"- Provider-request reduction: **{baseline_summary['requestReduction']}** "
                f"(**{Decimal(cast(str, baseline_summary['requestReductionFraction'])):.2%}**).",
                "- Certified-document delta: "
                f"**{int(baseline_summary['certifiedDocumentDelta']):+d}**.",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "A fresh first pass requires a non-resumed execution with exactly one successful "
            "compiler stage, one successful pass-verdict critic stage, and exactly two provider "
            "requests. This resumed optimization lineage therefore cannot estimate fresh-run "
            "first-pass yield. Cost projections are linear descriptive extrapolations over "
            "selected documents; the selection-strata and outlier tables must be reviewed before "
            "a scale-up decision.",
            "",
            "## Artifact map",
            "",
            "- `documents.csv`: joined selection, complexity, usage, cost, and coverage metrics.",
            "- `stages.csv` and `role-summary.csv`: compiler/critic request, token, and cost "
            "detail.",
            "- `quality-gates.csv`: aggregate certified-template gate outcomes.",
            "- `mode-counts.csv`: binding, render, realization, and value-kind distributions.",
            "- `complexity-correlations.csv`, `selection-strata.csv`, and `outliers.csv`: transfer "
            "and tail diagnostics.",
            "- `projections.csv`: observed and cold-normalized sample/full-corpus economics.",
            "- `paired-baseline.csv` (when a baseline is supplied): exact-document request and "
            "cost deltas.",
            "- Eight numbered PNGs: acceptance, retries, cost, token, coverage, complexity, and "
            "baseline/projection views.",
            "",
        ]
    )
    report = "\n".join(report_lines)

    baseline_commit_sha = baseline.commit_sha256 if baseline is not None else None
    transaction = sha256_bytes(
        canonical_json_bytes(
            {
                "schemaVersion": 1,
                "inputCommitSha256": current.commit_sha256,
                "baselineCommitSha256": baseline_commit_sha,
                "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
                "expectedDocuments": expected_documents,
                "outlierLimit": outlier_limit,
                "runName": run_name,
            }
        )
    )
    staged = StagedArtifactRun(
        output_parent=output_parent.resolve(),
        run_name=run_name,
        transaction_sha256=transaction,
    )
    staged.recover_interrupted_temporary_files()
    staged.publish_json("input-commit.json", current.commit)
    if baseline is not None:
        staged.publish_json("baseline-commit.json", baseline.commit)
    staged.publish_json("summary.json", summary)
    staged.publish_bytes("REPORT.md", (report + "\n").encode("utf-8"))
    document_columns = tuple(documents[0])
    stage_columns = (
        tuple(stages[0])
        if stages
        else (
            "document_id",
            "role",
            "pass_number",
            "status",
            "verdict",
            "duration_seconds",
            "requests",
            "structured_output_retry_requests",
            "input_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "output_tokens",
            "reasoning_tokens",
            "visible_output_tokens",
            "observed_cost_usd",
            "cold_cost_usd",
            "critic_findings",
        )
    )
    staged.publish_bytes("documents.csv", _csv_bytes(documents, document_columns))
    staged.publish_bytes("stages.csv", _csv_bytes(stages, stage_columns))
    staged.publish_bytes("role-summary.csv", _csv_bytes(role_summary, tuple(role_summary[0])))
    gate_columns = (
        "gate",
        "passed",
        "failed",
        "not_applicable",
        "documents_evaluated",
    )
    staged.publish_bytes("quality-gates.csv", _csv_bytes(current.quality_gates, gate_columns))
    staged.publish_bytes("mode-counts.csv", _csv_bytes(modes, tuple(modes[0])))
    staged.publish_bytes(
        "complexity-correlations.csv", _csv_bytes(correlations, tuple(correlations[0]))
    )
    staged.publish_bytes("selection-strata.csv", _csv_bytes(strata, tuple(strata[0])))
    outlier_columns = (
        "metric",
        "rank",
        "value",
        "document_id",
        "ordinal",
        "status",
        "carrier_family",
        "document_type",
        "pages",
        "source_lines",
        "provider_requests",
        "cold_cost_usd",
    )
    staged.publish_bytes("outliers.csv", _csv_bytes(outliers, outlier_columns))
    finding_columns = ("document_id", "critic_pass", "finding_kind", "line_count")
    staged.publish_bytes(
        "critic-findings.csv", _csv_bytes(current.critic_findings, finding_columns)
    )
    staged.publish_bytes("projections.csv", _csv_bytes(projections, tuple(projections[0])))
    if paired_baseline:
        staged.publish_bytes(
            "paired-baseline.csv",
            _csv_bytes(paired_baseline, tuple(paired_baseline[0])),
        )
    staged.publish_bytes("01-acceptance-first-pass.png", _acceptance_figure(documents))
    staged.publish_bytes("02-requests-repairs.png", _request_figure(documents))
    staged.publish_bytes(
        "03-observed-cold-cost-distribution.png", _cost_distribution_figure(documents)
    )
    staged.publish_bytes("04-cost-drivers.png", _cost_drivers_figure(documents, role_summary))
    staged.publish_bytes("05-role-token-composition.png", _role_token_figure(role_summary))
    staged.publish_bytes("06-deterministic-agent-coverage.png", _coverage_figure(documents))
    staged.publish_bytes("07-complexity-relationships.png", _complexity_figure(documents))
    staged.publish_bytes("08-baseline-projection.png", _baseline_figure(projections))
    expected = [
        "01-acceptance-first-pass.png",
        "02-requests-repairs.png",
        "03-observed-cold-cost-distribution.png",
        "04-cost-drivers.png",
        "05-role-token-composition.png",
        "06-deterministic-agent-coverage.png",
        "07-complexity-relationships.png",
        "08-baseline-projection.png",
        "REPORT.md",
        "complexity-correlations.csv",
        "critic-findings.csv",
        "documents.csv",
        "input-commit.json",
        "mode-counts.csv",
        "outliers.csv",
        "projections.csv",
        "quality-gates.csv",
        "role-summary.csv",
        "selection-strata.csv",
        "stages.csv",
        "summary.json",
    ]
    if baseline is not None:
        expected.extend(("baseline-commit.json", "paired-baseline.csv"))
    staged.commit(
        expected_artifacts=expected,
        metadata={
            "schemaVersion": 1,
            "inputRun": current.root.name,
            "documents": expected_documents,
            "certified": certified,
            "firstPassCertified": first_pass,
            "baselineRun": baseline.root.name if baseline is not None else None,
        },
    )
    return staged.final_root


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Publish integrity-checked success EDA for a template-extraction run."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-parent", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--baseline-run-dir", type=Path)
    parser.add_argument("--expected-documents", type=int, default=200)
    parser.add_argument("--outlier-limit", type=int, default=10)
    arguments = parser.parse_args(argv)
    output = analyze_template_extraction(
        run_dir=arguments.run_dir,
        output_parent=arguments.output_parent,
        run_name=arguments.run_name,
        baseline_run_dir=arguments.baseline_run_dir,
        expected_documents=arguments.expected_documents,
        outlier_limit=arguments.outlier_limit,
    )
    print(output)


if __name__ == "__main__":
    main()
