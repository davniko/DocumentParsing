#!/usr/bin/env python3
"""Build an immutable accepted-versus-quarantined audit of a raw-text pipeline run.

The production pipeline deliberately publishes no training records when any case is blocked.
This analyzer preserves that boundary: it validates the committed pipeline and every referenced
component run, replays the deterministic certification host audit, and writes only review
workbooks, metrics, and manifests.  It never copies candidates into a training dataset.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher, unified_diff
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.raw_text_certification import (
    CertificationCaseResult,
    CertificationStage,
    SemanticAuditOutput,
    _host_audit,
)
from document_ocr.synthesis.raw_text_pipeline import _PipelineResumeLineageRow
from document_ocr.synthesis.run_safety import StagedArtifactRun, StagedCommitReceipt
from document_ocr.training.config import resolve_config_path

_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)
_SHA256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_DOCUMENT_ID = Annotated[str, StringConstraints(pattern=r"^doc_[0-9a-f]{64}$")]
_PAGE_MARKER = re.compile(r"--- PAGE [0-9]+ ---")
_RUN_VERSION = re.compile(r"(?:^|[-_])(v[0-9]+)(?:[-_]|$)")
_RANDOM_SELECTION = "deterministic_random"


class ManualAuditCase(BaseModel):
    """One independent reviewer decision over a machine-certified candidate."""

    model_config = _STRICT

    documentId: _DOCUMENT_ID
    semanticConsistency: bool
    templateFidelity: bool
    privacyRewritten: bool
    selectionBasis: Annotated[tuple[str, ...], Field(min_length=1)]
    notes: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

    @model_validator(mode="after")
    def selection_basis_is_unique(self) -> ManualAuditCase:
        if len(self.selectionBasis) != len(set(self.selectionBasis)):
            raise ValueError("manual-audit selectionBasis contains duplicates")
        return self


class ManualAudit(BaseModel):
    """Pinned manual audit input with a reproducible random accepted-case subset."""

    model_config = _STRICT

    schemaVersion: Literal[1]
    pipelineCommitSha256: _SHA256
    randomSampleSize: Annotated[int, Field(ge=1)]
    cases: Annotated[tuple[ManualAuditCase, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def cases_are_unique(self) -> ManualAudit:
        ids = tuple(row.documentId for row in self.cases)
        if len(ids) != len(set(ids)):
            raise ValueError("manual audit repeats a document")
        return self


@dataclass(frozen=True, slots=True)
class RunReference:
    path: str
    root: Path
    commit_sha256: str
    transaction_sha256: str


@dataclass(frozen=True, slots=True)
class LoadedCase:
    lineage: _PipelineResumeLineageRow
    inventory: RunReference
    certification: RunReference | None
    source: str
    candidate: str
    source_label: dict[str, Any]
    target_label: dict[str, Any]
    contract: dict[str, Any]
    certification_result: CertificationCaseResult | None
    findings: tuple[dict[str, Any], ...]


def _ensure_under_project_root(project_root: Path, value: Path, *, label: str) -> Path:
    path = value.resolve()
    try:
        path.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"{label} must resolve inside project_root: {path}") from error
    return path


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(read_regular_file_bytes(path))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _json_rows(path: Path) -> tuple[dict[str, Any], ...]:
    payload = read_regular_file_bytes(path)
    if payload and not payload.endswith(b"\n"):
        raise ValueError(f"JSONL artifact lacks a terminal newline: {path}")
    output: list[dict[str, Any]] = []
    for number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            raise ValueError(f"JSONL artifact contains a blank row: {path}:{number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object: {path}:{number}")
        output.append(value)
    return tuple(output)


def _validate_run_root(root: Path) -> tuple[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"run must be a regular directory: {root}")
    commit_path = root / "_COMMIT.json"
    receipt = StagedCommitReceipt.model_validate_json(
        read_regular_file_bytes(commit_path), strict=True
    )
    if receipt.run_name != root.name:
        raise ValueError(f"commit receipt run name differs: {root}")
    StagedArtifactRun(
        output_parent=root.parent,
        run_name=root.name,
        transaction_sha256=receipt.transaction_sha256,
    ).validate_committed_run()
    return sha256_file(commit_path), receipt.transaction_sha256


def _run_reference(
    *,
    project_root: Path,
    raw: Any,
    cache: dict[tuple[str, str, str], RunReference],
) -> RunReference:
    path = cast(str, raw.path)
    commit_sha256 = cast(str, raw.commitSha256)
    transaction_sha256 = cast(str, raw.transactionSha256)
    key = (path, commit_sha256, transaction_sha256)
    existing = cache.get(key)
    if existing is not None:
        return existing
    root = _ensure_under_project_root(
        project_root,
        resolve_config_path(project_root, path),
        label="pipeline component run",
    )
    observed_commit, observed_transaction = _validate_run_root(root)
    if observed_commit != commit_sha256 or observed_transaction != transaction_sha256:
        raise ValueError(f"pipeline component run differs from its lineage pin: {root}")
    reference = RunReference(
        path=path,
        root=root,
        commit_sha256=commit_sha256,
        transaction_sha256=transaction_sha256,
    )
    cache[key] = reference
    return reference


def _certification_findings(case_root: Path) -> tuple[dict[str, Any], ...]:
    values = json.loads(read_regular_file_bytes(case_root / "stages.json"))
    if not isinstance(values, list):
        raise ValueError(f"expected certification stage array: {case_root}")
    stages = tuple(
        CertificationStage.model_validate_json(canonical_json_bytes(row), strict=True)
        for row in values
    )
    outputs = tuple(row.modelOutput for row in stages if row.modelOutput is not None)
    if not outputs:
        return ()
    parsed = SemanticAuditOutput.model_validate_json(
        canonical_json_bytes(outputs[-1]), strict=True
    )
    return tuple(row.model_dump(mode="json") for row in parsed.findings)


def _matching_candidate_reference(
    *,
    row: _PipelineResumeLineageRow,
    project_root: Path,
    cache: dict[tuple[str, str, str], RunReference],
) -> RunReference:
    matches = tuple(
        history.run
        for history in row.history
        if getattr(history, "candidateSha256", None) == row.currentCandidateSha256
    )
    if not matches:
        raise ValueError(f"current candidate has no lineage owner: {row.documentId}")
    return _run_reference(project_root=project_root, raw=matches[-1], cache=cache)


def _load_case(
    *,
    project_root: Path,
    row: _PipelineResumeLineageRow,
    cache: dict[tuple[str, str, str], RunReference],
) -> LoadedCase:
    inventory = _run_reference(project_root=project_root, raw=row.inventoryRun, cache=cache)
    inventory_case = inventory.root / "cases" / row.documentId
    source = read_regular_file_bytes(inventory_case / "source.txt").decode("utf-8")
    source_label = _json_object(inventory_case / "source-label.json")
    target_label = _json_object(inventory_case / "target-label.json")
    if sha256_bytes(source.encode()) != row.sourceTextSha256:
        raise ValueError(f"source text differs from pipeline lineage: {row.documentId}")

    certification_rows = tuple(
        history for history in row.history if history.stage == "certification"
    )
    certification: RunReference | None = None
    result: CertificationCaseResult | None = None
    findings: tuple[dict[str, Any], ...] = ()
    if certification_rows:
        latest = certification_rows[-1]
        certification = _run_reference(project_root=project_root, raw=latest.run, cache=cache)
        certification_case = certification.root / "cases" / row.documentId
        candidate = read_regular_file_bytes(certification_case / "final.txt").decode("utf-8")
        result = CertificationCaseResult.model_validate_json(
            read_regular_file_bytes(certification_case / "result.json"), strict=True
        )
        contract = _json_object(certification_case / "source-contract.json")
        findings = _certification_findings(certification_case)
        if (
            result.documentId != row.documentId
            or result.inputCandidateSha256 != row.currentCandidateSha256
            or result.finalTextSha256 != row.currentCandidateSha256
            or result.sourceTextSha256 != row.sourceTextSha256
            or result.status != latest.status
            or result.semanticFindings != len(findings)
        ):
            raise ValueError(f"latest certification differs from lineage: {row.documentId}")
        for filename, expected in (
            ("source.txt", source.encode()),
            ("source-label.json", canonical_json_bytes(source_label) + b"\n"),
            ("target-label.json", canonical_json_bytes(target_label) + b"\n"),
        ):
            observed = read_regular_file_bytes(certification_case / filename)
            if filename == "source.txt":
                same = observed == expected
            else:
                same = json.loads(observed) == json.loads(expected)
            if not same:
                raise ValueError(
                    f"certification input identity differs for {row.documentId}: {filename}"
                )
    else:
        candidate_owner = _matching_candidate_reference(
            row=row, project_root=project_root, cache=cache
        )
        candidate_case = candidate_owner.root / "cases" / row.documentId
        candidate = read_regular_file_bytes(candidate_case / "final.txt").decode("utf-8")
        contract_name = (
            "source-contract.json"
            if (candidate_case / "source-contract.json").is_file()
            else "contract.json"
        )
        contract = _json_object(candidate_case / contract_name)

    if sha256_bytes(candidate.encode()) != row.currentCandidateSha256:
        raise ValueError(f"current candidate bytes differ from lineage: {row.documentId}")
    audit = _host_audit(source=source, output=candidate, contract=contract)
    if result is not None and audit != result.hostAudit:
        raise ValueError(f"current host-audit replay differs: {row.documentId}")
    if row.status == "certified":
        if (
            result is None
            or result.status != "certified"
            or findings
            or not audit.passed
            or row.finalCertificationRun is None
            or certification is None
            or certification.path != row.finalCertificationRun.path
            or certification.commit_sha256 != row.finalCertificationRun.commitSha256
            or certification.transaction_sha256 != row.finalCertificationRun.transactionSha256
        ):
            raise ValueError(f"certified case lacks exact clean evidence: {row.documentId}")
    elif row.blockingReason is None:
        raise ValueError(f"blocked case lacks a blocking reason: {row.documentId}")

    return LoadedCase(
        lineage=row,
        inventory=inventory,
        certification=certification,
        source=source,
        candidate=candidate,
        source_label=source_label,
        target_label=target_label,
        contract=contract,
        certification_result=result,
        findings=findings,
    )


def _edit_metrics(source: str, candidate: str) -> dict[str, Any]:
    source_lines = source.splitlines()
    candidate_lines = candidate.splitlines()
    matcher = SequenceMatcher(a=source_lines, b=candidate_lines, autojunk=False)
    equal = changed_source = changed_candidate = blocks = 0
    for tag, first_start, first_end, second_start, second_end in matcher.get_opcodes():
        if tag == "equal":
            equal += first_end - first_start
        else:
            blocks += 1
            changed_source += first_end - first_start
            changed_candidate += second_end - second_start
    denominator = max(len(source_lines), len(candidate_lines), 1)
    source_pages = tuple(line for line in source_lines if _PAGE_MARKER.fullmatch(line))
    candidate_pages = tuple(line for line in candidate_lines if _PAGE_MARKER.fullmatch(line))
    return {
        "source_characters": len(source),
        "candidate_characters": len(candidate),
        "character_delta": len(candidate) - len(source),
        "source_lines": len(source_lines),
        "candidate_lines": len(candidate_lines),
        "line_count_delta": len(candidate_lines) - len(source_lines),
        "blank_line_delta": sum(not line.strip() for line in candidate_lines)
        - sum(not line.strip() for line in source_lines),
        "changed_source_lines": changed_source,
        "changed_candidate_lines": changed_candidate,
        "changed_blocks": blocks,
        "unchanged_line_fraction": equal / denominator,
        "pages": len(source_pages),
        "page_markers_preserved": source_pages == candidate_pages,
        "terminal_newline_preserved": source.endswith("\n") == candidate.endswith("\n"),
    }


def _target_features(label: Mapping[str, Any]) -> dict[str, int]:
    patch = label.get("documentPatch")
    if not isinstance(patch, Mapping):
        raise ValueError("target label lacks documentPatch")
    parties = patch.get("parties") or {}
    containers = patch.get("containers") or []
    cargo_groups = patch.get("cargoGroups") or []
    packages = patch.get("cargoPackages") or []
    allocation_groups = patch.get("cargoAllocationGroups") or []
    if not all(isinstance(value, Sequence) for value in (containers, cargo_groups, packages)):
        raise ValueError("target topology collections are invalid")
    return {
        "containers": len(containers),
        "cargo_groups": len(cargo_groups),
        "packages": len(packages),
        "allocation_groups": len(allocation_groups),
        "allocations": sum(
            len(row.get("allocations") or [])
            for row in allocation_groups
            if isinstance(row, Mapping)
        ),
        "dangerous_goods": sum(
            len(row.get("dangerousGoods") or []) for row in cargo_groups if isinstance(row, Mapping)
        ),
        "hs_codes": sum(
            len(row.get("hsCodes") or []) for row in cargo_groups if isinstance(row, Mapping)
        ),
        "temperature_containers": sum(
            isinstance(row, Mapping) and row.get("temperatureSetpoint") is not None
            for row in containers
        ),
        "party_roles": len(parties) if isinstance(parties, Mapping) else 0,
    }


def _semantic_values(
    *, document_id: str, status: str, variant: str, label: Mapping[str, Any]
) -> list[dict[str, str]]:
    patch = label.get("documentPatch")
    if not isinstance(patch, Mapping):
        return []
    output: list[dict[str, str]] = []

    def add(dimension: str, value: object, role: str = "") -> None:
        if isinstance(value, str) and value.strip():
            output.append(
                {
                    "document_id": document_id,
                    "status": status,
                    "variant": variant,
                    "dimension": dimension,
                    "role": role,
                    "value": value.strip(),
                }
            )

    for row in patch.get("cargoPackages") or ():
        if isinstance(row, Mapping):
            add("package_type", row.get("typeCategory"), str(row.get("groupId") or ""))
    for row in patch.get("containers") or ():
        if isinstance(row, Mapping):
            add("container_type", row.get("typeCategory"))
            add("container_size", row.get("sizeCategory"))
    parties = patch.get("parties") or {}
    if isinstance(parties, Mapping):
        for role, raw in parties.items():
            values = raw if role == "notifyParties" and isinstance(raw, list) else [raw]
            for row in values:
                if isinstance(row, Mapping):
                    add("party_country", row.get("country"), str(role))
    route = patch.get("route") or {}
    if isinstance(route, Mapping):
        for role, row in route.items():
            if isinstance(row, Mapping):
                add("route_country", row.get("country"), str(role))
    for group in patch.get("cargoGroups") or ():
        if not isinstance(group, Mapping):
            continue
        group_id = str(group.get("groupId") or "")
        for code in group.get("hsCodes") or ():
            if isinstance(code, str) and len(code) >= 2 and code[:2].isdigit():
                add("hs_chapter", code[:2], group_id)
        for row in group.get("dangerousGoods") or ():
            if isinstance(row, Mapping):
                add("hazard_category", row.get("hazardCategory"), group_id)
    return output


def _case_metrics(case: LoadedCase) -> dict[str, Any]:
    result = case.certification_result
    history_counts = Counter(row.stage for row in case.lineage.history)
    return {
        "document_id": case.lineage.documentId,
        "status": case.lineage.status,
        "blocking_reason": case.lineage.blockingReason or "",
        "inventory_round": case.lineage.inventoryRound,
        "correction_rounds": case.lineage.correctionRounds,
        "inventory_attempts": history_counts["inventory"],
        "certification_attempts": history_counts["certification"],
        "correction_attempts": history_counts["correction"],
        "latest_certification_status": result.status if result is not None else "not_audited",
        "latest_semantic_findings": len(case.findings),
        "latest_host_audit_passed": bool(result and result.hostAudit.passed),
        "source_sha256": case.lineage.sourceTextSha256,
        "candidate_sha256": case.lineage.currentCandidateSha256,
        "target_sha256": sha256_bytes(canonical_json_bytes(case.target_label)),
        "inventory_run": case.inventory.path,
        "latest_certification_run": case.certification.path if case.certification else "",
        **_edit_metrics(case.source, case.candidate),
        **_target_features(case.target_label),
    }


def _cost_rows(
    *,
    project_root: Path,
    lineage: Sequence[_PipelineResumeLineageRow],
    cache: dict[tuple[str, str, str], RunReference],
) -> list[dict[str, Any]]:
    references: dict[tuple[str, str, str, str], RunReference] = {}
    for case in lineage:
        for history in case.history:
            reference = _run_reference(project_root=project_root, raw=history.run, cache=cache)
            key = (
                history.stage,
                reference.path,
                reference.commit_sha256,
                reference.transaction_sha256,
            )
            references[key] = reference
    output: list[dict[str, Any]] = []
    for (stage, _path, _commit, _transaction), reference in sorted(references.items()):
        summary = _json_object(reference.root / "summary.json")
        match = _RUN_VERSION.search(reference.root.name)
        reported = summary.get("providerReportedCostUsd")
        estimated = summary.get("estimatedCostUsd")
        output.append(
            {
                "stage": stage,
                "lineage_version": match.group(1) if match is not None else "unversioned",
                "run_id": reference.root.name,
                "path": reference.path,
                "documents": int(summary.get("liveDocuments") or summary.get("documents") or 0),
                "requests": int(summary.get("requests") or 0),
                "provider_attempts": int(summary.get("providerAttempts") or 0),
                "failed_provider_attempts": int(summary.get("failedProviderAttempts") or 0),
                "input_tokens": int(summary.get("inputTokens") or 0),
                "reasoning_tokens": int(summary.get("reasoningTokens") or 0),
                "visible_output_tokens": int(summary.get("visibleOutputTokens") or 0),
                "provider_reported_cost_usd": float(reported or 0),
                "estimated_cost_usd": float(estimated or 0),
                "cost_receipt_available": reported is not None,
                "wall_seconds": float(summary.get("wallSeconds") or 0),
                "commit_sha256": reference.commit_sha256,
                "transaction_sha256": reference.transaction_sha256,
            }
        )
    return output


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        return b""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _safe_markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("`", "\\`").replace("\n", " ")


def _target_summary(label: Mapping[str, Any]) -> dict[str, Any]:
    patch = label.get("documentPatch")
    if not isinstance(patch, Mapping):
        return {}
    return {
        key: patch.get(key)
        for key in (
            "billOfLadingNumber",
            "parties",
            "route",
            "transport",
            "shippedOnBoardDate",
            "freight",
            "containers",
            "cargoGroups",
            "cargoPackages",
            "cargoAllocationGroups",
        )
        if patch.get(key) is not None
    }


def _case_workbook(case: LoadedCase) -> bytes:
    source_lines = case.source.splitlines()
    candidate_lines = case.candidate.splitlines()
    if len(source_lines) == len(candidate_lines):
        changes = [
            (number, before, after)
            for number, (before, after) in enumerate(
                zip(source_lines, candidate_lines, strict=True), start=1
            )
            if before != after
        ]
    else:
        changes = []
    rows = [
        f"# {case.lineage.documentId}",
        "",
        f"- Outcome: **{case.lineage.status}**",
        f"- Correction rounds: **{case.lineage.correctionRounds}**",
        f"- Current candidate SHA-256: `{case.lineage.currentCandidateSha256}`",
        f"- Blocking reason: {_safe_markdown(case.lineage.blockingReason or 'none')}",
        "",
        "## Latest independent findings",
        "",
    ]
    if not case.findings:
        rows.append("No semantic findings in the latest certification.")
    for finding in case.findings:
        evidence = "; ".join(
            f"{row['lineId']} `{_safe_markdown(row['currentFragment'])}`"
            for row in finding["evidence"]
        )
        rows.append(
            f"- **{finding['findingKind']}** — {evidence}: {_safe_markdown(finding['problem'])}"
        )
    rows.extend(
        [
            "",
            "## Synthetic target summary",
            "",
            "```json",
            json.dumps(_target_summary(case.target_label), indent=2, ensure_ascii=False),
            "```",
            "",
            "## Changed-line table",
            "",
        ]
    )
    if changes:
        rows.extend(
            [
                "| Line | Source | Current candidate |",
                "|---:|---|---|",
                *(
                    f"| {number} | `{_safe_markdown(before)}` | `{_safe_markdown(after)}` |"
                    for number, before, after in changes
                ),
            ]
        )
    else:
        rows.append(
            "Physical line topology differs; inspect the unified diff below."
            if case.source != case.candidate
            else "Candidate is byte-identical to the source."
        )
    difference = "".join(
        unified_diff(
            case.source.splitlines(keepends=True),
            case.candidate.splitlines(keepends=True),
            fromfile="source.txt",
            tofile="current-candidate.txt",
        )
    )
    rows.extend(("", "## Unified diff", "", "```diff", difference.rstrip(), "```", ""))
    return "\n".join(rows).encode("utf-8")


def _manifest_row(
    case: LoadedCase,
    metrics: Mapping[str, Any],
    *,
    audit_disposition: str,
    manual_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "documentId": case.lineage.documentId,
        "status": case.lineage.status,
        "auditDisposition": audit_disposition,
        "sourceSha256": case.lineage.sourceTextSha256,
        "candidateSha256": case.lineage.currentCandidateSha256,
        "targetSha256": metrics["target_sha256"],
        "inventoryRun": {
            "path": case.inventory.path,
            "commitSha256": case.inventory.commit_sha256,
            "transactionSha256": case.inventory.transaction_sha256,
        },
        "finalCertificationRun": (
            {
                "path": case.certification.path,
                "commitSha256": case.certification.commit_sha256,
                "transactionSha256": case.certification.transaction_sha256,
            }
            if case.lineage.status == "certified" and case.certification is not None
            else None
        ),
        "blockingReason": case.lineage.blockingReason,
        "latestSemanticFindings": len(case.findings),
        "hostAuditReplayPassed": bool(
            case.certification_result and case.certification_result.hostAudit.passed
        ),
        "manualAudit": dict(manual_audit) if manual_audit is not None else None,
    }


def _audit_disposition(
    *, status: str, manual_audit: Mapping[str, Any] | None
) -> Literal["confirmed", "manual_quarantine", "provisional", "pipeline_quarantine"]:
    if status == "blocked":
        if manual_audit is not None:
            raise ValueError("pipeline-quarantined case cannot have a manual accepted-case audit")
        return "pipeline_quarantine"
    if status != "certified":
        raise ValueError(f"unknown pipeline case status: {status}")
    if manual_audit is None:
        return "provisional"
    return "confirmed" if bool(manual_audit["all_checks_passed"]) else "manual_quarantine"


def _bar_svg(
    *, title: str, rows: Sequence[tuple[str, float]], x_label: str, color: str = "#4472C4"
) -> bytes:
    width = 1000
    left = 310
    right = 55
    top = 88
    row_height = 42
    height = max(220, top + len(rows) * row_height + 70)
    maximum = max((value for _label, value in rows), default=1.0) or 1.0
    usable = width - left - right
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="38" text-anchor="middle" font-family="sans-serif" '
        f'font-size="24" font-weight="bold">{html.escape(title)}</text>',
    ]
    for index, (label, value) in enumerate(rows):
        y = top + index * row_height
        bar_width = usable * value / maximum
        parts.extend(
            [
                f'<text x="{left - 12}" y="{y + 20}" text-anchor="end" '
                f'font-family="sans-serif" font-size="14">{html.escape(label)}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_width:.3f}" height="26" fill="{color}"/>',
                f'<text x="{left + bar_width + 8:.3f}" y="{y + 19}" '
                f'font-family="sans-serif" font-size="14">{value:g}</text>',
            ]
        )
    parts.extend(
        [
            f'<text x="{left + usable / 2}" y="{height - 20}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14">{html.escape(x_label)}</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts).encode("utf-8")


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial observation")
    z = 1.959963984540054
    observed = successes / trials
    denominator = 1 + z * z / trials
    center = (observed + z * z / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(observed * (1 - observed) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _manual_audit_rows(
    *,
    audit: ManualAudit,
    cases: Sequence[LoadedCase],
    pipeline_commit_sha256: str,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    if audit.pipelineCommitSha256 != pipeline_commit_sha256:
        raise ValueError("manual audit is pinned to a different pipeline commit")
    certified = tuple(
        case.lineage.documentId for case in cases if case.lineage.status == "certified"
    )
    certified_set = set(certified)
    unknown = {row.documentId for row in audit.cases} - certified_set
    if unknown:
        raise ValueError(f"manual audit includes non-certified cases: {sorted(unknown)}")
    if audit.randomSampleSize > len(certified):
        raise ValueError("manual-audit random sample exceeds the certified cohort")
    random_ids = tuple(
        sorted(
            certified,
            key=lambda document_id: sha256_bytes(
                f"{pipeline_commit_sha256}:{document_id}".encode()
            ),
        )[: audit.randomSampleSize]
    )
    audited_by_id = {row.documentId: row for row in audit.cases}
    missing_random = [
        document_id
        for document_id in random_ids
        if document_id not in audited_by_id
        or _RANDOM_SELECTION not in audited_by_id[document_id].selectionBasis
    ]
    if missing_random:
        raise ValueError(
            f"manual audit does not cover the exact deterministic random sample: {missing_random}"
        )
    rows = [
        {
            "document_id": row.documentId,
            "semantic_consistency": row.semanticConsistency,
            "template_fidelity": row.templateFidelity,
            "privacy_rewritten": row.privacyRewritten,
            "all_checks_passed": (
                row.semanticConsistency and row.templateFidelity and row.privacyRewritten
            ),
            "selection_basis": ";".join(row.selectionBasis),
            "notes": row.notes,
        }
        for row in audit.cases
    ]
    return rows, random_ids


def _report(
    *,
    summary: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
    costs: Sequence[Mapping[str, Any]],
    manual: Sequence[Mapping[str, Any]],
    random_ids: Sequence[str],
) -> bytes:
    status = Counter(row["status"] for row in metrics)
    finding_kinds = Counter(row["finding_kind"] for row in findings)
    certified = status["certified"]
    documents = len(metrics)
    provider_cost = sum(float(row["provider_reported_cost_usd"]) for row in costs)
    requests = sum(int(row["requests"]) for row in costs)
    provider_attempts = sum(int(row["provider_attempts"]) for row in costs)
    failed_attempts = sum(int(row["failed_provider_attempts"]) for row in costs)
    manual_passes = sum(bool(row["all_checks_passed"]) for row in manual)
    random_rows = [row for row in manual if row["document_id"] in set(random_ids)]
    random_passes = sum(bool(row["all_checks_passed"]) for row in random_rows)
    lower, upper = _wilson_interval(random_passes, len(random_rows))
    random_pass_fraction = random_passes / len(random_rows)
    estimated_clean_fraction = certified / documents * random_pass_fraction
    estimated_clean_lower = certified / documents * lower
    estimated_clean_upper = certified / documents * upper
    manually_rejected = len(manual) - manual_passes
    semantic_failures = sum(not bool(row["semantic_consistency"]) for row in manual)
    template_failures = sum(not bool(row["template_fidelity"]) for row in manual)
    privacy_failures = sum(not bool(row["privacy_rewritten"]) for row in manual)
    provisional = certified - len(manual)
    total_lines = sum(int(row["source_lines"]) for row in metrics)
    changed_lines = sum(int(row["changed_source_lines"]) for row in metrics)
    median_retention = sorted(float(row["unchanged_line_fraction"]) for row in metrics)[
        len(metrics) // 2
    ]
    host_passes = sum(bool(row["latest_host_audit_passed"]) for row in metrics)
    page_passes = sum(bool(row["page_markers_preserved"]) for row in metrics)
    line_passes = sum(int(row["line_count_delta"]) == 0 for row in metrics)
    blank_passes = sum(int(row["blank_line_delta"]) == 0 for row in metrics)
    publication_status = str(bool(summary["trainingRecordsPublished"])).lower()
    rows = [
        "# Raw-text pipeline outcome and quality audit",
        "",
        "## Decision result",
        "",
        f"- Pipeline machine-certified: **{certified}/{documents} "
        f"({certified / documents:.1%})**.",
        f"- Pipeline-quarantined: **{status['blocked']}/{documents}**.",
        f"- Manually confirmed machine-certified cases: **{manual_passes}/{len(manual)}**; "
        f"manually rejected false positives: **{manually_rejected}/{len(manual)}**; "
        f"remaining provisional and unaudited: **{provisional}/{documents}**.",
        f"- Training records published: **{publication_status}**.",
        f"- Deterministic random machine-certified-case review: "
        f"**{random_passes}/{len(random_rows)}**; "
        f"Wilson 95% interval **{lower:.1%}-{upper:.1%}**. This interval measures the manual "
        "review sample, not unseen-template transformability.",
        f"- Manual failure dimensions across all {len(manual)} reviewed cases: semantic "
        f"**{semantic_failures}**, template **{template_failures}**, privacy/source-residue "
        f"**{privacy_failures}**; dimensions can overlap.",
        f"- Estimated clean yield across the attempted cohort: **{estimated_clean_fraction:.1%}** "
        f"(screening-adjusted Wilson range **{estimated_clean_lower:.1%}-"
        f"{estimated_clean_upper:.1%}**), assuming the deterministic random review is "
        "representative of machine-certified cases.",
        "",
        "Machine certification is not an acceptance decision: audited false positives are moved "
        "to `manual-quarantine/`, audited passes to `confirmed/`, and unaudited machine passes to "
        "`provisional/`. The analyzer never promotes any subset into a training publication.",
        "",
        "## Deterministic and provenance checks",
        "",
        f"- Complete pipeline and component commit receipts validated: **{len(costs) + 1} runs**.",
        f"- Latest host-audit replay passed: **{host_passes}/{documents}**.",
        f"- Page-marker topology preserved: **{page_passes}/{documents}**.",
        f"- Physical line count preserved: **{line_passes}/{documents}**; "
        f"blank-line count preserved: **{blank_passes}/{documents}**.",
        f"- Changed source lines: **{changed_lines:,}/{total_lines:,} "
        f"({changed_lines / total_lines:.1%})**; "
        f"median unchanged-line fraction: **{median_retention:.1%}**.",
        "",
        "## Quarantine evidence",
        "",
        f"- Latest semantic findings by category: **{dict(finding_kinds)}**.",
        "- Every pipeline rejection has its exact current-candidate hash, blocking reason, "
        "evidence fragments, and workbook under `pipeline-quarantine/`.",
        "- Every manually detected false positive is segregated under `manual-quarantine/` with "
        "the reviewer decision and concrete evidence note.",
        "",
        "## Cost of the exact attempted lineage",
        "",
        f"- Unique provider-bearing component runs: **{len(costs)}**.",
        f"- Provider-reported cost: **${provider_cost:.9f}**, or "
        f"**${provider_cost / documents * 1000:.3f}/1,000 attempted documents** and "
        f"**${provider_cost / certified * 1000:.3f}/1,000 certified documents**.",
        f"- Successful structured responses: **{requests}**; provider-route attempts: "
        f"**{provider_attempts}**, of which **{failed_attempts}** failed before an accepted "
        "structured response.",
        "- This is full calibration/continuation spend for the exact lineage, not a clean-run "
        "steady-state estimate.",
        "",
        "## Artifact map",
        "",
        "- `machine-certified/manifest.json`: all pipeline-certified cases with their later audit "
        "disposition; this is not an acceptance manifest.",
        "- `confirmed/manifest.json`: manually reviewed cases passing all three checks.",
        "- `manual-quarantine/manifest.json`: manually detected certification false positives.",
        "- `provisional/manifest.json`: machine-certified but not manually reviewed cases.",
        "- `pipeline-quarantine/manifest.json`: pipeline-blocked cases and their findings.",
        "- Each disposition's `cases/` directory contains the target summary and exact "
        "source/candidate diff.",
        "- `case-metrics.csv`, `findings.csv`, `lineage-costs.csv`, and "
        "`semantic-values.csv`: reproducible EDA tables.",
        "- `manual-audit.csv`: reviewer decisions and selection provenance.",
        "- `plots/`: deterministic SVG summaries.",
        "",
    ]
    return "\n".join(rows).encode("utf-8")


def analyze(
    *,
    project_root: Path,
    pipeline_root: Path,
    manual_audit_path: Path,
    output_parent: Path,
    run_id: str,
) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    pipeline_root = _ensure_under_project_root(project_root, pipeline_root, label="pipeline root")
    output_parent = _ensure_under_project_root(
        project_root, output_parent, label="analysis output parent"
    )
    pipeline_commit_sha256, pipeline_transaction_sha256 = _validate_run_root(pipeline_root)
    manual_audit_path = _ensure_under_project_root(
        project_root, manual_audit_path, label="manual audit"
    )
    if manual_audit_path.is_symlink() or not manual_audit_path.is_file():
        raise ValueError("manual audit must be a regular file")
    manual_payload = read_regular_file_bytes(manual_audit_path)
    manual_audit = ManualAudit.model_validate_json(manual_payload, strict=True)
    transaction = {
        "schemaVersion": 1,
        "runId": run_id,
        "pipelineRun": {
            "path": pipeline_root.relative_to(project_root).as_posix(),
            "commitSha256": pipeline_commit_sha256,
            "transactionSha256": pipeline_transaction_sha256,
        },
        "manualAudit": {
            "path": manual_audit_path.relative_to(project_root).as_posix(),
            "sha256": sha256_bytes(manual_payload),
        },
        "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
    }
    transaction_sha256 = sha256_bytes(canonical_json_bytes(transaction))
    staged = StagedArtifactRun(
        output_parent=output_parent,
        run_name=run_id,
        transaction_sha256=transaction_sha256,
    )
    if staged.completed:
        result = _json_object(staged.final_root / "summary.json")
        return {
            "runId": run_id,
            "created": False,
            "output": staged.final_root.relative_to(project_root).as_posix(),
            "commitSha256": sha256_file(staged.final_root / "_COMMIT.json"),
            "transactionSha256": transaction_sha256,
            "summary": result,
        }

    summary = _json_object(pipeline_root / "summary.json")
    raw_rows = _json_rows(pipeline_root / "lineage/cases.jsonl")
    lineage = tuple(
        _PipelineResumeLineageRow.model_validate_json(canonical_json_bytes(row), strict=True)
        for row in raw_rows
    )
    document_ids = tuple(row.documentId for row in lineage)
    status_counts = Counter(row.status for row in lineage)
    if (
        not lineage
        or len(document_ids) != len(set(document_ids))
        or summary.get("documents") != len(lineage)
        or summary.get("certifiedDocuments") != status_counts["certified"]
        or summary.get("blockedDocuments") != status_counts["blocked"]
        or summary.get("caseLineageSha256")
        != sha256_bytes(read_regular_file_bytes(pipeline_root / "lineage/cases.jsonl"))
        or summary.get("trainingRecordsPublished") is not False
    ):
        raise ValueError("pipeline summary differs from its exact non-published lineage")

    cache: dict[tuple[str, str, str], RunReference] = {}
    cases = tuple(_load_case(project_root=project_root, row=row, cache=cache) for row in lineage)
    metrics = [_case_metrics(case) for case in cases]
    metric_by_id = {row["document_id"]: row for row in metrics}
    costs = _cost_rows(project_root=project_root, lineage=lineage, cache=cache)
    manual, random_ids = _manual_audit_rows(
        audit=manual_audit,
        cases=cases,
        pipeline_commit_sha256=pipeline_commit_sha256,
    )
    findings = [
        {
            "document_id": case.lineage.documentId,
            "status": case.lineage.status,
            "finding_number": number,
            "finding_kind": finding["findingKind"],
            "line_ids": ";".join(row["lineId"] for row in finding["evidence"]),
            "evidence_fragments": " || ".join(
                row["currentFragment"] for row in finding["evidence"]
            ),
            "problem": finding["problem"],
        }
        for case in cases
        for number, finding in enumerate(case.findings, start=1)
    ]
    semantic_values = [
        row
        for case in cases
        for variant, label in (
            ("source", case.source_label),
            ("synthetic_target", case.target_label),
        )
        for row in _semantic_values(
            document_id=case.lineage.documentId,
            status=case.lineage.status,
            variant=variant,
            label=label,
        )
    ]

    manual_passes = sum(bool(row["all_checks_passed"]) for row in manual)
    manual_by_id = {str(row["document_id"]): row for row in manual}
    disposition_by_id = {
        case.lineage.documentId: _audit_disposition(
            status=case.lineage.status,
            manual_audit=manual_by_id.get(case.lineage.documentId),
        )
        for case in cases
    }
    manifest_rows = {
        case.lineage.documentId: _manifest_row(
            case,
            metric_by_id[case.lineage.documentId],
            audit_disposition=disposition_by_id[case.lineage.documentId],
            manual_audit=manual_by_id.get(case.lineage.documentId),
        )
        for case in cases
    }
    machine_certified = [
        manifest_rows[case.lineage.documentId]
        for case in cases
        if case.lineage.status == "certified"
    ]
    disposition_rows = {
        disposition: [
            manifest_rows[case.lineage.documentId]
            for case in cases
            if disposition_by_id[case.lineage.documentId] == disposition
        ]
        for disposition in (
            "confirmed",
            "manual_quarantine",
            "provisional",
            "pipeline_quarantine",
        )
    }
    random_rows = [row for row in manual if row["document_id"] in set(random_ids)]
    random_passes = sum(bool(row["all_checks_passed"]) for row in random_rows)
    lower, upper = _wilson_interval(random_passes, len(random_rows))
    provider_cost = sum(float(row["provider_reported_cost_usd"]) for row in costs)
    analysis_summary = {
        "schemaVersion": 1,
        "runId": run_id,
        "pipelineRunId": summary["runId"],
        "pipelineCommitSha256": pipeline_commit_sha256,
        "pipelineTransactionSha256": pipeline_transaction_sha256,
        "documents": len(cases),
        "pipelineCertifiedDocuments": len(machine_certified),
        "pipelineQuarantinedDocuments": len(disposition_rows["pipeline_quarantine"]),
        "pipelineCertifiedFraction": len(machine_certified) / len(cases),
        "manuallyConfirmedDocuments": len(disposition_rows["confirmed"]),
        "manuallyRejectedDocuments": len(disposition_rows["manual_quarantine"]),
        "provisionalDocuments": len(disposition_rows["provisional"]),
        "hostAuditReplayPasses": sum(bool(row["latest_host_audit_passed"]) for row in metrics),
        "pageMarkersPreserved": sum(bool(row["page_markers_preserved"]) for row in metrics),
        "zeroLineCountDelta": sum(int(row["line_count_delta"]) == 0 for row in metrics),
        "zeroBlankLineDelta": sum(int(row["blank_line_delta"]) == 0 for row in metrics),
        "findingKinds": dict(Counter(row["finding_kind"] for row in findings)),
        "manualAuditDocuments": len(manual),
        "manualAuditPasses": manual_passes,
        "manualSemanticFailures": sum(
            not bool(row["semantic_consistency"]) for row in manual
        ),
        "manualTemplateFailures": sum(
            not bool(row["template_fidelity"]) for row in manual
        ),
        "manualPrivacyFailures": sum(
            not bool(row["privacy_rewritten"]) for row in manual
        ),
        "randomAuditDocuments": len(random_rows),
        "randomAuditPasses": random_passes,
        "randomAuditPassFraction": random_passes / len(random_rows),
        "randomAuditWilson95Lower": lower,
        "randomAuditWilson95Upper": upper,
        "estimatedCleanCohortFraction": (
            len(machine_certified) / len(cases) * random_passes / len(random_rows)
        ),
        "estimatedCleanCohortWilson95Lower": (
            len(machine_certified) / len(cases) * lower
        ),
        "estimatedCleanCohortWilson95Upper": (
            len(machine_certified) / len(cases) * upper
        ),
        "uniqueComponentRuns": len(costs),
        "providerReportedLineageCostUsd": provider_cost,
        "providerReportedLineageCostPerThousandAttemptedUsd": (provider_cost / len(cases) * 1000),
        "providerReportedLineageCostPerThousandCertifiedUsd": (
            provider_cost / len(machine_certified) * 1000
        ),
        "requests": sum(int(row["requests"]) for row in costs),
        "providerAttempts": sum(int(row["provider_attempts"]) for row in costs),
        "failedProviderAttempts": sum(int(row["failed_provider_attempts"]) for row in costs),
        "inputTokens": sum(int(row["input_tokens"]) for row in costs),
        "reasoningTokens": sum(int(row["reasoning_tokens"]) for row in costs),
        "visibleOutputTokens": sum(int(row["visible_output_tokens"]) for row in costs),
        "trainingRecordsPublished": False,
        "machineCertifiedManifestIsAcceptanceManifest": False,
        "confirmedManifestIsTrainingPublication": False,
    }

    artifacts: dict[str, bytes] = {
        "provenance/transaction.json": canonical_json_bytes(transaction) + b"\n",
        "inputs/manual-audit.json": manual_payload,
        "summary.json": canonical_json_bytes(analysis_summary) + b"\n",
        "case-metrics.csv": _csv_bytes(metrics),
        "findings.csv": _csv_bytes(findings),
        "lineage-costs.csv": _csv_bytes(costs),
        "semantic-values.csv": _csv_bytes(semantic_values),
        "manual-audit.csv": _csv_bytes(manual),
        "machine-certified/manifest.json": canonical_json_bytes(
            {
                "schemaVersion": 1,
                "documents": len(machine_certified),
                "isAcceptanceManifest": False,
                "cases": machine_certified,
            }
        )
        + b"\n",
        "REPORT.md": _report(
            summary=summary,
            metrics=metrics,
            findings=findings,
            costs=costs,
            manual=manual,
            random_ids=random_ids,
        ),
    }
    for disposition, rows in disposition_rows.items():
        artifacts[f"{disposition.replace('_', '-')}/manifest.json"] = canonical_json_bytes(
            {"schemaVersion": 1, "documents": len(rows), "cases": rows}
        ) + b"\n"
    for case in cases:
        category = disposition_by_id[case.lineage.documentId].replace("_", "-")
        artifacts[f"{category}/cases/{case.lineage.documentId}.md"] = _case_workbook(case)

    outcome_counts = Counter(disposition_by_id.values())
    artifacts["plots/01-outcomes.svg"] = _bar_svg(
        title="Raw-text audit dispositions",
        rows=[
            ("confirmed", outcome_counts["confirmed"]),
            ("manual quarantine", outcome_counts["manual_quarantine"]),
            ("provisional", outcome_counts["provisional"]),
            ("pipeline quarantine", outcome_counts["pipeline_quarantine"]),
        ],
        x_label="documents",
    )
    correction_counts = Counter((row["status"], int(row["correction_rounds"])) for row in metrics)
    artifacts["plots/02-correction-rounds.svg"] = _bar_svg(
        title="Outcome by successful correction rounds",
        rows=[
            (f"{status}, round {rounds}", correction_counts[(status, rounds)])
            for status in ("certified", "blocked")
            for rounds in sorted({key[1] for key in correction_counts if key[0] == status})
        ],
        x_label="documents",
    )
    finding_counts = Counter(row["finding_kind"] for row in findings)
    artifacts["plots/03-quarantine-findings.svg"] = _bar_svg(
        title="Latest quarantine findings",
        rows=sorted(finding_counts.items(), key=lambda row: (-row[1], row[0])),
        x_label="findings",
        color="#C0504D",
    )
    stage_cost: defaultdict[str, float] = defaultdict(float)
    for row in costs:
        stage_cost[str(row["stage"])] += float(row["provider_reported_cost_usd"])
    artifacts["plots/04-lineage-cost.svg"] = _bar_svg(
        title="Provider-reported lineage cost by stage",
        rows=sorted(stage_cost.items()),
        x_label="USD",
        color="#70AD47",
    )
    feature_rows = [
        (feature.replace("_", " "), sum(int(row[feature]) > 0 for row in metrics))
        for feature in (
            "containers",
            "cargo_groups",
            "packages",
            "allocations",
            "dangerous_goods",
            "hs_codes",
            "temperature_containers",
        )
    ]
    artifacts["plots/05-feature-coverage.svg"] = _bar_svg(
        title="Synthetic-target feature coverage",
        rows=feature_rows,
        x_label="documents",
        color="#8064A2",
    )

    for relative_path, payload in sorted(artifacts.items()):
        staged.publish_bytes(relative_path, payload)
    committed = staged.commit(
        expected_artifacts=artifacts,
        metadata={
            "kind": "raw_text_pipeline_outcome_analysis",
            "pipeline_commit_sha256": pipeline_commit_sha256,
            "documents": len(cases),
            "pipeline_certified_documents": len(machine_certified),
            "manually_confirmed_documents": len(disposition_rows["confirmed"]),
            "manually_rejected_documents": len(disposition_rows["manual_quarantine"]),
            "provisional_documents": len(disposition_rows["provisional"]),
            "pipeline_quarantined_documents": len(disposition_rows["pipeline_quarantine"]),
            "training_records_published": False,
        },
    )
    return {
        "runId": run_id,
        "created": committed.created,
        "output": staged.final_root.relative_to(project_root).as_posix(),
        "commitSha256": sha256_file(staged.final_root / "_COMMIT.json"),
        "transactionSha256": transaction_sha256,
        "summary": analysis_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--manual-audit", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()
    result = analyze(
        project_root=arguments.project_root,
        pipeline_root=arguments.pipeline_root,
        manual_audit_path=arguments.manual_audit,
        output_parent=arguments.output_parent,
        run_id=arguments.run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
