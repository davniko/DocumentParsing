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
import io
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from difflib import SequenceMatcher, unified_diff
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.raw_text_certification import (
    CertificationCaseResult,
    CertificationStage,
    LegacySemanticAuditOutput,
    SemanticAuditOutput,
    _host_audit,
)
from document_ocr.synthesis.raw_text_certification_invariants import (
    CertificationInvariantAudit,
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
_WORD_TOKEN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_RANDOM_SELECTION = "deterministic_random"
_FINDING_DIMENSION = {
    "target_fact_mismatch": "target_fact_fidelity",
    "repeated_or_derived_fact_mismatch": "repeated_and_derived_relations",
    "source_only_private_or_auxiliary_fact": "source_private_and_auxiliary_replacement",
    "party_or_legal_identity_mismatch": "party_legal_and_negotiability",
    "route_or_jurisdiction_mismatch": "route_jurisdiction_and_identifiers",
    "equipment_or_temperature_mismatch": "equipment_temperature_and_capacity",
    "cargo_or_dangerous_goods_mismatch": "cargo_packages_and_dangerous_goods",
    "format_or_model_artifact": "template_format_and_model_artifacts",
}


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
    certification_stages: tuple[CertificationStage, ...]
    semantic_findings: tuple[dict[str, Any], ...]
    invariant_audit: CertificationInvariantAudit | None


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


def _certification_artifacts(
    case_root: Path, result: CertificationCaseResult
) -> tuple[
    tuple[CertificationStage, ...],
    tuple[dict[str, Any], ...],
    CertificationInvariantAudit | None,
]:
    values = json.loads(read_regular_file_bytes(case_root / "stages.json"))
    if not isinstance(values, list):
        raise ValueError(f"expected certification stage array: {case_root}")
    stages = tuple(
        CertificationStage.model_validate_json(canonical_json_bytes(row), strict=True)
        for row in values
    )
    candidate_lines = read_regular_file_bytes(case_root / "final.txt").decode("utf-8").splitlines()
    findings: dict[str, dict[str, Any]] = {}
    for stage in stages:
        if stage.modelOutput is None or stage.hostError is not None or stage.errorType is not None:
            continue
        output_model = (
            SemanticAuditOutput
            if stage.auditContractVersion in {2, 3}
            else LegacySemanticAuditOutput
        )
        parsed = output_model.model_validate_json(
            canonical_json_bytes(stage.modelOutput), strict=True
        )
        for finding in parsed.findings:
            raw = finding.model_dump(mode="json")
            evidence = []
            for row in raw["evidence"]:
                line_id = str(row["lineId"])
                line_number = int(line_id[1:])
                if line_number < 1 or line_number > len(candidate_lines):
                    raise ValueError(
                        f"semantic finding cites an absent line: {case_root}:{line_id}"
                    )
                evidence.append(
                    {
                        "lineId": line_id,
                        "currentFragment": str(
                            row.get("currentFragment") or candidate_lines[line_number - 1]
                        ),
                    }
                )
            finding_kind = str(raw["findingKind"])
            value = {
                "findingKind": finding_kind,
                "dimension": _FINDING_DIMENSION[finding_kind],
                "evidence": evidence,
                "obligationIds": list(raw.get("obligationIds") or ()),
                "problem": str(raw["problem"]),
            }
            findings[sha256_bytes(canonical_json_bytes(value))] = value
    semantic_findings = tuple(findings[key] for key in sorted(findings))

    invariant_audit: CertificationInvariantAudit | None = None
    invariant_sha256 = getattr(result, "invariantAuditSha256", None)
    if invariant_sha256 is not None:
        audit_path = case_root / "invariant-audit.json"
        envelope_path = case_root / "invariant-envelope.json"
        audit_payload = _json_object(audit_path)
        envelope_payload = _json_object(envelope_path)
        if sha256_bytes(canonical_json_bytes(audit_payload)) != invariant_sha256 or sha256_bytes(
            canonical_json_bytes(envelope_payload)
        ) != getattr(result, "invariantEnvelopeSha256", None):
            raise ValueError(f"certification invariant artifacts differ from result: {case_root}")
        invariant_audit = CertificationInvariantAudit.model_validate_json(
            canonical_json_bytes(audit_payload), strict=True
        )
        if invariant_audit.envelopeSha256 != sha256_bytes(canonical_json_bytes(envelope_payload)):
            raise ValueError(f"invariant audit is pinned to another envelope: {case_root}")
    if result.semanticFindings != len(semantic_findings) or result.deterministicFindings != (
        len(invariant_audit.findings) if invariant_audit is not None else 0
    ):
        raise ValueError(f"certification finding counts differ from artifacts: {case_root}")
    return stages, semantic_findings, invariant_audit


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
    certification_stages: tuple[CertificationStage, ...] = ()
    semantic_findings: tuple[dict[str, Any], ...] = ()
    invariant_audit: CertificationInvariantAudit | None = None
    if certification_rows:
        latest = certification_rows[-1]
        certification = _run_reference(project_root=project_root, raw=latest.run, cache=cache)
        certification_case = certification.root / "cases" / row.documentId
        candidate = read_regular_file_bytes(certification_case / "final.txt").decode("utf-8")
        result = CertificationCaseResult.model_validate_json(
            read_regular_file_bytes(certification_case / "result.json"), strict=True
        )
        contract = _json_object(certification_case / "source-contract.json")
        certification_stages, semantic_findings, invariant_audit = _certification_artifacts(
            certification_case, result
        )
        if (
            result.documentId != row.documentId
            or result.inputCandidateSha256 != row.currentCandidateSha256
            or result.finalTextSha256 != row.currentCandidateSha256
            or result.sourceTextSha256 != row.sourceTextSha256
            or result.status != latest.status
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
            or semantic_findings
            or (invariant_audit is not None and invariant_audit.findings)
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
        certification_stages=certification_stages,
        semantic_findings=semantic_findings,
        invariant_audit=invariant_audit,
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
    source_tokens = _WORD_TOKEN.findall(source)
    candidate_tokens = _WORD_TOKEN.findall(candidate)
    source_alpha = tuple(character for character in source if character.isalpha())
    candidate_alpha = tuple(character for character in candidate if character.isalpha())
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
        "source_blank_lines": sum(not line.strip() for line in source_lines),
        "candidate_blank_lines": sum(not line.strip() for line in candidate_lines),
        "source_nonblank_lines": sum(bool(line.strip()) for line in source_lines),
        "candidate_nonblank_lines": sum(bool(line.strip()) for line in candidate_lines),
        "source_tokens": len(source_tokens),
        "candidate_tokens": len(candidate_tokens),
        "source_unique_tokens": len({token.casefold() for token in source_tokens}),
        "candidate_unique_tokens": len({token.casefold() for token in candidate_tokens}),
        "source_digit_fraction": sum(character.isdigit() for character in source)
        / max(len(source), 1),
        "candidate_digit_fraction": sum(character.isdigit() for character in candidate)
        / max(len(candidate), 1),
        "source_uppercase_fraction": sum(character.isupper() for character in source_alpha)
        / max(len(source_alpha), 1),
        "candidate_uppercase_fraction": sum(character.isupper() for character in candidate_alpha)
        / max(len(candidate_alpha), 1),
        "pages": len(source_pages),
        "page_markers_preserved": source_pages == candidate_pages,
        "terminal_newline_preserved": source.endswith("\n") == candidate.endswith("\n"),
    }


def _flatten_scalar_values(value: object, *, prefix: str = "") -> dict[str, object]:
    output: dict[str, object] = {}
    if isinstance(value, Mapping):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten_scalar_values(value[key], prefix=child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, row in enumerate(value):
            child = f"{prefix}[{index}]"
            output.update(_flatten_scalar_values(row, prefix=child))
    elif value is not None:
        output[prefix] = value
    return output


def _label_metrics(source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, int]:
    source_values = _flatten_scalar_values(source.get("documentPatch"))
    target_values = _flatten_scalar_values(target.get("documentPatch"))
    all_paths = set(source_values) | set(target_values)
    return {
        "source_label_scalar_leaves": len(source_values),
        "target_label_scalar_leaves": len(target_values),
        "label_changed_scalar_paths": sum(
            source_values.get(path) != target_values.get(path) for path in all_paths
        ),
        "label_added_scalar_paths": sum(path not in source_values for path in target_values),
        "label_removed_scalar_paths": sum(path not in target_values for path in source_values),
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
        "additional_information": sum(
            len(row.get("additionalInformation") or [])
            for row in cargo_groups
            if isinstance(row, Mapping)
        ),
        "marks_and_numbers": sum(
            len(row.get("marksAndNumbers") or [])
            for row in cargo_groups
            if isinstance(row, Mapping)
        ),
        "temperature_containers": sum(
            isinstance(row, Mapping) and row.get("temperatureSetpoint") is not None
            for row in containers
        ),
        "notify_parties": (
            len(parties.get("notifyParties") or []) if isinstance(parties, Mapping) else 0
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
    add("negotiability", patch.get("negotiability"))
    freight = patch.get("freight") or {}
    if isinstance(freight, Mapping):
        add("freight_arrangement", freight.get("paymentArrangement"))
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


def _numeric_values(
    *, document_id: str, status: str, variant: str, label: Mapping[str, Any]
) -> list[dict[str, Any]]:
    patch = label.get("documentPatch")
    if not isinstance(patch, Mapping):
        return []
    output: list[dict[str, Any]] = []

    def add(dimension: str, raw: object, unit: str = "count") -> None:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return
        output.append(
            {
                "document_id": document_id,
                "status": status,
                "variant": variant,
                "dimension": dimension,
                "unit": unit,
                "value": float(raw),
            }
        )

    for row in patch.get("cargoPackages") or ():
        if isinstance(row, Mapping):
            add("package_quantity", row.get("quantity"))
    for row in patch.get("cargoAllocationGroups") or ():
        if not isinstance(row, Mapping):
            continue
        for allocation in row.get("allocations") or ():
            if isinstance(allocation, Mapping):
                add("allocation_package_quantity", allocation.get("packageQuantity"))
    for row in patch.get("cargoGroups") or ():
        if not isinstance(row, Mapping):
            continue
        for field, dimension in (
            ("grossWeight", "gross_weight"),
            ("netWeight", "net_weight"),
            ("volume", "volume"),
        ):
            measure = row.get(field)
            if isinstance(measure, Mapping):
                add(dimension, measure.get("value"), str(measure.get("unit") or ""))
    for row in patch.get("containers") or ():
        if not isinstance(row, Mapping):
            continue
        measure = row.get("temperatureSetpoint")
        if isinstance(measure, Mapping):
            add("temperature_setpoint", measure.get("value"), str(measure.get("unit") or ""))
    return output


def _semantic_transition_rows(
    values: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str, str], list[str]] = defaultdict(list)
    statuses: dict[str, str] = {}
    for row in values:
        document_id = str(row["document_id"])
        dimension = str(row["dimension"])
        variant = str(row["variant"])
        statuses[document_id] = str(row["status"])
        grouped[(document_id, dimension, variant)].append(str(row["value"]))
    dimensions = sorted({str(row["dimension"]) for row in values})
    output: list[dict[str, Any]] = []
    for document_id in sorted(statuses):
        for dimension in dimensions:
            source = " | ".join(sorted(grouped[(document_id, dimension, "source")]))
            target = " | ".join(sorted(grouped[(document_id, dimension, "synthetic_target")]))
            if not source and not target:
                continue
            output.append(
                {
                    "document_id": document_id,
                    "status": statuses[document_id],
                    "dimension": dimension,
                    "source_values": source,
                    "target_values": target,
                    "changed": source != target,
                }
            )
    return output


def _case_metrics(case: LoadedCase) -> dict[str, Any]:
    result = case.certification_result
    history_counts = Counter(row.stage for row in case.lineage.history)
    deterministic_findings = (
        len(case.invariant_audit.findings) if case.invariant_audit is not None else 0
    )
    semantic_findings = len(case.semantic_findings)
    if case.lineage.status == "certified":
        certification_pathway = "certified"
    elif deterministic_findings:
        certification_pathway = "deterministic_rejected"
    elif semantic_findings:
        certification_pathway = "semantic_rejected"
    else:
        certification_pathway = "other_blocked"
    source_features = {
        f"source_{key}": value for key, value in _target_features(case.source_label).items()
    }
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
        "certification_pathway": certification_pathway,
        "latest_deterministic_findings": deterministic_findings,
        "latest_semantic_findings": semantic_findings,
        "audit_passes": int(result.auditPasses) if result is not None else 0,
        "clean_audit_passes": int(result.cleanAuditPasses) if result is not None else 0,
        "latest_host_audit_passed": bool(result and result.hostAudit.passed),
        "source_sha256": case.lineage.sourceTextSha256,
        "candidate_sha256": case.lineage.currentCandidateSha256,
        "target_sha256": sha256_bytes(canonical_json_bytes(case.target_label)),
        "inventory_run": case.inventory.path,
        "latest_certification_run": case.certification.path if case.certification else "",
        **_edit_metrics(case.source, case.candidate),
        **_label_metrics(case.source_label, case.target_label),
        **source_features,
        **_target_features(case.target_label),
    }


def _finding_rows(cases: Sequence[LoadedCase]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for case in cases:
        if case.invariant_audit is not None:
            for number, finding in enumerate(case.invariant_audit.findings, start=1):
                output.append(
                    {
                        "document_id": case.lineage.documentId,
                        "status": case.lineage.status,
                        "origin": "deterministic",
                        "finding_number": number,
                        "finding_kind": finding.findingKind,
                        "dimension": finding.dimension,
                        "invariant_id": finding.invariantId,
                        "obligation_ids": "",
                        "line_ids": ";".join(row.lineId for row in finding.evidence),
                        "evidence_fragments": " || ".join(
                            row.currentLine for row in finding.evidence
                        ),
                        "target_paths": ";".join(finding.targetPaths),
                        "repair_count": len(finding.repairs),
                        "problem": finding.problem,
                    }
                )
        for number, semantic_finding in enumerate(case.semantic_findings, start=1):
            output.append(
                {
                    "document_id": case.lineage.documentId,
                    "status": case.lineage.status,
                    "origin": "semantic",
                    "finding_number": number,
                    "finding_kind": semantic_finding["findingKind"],
                    "dimension": semantic_finding["dimension"],
                    "invariant_id": "",
                    "obligation_ids": ";".join(semantic_finding["obligationIds"]),
                    "line_ids": ";".join(row["lineId"] for row in semantic_finding["evidence"]),
                    "evidence_fragments": " || ".join(
                        row["currentFragment"] for row in semantic_finding["evidence"]
                    ),
                    "target_paths": "",
                    "repair_count": 0,
                    "problem": semantic_finding["problem"],
                }
            )
    return output


def _invariant_check_rows(cases: Sequence[LoadedCase]) -> list[dict[str, Any]]:
    return [
        {
            "document_id": case.lineage.documentId,
            "status": case.lineage.status,
            "check_id": check.checkId,
            "passed": check.passed,
            "findings": check.findings,
        }
        for case in cases
        if case.invariant_audit is not None
        for check in case.invariant_audit.checks
    ]


def _usage_number(usage: Mapping[str, Any], key: str) -> int:
    value = usage.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid usage counter {key}: {value!r}")
    return value


def _usage_cost(usage: Mapping[str, Any], key: str) -> Decimal:
    value = usage.get(key)
    if value is None:
        return Decimal(0)
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError(f"invalid usage cost {key}: {value!r}")
    return result


def _case_usage_and_route_rows(
    *,
    project_root: Path,
    lineage: Sequence[_PipelineResumeLineageRow],
    cache: dict[tuple[str, str, str], RunReference],
    costs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    usage_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    visited: set[tuple[str, str, str]] = set()
    for case in lineage:
        for history in case.history:
            reference = _run_reference(project_root=project_root, raw=history.run, cache=cache)
            key = (history.stage, reference.path, case.documentId)
            if key in visited:
                continue
            visited.add(key)
            case_root = reference.root / "cases" / case.documentId
            result = _json_object(case_root / "result.json")
            if result.get("documentId") != case.documentId:
                raise ValueError(f"component result belongs to another case: {case_root}")
            usage = result.get("usage")
            if not isinstance(usage, Mapping):
                raise ValueError(f"component result lacks a usage receipt: {case_root}")
            raw_stages = json.loads(read_regular_file_bytes(case_root / "stages.json"))
            if not isinstance(raw_stages, list) or any(
                not isinstance(stage, Mapping) for stage in raw_stages
            ):
                raise ValueError(f"component stages are not an object array: {case_root}")
            failed_attempts = 0
            host_rejections = 0
            total_latency = 0.0
            for attempt, stage in enumerate(raw_stages, start=1):
                started = stage.get("startedAtUnixSeconds")
                completed = stage.get("completedAtUnixSeconds")
                if not isinstance(started, (int, float)) or not isinstance(completed, (int, float)):
                    raise ValueError(f"component stage lacks valid timing: {case_root}")
                latency = float(completed) - float(started)
                if latency < 0:
                    raise ValueError(f"component stage completes before it starts: {case_root}")
                total_latency += latency
                error_type = stage.get("errorType")
                host_evaluation = stage.get("hostEvaluation")
                host_error = stage.get("hostError")
                if error_type is not None:
                    outcome = "transport_error"
                    failed_attempts += 1
                elif host_error is not None or (
                    isinstance(host_evaluation, Mapping) and host_evaluation.get("passed") is False
                ):
                    outcome = "host_rejected_response"
                    host_rejections += 1
                else:
                    outcome = "structured_response"
                route_rows.append(
                    {
                        "document_id": case.documentId,
                        "pipeline_status": case.status,
                        "stage": history.stage,
                        "run_id": reference.root.name,
                        "attempt": attempt,
                        "route_provider": str(stage.get("routeProvider") or "not_recorded"),
                        "route_round": int(stage.get("routeRound") or 0),
                        "route_attempt": int(stage.get("routeAttempt") or 0),
                        "request_kind": str(
                            stage.get("requestKind")
                            or f"semantic_audit_pass_{stage.get('auditPass', 0)}"
                        ),
                        "outcome": outcome,
                        "error_type": str(error_type or ""),
                        "latency_seconds": latency,
                    }
                )
            usage_rows.append(
                {
                    "document_id": case.documentId,
                    "pipeline_status": case.status,
                    "stage": history.stage,
                    "run_id": reference.root.name,
                    "path": reference.path,
                    "component_status": str(result.get("status") or ""),
                    "requests": _usage_number(usage, "requests"),
                    "provider_attempts": len(raw_stages),
                    "failed_provider_attempts": failed_attempts,
                    "host_rejected_responses": host_rejections,
                    "input_tokens": _usage_number(usage, "inputTokens"),
                    "reasoning_tokens": _usage_number(usage, "reasoningTokens"),
                    "visible_output_tokens": _usage_number(usage, "visibleOutputTokens"),
                    "provider_reported_cost_usd": str(
                        _usage_cost(usage, "providerReportedCostUsd")
                    ),
                    "estimated_cost_usd": str(_usage_cost(usage, "estimatedCostUsd")),
                    "route_latency_seconds": total_latency,
                }
            )

    expected = {(str(row["stage"]), str(row["path"])): row for row in costs}
    grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in usage_rows:
        grouped[(str(row["stage"]), str(row["path"]))].append(row)
    if set(grouped) != set(expected):
        raise ValueError("case usage rows do not cover the exact component-run set")
    for group_key, rows in grouped.items():
        aggregate = expected[group_key]
        for case_key, run_key in (
            ("requests", "requests"),
            ("provider_attempts", "provider_attempts"),
            ("failed_provider_attempts", "failed_provider_attempts"),
            ("input_tokens", "input_tokens"),
            ("reasoning_tokens", "reasoning_tokens"),
            ("visible_output_tokens", "visible_output_tokens"),
        ):
            if sum(int(row[case_key]) for row in rows) != int(aggregate[run_key]):
                raise ValueError(f"case usage does not reconcile {case_key}: {group_key}")
        observed_cost = sum(
            (Decimal(str(row["provider_reported_cost_usd"])) for row in rows),
            start=Decimal(0),
        )
        expected_cost = Decimal(str(aggregate["provider_reported_cost_usd"]))
        if observed_cost != expected_cost:
            raise ValueError(f"case usage does not reconcile provider cost: {group_key}")
    return usage_rows, route_rows


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
                "provider_reported_cost_usd": str(reported or 0),
                "estimated_cost_usd": str(estimated or 0),
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


def _case_workbook(
    case: LoadedCase,
    *,
    audit_disposition: str,
    manual_audit: Mapping[str, Any] | None,
) -> bytes:
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
        f"- Review disposition: **{audit_disposition}**",
        f"- Correction rounds: **{case.lineage.correctionRounds}**",
        f"- Current candidate SHA-256: `{case.lineage.currentCandidateSha256}`",
        f"- Blocking reason: {_safe_markdown(case.lineage.blockingReason or 'none')}",
    ]
    if manual_audit is not None:
        semantic_consistency = str(bool(manual_audit["semantic_consistency"])).lower()
        template_fidelity = str(bool(manual_audit["template_fidelity"])).lower()
        privacy_rewritten = str(bool(manual_audit["privacy_rewritten"])).lower()
        all_checks_passed = str(bool(manual_audit["all_checks_passed"])).lower()
        rows.extend(
            [
                "",
                "## Independent manual review",
                "",
                f"- Semantic consistency: **{semantic_consistency}**",
                f"- Template fidelity: **{template_fidelity}**",
                f"- Privacy/source replacement: **{privacy_rewritten}**",
                f"- All checks passed: **{all_checks_passed}**",
                f"- Selection basis: `{_safe_markdown(manual_audit['selection_basis'])}`",
                f"- Evidence note: {_safe_markdown(manual_audit['notes'])}",
            ]
        )
    rows.extend(("", "## Latest deterministic and independent findings", ""))
    deterministic_findings = (
        case.invariant_audit.findings if case.invariant_audit is not None else ()
    )
    if not deterministic_findings and not case.semantic_findings:
        rows.append("No deterministic or semantic findings in the latest certification.")
    for finding in deterministic_findings:
        evidence = "; ".join(
            f"{row.lineId} `{_safe_markdown(row.currentLine)}`" for row in finding.evidence
        )
        rows.append(
            f"- **deterministic / {finding.findingKind} / {finding.dimension}** — "
            f"{evidence}: {_safe_markdown(finding.problem)}"
        )
    for semantic_finding in case.semantic_findings:
        evidence = "; ".join(
            f"{row['lineId']} `{_safe_markdown(row['currentFragment'])}`"
            for row in semantic_finding["evidence"]
        )
        rows.append(
            f"- **semantic / {semantic_finding['findingKind']} / "
            f"{semantic_finding['dimension']}** — "
            f"{evidence}: {_safe_markdown(semantic_finding['problem'])}"
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
        "latestDeterministicFindings": (
            len(case.invariant_audit.findings) if case.invariant_audit is not None else 0
        ),
        "latestSemanticFindings": len(case.semantic_findings),
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


def _png_bytes(figure: Any) -> bytes:
    stream = io.BytesIO()
    figure.tight_layout()
    figure.savefig(
        stream,
        format="png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "document-ocr raw-text outcome analyzer"},
    )
    plt.close(figure)
    payload = stream.getvalue()
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("matplotlib did not produce a PNG artifact")
    return payload


def _label_bars(axis: Any, *, decimals: int = 0) -> None:
    pattern = f"%.{decimals}f"
    for container in axis.containers:
        axis.bar_label(container, fmt=pattern, padding=3, fontsize=8)


def _paired_category_png(
    values: pd.DataFrame,
    *,
    dimension: str,
    title: str,
    maximum_values: int = 20,
) -> bytes:
    selected = values[values["dimension"] == dimension].copy()
    if selected.empty:
        raise ValueError(f"semantic-value dimension has no observations: {dimension}")
    totals = selected.groupby("value", observed=True).size().sort_values(ascending=False)
    retained = list(totals.head(maximum_values).index)
    selected = selected[selected["value"].isin(retained)]
    counts = (
        selected.groupby(["value", "variant"], observed=True).size().reset_index(name="occurrences")
    )
    order = list(totals.loc[retained].sort_values(ascending=True).index)
    figure, axis = plt.subplots(figsize=(13, max(5.5, min(13.0, 0.42 * len(order) + 2.0))))
    sns.barplot(
        data=counts,
        y="value",
        x="occurrences",
        hue="variant",
        order=order,
        errorbar=None,
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("field occurrences")
    axis.set_ylabel("")
    return _png_bytes(figure)


def _plot_artifacts(
    *,
    summary: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
    invariant_checks: Sequence[Mapping[str, Any]],
    costs: Sequence[Mapping[str, Any]],
    case_usage: Sequence[Mapping[str, Any]],
    route_attempts: Sequence[Mapping[str, Any]],
    semantic_values: Sequence[Mapping[str, Any]],
    numeric_values: Sequence[Mapping[str, Any]],
    semantic_transitions: Sequence[Mapping[str, Any]],
    manual: Sequence[Mapping[str, Any]],
) -> dict[str, bytes]:
    sns.set_theme(style="whitegrid", context="notebook")
    case_frame = pd.DataFrame(metrics).copy()
    finding_frame = pd.DataFrame(findings).copy()
    check_frame = pd.DataFrame(invariant_checks).copy()
    cost_frame = pd.DataFrame(costs).copy()
    route_frame = pd.DataFrame(route_attempts).copy()
    semantic_frame = pd.DataFrame(semantic_values).copy()
    numeric_frame = pd.DataFrame(numeric_values).copy()
    transition_frame = pd.DataFrame(semantic_transitions).copy()
    manual_frame = pd.DataFrame(manual).copy()
    output: dict[str, bytes] = {}

    deterministic_rejected = int(
        (case_frame["certification_pathway"] == "deterministic_rejected").sum()
    )
    semantic_audited = len(case_frame) - deterministic_rejected
    machine_certified = int((case_frame["status"] == "certified").sum())
    manual_accepted = int((case_frame["audit_disposition"] == "confirmed").sum())
    funnel = pd.DataFrame(
        {
            "stage": [
                "attempted",
                "inventory ready",
                "semantic-audit eligible",
                "machine certified",
                "manual accepted",
                "training published",
            ],
            "documents": [
                len(case_frame),
                int(summary["inventoryReadyDocuments"]),
                semantic_audited,
                machine_certified,
                manual_accepted,
                0,
            ],
        }
    )
    figure, axis = plt.subplots(figsize=(12, 6))
    sns.barplot(data=funnel, x="stage", y="documents", hue="stage", legend=False, ax=axis)
    _label_bars(axis)
    axis.set_title("Contract-v3 pipeline funnel")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=18)
    output["plots/01_pipeline_funnel.png"] = _png_bytes(figure)

    disposition_order = [
        "confirmed",
        "manual_quarantine",
        "provisional",
        "pipeline_quarantine",
    ]
    disposition = (
        case_frame["audit_disposition"]
        .value_counts()
        .reindex(disposition_order, fill_value=0)
        .rename_axis("disposition")
        .reset_index(name="documents")
    )
    figure, axis = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=disposition,
        x="disposition",
        y="documents",
        hue="disposition",
        legend=False,
        ax=axis,
    )
    _label_bars(axis)
    axis.set_title("Post-review disposition (no training publication)")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=15)
    output["plots/02_review_dispositions.png"] = _png_bytes(figure)

    pathway_order = ["deterministic_rejected", "semantic_rejected", "certified", "other_blocked"]
    pathway = (
        case_frame["certification_pathway"]
        .value_counts()
        .reindex(pathway_order, fill_value=0)
        .rename_axis("pathway")
        .reset_index(name="documents")
    )
    figure, axis = plt.subplots(figsize=(11, 6))
    sns.barplot(data=pathway, x="pathway", y="documents", hue="pathway", legend=False, ax=axis)
    _label_bars(axis)
    axis.set_title("Terminal certification pathway")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=15)
    output["plots/03_certification_pathway.png"] = _png_bytes(figure)

    kind_counts = (
        finding_frame.groupby(["finding_kind", "origin"], observed=True)
        .size()
        .reset_index(name="findings")
        .sort_values("findings", ascending=True)
    )
    figure, axis = plt.subplots(figsize=(13, 7))
    sns.barplot(data=kind_counts, y="finding_kind", x="findings", hue="origin", ax=axis)
    axis.set_title("Deterministic and semantic findings by kind")
    axis.set_xlabel("unique latest findings")
    axis.set_ylabel("")
    output["plots/04_findings_by_kind.png"] = _png_bytes(figure)

    dimension_counts = (
        finding_frame.groupby(["dimension", "origin"], observed=True)
        .size()
        .reset_index(name="findings")
        .sort_values("findings", ascending=True)
    )
    figure, axis = plt.subplots(figsize=(13, 7))
    sns.barplot(data=dimension_counts, y="dimension", x="findings", hue="origin", ax=axis)
    axis.set_title("Certification failures across the eight audit dimensions")
    axis.set_xlabel("unique latest findings")
    axis.set_ylabel("")
    output["plots/05_findings_by_dimension.png"] = _png_bytes(figure)

    failed_checks = (
        check_frame[check_frame["passed"] == False]  # noqa: E712
        .groupby("check_id", observed=True)
        .agg(documents=("document_id", "nunique"), findings=("findings", "sum"))
        .reset_index()
        .sort_values(["documents", "findings"], ascending=True)
        .tail(20)
    )
    check_long = failed_checks.melt(
        id_vars="check_id",
        value_vars=["documents", "findings"],
        var_name="measure",
        value_name="count",
    )
    figure, axis = plt.subplots(figsize=(13, max(6.0, 0.42 * len(failed_checks) + 2.0)))
    sns.barplot(data=check_long, y="check_id", x="count", hue="measure", ax=axis)
    axis.set_title("Top deterministic invariant failures")
    axis.set_xlabel("documents or findings")
    axis.set_ylabel("")
    output["plots/06_deterministic_check_failures.png"] = _png_bytes(figure)

    per_document_findings = (
        finding_frame.groupby(["document_id", "origin"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(case_frame["document_id"], fill_value=0)
    )
    for origin in ("deterministic", "semantic"):
        if origin not in per_document_findings:
            per_document_findings[origin] = 0
    finding_distribution = per_document_findings.reset_index().melt(
        id_vars="document_id",
        value_vars=["deterministic", "semantic"],
        var_name="origin",
        value_name="findings",
    )
    figure, axis = plt.subplots(figsize=(11, 6))
    sns.histplot(
        data=finding_distribution,
        x="findings",
        hue="origin",
        discrete=True,
        multiple="dodge",
        shrink=0.8,
        ax=axis,
    )
    axis.set_title("Latest findings per attempted document")
    axis.set_ylabel("document-origin observations")
    output["plots/07_findings_per_document.png"] = _png_bytes(figure)

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    sns.histplot(
        data=case_frame,
        x="source_lines",
        hue="certification_pathway",
        bins=20,
        element="step",
        ax=axes[0],
    )
    sns.histplot(
        data=case_frame,
        x="source_characters",
        hue="certification_pathway",
        bins=20,
        element="step",
        ax=axes[1],
    )
    axes[0].set_title("Source physical-line length")
    axes[1].set_title("Source character length")
    output["plots/08_source_length_distributions.png"] = _png_bytes(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    sns.boxplot(data=case_frame, x="certification_pathway", y="pages", ax=axes[0])
    sns.stripplot(
        data=case_frame,
        x="certification_pathway",
        y="pages",
        color="black",
        alpha=0.35,
        size=3,
        ax=axes[0],
    )
    sns.boxplot(data=case_frame, x="certification_pathway", y="source_nonblank_lines", ax=axes[1])
    for axis, title in zip(axes, ("Pages", "Nonblank source lines"), strict=True):
        axis.set_title(title)
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=18)
    output["plots/09_page_and_line_complexity.png"] = _png_bytes(figure)

    lexical = case_frame.melt(
        id_vars=["document_id", "certification_pathway"],
        value_vars=[
            "source_tokens",
            "source_unique_tokens",
            "source_digit_fraction",
            "source_uppercase_fraction",
        ],
        var_name="measure",
        value_name="value",
    )
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    for axis, measure in zip(axes.flat, lexical["measure"].unique(), strict=True):
        selected = lexical[lexical["measure"] == measure]
        sns.histplot(
            data=selected,
            x="value",
            hue="certification_pathway",
            bins=18,
            element="step",
            ax=axis,
        )
        axis.set_title(str(measure).replace("source_", "").replace("_", " "))
    output["plots/10_source_lexical_profile.png"] = _png_bytes(figure)

    figure, axis = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=case_frame,
        x="certification_pathway",
        y="unchanged_line_fraction",
        ax=axis,
    )
    sns.stripplot(
        data=case_frame,
        x="certification_pathway",
        y="unchanged_line_fraction",
        color="black",
        alpha=0.35,
        size=3,
        ax=axis,
    )
    axis.set_title("Template retention by certification pathway")
    axis.set_xlabel("")
    axis.set_ylabel("unchanged physical-line fraction")
    axis.tick_params(axis="x", rotation=15)
    output["plots/11_template_retention.png"] = _png_bytes(figure)

    figure, axis = plt.subplots(figsize=(11, 6))
    sns.scatterplot(
        data=case_frame,
        x="source_lines",
        y="changed_source_lines",
        hue="certification_pathway",
        size="changed_blocks",
        sizes=(25, 240),
        alpha=0.75,
        ax=axis,
    )
    axis.set_title("Edit scope versus source length")
    axis.set_xlabel("source physical lines")
    axis.set_ylabel("changed source lines")
    output["plots/12_edit_scope_vs_source_length.png"] = _png_bytes(figure)

    figure, axis = plt.subplots(figsize=(9, 7))
    sns.scatterplot(
        data=case_frame,
        x="source_characters",
        y="candidate_characters",
        hue="certification_pathway",
        alpha=0.75,
        ax=axis,
    )
    extent = max(case_frame["source_characters"].max(), case_frame["candidate_characters"].max())
    axis.plot([0, extent], [0, extent], linestyle="--", color="gray", linewidth=1)
    axis.set_title("Source and generated-candidate size")
    output["plots/13_source_vs_candidate_characters.png"] = _png_bytes(figure)

    label_complexity = case_frame.melt(
        id_vars=["document_id", "certification_pathway"],
        value_vars=[
            "source_label_scalar_leaves",
            "target_label_scalar_leaves",
            "label_changed_scalar_paths",
        ],
        var_name="measure",
        value_name="scalar_paths",
    )
    figure, axis = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=label_complexity, x="measure", y="scalar_paths", ax=axis)
    sns.stripplot(
        data=label_complexity,
        x="measure",
        y="scalar_paths",
        color="black",
        alpha=0.28,
        size=3,
        ax=axis,
    )
    axis.set_title("Structured-label size and mutation breadth")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=15)
    output["plots/14_label_scalar_complexity.png"] = _png_bytes(figure)

    topology_columns = ["containers", "cargo_groups", "packages", "allocations", "party_roles"]
    topology = case_frame.melt(
        id_vars="document_id",
        value_vars=topology_columns,
        var_name="feature",
        value_name="count",
    )
    figure, axis = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=topology, x="feature", y="count", showfliers=False, ax=axis)
    sns.stripplot(data=topology, x="feature", y="count", color="black", alpha=0.25, size=3)
    axis.set_title("Synthetic-target topology")
    axis.set_xlabel("")
    output["plots/15_target_topology.png"] = _png_bytes(figure)

    feature_columns = [
        "containers",
        "cargo_groups",
        "packages",
        "allocations",
        "dangerous_goods",
        "hs_codes",
        "additional_information",
        "marks_and_numbers",
        "temperature_containers",
        "notify_parties",
    ]
    prevalence = pd.DataFrame(
        [
            {"feature": column, "documents": int((case_frame[column] > 0).sum())}
            for column in feature_columns
        ]
    ).sort_values("documents", ascending=True)
    figure, axis = plt.subplots(figsize=(11, 7))
    sns.barplot(data=prevalence, y="feature", x="documents", hue="feature", legend=False, ax=axis)
    _label_bars(axis)
    axis.set_title("Synthetic-target feature prevalence")
    axis.set_ylabel("")
    output["plots/16_target_feature_prevalence.png"] = _png_bytes(figure)

    correlation_columns = [
        "source_lines",
        "source_tokens",
        "changed_source_lines",
        "changed_blocks",
        "unchanged_line_fraction",
        "target_label_scalar_leaves",
        "label_changed_scalar_paths",
        "containers",
        "cargo_groups",
        "packages",
        "allocations",
        "latest_deterministic_findings",
        "latest_semantic_findings",
        "provider_cost_usd",
        "provider_attempts",
    ]
    correlations = case_frame[correlation_columns].corr(numeric_only=True)
    figure, axis = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        correlations,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7},
        ax=axis,
    )
    axis.set_title("Raw-text, label, finding, and provider-cost correlations")
    output["plots/17_complexity_correlation.png"] = _png_bytes(figure)

    for number, dimension, title, maximum in (
        (18, "package_type", "Source vs target package categories", 20),
        (19, "container_type", "Source vs target container types", 20),
        (20, "container_size", "Source vs target container sizes", 20),
        (21, "party_country", "Source vs target party countries", 20),
        (22, "route_country", "Source vs target route countries", 20),
        (23, "hs_chapter", "Source vs target HS chapters", 20),
        (24, "freight_arrangement", "Source vs target freight arrangements", 12),
        (25, "negotiability", "Source vs target negotiability", 12),
    ):
        output[f"plots/{number:02d}_{dimension}.png"] = _paired_category_png(
            semantic_frame,
            dimension=dimension,
            title=title,
            maximum_values=maximum,
        )

    numeric_dimensions = [
        dimension
        for dimension in (
            "package_quantity",
            "allocation_package_quantity",
            "gross_weight",
            "net_weight",
            "volume",
            "temperature_setpoint",
        )
        if dimension in set(numeric_frame["dimension"])
    ]
    figure, axes = plt.subplots(2, 3, figsize=(17, 10))
    for axis, dimension in zip(axes.flat, numeric_dimensions, strict=False):
        selected = numeric_frame[numeric_frame["dimension"] == dimension]
        sns.boxplot(data=selected, x="variant", y="value", ax=axis)
        sns.stripplot(
            data=selected,
            x="variant",
            y="value",
            color="black",
            alpha=0.3,
            size=2.5,
            ax=axis,
        )
        if dimension != "temperature_setpoint" and (selected["value"] > 0).all():
            axis.set_yscale("log")
        axis.set_title(dimension.replace("_", " "))
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=12)
    for axis in axes.flat[len(numeric_dimensions) :]:
        axis.set_visible(False)
    figure.suptitle("Source and synthetic-target numeric facts (log scale except temperature)")
    output["plots/26_numeric_label_facts.png"] = _png_bytes(figure)

    transitions = (
        transition_frame.groupby("dimension", observed=True)["changed"]
        .agg([("documents", "size"), ("changed_documents", "sum")])
        .reset_index()
    )
    transitions["changed_fraction"] = transitions["changed_documents"] / transitions["documents"]
    transitions = transitions.sort_values("changed_fraction", ascending=True)
    figure, axis = plt.subplots(figsize=(12, 7))
    sns.barplot(
        data=transitions,
        y="dimension",
        x="changed_fraction",
        hue="dimension",
        legend=False,
        ax=axis,
    )
    axis.set_xlim(0, 1)
    axis.set_title("Per-document source-to-target categorical change rate")
    axis.set_xlabel("documents with changed category signature")
    axis.set_ylabel("")
    output["plots/27_categorical_change_rates.png"] = _png_bytes(figure)

    cost_frame["provider_reported_cost_usd"] = pd.to_numeric(
        cost_frame["provider_reported_cost_usd"]
    )
    stage_cost = (
        cost_frame.groupby("stage", observed=True)["provider_reported_cost_usd"]
        .sum()
        .reset_index(name="cost_usd")
    )
    figure, axis = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=stage_cost, x="stage", y="cost_usd", hue="stage", legend=False, ax=axis)
    _label_bars(axis, decimals=4)
    axis.set_title("Provider-reported cost by pipeline stage")
    axis.set_ylabel("USD")
    axis.set_xlabel("")
    output["plots/28_cost_by_stage.png"] = _png_bytes(figure)

    stage_tokens = (
        cost_frame.groupby("stage", observed=True)[
            ["input_tokens", "reasoning_tokens", "visible_output_tokens"]
        ]
        .sum()
        .reset_index()
        .melt(id_vars="stage", var_name="token_class", value_name="tokens")
    )
    figure, axis = plt.subplots(figsize=(11, 6))
    sns.barplot(data=stage_tokens, x="stage", y="tokens", hue="token_class", ax=axis)
    axis.set_title("Token usage by pipeline stage")
    axis.set_xlabel("")
    output["plots/29_tokens_by_stage.png"] = _png_bytes(figure)

    route_counts = (
        route_frame.groupby(["route_provider", "outcome"], observed=True)
        .size()
        .reset_index(name="attempts")
    )
    figure, axis = plt.subplots(figsize=(12, 6))
    sns.barplot(data=route_counts, x="route_provider", y="attempts", hue="outcome", ax=axis)
    axis.set_title("Provider routing outcomes")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=15)
    output["plots/30_provider_route_outcomes.png"] = _png_bytes(figure)

    responses = route_frame[route_frame["outcome"] != "transport_error"]
    figure, axis = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=responses, x="route_provider", y="latency_seconds", hue="stage", ax=axis)
    sns.stripplot(
        data=responses,
        x="route_provider",
        y="latency_seconds",
        color="black",
        alpha=0.25,
        size=2,
        ax=axis,
    )
    logarithmic_latency = float(responses["latency_seconds"].max()) > 100
    if logarithmic_latency:
        axis.set_yscale("log")
    latency_scale = "log" if logarithmic_latency else "linear"
    axis.set_title(f"Structured-response latency by provider ({latency_scale} scale)")
    axis.set_xlabel("")
    axis.set_ylabel("seconds")
    axis.tick_params(axis="x", rotation=15)
    output["plots/31_provider_latency.png"] = _png_bytes(figure)

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    sns.histplot(
        data=case_frame,
        x="provider_cost_usd",
        hue="certification_pathway",
        bins=18,
        element="step",
        ax=axes[0],
    )
    sns.boxplot(
        data=case_frame,
        x="certification_pathway",
        y="provider_attempts",
        ax=axes[1],
    )
    axes[0].set_title("Per-document provider-reported cost")
    axes[0].set_xlabel("USD")
    axes[1].set_title("Provider attempts by certification pathway")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=15)
    output["plots/32_document_cost_and_attempts.png"] = _png_bytes(figure)

    figure, axis = plt.subplots(figsize=(11, 6))
    sns.scatterplot(
        data=case_frame,
        x="source_tokens",
        y="provider_cost_usd",
        hue="certification_pathway",
        size="provider_attempts",
        sizes=(25, 220),
        alpha=0.75,
        ax=axis,
    )
    axis.set_title("Raw-text complexity and provider cost")
    axis.set_xlabel("source tokens")
    axis.set_ylabel("provider-reported USD")
    output["plots/33_cost_vs_source_tokens.png"] = _png_bytes(figure)

    ordered_cost = case_frame.sort_values("provider_cost_usd", ascending=False).copy()
    ordered_cost["documents"] = range(1, len(ordered_cost) + 1)
    total_cost = float(ordered_cost["provider_cost_usd"].sum())
    ordered_cost["cumulative_cost_fraction"] = (
        ordered_cost["provider_cost_usd"].cumsum() / total_cost
    )
    figure, axis = plt.subplots(figsize=(10, 5.8))
    sns.lineplot(
        data=ordered_cost,
        x="documents",
        y="cumulative_cost_fraction",
        marker="o",
        markersize=3,
        ax=axis,
    )
    axis.axhline(0.5, color="gray", linestyle="--")
    axis.set_title("Cost concentration (most expensive documents first)")
    output["plots/34_cost_pareto.png"] = _png_bytes(figure)

    audit_measures = pd.DataFrame(
        [
            {
                "check": field,
                "passed": int(manual_frame[field].sum()),
                "failed": int((~manual_frame[field]).sum()),
            }
            for field in ("semantic_consistency", "template_fidelity", "privacy_rewritten")
        ]
    ).melt(id_vars="check", var_name="verdict", value_name="documents")
    figure, axis = plt.subplots(figsize=(10, 5.8))
    sns.barplot(data=audit_measures, x="check", y="documents", hue="verdict", ax=axis)
    _label_bars(axis)
    axis.set_title(f"Manual review of every machine-certified case (n={len(manual_frame)})")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=12)
    output["plots/35_manual_certified_census.png"] = _png_bytes(figure)
    return output


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


def _clean_yield_estimate(
    *,
    documents: int,
    certified: int,
    manual_passes: int,
    random_passes: int,
    random_documents: int,
    certified_census_complete: bool,
) -> tuple[float, float, float, str]:
    """Estimate accepted yield without understating uncertainty for a complete census."""

    if documents <= 0 or certified < 0 or certified > documents:
        raise ValueError("invalid pipeline cohort counts")
    if manual_passes < 0 or manual_passes > certified:
        raise ValueError("invalid manual-pass count")
    if random_documents <= 0 or random_passes < 0 or random_passes > random_documents:
        raise ValueError("invalid manual random-sample counts")
    if certified_census_complete:
        estimate = manual_passes / documents
        lower, upper = _wilson_interval(manual_passes, documents)
        return estimate, lower, upper, "complete_machine_certified_census"
    conditional_lower, conditional_upper = _wilson_interval(random_passes, random_documents)
    certified_fraction = certified / documents
    return (
        certified_fraction * random_passes / random_documents,
        certified_fraction * conditional_lower,
        certified_fraction * conditional_upper,
        "screening_fraction_times_sampled_conditional_precision",
    )


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
    invariant_checks: Sequence[Mapping[str, Any]],
    costs: Sequence[Mapping[str, Any]],
    case_usage: Sequence[Mapping[str, Any]],
    route_attempts: Sequence[Mapping[str, Any]],
    semantic_values: Sequence[Mapping[str, Any]],
    numeric_values: Sequence[Mapping[str, Any]],
    manual: Sequence[Mapping[str, Any]],
    random_ids: Sequence[str],
    plot_count: int,
) -> bytes:
    status = Counter(row["status"] for row in metrics)
    pathways = Counter(row["certification_pathway"] for row in metrics)
    finding_origins = Counter(row["origin"] for row in findings)
    finding_kinds = Counter(row["finding_kind"] for row in findings)
    finding_dimensions = Counter(row["dimension"] for row in findings)
    certified = status["certified"]
    documents = len(metrics)
    provider_cost_decimal = sum(
        (Decimal(str(row["provider_reported_cost_usd"])) for row in costs),
        start=Decimal(0),
    )
    provider_cost = float(provider_cost_decimal)
    requests = sum(int(row["requests"]) for row in costs)
    provider_attempts = sum(int(row["provider_attempts"]) for row in costs)
    failed_attempts = sum(int(row["failed_provider_attempts"]) for row in costs)
    manual_passes = sum(bool(row["all_checks_passed"]) for row in manual)
    random_rows = [row for row in manual if row["document_id"] in set(random_ids)]
    random_passes = sum(bool(row["all_checks_passed"]) for row in random_rows)
    lower, upper = _wilson_interval(random_passes, len(random_rows))
    observed_accepted_fraction = manual_passes / documents
    census_complete = len(manual) == certified and set(random_ids) == {
        str(row["document_id"]) for row in manual
    }
    (
        estimated_future_clean_fraction,
        estimated_clean_lower,
        estimated_clean_upper,
        clean_yield_method,
    ) = _clean_yield_estimate(
        documents=documents,
        certified=certified,
        manual_passes=manual_passes,
        random_passes=random_passes,
        random_documents=len(random_rows),
        certified_census_complete=census_complete,
    )
    manually_rejected = len(manual) - manual_passes
    semantic_failures = sum(not bool(row["semantic_consistency"]) for row in manual)
    template_failures = sum(not bool(row["template_fidelity"]) for row in manual)
    privacy_failures = sum(not bool(row["privacy_rewritten"]) for row in manual)
    provisional = certified - len(manual)
    total_lines = sum(int(row["source_lines"]) for row in metrics)
    changed_lines = sum(int(row["changed_source_lines"]) for row in metrics)
    median_retention = float(
        pd.Series([float(row["unchanged_line_fraction"]) for row in metrics]).median()
    )
    source_lines = [int(row["source_lines"]) for row in metrics]
    source_characters = [int(row["source_characters"]) for row in metrics]
    source_tokens = [int(row["source_tokens"]) for row in metrics]
    host_passes = sum(bool(row["latest_host_audit_passed"]) for row in metrics)
    host_failure_ids = sorted(
        str(row["document_id"]) for row in metrics if not bool(row["latest_host_audit_passed"])
    )
    host_failure_display = ", ".join(f"`{document_id}`" for document_id in host_failure_ids)
    page_passes = sum(bool(row["page_markers_preserved"]) for row in metrics)
    line_passes = sum(int(row["line_count_delta"]) == 0 for row in metrics)
    blank_passes = sum(int(row["blank_line_delta"]) == 0 for row in metrics)
    publication_status = str(bool(summary["trainingRecordsPublished"])).lower()
    failed_checks: Counter[str] = Counter()
    for row in invariant_checks:
        if not bool(row["passed"]):
            failed_checks[str(row["check_id"])] += 1
    feature_fields = (
        "containers",
        "cargo_groups",
        "packages",
        "allocations",
        "dangerous_goods",
        "hs_codes",
        "temperature_containers",
    )
    feature_totals = {field: sum(int(row[field]) for row in metrics) for field in feature_fields}
    feature_coverage = {
        field: sum(int(row[field]) > 0 for row in metrics) for field in feature_fields
    }
    categorical_diversity = {
        f"{dimension}:{variant}": len(
            {
                str(row["value"])
                for row in semantic_values
                if row["dimension"] == dimension and row["variant"] == variant
            }
        )
        for dimension in sorted({str(row["dimension"]) for row in semantic_values})
        for variant in ("source", "synthetic_target")
    }
    route_outcomes = Counter(str(row["outcome"]) for row in route_attempts)
    stage_cost: defaultdict[str, Decimal] = defaultdict(Decimal)
    for row in costs:
        stage_cost[str(row["stage"])] += Decimal(str(row["provider_reported_cost_usd"]))
    stage_cost_display = ", ".join(
        f"{stage} ${cost:.9f}" for stage, cost in sorted(stage_cost.items())
    )
    review_scope = (
        "an exhaustive review of every machine-certified case"
        if census_complete
        else "the exact deterministic random machine-certified-case sample"
    )
    rows = [
        "# Raw-text pipeline outcome and quality audit",
        "",
        "## Decision result",
        "",
        f"- Pipeline machine-certified: **{certified}/{documents} ({certified / documents:.1%})**.",
        f"- Pipeline-quarantined: **{status['blocked']}/{documents}**.",
        f"- Manually confirmed machine-certified cases: **{manual_passes}/{len(manual)}**; "
        f"manually rejected false positives: **{manually_rejected}/{len(manual)}**; "
        f"remaining provisional and unaudited: **{provisional}/{documents}**.",
        f"- Training records published: **{publication_status}**.",
        f"- Manual review scope: **{review_scope}**. Passed "
        f"**{random_passes}/{len(random_rows)}**; "
        f"the Wilson 95% interval for future conditional certification precision is "
        f"**{lower:.1%}-{upper:.1%}**. It is not an interval for this completed census.",
        f"- Manual failure dimensions across all {len(manual)} reviewed cases: semantic "
        f"**{semantic_failures}**, template **{template_failures}**, privacy/source-residue "
        f"**{privacy_failures}**; dimensions can overlap.",
        (
            f"- Observed manually accepted yield for this attempted cohort: "
            f"**{manual_passes}/{documents} ({observed_accepted_fraction:.1%})**. Because every "
            "machine-certified case was reviewed, the direct attempted-cohort Wilson 95% "
            f"interval is **{estimated_clean_lower:.1%}-{estimated_clean_upper:.1%}**. This "
            "interval assumes comparable future sampling and does not establish unseen-template "
            "performance."
            if clean_yield_method == "complete_machine_certified_census"
            else f"- Observed manually accepted cases in the reviewed subset: "
            f"**{manual_passes}/{documents} ({observed_accepted_fraction:.1%})**. The "
            f"screening-adjusted clean-yield estimate is **{estimated_future_clean_fraction:.1%}** "
            f"with a conditional-precision range of **{estimated_clean_lower:.1%}-"
            f"{estimated_clean_upper:.1%}**; this incomplete-census estimate does not establish "
            "unseen-template performance."
        ),
        f"- Terminal pathways: **{dict(pathways)}**.",
        "",
        "Machine certification is not an acceptance decision: audited false positives are moved "
        "to `manual-quarantine/`, audited passes to `confirmed/`, and unaudited machine passes to "
        "`provisional/`. The analyzer never promotes any subset into a training publication.",
        "",
        "## Deterministic and provenance checks",
        "",
        f"- Complete pipeline and component commit receipts validated: **{len(costs) + 1} runs**.",
        f"- Latest host-audit replay passed: **{host_passes}/{documents}**; failures already "
        f"pipeline-quarantined: {host_failure_display or 'none'}.",
        f"- Page-marker topology preserved: **{page_passes}/{documents}**.",
        f"- Physical line count preserved: **{line_passes}/{documents}**; "
        f"blank-line count preserved: **{blank_passes}/{documents}**.",
        f"- Changed source lines: **{changed_lines:,}/{total_lines:,} "
        f"({changed_lines / total_lines:.1%})**; "
        f"median unchanged-line fraction: **{median_retention:.1%}**.",
        "",
        "## Quarantine evidence",
        "",
        f"- Latest findings by origin: **{dict(finding_origins)}**; by category: "
        f"**{dict(finding_kinds)}**.",
        f"- Findings across audit dimensions: **{dict(finding_dimensions)}**.",
        f"- Most frequent deterministic failing checks by affected documents: "
        f"**{dict(failed_checks.most_common(12))}**.",
        "- Every pipeline rejection has its exact current-candidate hash, blocking reason, "
        "evidence fragments, and workbook under `pipeline-quarantine/`.",
        "- Every manually detected false positive is segregated under `manual-quarantine/` with "
        "the reviewer decision and concrete evidence note.",
        "",
        "## Raw-text and label exploration",
        "",
        f"- Source corpus size: **{sum(source_lines):,} physical lines**, "
        f"**{sum(source_characters):,} characters**, and **{sum(source_tokens):,} lexical "
        f"tokens**. Per document, source lines have median "
        f"**{pd.Series(source_lines).median():.1f}** "
        f"and range **{min(source_lines)}-{max(source_lines)}**; characters have median "
        f"**{pd.Series(source_characters).median():.1f}** and range "
        f"**{min(source_characters):,}-{max(source_characters):,}**.",
        f"- Changed source lines: **{changed_lines:,}/{total_lines:,} "
        f"({changed_lines / total_lines:.1%})**; median unchanged-line fraction: "
        f"**{median_retention:.1%}**.",
        f"- Synthetic-target topology totals: **{feature_totals}**; document coverage: "
        f"**{feature_coverage}**.",
        f"- Flattened categorical diversity (source and target): "
        f"**{categorical_diversity}**. Numeric label observations: **{len(numeric_values)}**.",
        "- All source, candidate, and target hashes are retained per case; no plot or manifest "
        "is a training-data publication.",
        "",
        "## Cost of the exact attempted lineage",
        "",
        f"- Unique provider-bearing component runs: **{len(costs)}**.",
        f"- Provider-reported cost: **${provider_cost_decimal:.9f}**, or "
        f"**${provider_cost / documents * 1000:.3f}/1,000 attempted documents** and "
        f"**${provider_cost / max(manual_passes, 1) * 1000:.3f}/1,000 manually accepted "
        "documents** for this calibration lineage.",
        f"- Stage cost split: **{stage_cost_display}**.",
        f"- Successful structured responses: **{requests}**; provider-route attempts: "
        f"**{provider_attempts}**, of which **{failed_attempts}** failed before an accepted "
        "structured response.",
        f"- Route-attempt outcome audit: **{dict(route_outcomes)}**; component-case usage rows: "
        f"**{len(case_usage)}**.",
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
        "- `case-metrics.csv`, `findings.csv`, `invariant-checks.csv`, `lineage-costs.csv`, "
        "`case-usage.csv`, `route-attempts.csv`, `semantic-values.csv`, "
        "`semantic-transitions.csv`, and `numeric-label-values.csv`: reproducible EDA tables.",
        "- `manual-audit.csv`: reviewer decisions and selection provenance.",
        f"- `plots/`: **{plot_count}** deterministic Matplotlib/Seaborn PNG figures.",
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
    costs = _cost_rows(project_root=project_root, lineage=lineage, cache=cache)
    case_usage, route_attempts = _case_usage_and_route_rows(
        project_root=project_root,
        lineage=lineage,
        cache=cache,
        costs=costs,
    )
    usage_by_id: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in case_usage:
        usage_by_id[str(row["document_id"])].append(row)
    for row in metrics:
        usage = usage_by_id[str(row["document_id"])]
        row.update(
            {
                "model_requests": sum(int(value["requests"]) for value in usage),
                "provider_attempts": sum(int(value["provider_attempts"]) for value in usage),
                "failed_provider_attempts": sum(
                    int(value["failed_provider_attempts"]) for value in usage
                ),
                "host_rejected_responses": sum(
                    int(value["host_rejected_responses"]) for value in usage
                ),
                "input_tokens": sum(int(value["input_tokens"]) for value in usage),
                "reasoning_tokens": sum(int(value["reasoning_tokens"]) for value in usage),
                "visible_output_tokens": sum(
                    int(value["visible_output_tokens"]) for value in usage
                ),
                "provider_cost_usd": sum(
                    float(value["provider_reported_cost_usd"]) for value in usage
                ),
                "route_latency_seconds": sum(
                    float(value["route_latency_seconds"]) for value in usage
                ),
            }
        )
    metric_by_id = {row["document_id"]: row for row in metrics}
    manual, random_ids = _manual_audit_rows(
        audit=manual_audit,
        cases=cases,
        pipeline_commit_sha256=pipeline_commit_sha256,
    )
    findings = _finding_rows(cases)
    invariant_checks = _invariant_check_rows(cases)
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
    numeric_values = [
        row
        for case in cases
        for variant, label in (
            ("source", case.source_label),
            ("synthetic_target", case.target_label),
        )
        for row in _numeric_values(
            document_id=case.lineage.documentId,
            status=case.lineage.status,
            variant=variant,
            label=label,
        )
    ]
    semantic_transitions = _semantic_transition_rows(semantic_values)

    manual_passes = sum(bool(row["all_checks_passed"]) for row in manual)
    manual_by_id = {str(row["document_id"]): row for row in manual}
    disposition_by_id = {
        case.lineage.documentId: _audit_disposition(
            status=case.lineage.status,
            manual_audit=manual_by_id.get(case.lineage.documentId),
        )
        for case in cases
    }
    for row in metrics:
        row["audit_disposition"] = disposition_by_id[str(row["document_id"])]
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
    certified_census_complete = len(manual) == len(machine_certified) and set(random_ids) == {
        str(row["document_id"]) for row in manual
    }
    (
        clean_yield_estimate,
        clean_yield_lower,
        clean_yield_upper,
        clean_yield_method,
    ) = _clean_yield_estimate(
        documents=len(cases),
        certified=len(machine_certified),
        manual_passes=manual_passes,
        random_passes=random_passes,
        random_documents=len(random_rows),
        certified_census_complete=certified_census_complete,
    )
    provider_cost = float(
        sum(
            (Decimal(str(row["provider_reported_cost_usd"])) for row in costs),
            start=Decimal(0),
        )
    )
    plot_artifacts = _plot_artifacts(
        summary=summary,
        metrics=metrics,
        findings=findings,
        invariant_checks=invariant_checks,
        costs=costs,
        case_usage=case_usage,
        route_attempts=route_attempts,
        semantic_values=semantic_values,
        numeric_values=numeric_values,
        semantic_transitions=semantic_transitions,
        manual=manual,
    )
    feature_fields = (
        "containers",
        "cargo_groups",
        "packages",
        "allocations",
        "dangerous_goods",
        "hs_codes",
        "additional_information",
        "marks_and_numbers",
        "temperature_containers",
        "notify_parties",
    )
    finding_origins = Counter(str(row["origin"]) for row in findings)
    analysis_summary = {
        "schemaVersion": 2,
        "runId": run_id,
        "pipelineRunId": summary["runId"],
        "pipelineCommitSha256": pipeline_commit_sha256,
        "pipelineTransactionSha256": pipeline_transaction_sha256,
        "documents": len(cases),
        "pipelineCertifiedDocuments": len(machine_certified),
        "pipelineQuarantinedDocuments": len(disposition_rows["pipeline_quarantine"]),
        "pipelineCertifiedFraction": len(machine_certified) / len(cases),
        "pipelineDeterministicRejectedDocuments": sum(
            row["certification_pathway"] == "deterministic_rejected" for row in metrics
        ),
        "pipelineSemanticRejectedDocuments": sum(
            row["certification_pathway"] == "semantic_rejected" for row in metrics
        ),
        "manuallyConfirmedDocuments": len(disposition_rows["confirmed"]),
        "manuallyRejectedDocuments": len(disposition_rows["manual_quarantine"]),
        "provisionalDocuments": len(disposition_rows["provisional"]),
        "manualCertifiedCensusComplete": certified_census_complete,
        "observedManuallyAcceptedCohortFraction": manual_passes / len(cases),
        "hostAuditReplayPasses": sum(bool(row["latest_host_audit_passed"]) for row in metrics),
        "hostAuditReplayFailureDocumentIds": sorted(
            str(row["document_id"]) for row in metrics if not bool(row["latest_host_audit_passed"])
        ),
        "pageMarkersPreserved": sum(bool(row["page_markers_preserved"]) for row in metrics),
        "zeroLineCountDelta": sum(int(row["line_count_delta"]) == 0 for row in metrics),
        "zeroBlankLineDelta": sum(int(row["blank_line_delta"]) == 0 for row in metrics),
        "findingOrigins": dict(finding_origins),
        "findingKinds": dict(Counter(str(row["finding_kind"]) for row in findings)),
        "findingDimensions": dict(Counter(str(row["dimension"]) for row in findings)),
        "manualAuditDocuments": len(manual),
        "manualAuditPasses": manual_passes,
        "manualSemanticFailures": sum(not bool(row["semantic_consistency"]) for row in manual),
        "manualTemplateFailures": sum(not bool(row["template_fidelity"]) for row in manual),
        "manualPrivacyFailures": sum(not bool(row["privacy_rewritten"]) for row in manual),
        "randomAuditDocuments": len(random_rows),
        "randomAuditPasses": random_passes,
        "randomAuditPassFraction": random_passes / len(random_rows),
        "randomAuditWilson95Lower": lower,
        "randomAuditWilson95Upper": upper,
        "estimatedCleanCohortFraction": clean_yield_estimate,
        "estimatedCleanCohortWilson95Lower": clean_yield_lower,
        "estimatedCleanCohortWilson95Upper": clean_yield_upper,
        "estimatedCleanCohortMethod": clean_yield_method,
        "uniqueSourceTexts": len({case.lineage.sourceTextSha256 for case in cases}),
        "uniqueCandidateTexts": len({case.lineage.currentCandidateSha256 for case in cases}),
        "uniqueTargetLabels": len({str(row["target_sha256"]) for row in metrics}),
        "sourceCharacters": sum(int(row["source_characters"]) for row in metrics),
        "sourcePhysicalLines": sum(int(row["source_lines"]) for row in metrics),
        "sourceLexicalTokens": sum(int(row["source_tokens"]) for row in metrics),
        "changedSourceLines": sum(int(row["changed_source_lines"]) for row in metrics),
        "targetFeatureTotals": {
            field: sum(int(row[field]) for row in metrics) for field in feature_fields
        },
        "targetFeatureDocumentCoverage": {
            field: sum(int(row[field]) > 0 for row in metrics) for field in feature_fields
        },
        "semanticValueRows": len(semantic_values),
        "numericLabelValueRows": len(numeric_values),
        "semanticTransitionRows": len(semantic_transitions),
        "uniqueComponentRuns": len(costs),
        "providerReportedLineageCostUsd": provider_cost,
        "providerReportedLineageCostPerThousandAttemptedUsd": (provider_cost / len(cases) * 1000),
        "providerReportedLineageCostPerThousandCertifiedUsd": (
            provider_cost / len(machine_certified) * 1000
        ),
        "providerReportedLineageCostPerThousandManuallyAcceptedUsd": (
            provider_cost / max(manual_passes, 1) * 1000
        ),
        "requests": sum(int(row["requests"]) for row in costs),
        "providerAttempts": sum(int(row["provider_attempts"]) for row in costs),
        "failedProviderAttempts": sum(int(row["failed_provider_attempts"]) for row in costs),
        "hostRejectedProviderResponses": sum(
            int(row["host_rejected_responses"]) for row in case_usage
        ),
        "routeAttemptOutcomes": dict(Counter(str(row["outcome"]) for row in route_attempts)),
        "inputTokens": sum(int(row["input_tokens"]) for row in costs),
        "reasoningTokens": sum(int(row["reasoning_tokens"]) for row in costs),
        "visibleOutputTokens": sum(int(row["visible_output_tokens"]) for row in costs),
        "plotArtifacts": len(plot_artifacts),
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
        "invariant-checks.csv": _csv_bytes(invariant_checks),
        "lineage-costs.csv": _csv_bytes(costs),
        "case-usage.csv": _csv_bytes(case_usage),
        "route-attempts.csv": _csv_bytes(route_attempts),
        "semantic-values.csv": _csv_bytes(semantic_values),
        "semantic-transitions.csv": _csv_bytes(semantic_transitions),
        "numeric-label-values.csv": _csv_bytes(numeric_values),
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
            invariant_checks=invariant_checks,
            costs=costs,
            case_usage=case_usage,
            route_attempts=route_attempts,
            semantic_values=semantic_values,
            numeric_values=numeric_values,
            manual=manual,
            random_ids=random_ids,
            plot_count=len(plot_artifacts),
        ),
    }
    artifacts.update(plot_artifacts)
    for disposition, rows in disposition_rows.items():
        artifacts[f"{disposition.replace('_', '-')}/manifest.json"] = (
            canonical_json_bytes({"schemaVersion": 1, "documents": len(rows), "cases": rows})
            + b"\n"
        )
    for case in cases:
        disposition = disposition_by_id[case.lineage.documentId]
        category = disposition.replace("_", "-")
        artifacts[f"{category}/cases/{case.lineage.documentId}.md"] = _case_workbook(
            case,
            audit_disposition=disposition,
            manual_audit=manual_by_id.get(case.lineage.documentId),
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
