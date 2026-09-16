#!/usr/bin/env python3
"""Materialize, pre-register, and score the frozen contract-v2 benchmark.

``build`` copies exact manually reviewed bytes into an immutable non-training cohort. ``plan``
binds that cohort to exact baseline/challenger configuration bytes and reasoning efforts before
provider calls. ``score`` replays only those planned committed arms and records classification,
issue-evidence quality, reasoning, and cost without making provider calls itself.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

import document_ocr.synthesis.config as synthesis_config_module
import document_ocr.synthesis.raw_text_certified_correction as raw_text_correction_module
import document_ocr.synthesis.raw_text_pipeline as raw_text_pipeline_module
from document_ocr.atomic import json_artifact_bytes, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.config import (
    RawTextCertificationInvariantInputsConfig,
    SynthesisRawTextCertificationConfig,
    load_synthesis_raw_text_certification_config,
    load_synthesis_raw_text_inventory_batch_config,
)
from document_ocr.synthesis.raw_text_certification import (
    _AUDIT_DIMENSIONS,
    _FINDING_DIMENSIONS,
    AuditDimension,
    CertificationCaseResult,
    _host_audit,
    _validate_case_contract_labels,
    certification_implementation_contract,
    run_raw_text_certification,
    validate_certification_run_stage_identities,
)
from document_ocr.synthesis.raw_text_certification_artifacts import (
    ValidatedCertificationCase,
    ValidatedCertificationRun,
    load_validated_certification_run,
)
from document_ocr.synthesis.raw_text_certification_host import (
    evaluate_certification_invariant_case,
    load_certification_invariant_context,
)
from document_ocr.synthesis.raw_text_inventory import (
    DeterministicInventoryEdit,
    InventoryCandidate,
)
from document_ocr.synthesis.raw_text_inventory_probe import (
    RawTextInventoryCompilerCase,
    compile_raw_text_inventory_cases,
    raw_text_inventory_compiler_contract,
)
from document_ocr.synthesis.raw_text_pipeline_lineage import (
    PipelineResumeLineageRow as _PipelineResumeLineageRow,
)
from document_ocr.synthesis.run_safety import StagedArtifactRun

_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)
_SHA256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_DOCUMENT_ID = Annotated[str, StringConstraints(pattern=r"^doc_[0-9a-f]{64}$")]
_LINE_ID = Annotated[str, StringConstraints(pattern=r"^L[0-9]{5}$")]
BenchmarkEffort = Literal["low", "medium", "high"]
_EFFORT_RANK: dict[BenchmarkEffort, int] = {"low": 0, "medium": 1, "high": 2}
_SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
_ISOLATED_BENCHMARK_ROUTES = frozenset(
    {"deepinfra/fp4", "coreweave/fp8", "fireworks", "nextbit/fp8"}
)
_QUALITY_TRANSACTION_KEYS = {
    "schemaVersion",
    "runId",
    "pipelineRun",
    "manualAudit",
    "implementationSha256",
}
_QUALITY_MANIFEST_KEYS = {"schemaVersion", "documents", "cases"}
_QUALITY_CASE_KEYS = {
    "documentId",
    "status",
    "auditDisposition",
    "sourceSha256",
    "candidateSha256",
    "targetSha256",
    "inventoryRun",
    "finalCertificationRun",
    "blockingReason",
    "latestSemanticFindings",
    "hostAuditReplayPassed",
    "manualAudit",
}
_MANUAL_ROW_KEYS = {
    "document_id",
    "semantic_consistency",
    "template_fidelity",
    "privacy_rewritten",
    "selection_basis",
    "notes",
    "all_checks_passed",
}


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\x00" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe relative path: {value!r}")
    return path.as_posix()


class RunPin(BaseModel):
    """One exact committed staged-run reference."""

    model_config = _STRICT

    path: str
    commitSha256: _SHA256
    transactionSha256: _SHA256

    @model_validator(mode="after")
    def path_is_safe(self) -> RunPin:
        _safe_relative_path(self.path)
        return self


class FilePin(BaseModel):
    model_config = _STRICT

    path: str
    sha256: _SHA256

    @model_validator(mode="after")
    def path_is_safe(self) -> FilePin:
        _safe_relative_path(self.path)
        return self


class BenchmarkRunConfig(BaseModel):
    model_config = _STRICT

    runId: Annotated[str, StringConstraints(min_length=1, max_length=255)]
    outputDir: str

    @model_validator(mode="after")
    def paths_are_safe(self) -> BenchmarkRunConfig:
        if _SAFE_RUN_NAME.fullmatch(self.runId) is None:
            raise ValueError("benchmark runId is not one safe path component")
        _safe_relative_path(self.outputDir)
        return self


class ExpectedIssue(BaseModel):
    """One independently adjudicated issue and its candidate evidence boundary."""

    model_config = _STRICT

    issueId: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{2,79}$")]
    primaryDimension: AuditDimension
    acceptableDimensions: Annotated[tuple[AuditDimension, ...], Field(min_length=1)]
    requiredEvidenceGroups: Annotated[
        tuple[Annotated[tuple[_LINE_ID, ...], Field(min_length=1, max_length=16)], ...],
        Field(min_length=1, max_length=16),
    ]
    description: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)
    ]

    @model_validator(mode="after")
    def values_are_unique(self) -> ExpectedIssue:
        if self.primaryDimension not in self.acceptableDimensions:
            raise ValueError("expected issue primary dimension is not acceptable")
        if len(self.acceptableDimensions) != len(set(self.acceptableDimensions)):
            raise ValueError("expected issue repeats an acceptable dimension")
        frozen_groups = tuple(tuple(group) for group in self.requiredEvidenceGroups)
        if any(len(group) != len(set(group)) for group in frozen_groups):
            raise ValueError("expected issue repeats an evidence line within a group")
        if len(frozen_groups) != len(set(frozen_groups)):
            raise ValueError("expected issue repeats an evidence group")
        return self


class BenchmarkCase(BaseModel):
    model_config = _STRICT

    documentId: _DOCUMENT_ID
    expectedClean: bool
    manualAuditNotesSha256: _SHA256
    issues: tuple[ExpectedIssue, ...]

    @model_validator(mode="after")
    def outcome_matches_issues(self) -> BenchmarkCase:
        if self.expectedClean == bool(self.issues):
            raise ValueError("clean benchmark cases must have no issues and defects need issues")
        issue_ids = tuple(row.issueId for row in self.issues)
        if len(issue_ids) != len(set(issue_ids)):
            raise ValueError("benchmark case repeats an issueId")
        return self


class BenchmarkGates(BaseModel):
    """Pre-registered promotion gates; values are part of the score transaction."""

    model_config = _STRICT

    requireAllChallengerDefectsCaught: Literal[True]
    requireAllChallengerControlsClean: Literal[True]
    requirePerfectCounterfactualCascade: Literal[True]
    requireEveryExpectedDimensionObserved: Literal[True]
    minimumChallengerIssueRecall: Annotated[Decimal, Field(ge=0, le=1)]
    minimumChallengerFindingPrecision: Annotated[Decimal, Field(ge=0, le=1)]
    minimumCascadeIssueRecall: Annotated[Decimal, Field(ge=0, le=1)]
    minimumCascadeFindingPrecision: Annotated[Decimal, Field(ge=0, le=1)]
    maximumCombinedArmCostUsd: Annotated[Decimal, Field(gt=0)]
    minimumMedianReasoningRatio: Annotated[Decimal, Field(gt=1)]
    maximumMedianCostRatio: Annotated[Decimal, Field(gt=1)]


class CohortBuildConfig(BaseModel):
    """Pre-registered source, truth, model contract, and gates for one frozen cohort."""

    model_config = _STRICT

    schemaVersion: Literal[1, 2]
    task: Literal[
        "raw_text_certification_contract_v2_benchmark_cohort_v1",
        "raw_text_certification_contract_v3_benchmark_cohort_v2",
    ]
    run: BenchmarkRunConfig
    sourceQualityAudit: RunPin
    sourcePipeline: RunPin
    manualAudit: FilePin
    prompt: FilePin
    compilerInventoryConfig: FilePin | None = None
    invariantInputs: RawTextCertificationInvariantInputsConfig | None = None
    gates: BenchmarkGates
    cases: Annotated[tuple[BenchmarkCase, ...], Field(min_length=15, max_length=15)]

    @model_validator(mode="after")
    def cases_are_the_complete_preregistered_design(self) -> CohortBuildConfig:
        legacy = self.schemaVersion == 1
        if legacy != (self.task == "raw_text_certification_contract_v2_benchmark_cohort_v1"):
            raise ValueError("benchmark cohort schema and task versions differ")
        legacy_has_current_inputs = (
            self.compilerInventoryConfig is not None or self.invariantInputs is not None
        )
        current_lacks_inputs = self.compilerInventoryConfig is None or self.invariantInputs is None
        if (legacy and legacy_has_current_inputs) or (not legacy and current_lacks_inputs):
            raise ValueError(
                "benchmark cohort v2 requires compiler and invariant inputs; v1 forbids them"
            )
        ids = tuple(row.documentId for row in self.cases)
        if len(ids) != len(set(ids)):
            raise ValueError("benchmark cohort repeats a document")
        clean = sum(row.expectedClean for row in self.cases)
        if clean != 6 or len(self.cases) - clean != 9:
            raise ValueError("benchmark cohort requires exactly 9 defects and 6 controls")
        primary_dimensions = {issue.primaryDimension for row in self.cases for issue in row.issues}
        if primary_dimensions != set(_AUDIT_DIMENSIONS):
            raise ValueError("benchmark cohort must cover every audit dimension")
        return self


class BenchmarkScoreConfig(BaseModel):
    model_config = _STRICT

    schemaVersion: Literal[2]
    task: Literal["raw_text_certification_contract_v2_benchmark_score_v2"]
    run: BenchmarkRunConfig
    planRun: RunPin
    baselineRun: RunPin
    challengerRun: RunPin


class BenchmarkPlannedArm(BaseModel):
    """One named arm whose effort and exact template bytes are fixed by the plan."""

    model_config = _STRICT

    effort: BenchmarkEffort
    config: FilePin


class BenchmarkPlanConfig(BaseModel):
    """Exact A/B configuration files committed before either provider arm runs."""

    model_config = _STRICT

    schemaVersion: Literal[2]
    task: Literal["raw_text_certification_contract_v2_benchmark_plan_v2"]
    run: BenchmarkRunConfig
    cohortRun: RunPin
    baseline: BenchmarkPlannedArm
    challenger: BenchmarkPlannedArm

    @model_validator(mode="after")
    def inputs_are_distinct(self) -> BenchmarkPlanConfig:
        if self.baseline.config.path == self.challenger.config.path:
            raise ValueError("benchmark baseline and challenger configurations must be distinct")
        if self.baseline.effort == self.challenger.effort:
            raise ValueError("benchmark baseline and challenger efforts must be distinct")
        if _EFFORT_RANK[self.baseline.effort] >= _EFFORT_RANK[self.challenger.effort]:
            raise ValueError("benchmark challenger effort must be higher than baseline effort")
        return self


class BenchmarkLaunchConfig(BaseModel):
    """Post-plan paths whose configs must be deterministic plan-authorized derivatives."""

    model_config = _STRICT

    schemaVersion: Literal[2]
    task: Literal["raw_text_certification_contract_v2_benchmark_launch_v2"]
    planRun: RunPin
    baselineConfig: FilePin
    challengerConfig: FilePin

    @model_validator(mode="after")
    def inputs_are_distinct(self) -> BenchmarkLaunchConfig:
        if self.baselineConfig.path == self.challengerConfig.path:
            raise ValueError("benchmark launch baseline and challenger configs must be distinct")
        return self


class DeterministicBenchmarkGates(BaseModel):
    """Pre-registered contract-v3 promotion gates for provider-free screening."""

    model_config = _STRICT

    requireEveryDefectCaseCaught: Literal[True]
    requireEveryControlClean: Literal[True]
    requireEveryExpectedDimensionObserved: Literal[True]
    minimumIssueRecall: Annotated[Decimal, Field(ge=0, le=1)]
    minimumFindingPrecision: Annotated[Decimal, Field(ge=0, le=1)]


class BenchmarkDeterministicAuditConfig(BaseModel):
    model_config = _STRICT

    schemaVersion: Literal[1]
    task: Literal["raw_text_certification_contract_v3_deterministic_benchmark_v1"]
    run: BenchmarkRunConfig
    cohortRun: RunPin
    gates: DeterministicBenchmarkGates


@dataclass(frozen=True, slots=True)
class ResolvedRun:
    pin: RunPin
    root: Path


@dataclass(frozen=True, slots=True)
class LoadedBenchmarkCase:
    specification: BenchmarkCase
    source: bytes
    candidate: bytes
    source_label: dict[str, Any]
    target_label: dict[str, Any]
    contract: dict[str, Any]
    inventory: tuple[InventoryCandidate, ...] | None
    deterministic_edits: tuple[DeterministicInventoryEdit, ...] | None
    ground_truth: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LoadedBenchmarkCohort:
    reference: ResolvedRun
    config: CohortBuildConfig
    cases: tuple[LoadedBenchmarkCase, ...]


@dataclass(frozen=True, slots=True)
class LoadedBenchmarkPlan:
    reference: ResolvedRun
    config: BenchmarkPlanConfig
    cohort: LoadedBenchmarkCohort
    baseline_config: SynthesisRawTextCertificationConfig
    challenger_config: SynthesisRawTextCertificationConfig
    baseline_config_payload: bytes
    challenger_config_payload: bytes


def _certification_runtime_contract() -> dict[str, str]:
    return {
        "pydanticAiVersion": version("pydantic-ai-slim"),
        "openaiVersion": version("openai"),
    }


def _cohort_dependency_contract(schema_version: Literal[1, 2]) -> dict[str, Any]:
    common = {
        "configSha256": sha256_file(Path(synthesis_config_module.__file__).resolve(strict=True)),
        "pydanticVersion": version("pydantic"),
    }
    if schema_version == 2:
        return {
            **common,
            "inventoryCompiler": raw_text_inventory_compiler_contract(),
        }
    return {
        **common,
        "certification": certification_implementation_contract(),
        "correctionSha256": sha256_file(
            Path(raw_text_correction_module.__file__).resolve(strict=True)
        ),
        "pipelineSha256": sha256_file(Path(raw_text_pipeline_module.__file__).resolve(strict=True)),
    }


def _resolve_inside(project_root: Path, value: str, *, label: str) -> Path:
    path = (project_root / _safe_relative_path(value)).resolve()
    try:
        path.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"{label} resolves outside project_root: {path}") from error
    return path


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(read_regular_file_bytes(path))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _read_json_rows(path: Path) -> tuple[dict[str, Any], ...]:
    payload = read_regular_file_bytes(path)
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"JSONL artifact lacks a terminal newline: {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            raise ValueError(f"JSONL artifact contains a blank row: {path}:{number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL artifact row is not an object: {path}:{number}")
        rows.append(cast(dict[str, Any], value))
    return tuple(rows)


def _load_config(path: Path, model: type[BaseModel]) -> BaseModel:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"benchmark configuration must be a regular file: {path}")
    return model.model_validate_json(read_regular_file_bytes(path), strict=True)


def _validate_run_pin(project_root: Path, pin: RunPin, *, label: str) -> ResolvedRun:
    root = _resolve_inside(project_root, pin.path, label=label)
    commit_path = root / "_COMMIT.json"
    if (
        root.is_symlink()
        or not root.is_dir()
        or commit_path.is_symlink()
        or not commit_path.is_file()
    ):
        raise ValueError(f"{label} is not a committed regular directory: {root}")
    if sha256_file(commit_path) != pin.commitSha256:
        raise ValueError(f"{label} commit differs from its pin")
    receipt = StagedArtifactRun(
        output_parent=root.parent,
        run_name=root.name,
        transaction_sha256=pin.transactionSha256,
    ).validate_committed_run()
    if receipt.transaction_sha256 != pin.transactionSha256:
        raise ValueError(f"{label} transaction differs from its pin")
    provenance = _read_json_object(root / "provenance/transaction.json")
    if sha256_bytes(canonical_json_bytes(provenance)) != pin.transactionSha256:
        raise ValueError(f"{label} transaction artifact is not bound to its staged-run marker")
    return ResolvedRun(pin=pin, root=root)


def _manifest_rows(path: Path, *, disposition: str) -> tuple[dict[str, Any], ...]:
    manifest = _read_json_object(path)
    if set(manifest) != _QUALITY_MANIFEST_KEYS or manifest.get("schemaVersion") != 1:
        raise ValueError(f"quality-audit {disposition} manifest has an unexpected shape")
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or manifest.get("documents") != len(raw_cases):
        raise ValueError(f"quality-audit {disposition} manifest count differs")
    rows: list[dict[str, Any]] = []
    for raw in raw_cases:
        if not isinstance(raw, dict) or set(raw) != _QUALITY_CASE_KEYS:
            raise ValueError(f"quality-audit {disposition} case has an unexpected shape")
        if raw.get("auditDisposition") != disposition:
            raise ValueError(f"quality-audit {disposition} case has the wrong disposition")
        rows.append(cast(dict[str, Any], raw))
    return tuple(rows)


def _quality_cases(
    *,
    project_root: Path,
    config: CohortBuildConfig,
) -> tuple[ResolvedRun, dict[str, dict[str, Any]], bytes]:
    quality = _validate_run_pin(
        project_root, config.sourceQualityAudit, label="source quality-audit run"
    )
    pipeline = _validate_run_pin(project_root, config.sourcePipeline, label="source pipeline run")
    manual_path = _resolve_inside(project_root, config.manualAudit.path, label="manual audit")
    manual_payload = read_regular_file_bytes(manual_path)
    if sha256_bytes(manual_payload) != config.manualAudit.sha256:
        raise ValueError("manual audit differs from its pin")
    transaction = _read_json_object(quality.root / "provenance/transaction.json")
    if set(transaction) != _QUALITY_TRANSACTION_KEYS:
        raise ValueError("source quality-audit transaction has an unexpected shape")
    expected_pipeline = config.sourcePipeline.model_dump(mode="json")
    expected_manual = config.manualAudit.model_dump(mode="json")
    if (
        transaction.get("schemaVersion") != 1
        or transaction.get("runId") != quality.root.name
        or transaction.get("pipelineRun") != expected_pipeline
        or transaction.get("manualAudit") != expected_manual
        or not isinstance(transaction.get("implementationSha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", cast(str, transaction["implementationSha256"])) is None
        or pipeline.root.name != Path(config.sourcePipeline.path).name
        or read_regular_file_bytes(quality.root / "inputs/manual-audit.json") != manual_payload
    ):
        raise ValueError("source quality-audit provenance differs from benchmark pins")
    summary = _read_json_object(quality.root / "summary.json")
    pipeline_summary = _read_json_object(pipeline.root / "summary.json")
    lineage_path = pipeline.root / "lineage/cases.jsonl"
    lineage_payload = read_regular_file_bytes(lineage_path)
    lineage = tuple(
        _PipelineResumeLineageRow.model_validate_json(canonical_json_bytes(row), strict=True)
        for row in _read_json_rows(lineage_path)
    )
    lineage_by_id = {row.documentId: row for row in lineage}
    if (
        summary.get("schemaVersion") != 1
        or summary.get("pipelineCommitSha256") != config.sourcePipeline.commitSha256
        or summary.get("trainingRecordsPublished") is not False
        or summary.get("machineCertifiedManifestIsAcceptanceManifest") is not False
        or summary.get("confirmedManifestIsTrainingPublication") is not False
        or pipeline_summary.get("schemaVersion") != 1
        or pipeline_summary.get("runId") != pipeline.root.name
        or pipeline_summary.get("documents") != len(lineage)
        or pipeline_summary.get("certifiedDocuments")
        != sum(row.status == "certified" for row in lineage)
        or pipeline_summary.get("blockedDocuments")
        != sum(row.status == "blocked" for row in lineage)
        or pipeline_summary.get("caseLineageSha256") != sha256_bytes(lineage_payload)
        or pipeline_summary.get("trainingRecordsPublished") is not False
        or len(lineage_by_id) != len(lineage)
    ):
        raise ValueError("source quality-audit summary does not preserve quarantine semantics")
    rows = (
        *_manifest_rows(quality.root / "confirmed/manifest.json", disposition="confirmed"),
        *_manifest_rows(
            quality.root / "manual-quarantine/manifest.json",
            disposition="manual_quarantine",
        ),
    )
    by_id = {cast(str, row["documentId"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("source quality-audit repeats a manually reviewed case")
    manual_value = json.loads(manual_payload)
    if (
        not isinstance(manual_value, dict)
        or set(manual_value)
        != {"schemaVersion", "pipelineCommitSha256", "randomSampleSize", "cases"}
        or manual_value.get("schemaVersion") != 1
        or manual_value.get("pipelineCommitSha256") != config.sourcePipeline.commitSha256
        or not isinstance(manual_value.get("cases"), list)
    ):
        raise ValueError("manual audit has an unexpected root contract")
    manual_by_id: dict[str, dict[str, Any]] = {}
    for raw in cast(list[Any], manual_value["cases"]):
        if not isinstance(raw, dict) or set(raw) != {
            "documentId",
            "semanticConsistency",
            "templateFidelity",
            "privacyRewritten",
            "selectionBasis",
            "notes",
        }:
            raise ValueError("manual audit case has an unexpected shape")
        document_id = raw.get("documentId")
        if not isinstance(document_id, str) or document_id in manual_by_id:
            raise ValueError("manual audit repeats or malforms a document identity")
        manual_by_id[document_id] = cast(dict[str, Any], raw)
    for specification in config.cases:
        document_id = specification.documentId
        quality_row = by_id.get(document_id)
        lineage_row = lineage_by_id.get(document_id)
        manual_row = manual_by_id.get(document_id)
        if quality_row is None or lineage_row is None or manual_row is None:
            raise ValueError(f"benchmark source lacks reviewed lineage: {document_id}")
        quality_manual = quality_row.get("manualAudit")
        expected_manual = {
            "document_id": document_id,
            "semantic_consistency": manual_row["semanticConsistency"],
            "template_fidelity": manual_row["templateFidelity"],
            "privacy_rewritten": manual_row["privacyRewritten"],
            "selection_basis": ";".join(cast(list[str], manual_row["selectionBasis"])),
            "notes": manual_row["notes"],
            "all_checks_passed": all(
                manual_row[key]
                for key in ("semanticConsistency", "templateFidelity", "privacyRewritten")
            ),
        }
        if (
            quality_manual != expected_manual
            or lineage_row.status != "certified"
            or lineage_row.blockingReason is not None
            or lineage_row.inventoryRun.model_dump(mode="json") != quality_row.get("inventoryRun")
            or lineage_row.finalCertificationRun is None
            or lineage_row.finalCertificationRun.model_dump(mode="json")
            != quality_row.get("finalCertificationRun")
            or lineage_row.sourceTextSha256 != quality_row.get("sourceSha256")
            or lineage_row.currentCandidateSha256 != quality_row.get("candidateSha256")
        ):
            raise ValueError(
                f"quality-audit case differs from v8 lineage/manual audit: {document_id}"
            )
    return quality, by_id, manual_payload


def _line_text(candidate: bytes, line_id: str) -> str:
    lines = candidate.decode("utf-8").splitlines()
    number = int(line_id[1:])
    if not 1 <= number <= len(lines):
        raise ValueError(f"ground-truth evidence line is outside candidate: {line_id}")
    line = lines[number - 1]
    if not line.strip() or re.fullmatch(r"--- PAGE [0-9]+ ---", line):
        raise ValueError(f"ground-truth evidence cites a blank/page-marker line: {line_id}")
    return line


def _validate_inventory_certification_contract_handoff(
    *,
    source: str,
    inventory_contract: dict[str, Any],
    certification_contract: dict[str, Any],
) -> None:
    """Accept only the two contracts reachable through the production pipeline."""

    refined_contract = raw_text_correction_module._refine_carrier_occurrence_contract(
        source=source,
        contract=inventory_contract,
    )
    if certification_contract not in (inventory_contract, refined_contract):
        raise ValueError("benchmark inventory/certification contract handoff differs")


def _materialize_case(
    *,
    project_root: Path,
    specification: BenchmarkCase,
    quality_row: dict[str, Any],
    run_cache: dict[tuple[str, str, str], ResolvedRun],
    compiler_case: RawTextInventoryCompilerCase | None,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    manual = quality_row.get("manualAudit")
    expected_disposition = "confirmed" if specification.expectedClean else "manual_quarantine"
    if (
        quality_row.get("documentId") != specification.documentId
        or quality_row.get("status") != "certified"
        or quality_row.get("auditDisposition") != expected_disposition
        or quality_row.get("blockingReason") is not None
        or quality_row.get("latestSemanticFindings") != 0
        or quality_row.get("hostAuditReplayPassed") is not True
        or not isinstance(manual, dict)
        or set(manual) != _MANUAL_ROW_KEYS
        or manual.get("document_id") != specification.documentId
        or manual.get("all_checks_passed") is not specification.expectedClean
        or sha256_bytes(cast(str, manual.get("notes")).encode())
        != specification.manualAuditNotesSha256
    ):
        raise ValueError(
            f"benchmark truth differs from quality-audit adjudication: {specification.documentId}"
        )
    raw_final = quality_row.get("finalCertificationRun")
    raw_inventory = quality_row.get("inventoryRun")
    if not isinstance(raw_final, dict) or not isinstance(raw_inventory, dict):
        raise ValueError("benchmark source lacks final certification or inventory provenance")
    final_pin = RunPin.model_validate(raw_final, strict=True)
    inventory_pin = RunPin.model_validate(raw_inventory, strict=True)

    def resolve(pin: RunPin, label: str) -> ResolvedRun:
        key = (pin.path, pin.commitSha256, pin.transactionSha256)
        if key not in run_cache:
            run_cache[key] = _validate_run_pin(project_root, pin, label=label)
        return run_cache[key]

    final_run = resolve(final_pin, "legacy final-certification source")
    inventory_run = resolve(inventory_pin, "inventory source")
    final_root = final_run.root / "cases" / specification.documentId
    inventory_root = inventory_run.root / "cases" / specification.documentId
    required_final = (
        "source.txt",
        "input-candidate.txt",
        "final.txt",
        "source-label.json",
        "target-label.json",
        "source-contract.json",
        "result.json",
    )
    for filename in required_final:
        path = final_root / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"legacy certification source lacks {filename}")
    source = read_regular_file_bytes(final_root / "source.txt")
    candidate = read_regular_file_bytes(final_root / "final.txt")
    source_label_payload = read_regular_file_bytes(final_root / "source-label.json")
    target_label_payload = read_regular_file_bytes(final_root / "target-label.json")
    legacy_contract_payload = read_regular_file_bytes(final_root / "source-contract.json")
    source_label = _read_json_object(final_root / "source-label.json")
    target_label = _read_json_object(final_root / "target-label.json")
    legacy_contract = _read_json_object(final_root / "source-contract.json")
    _validate_case_contract_labels(
        document_id=specification.documentId,
        contract=legacy_contract,
        source_label=source_label,
        target_label=target_label,
    )
    for filename, expected in (
        ("source.txt", source),
        ("source-label.json", source_label_payload),
        ("target-label.json", target_label_payload),
    ):
        if read_regular_file_bytes(inventory_root / filename) != expected:
            raise ValueError(
                f"benchmark inventory/certification handoff differs: "
                f"{specification.documentId}/{filename}"
            )
    _validate_inventory_certification_contract_handoff(
        source=source.decode("utf-8"),
        inventory_contract=_read_json_object(inventory_root / "contract.json"),
        certification_contract=legacy_contract,
    )
    result = CertificationCaseResult.model_validate_json(
        read_regular_file_bytes(final_root / "result.json"), strict=True
    )
    observed_host = _host_audit(
        source=source.decode("utf-8"),
        output=candidate.decode("utf-8"),
        contract=legacy_contract,
    )
    if (
        read_regular_file_bytes(final_root / "input-candidate.txt") != candidate
        or sha256_bytes(source) != quality_row.get("sourceSha256")
        or sha256_bytes(candidate) != quality_row.get("candidateSha256")
        or sha256_bytes(canonical_json_bytes(target_label)) != quality_row.get("targetSha256")
        or result.documentId != specification.documentId
        or result.status != "certified"
        or result.semanticFindings != 0
        or not result.hostAudit.passed
        or (compiler_case is None and not observed_host.passed)
        or result.sourceTextSha256 != sha256_bytes(source)
        or result.inputCandidateSha256 != sha256_bytes(candidate)
        or result.finalTextSha256 != sha256_bytes(candidate)
    ):
        raise ValueError(f"benchmark source case fails identity replay: {specification.documentId}")
    source_label_bytes = source_label_payload
    target_label_bytes = target_label_payload
    contract_payload = legacy_contract_payload
    inventory_payload: bytes | None = None
    deterministic_edits_payload: bytes | None = None
    deterministic_text_sha256: str | None = None
    if compiler_case is not None:
        compiled_source_label = json.loads(compiler_case.source_label_json)
        compiled_target_label = json.loads(compiler_case.target_label_json)
        compiled_contract = json.loads(compiler_case.contract_json)
        if (
            compiler_case.document_id != specification.documentId
            or compiler_case.source_text != source
            or compiled_source_label != source_label
            or compiled_target_label != target_label
            or not isinstance(compiled_contract, dict)
        ):
            raise ValueError(
                f"current compiler inputs differ from reviewed lineage: {specification.documentId}"
            )
        _validate_case_contract_labels(
            document_id=specification.documentId,
            contract=compiled_contract,
            source_label=source_label,
            target_label=target_label,
        )
        source_label_bytes = compiler_case.source_label_json
        target_label_bytes = compiler_case.target_label_json
        contract_payload = compiler_case.contract_json
        inventory_payload = compiler_case.inventory_json
        deterministic_edits_payload = compiler_case.deterministic_edits_json
        deterministic_text_sha256 = sha256_bytes(compiler_case.deterministic_text)
    issues = []
    for issue in specification.issues:
        evidence_groups = [
            [
                {"lineId": line_id, "currentLine": _line_text(candidate, line_id)}
                for line_id in group
            ]
            for group in issue.requiredEvidenceGroups
        ]
        issues.append({**issue.model_dump(mode="json"), "evidenceGroups": evidence_groups})
    ground_truth = {
        "schemaVersion": 2 if compiler_case is not None else 1,
        "documentId": specification.documentId,
        "expectedClean": specification.expectedClean,
        "manualAudit": manual,
        "issues": issues,
        "sourceSha256": sha256_bytes(source),
        "candidateSha256": sha256_bytes(candidate),
        "sourceLabelSha256": sha256_bytes(source_label_bytes),
        "targetLabelSha256": sha256_bytes(target_label_bytes),
        "sourceContractSha256": sha256_bytes(contract_payload),
        "sourceQualityManifestRowSha256": sha256_bytes(canonical_json_bytes(quality_row)),
        "inventoryRun": inventory_pin.model_dump(mode="json"),
        "legacyFinalCertificationRun": final_pin.model_dump(mode="json"),
        "trainingRecord": False,
    }
    if compiler_case is not None:
        assert inventory_payload is not None
        assert deterministic_edits_payload is not None
        assert deterministic_text_sha256 is not None
        ground_truth.update(
            {
                "legacySourceContractSha256": sha256_bytes(legacy_contract_payload),
                "inventorySha256": sha256_bytes(inventory_payload),
                "deterministicEditsSha256": sha256_bytes(deterministic_edits_payload),
                "deterministicCompilerTextSha256": deterministic_text_sha256,
            }
        )
    prefix = f"cases/{specification.documentId}"
    artifacts = {
        f"{prefix}/source.txt": source,
        f"{prefix}/final.txt": candidate,
        f"{prefix}/source-label.json": source_label_bytes,
        f"{prefix}/target-label.json": target_label_bytes,
        f"{prefix}/source-contract.json": contract_payload,
        f"{prefix}/ground-truth.json": canonical_json_bytes(ground_truth) + b"\n",
    }
    if inventory_payload is not None and deterministic_edits_payload is not None:
        artifacts[f"{prefix}/inventory.json"] = inventory_payload
        artifacts[f"{prefix}/deterministic-edits.json"] = deterministic_edits_payload
    return artifacts, ground_truth


def build_cohort(*, project_root: Path, config_path: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    try:
        config_path.relative_to(project_root)
    except ValueError as error:
        raise ValueError("benchmark build config must be inside project_root") from error
    config = cast(CohortBuildConfig, _load_config(config_path, CohortBuildConfig))
    quality, quality_by_id, manual_payload = _quality_cases(
        project_root=project_root, config=config
    )
    prompt_path = _resolve_inside(project_root, config.prompt.path, label="benchmark prompt")
    prompt_payload = read_regular_file_bytes(prompt_path)
    if sha256_bytes(prompt_payload) != config.prompt.sha256:
        raise ValueError("benchmark prompt differs from its pre-registered pin")
    if set(row.documentId for row in config.cases) - set(quality_by_id):
        raise ValueError("benchmark config selects a case absent from the manual quality audit")
    compiler_config_payload: bytes | None = None
    compiler_cases_by_id: dict[str, RawTextInventoryCompilerCase] = {}
    if config.schemaVersion == 2:
        compiler_pin = config.compilerInventoryConfig
        if compiler_pin is None or config.invariantInputs is None:
            raise RuntimeError("validated benchmark v2 config lost its compiler inputs")
        compiler_path = _resolve_inside(
            project_root,
            compiler_pin.path,
            label="benchmark inventory compiler config",
        )
        compiler_config_payload = read_regular_file_bytes(compiler_path)
        if sha256_bytes(compiler_config_payload) != compiler_pin.sha256:
            raise ValueError("benchmark inventory compiler config differs from its pin")
        compiler_config = load_synthesis_raw_text_inventory_batch_config(compiler_path)
        compiled_cases = compile_raw_text_inventory_cases(
            project_root=project_root,
            config=compiler_config,
            document_ids=tuple(row.documentId for row in config.cases),
        )
        compiler_cases_by_id = {row.document_id: row for row in compiled_cases}
        if len(compiler_cases_by_id) != len(config.cases):
            raise RuntimeError("inventory compiler returned incomplete benchmark coverage")
    transaction = {
        "schemaVersion": config.schemaVersion,
        "runId": config.run.runId,
        "configSha256": sha256_file(config_path),
        "resolvedConfigSha256": sha256_bytes(canonical_json_bytes(config.model_dump(mode="json"))),
        "sourceQualityAudit": config.sourceQualityAudit.model_dump(mode="json"),
        "sourcePipeline": config.sourcePipeline.model_dump(mode="json"),
        "manualAudit": config.manualAudit.model_dump(mode="json"),
        "prompt": config.prompt.model_dump(mode="json"),
        "gates": config.gates.model_dump(mode="json"),
        "documentIds": [row.documentId for row in config.cases],
        "dependencyContract": _cohort_dependency_contract(config.schemaVersion),
        "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
    }
    if config.schemaVersion == 2:
        assert config.compilerInventoryConfig is not None
        assert config.invariantInputs is not None
        transaction.update(
            {
                "compilerInventoryConfig": config.compilerInventoryConfig.model_dump(mode="json"),
                "invariantInputs": config.invariantInputs.model_dump(mode="json"),
            }
        )
    transaction_sha256 = sha256_bytes(canonical_json_bytes(transaction))
    output_parent = _resolve_inside(
        project_root, config.run.outputDir, label="benchmark output directory"
    )
    staged = StagedArtifactRun(
        output_parent=output_parent,
        run_name=config.run.runId,
        transaction_sha256=transaction_sha256,
    )
    artifacts: dict[str, bytes] = {
        "config.json": read_regular_file_bytes(config_path),
        "inputs/manual-audit.json": manual_payload,
        "inputs/auditor-prompt.md": prompt_payload,
        "inputs/source-quality-audit-transaction.json": read_regular_file_bytes(
            quality.root / "provenance/transaction.json"
        ),
        "provenance/transaction.json": canonical_json_bytes(transaction) + b"\n",
    }
    if compiler_config_payload is not None:
        artifacts["inputs/inventory-compiler-config.yaml"] = compiler_config_payload
    ground_truth_rows: list[dict[str, Any]] = []
    run_cache: dict[tuple[str, str, str], ResolvedRun] = {}
    for specification in config.cases:
        case_artifacts, ground_truth = _materialize_case(
            project_root=project_root,
            specification=specification,
            quality_row=quality_by_id[specification.documentId],
            run_cache=run_cache,
            compiler_case=compiler_cases_by_id.get(specification.documentId),
        )
        artifacts.update(case_artifacts)
        ground_truth_rows.append(ground_truth)
    ground_truth_payload = (
        canonical_json_bytes(
            {
                "schemaVersion": config.schemaVersion,
                "documents": len(ground_truth_rows),
                "cleanControls": sum(row["expectedClean"] is True for row in ground_truth_rows),
                "knownDefects": sum(row["expectedClean"] is False for row in ground_truth_rows),
                "cases": ground_truth_rows,
                "trainingRecordsPublished": False,
            }
        )
        + b"\n"
    )
    summary = {
        "schemaVersion": config.schemaVersion,
        "runId": config.run.runId,
        "status": "complete",
        "documents": len(config.cases),
        "knownDefects": sum(not row.expectedClean for row in config.cases),
        "cleanControls": sum(row.expectedClean for row in config.cases),
        "expectedIssues": sum(len(row.issues) for row in config.cases),
        "sourceComponentRuns": len(run_cache),
        "sourceQualityAuditCommitSha256": config.sourceQualityAudit.commitSha256,
        "sourcePipelineCommitSha256": config.sourcePipeline.commitSha256,
        "groundTruthSha256": sha256_bytes(ground_truth_payload),
        "promptSha256": config.prompt.sha256,
        "gatePolicySha256": sha256_bytes(
            canonical_json_bytes(config.gates.model_dump(mode="json"))
        ),
        "trainingRecordsPublished": False,
    }
    if config.schemaVersion == 2:
        assert config.compilerInventoryConfig is not None
        summary.update(
            {
                "compilerInventoryConfigSha256": config.compilerInventoryConfig.sha256,
                "deterministicAuditContractVersion": 3,
            }
        )
    artifacts["ground-truth.json"] = ground_truth_payload
    artifacts["summary.json"] = canonical_json_bytes(summary) + b"\n"
    for relative, payload in sorted(artifacts.items()):
        staged.publish_bytes(relative, payload)
    committed = staged.commit(
        expected_artifacts=artifacts,
        metadata={
            "kind": (
                "raw_text_certification_v3_benchmark_cohort"
                if config.schemaVersion == 2
                else "raw_text_certification_benchmark_cohort"
            ),
            "documents": len(config.cases),
            "known_defects": sum(not row.expectedClean for row in config.cases),
            "clean_controls": sum(row.expectedClean for row in config.cases),
            "training_records_published": False,
        },
    )
    commit_sha256 = sha256_file(staged.final_root / "_COMMIT.json")
    replayed = _load_cohort(
        project_root,
        RunPin(
            path=staged.final_root.relative_to(project_root).as_posix(),
            commitSha256=commit_sha256,
            transactionSha256=transaction_sha256,
        ),
    )
    if replayed.config != config or len(replayed.cases) != len(config.cases):
        raise ValueError("committed benchmark cohort differs after full replay")
    return {
        "runId": config.run.runId,
        "created": committed.created,
        "output": staged.final_root.relative_to(project_root).as_posix(),
        "commitSha256": commit_sha256,
        "transactionSha256": transaction_sha256,
        "summary": summary,
    }


def _load_cohort(project_root: Path, pin: RunPin) -> LoadedBenchmarkCohort:
    reference = _validate_run_pin(project_root, pin, label="benchmark cohort")
    config = cast(
        CohortBuildConfig,
        _load_config(reference.root / "config.json", CohortBuildConfig),
    )
    compiler_cases_by_id: dict[str, RawTextInventoryCompilerCase] = {}
    if config.schemaVersion == 2:
        compiler_pin = config.compilerInventoryConfig
        if compiler_pin is None or config.invariantInputs is None:
            raise RuntimeError("validated benchmark v2 config lost its compiler inputs")
        compiler_source = _resolve_inside(
            project_root,
            compiler_pin.path,
            label="benchmark inventory compiler config",
        )
        compiler_copy = reference.root / "inputs/inventory-compiler-config.yaml"
        if (
            compiler_copy.is_symlink()
            or not compiler_copy.is_file()
            or read_regular_file_bytes(compiler_copy) != read_regular_file_bytes(compiler_source)
            or sha256_file(compiler_copy) != compiler_pin.sha256
        ):
            raise ValueError("benchmark inventory compiler config copy differs from its pin")
        compiler_config = load_synthesis_raw_text_inventory_batch_config(compiler_copy)
        compiler_cases = compile_raw_text_inventory_cases(
            project_root=project_root,
            config=compiler_config,
            document_ids=tuple(row.documentId for row in config.cases),
        )
        compiler_cases_by_id = {row.document_id: row for row in compiler_cases}
    transaction = _read_json_object(reference.root / "provenance/transaction.json")
    expected_transaction = {
        "schemaVersion": config.schemaVersion,
        "runId": config.run.runId,
        "configSha256": sha256_file(reference.root / "config.json"),
        "resolvedConfigSha256": sha256_bytes(canonical_json_bytes(config.model_dump(mode="json"))),
        "sourceQualityAudit": config.sourceQualityAudit.model_dump(mode="json"),
        "sourcePipeline": config.sourcePipeline.model_dump(mode="json"),
        "manualAudit": config.manualAudit.model_dump(mode="json"),
        "prompt": config.prompt.model_dump(mode="json"),
        "gates": config.gates.model_dump(mode="json"),
        "documentIds": [row.documentId for row in config.cases],
        "dependencyContract": _cohort_dependency_contract(config.schemaVersion),
        "implementationSha256": transaction.get("implementationSha256"),
    }
    if config.schemaVersion == 2:
        assert config.compilerInventoryConfig is not None
        assert config.invariantInputs is not None
        expected_transaction.update(
            {
                "compilerInventoryConfig": config.compilerInventoryConfig.model_dump(mode="json"),
                "invariantInputs": config.invariantInputs.model_dump(mode="json"),
            }
        )
    implementation_sha256 = transaction.get("implementationSha256")
    prompt_copy_path = reference.root / "inputs/auditor-prompt.md"
    prompt_source_path = _resolve_inside(project_root, config.prompt.path, label="benchmark prompt")
    if (
        set(transaction) != set(expected_transaction)
        or not isinstance(implementation_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", implementation_sha256) is None
        or transaction != expected_transaction
        or config.run.runId != reference.root.name
        or reference.root.parent.relative_to(project_root).as_posix() != config.run.outputDir
        or prompt_copy_path.is_symlink()
        or not prompt_copy_path.is_file()
        or read_regular_file_bytes(prompt_copy_path) != read_regular_file_bytes(prompt_source_path)
        or sha256_file(prompt_copy_path) != config.prompt.sha256
    ):
        raise ValueError("benchmark cohort transaction fails deterministic replay")
    cases: list[LoadedBenchmarkCase] = []
    ground_truth_rows: list[dict[str, Any]] = []
    for specification in config.cases:
        case_root = reference.root / "cases" / specification.documentId
        required = (
            "source.txt",
            "final.txt",
            "source-label.json",
            "target-label.json",
            "source-contract.json",
            "ground-truth.json",
            *(("inventory.json", "deterministic-edits.json") if config.schemaVersion == 2 else ()),
        )
        if any(
            (case_root / name).is_symlink() or not (case_root / name).is_file() for name in required
        ):
            raise ValueError(f"benchmark cohort case is incomplete: {specification.documentId}")
        source = read_regular_file_bytes(case_root / "source.txt")
        candidate = read_regular_file_bytes(case_root / "final.txt")
        source_label = _read_json_object(case_root / "source-label.json")
        target_label = _read_json_object(case_root / "target-label.json")
        contract = _read_json_object(case_root / "source-contract.json")
        ground_truth = _read_json_object(case_root / "ground-truth.json")
        inventory: tuple[InventoryCandidate, ...] | None = None
        deterministic_edits: tuple[DeterministicInventoryEdit, ...] | None = None
        if config.schemaVersion == 2:
            inventory_value = json.loads(read_regular_file_bytes(case_root / "inventory.json"))
            deterministic_edits_value = json.loads(
                read_regular_file_bytes(case_root / "deterministic-edits.json")
            )
            if not isinstance(inventory_value, list) or not isinstance(
                deterministic_edits_value, list
            ):
                raise ValueError("benchmark compiler artifacts are not JSON arrays")
            inventory = tuple(
                InventoryCandidate.model_validate_json(canonical_json_bytes(row), strict=True)
                for row in inventory_value
            )
            deterministic_edits = tuple(
                DeterministicInventoryEdit.model_validate_json(
                    canonical_json_bytes(row), strict=True
                )
                for row in deterministic_edits_value
            )
        _validate_case_contract_labels(
            document_id=specification.documentId,
            contract=contract,
            source_label=source_label,
            target_label=target_label,
        )
        manual = ground_truth.get("manualAudit")
        inventory_run = ground_truth.get("inventoryRun")
        legacy_final_run = ground_truth.get("legacyFinalCertificationRun")
        ground_truth_keys = {
            "schemaVersion",
            "documentId",
            "expectedClean",
            "manualAudit",
            "issues",
            "sourceSha256",
            "candidateSha256",
            "sourceLabelSha256",
            "targetLabelSha256",
            "sourceContractSha256",
            "sourceQualityManifestRowSha256",
            "inventoryRun",
            "legacyFinalCertificationRun",
            "trainingRecord",
        }
        if config.schemaVersion == 2:
            ground_truth_keys.update(
                {
                    "legacySourceContractSha256",
                    "inventorySha256",
                    "deterministicEditsSha256",
                    "deterministicCompilerTextSha256",
                }
            )
        if (
            set(ground_truth) != ground_truth_keys
            or ground_truth.get("schemaVersion") != config.schemaVersion
            or ground_truth.get("documentId") != specification.documentId
            or ground_truth.get("expectedClean") is not specification.expectedClean
            or ground_truth.get("sourceSha256") != sha256_bytes(source)
            or ground_truth.get("candidateSha256") != sha256_bytes(candidate)
            or ground_truth.get("sourceLabelSha256") != sha256_file(case_root / "source-label.json")
            or ground_truth.get("targetLabelSha256") != sha256_file(case_root / "target-label.json")
            or ground_truth.get("sourceContractSha256")
            != sha256_file(case_root / "source-contract.json")
            or ground_truth.get("trainingRecord") is not False
            or not isinstance(manual, dict)
            or set(manual) != _MANUAL_ROW_KEYS
            or manual.get("document_id") != specification.documentId
            or manual.get("all_checks_passed") is not specification.expectedClean
            or not isinstance(manual.get("notes"), str)
            or sha256_bytes(cast(str, manual["notes"]).encode())
            != specification.manualAuditNotesSha256
            or not isinstance(ground_truth.get("sourceQualityManifestRowSha256"), str)
            or re.fullmatch(
                r"[0-9a-f]{64}",
                cast(str, ground_truth["sourceQualityManifestRowSha256"]),
            )
            is None
            or not isinstance(inventory_run, dict)
            or not isinstance(legacy_final_run, dict)
            or (
                config.schemaVersion == 1
                and not _host_audit(
                    source=source.decode("utf-8"),
                    output=candidate.decode("utf-8"),
                    contract=contract,
                ).passed
            )
        ):
            raise ValueError(f"benchmark cohort case fails replay: {specification.documentId}")
        if config.schemaVersion == 2:
            compiler_case = compiler_cases_by_id.get(specification.documentId)
            legacy_contract_sha256 = ground_truth.get("legacySourceContractSha256")
            if (
                compiler_case is None
                or not isinstance(legacy_contract_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", legacy_contract_sha256) is None
                or compiler_case.source_text != source
                or compiler_case.source_label_json
                != read_regular_file_bytes(case_root / "source-label.json")
                or compiler_case.target_label_json
                != read_regular_file_bytes(case_root / "target-label.json")
                or compiler_case.contract_json
                != read_regular_file_bytes(case_root / "source-contract.json")
                or compiler_case.inventory_json
                != read_regular_file_bytes(case_root / "inventory.json")
                or compiler_case.deterministic_edits_json
                != read_regular_file_bytes(case_root / "deterministic-edits.json")
                or ground_truth.get("inventorySha256") != sha256_file(case_root / "inventory.json")
                or ground_truth.get("deterministicEditsSha256")
                != sha256_file(case_root / "deterministic-edits.json")
                or ground_truth.get("deterministicCompilerTextSha256")
                != sha256_bytes(compiler_case.deterministic_text)
            ):
                raise ValueError(
                    f"benchmark compiler case fails exact replay: {specification.documentId}"
                )
        RunPin.model_validate(inventory_run, strict=True)
        RunPin.model_validate(legacy_final_run, strict=True)
        expected_issues = tuple(row.model_dump(mode="json") for row in specification.issues)
        observed_issues = ground_truth.get("issues")
        if not isinstance(observed_issues, list) or len(observed_issues) != len(expected_issues):
            raise ValueError("benchmark cohort issue inventory differs")
        for expected, observed in zip(expected_issues, observed_issues, strict=True):
            if not isinstance(observed, dict):
                raise ValueError("benchmark cohort issue is not an object")
            evidence = observed.get("evidenceGroups")
            if {
                key: value for key, value in observed.items() if key != "evidenceGroups"
            } != expected:
                raise ValueError("benchmark cohort issue metadata differs")
            if not isinstance(evidence, list) or evidence != [
                [
                    {"lineId": line_id, "currentLine": _line_text(candidate, line_id)}
                    for line_id in group
                ]
                for group in expected["requiredEvidenceGroups"]
            ]:
                raise ValueError("benchmark cohort issue evidence differs")
        ground_truth_rows.append(ground_truth)
        cases.append(
            LoadedBenchmarkCase(
                specification=specification,
                source=source,
                candidate=candidate,
                source_label=source_label,
                target_label=target_label,
                contract=contract,
                inventory=inventory,
                deterministic_edits=deterministic_edits,
                ground_truth=ground_truth,
            )
        )
    aggregate_path = reference.root / "ground-truth.json"
    aggregate = _read_json_object(aggregate_path)
    if (
        set(aggregate)
        != {
            "schemaVersion",
            "documents",
            "cleanControls",
            "knownDefects",
            "cases",
            "trainingRecordsPublished",
        }
        or aggregate.get("schemaVersion") != config.schemaVersion
        or aggregate.get("documents") != len(cases)
        or aggregate.get("cleanControls") != sum(row.specification.expectedClean for row in cases)
        or aggregate.get("knownDefects")
        != sum(not row.specification.expectedClean for row in cases)
        or aggregate.get("cases") != ground_truth_rows
        or aggregate.get("trainingRecordsPublished") is not False
    ):
        raise ValueError("benchmark cohort aggregate ground truth differs")
    component_runs = {
        (
            cast(dict[str, Any], row[run_key])["path"],
            cast(dict[str, Any], row[run_key])["commitSha256"],
            cast(dict[str, Any], row[run_key])["transactionSha256"],
        )
        for row in ground_truth_rows
        for run_key in ("inventoryRun", "legacyFinalCertificationRun")
    }
    summary = _read_json_object(reference.root / "summary.json")
    expected_summary = {
        "schemaVersion": config.schemaVersion,
        "runId": config.run.runId,
        "status": "complete",
        "documents": len(cases),
        "knownDefects": sum(not row.specification.expectedClean for row in cases),
        "cleanControls": sum(row.specification.expectedClean for row in cases),
        "expectedIssues": sum(len(row.specification.issues) for row in cases),
        "sourceComponentRuns": len(component_runs),
        "sourceQualityAuditCommitSha256": config.sourceQualityAudit.commitSha256,
        "sourcePipelineCommitSha256": config.sourcePipeline.commitSha256,
        "groundTruthSha256": sha256_file(aggregate_path),
        "promptSha256": config.prompt.sha256,
        "gatePolicySha256": sha256_bytes(
            canonical_json_bytes(config.gates.model_dump(mode="json"))
        ),
        "trainingRecordsPublished": False,
    }
    if config.schemaVersion == 2:
        assert config.compilerInventoryConfig is not None
        expected_summary.update(
            {
                "compilerInventoryConfigSha256": config.compilerInventoryConfig.sha256,
                "deterministicAuditContractVersion": 3,
            }
        )
    if summary != expected_summary:
        raise ValueError("benchmark cohort summary fails deterministic replay")
    return LoadedBenchmarkCohort(reference=reference, config=config, cases=tuple(cases))


def _match_deterministic_issues(
    issues: tuple[ExpectedIssue, ...],
    findings: tuple[Any, ...],
) -> dict[int, int]:
    """Return a maximum one-to-one match using exact evidence and audited dimensions."""

    finding_owner: dict[int, int] = {}

    def matches(issue: ExpectedIssue, finding: Any) -> bool:
        cited_ids = {row.lineId for row in finding.evidence}
        return finding.dimension in issue.acceptableDimensions and all(
            cited_ids.intersection(group) for group in issue.requiredEvidenceGroups
        )

    def assign(issue_index: int, visited: set[int]) -> bool:
        for finding_index, finding in enumerate(findings):
            if finding_index in visited or not matches(issues[issue_index], finding):
                continue
            visited.add(finding_index)
            previous = finding_owner.get(finding_index)
            if previous is None or assign(previous, visited):
                finding_owner[finding_index] = issue_index
                return True
        return False

    for issue_index in range(len(issues)):
        assign(issue_index, set())
    return {issue_index: finding_index for finding_index, issue_index in finding_owner.items()}


def _deterministic_benchmark_report(summary: Mapping[str, Any]) -> bytes:
    rows = [
        "# Contract-v3 deterministic certification benchmark",
        "",
        f"- Gate: **{'PASS' if summary['promotionGatePassed'] else 'BLOCKED'}**",
        f"- Documents: **{summary['documents']}** "
        f"({summary['knownDefects']} known defects, {summary['cleanControls']} controls)",
        f"- Correct case dispositions: **{summary['correct']}/{summary['documents']}**",
        f"- Defect cases caught with exact gold evidence: "
        f"**{summary['defectsCaught']}/{summary['knownDefects']}**",
        f"- Clean controls accepted: **{summary['controlsClean']}/{summary['cleanControls']}**",
        f"- Expected issue recall: **{summary['issueRecall']:.4f}**",
        f"- Structured invariant finding precision: **{summary['findingPrecision']:.4f}**",
        f"- Classic host findings (unstructured, reported separately): "
        f"**{summary['classicHostFindings']}**",
        "- Provider requests/cost: **0 / $0**",
        "",
        "This provider-free benchmark replays the exact current inventory compiler, classic host",
        "contract, and contract-v3 invariant audit. It is an evaluation artifact and does not",
        "modify candidates or publish training records.",
        "",
    ]
    return "\n".join(rows).encode("utf-8")


def _deterministic_finding_summary(
    case_rows: Sequence[Mapping[str, Any]], *, issues_matched: int
) -> dict[str, int | float]:
    """Keep exact-evidence precision separate from the legacy string-only host guard.

    Contract-v3 findings carry audited dimensions and physical evidence, so they can participate
    in one-to-one gold matching.  The classic host audit remains a required rejection guard but
    emits only strings.  Counting those strings in the precision denominator while making them
    ineligible for the numerator would systematically penalize duplicate true positives.
    """

    invariant_findings = sum(int(row["invariantFindings"]) for row in case_rows)
    classic_findings = sum(int(row["classicHostFindings"]) for row in case_rows)
    return {
        "findings": invariant_findings,
        "classicHostFindings": classic_findings,
        "allAuditFindings": invariant_findings + classic_findings,
        "findingPrecision": (issues_matched / invariant_findings if invariant_findings else 1.0),
    }


def run_deterministic_benchmark(*, project_root: Path, config_path: Path) -> dict[str, Any]:
    """Evaluate a contract-v3 cohort without loading credentials or invoking a model."""

    project_root = project_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    try:
        config_path.relative_to(project_root)
    except ValueError as error:
        raise ValueError("deterministic benchmark config must be inside project_root") from error
    config = cast(
        BenchmarkDeterministicAuditConfig,
        _load_config(config_path, BenchmarkDeterministicAuditConfig),
    )
    cohort = _load_cohort(project_root, config.cohortRun)
    if cohort.config.schemaVersion != 2 or cohort.config.invariantInputs is None:
        raise ValueError("deterministic benchmark requires a contract-v3 cohort")
    context = load_certification_invariant_context(
        project_root=project_root,
        config=cohort.config.invariantInputs,
    )
    case_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    case_artifacts: dict[str, bytes] = {}
    matched_dimensions: set[str] = set()
    for source_case in cohort.cases:
        document_id = source_case.specification.documentId
        if source_case.inventory is None or source_case.deterministic_edits is None:
            raise RuntimeError("contract-v3 cohort lost compiler-owned invariant inputs")
        source = source_case.source.decode("utf-8")
        candidate = source_case.candidate.decode("utf-8")
        host_audit = _host_audit(source=source, output=candidate, contract=source_case.contract)
        invariant_case = evaluate_certification_invariant_case(
            document_id=document_id,
            source=source,
            candidate=candidate,
            contract=source_case.contract,
            inventory=source_case.inventory,
            deterministic_edits=source_case.deterministic_edits,
            context=context,
        )
        findings = invariant_case.audit.findings
        issue_matches = _match_deterministic_issues(source_case.specification.issues, findings)
        for _issue_index, matched_finding_index in issue_matches.items():
            matched_dimensions.add(findings[matched_finding_index].dimension)
        predicted_clean = host_audit.passed and invariant_case.audit.passed
        gold_issue_caught = source_case.specification.expectedClean or bool(issue_matches)
        correct = (
            predicted_clean
            if source_case.specification.expectedClean
            else (not predicted_clean and gold_issue_caught)
        )
        observed_findings = len(findings)
        all_audit_findings = len(host_audit.findings) + observed_findings
        case_rows.append(
            {
                "documentId": document_id,
                "expectedClean": source_case.specification.expectedClean,
                "predictedClean": predicted_clean,
                "correct": correct,
                "goldIssueCaught": gold_issue_caught,
                "classicHostAuditPassed": host_audit.passed,
                "classicHostFindings": len(host_audit.findings),
                "invariantAuditPassed": invariant_case.audit.passed,
                "invariantFindings": len(findings),
                "findings": observed_findings,
                "allAuditFindings": all_audit_findings,
                "issuesMatched": len(issue_matches),
                "candidateSha256": sha256_bytes(source_case.candidate),
            }
        )
        for issue_index, issue in enumerate(source_case.specification.issues):
            finding_index = issue_matches.get(issue_index)
            issue_rows.append(
                {
                    "documentId": document_id,
                    "issueId": issue.issueId,
                    "acceptableDimensions": list(issue.acceptableDimensions),
                    "requiredEvidenceGroups": [
                        list(group) for group in issue.requiredEvidenceGroups
                    ],
                    "matched": finding_index is not None,
                    "findingIndex": finding_index,
                    "matchedFindingDimension": (
                        findings[finding_index].dimension if finding_index is not None else None
                    ),
                    "matchedInvariantId": (
                        findings[finding_index].invariantId if finding_index is not None else None
                    ),
                }
            )
        prefix = f"cases/{document_id}"
        case_artifacts[f"{prefix}/host-audit.json"] = json_artifact_bytes(
            host_audit.model_dump(mode="json")
        )
        case_artifacts[f"{prefix}/invariant-envelope.json"] = json_artifact_bytes(
            invariant_case.envelope.model_dump(mode="json")
        )
        case_artifacts[f"{prefix}/invariant-audit.json"] = json_artifact_bytes(
            invariant_case.audit.model_dump(mode="json")
        )

    defects = [row for row in case_rows if not row["expectedClean"]]
    controls = [row for row in case_rows if row["expectedClean"]]
    issues_matched = sum(bool(row["matched"]) for row in issue_rows)
    issue_recall = issues_matched / len(issue_rows) if issue_rows else 1.0
    finding_summary = _deterministic_finding_summary(case_rows, issues_matched=issues_matched)
    finding_precision = cast(float, finding_summary["findingPrecision"])
    expected_dimensions = {
        issue.primaryDimension
        for source_case in cohort.cases
        for issue in source_case.specification.issues
    }
    gates = config.gates
    defects_caught = sum(
        not bool(row["predictedClean"]) and bool(row["goldIssueCaught"]) for row in defects
    )
    controls_clean = sum(bool(row["predictedClean"]) for row in controls)
    promotion_gate_passed = (
        (not gates.requireEveryDefectCaseCaught or defects_caught == len(defects))
        and (not gates.requireEveryControlClean or controls_clean == len(controls))
        and (
            not gates.requireEveryExpectedDimensionObserved
            or expected_dimensions <= matched_dimensions
        )
        and Decimal(str(issue_recall)) >= gates.minimumIssueRecall
        and Decimal(str(finding_precision)) >= gates.minimumFindingPrecision
    )
    summary = {
        "schemaVersion": 1,
        "runId": config.run.runId,
        "status": "complete",
        "promotionGatePassed": promotion_gate_passed,
        "documents": len(case_rows),
        "knownDefects": len(defects),
        "cleanControls": len(controls),
        "correct": sum(bool(row["correct"]) for row in case_rows),
        "defectsCaught": defects_caught,
        "controlsClean": controls_clean,
        "expectedIssues": len(issue_rows),
        "issuesMatched": issues_matched,
        "issueRecall": issue_recall,
        **finding_summary,
        "expectedDimensions": sorted(expected_dimensions),
        "matchedDimensions": sorted(matched_dimensions),
        "providerRequests": 0,
        "providerCostUsd": "0",
        "candidateBytesModified": 0,
        "trainingRecordsPublished": False,
    }
    transaction = {
        "schemaVersion": 1,
        "runId": config.run.runId,
        "configSha256": sha256_file(config_path),
        "resolvedConfigSha256": sha256_bytes(canonical_json_bytes(config.model_dump(mode="json"))),
        "cohortRun": config.cohortRun.model_dump(mode="json"),
        "gates": config.gates.model_dump(mode="json"),
        "certificationImplementation": certification_implementation_contract(),
        "inventoryCompilerImplementation": raw_text_inventory_compiler_contract(),
        "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
        "providerAccess": False,
    }
    transaction_sha256 = sha256_bytes(canonical_json_bytes(transaction))
    artifacts = {
        "config.json": read_regular_file_bytes(config_path),
        "inputs/cohort-commit.json": read_regular_file_bytes(
            cohort.reference.root / "_COMMIT.json"
        ),
        "provenance/transaction.json": canonical_json_bytes(transaction) + b"\n",
        "results/cases.jsonl": b"".join(canonical_json_bytes(row) + b"\n" for row in case_rows),
        "results/issues.jsonl": b"".join(canonical_json_bytes(row) + b"\n" for row in issue_rows),
        "summary.json": canonical_json_bytes(summary) + b"\n",
        "REPORT.md": _deterministic_benchmark_report(summary),
        **case_artifacts,
    }
    staged = StagedArtifactRun(
        output_parent=_resolve_inside(
            project_root,
            config.run.outputDir,
            label="deterministic benchmark output directory",
        ),
        run_name=config.run.runId,
        transaction_sha256=transaction_sha256,
    )
    for relative, payload in sorted(artifacts.items()):
        staged.publish_bytes(relative, payload)
    committed = staged.commit(
        expected_artifacts=artifacts,
        metadata={
            "kind": "raw_text_certification_v3_deterministic_benchmark",
            "documents": len(case_rows),
            "promotion_gate_passed": promotion_gate_passed,
            "provider_requests": 0,
            "candidate_bytes_modified": 0,
            "training_records_published": False,
        },
    )
    return {
        "runId": config.run.runId,
        "created": committed.created,
        "output": staged.final_root.relative_to(project_root).as_posix(),
        "commitSha256": sha256_file(staged.final_root / "_COMMIT.json"),
        "transactionSha256": transaction_sha256,
        "summary": summary,
    }


def _load_certification_config_pin(
    project_root: Path,
    pin: FilePin,
    *,
    label: str,
) -> tuple[SynthesisRawTextCertificationConfig, bytes]:
    path = _resolve_inside(project_root, pin.path, label=label)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not a regular file")
    payload = read_regular_file_bytes(path)
    if sha256_bytes(payload) != pin.sha256:
        raise ValueError(f"{label} differs from its pin")
    return load_synthesis_raw_text_certification_config(path), payload


def _validate_planned_configs(
    *,
    cohort: LoadedBenchmarkCohort,
    baseline_config: SynthesisRawTextCertificationConfig,
    challenger_config: SynthesisRawTextCertificationConfig,
    baseline_effort: BenchmarkEffort,
    challenger_effort: BenchmarkEffort,
) -> None:
    if baseline_config.benchmark_plan is not None or challenger_config.benchmark_plan is not None:
        raise ValueError("pre-registered arm templates must not self-reference their plan")
    _validate_evaluation_config(
        config=baseline_config,
        cohort=cohort,
        effort=baseline_effort,
    )
    _validate_evaluation_config(
        config=challenger_config,
        cohort=cohort,
        effort=challenger_effort,
    )
    if baseline_config.run.run_id == challenger_config.run.run_id:
        raise ValueError("benchmark arms reuse one run identity")
    if _normalized_arm_config(baseline_config) != _normalized_arm_config(challenger_config):
        raise ValueError("planned arms differ beyond run identity and reasoning effort")


def _validate_authorized_config(
    *,
    config: SynthesisRawTextCertificationConfig,
    template: SynthesisRawTextCertificationConfig,
    plan: LoadedBenchmarkPlan,
    effort: BenchmarkEffort,
) -> None:
    _validate_evaluation_config(config=config, cohort=plan.cohort, effort=effort)
    authority = config.benchmark_plan
    if (
        authority is None
        or authority.path != plan.reference.pin.path
        or authority.commit_sha256 != plan.reference.pin.commitSha256
        or authority.transaction_sha256 != plan.reference.pin.transactionSha256
        or config.model_copy(update={"benchmark_plan": None}) != template
    ):
        raise ValueError(f"{effort} configuration is not the plan-authorized derivative")


def _plan_authorization(
    *,
    config: BenchmarkPlanConfig,
    baseline_config: SynthesisRawTextCertificationConfig,
    challenger_config: SynthesisRawTextCertificationConfig,
) -> dict[str, Any]:
    """Return the committed exact-template membership contract enforced by certification."""

    arms = []
    for role, specification, template in (
        ("baseline", config.baseline, baseline_config),
        ("challenger", config.challenger, challenger_config),
    ):
        arms.append(
            {
                "role": role,
                "effort": specification.effort,
                "templateConfigPath": specification.config.path,
                "templateConfigSha256": specification.config.sha256,
                "templateResolvedConfigSha256": sha256_bytes(
                    canonical_json_bytes(template.model_dump(mode="json"))
                ),
                "runId": template.run.run_id,
            }
        )
    return {
        "schemaVersion": 1,
        "task": "raw_text_certification_benchmark_plan_authorization_v1",
        "arms": arms,
    }


def _load_plan(project_root: Path, pin: RunPin) -> LoadedBenchmarkPlan:
    reference = _validate_run_pin(project_root, pin, label="benchmark experiment plan")
    config = cast(
        BenchmarkPlanConfig,
        _load_config(reference.root / "config.json", BenchmarkPlanConfig),
    )
    transaction = _read_json_object(reference.root / "provenance/transaction.json")
    expected_transaction = {
        "schemaVersion": 2,
        "runId": config.run.runId,
        "configSha256": sha256_file(reference.root / "config.json"),
        "resolvedConfigSha256": sha256_bytes(canonical_json_bytes(config.model_dump(mode="json"))),
        "cohortRun": config.cohortRun.model_dump(mode="json"),
        "baseline": config.baseline.model_dump(mode="json"),
        "challenger": config.challenger.model_dump(mode="json"),
        "certificationImplementation": certification_implementation_contract(),
        "certificationRuntime": _certification_runtime_contract(),
        "implementationSha256": transaction.get("implementationSha256"),
    }
    implementation_sha256 = transaction.get("implementationSha256")
    if (
        set(transaction) != set(expected_transaction)
        or transaction != expected_transaction
        or not isinstance(implementation_sha256, str)
        or implementation_sha256 != sha256_file(_IMPLEMENTATION_PATH)
        or config.run.runId != reference.root.name
        or reference.root.parent.relative_to(project_root).as_posix() != config.run.outputDir
    ):
        raise ValueError("benchmark plan transaction fails deterministic replay")
    cohort = _load_cohort(project_root, config.cohortRun)
    baseline_config, baseline_payload = _load_certification_config_pin(
        project_root, config.baseline.config, label="planned baseline configuration"
    )
    challenger_config, challenger_payload = _load_certification_config_pin(
        project_root, config.challenger.config, label="planned challenger configuration"
    )
    if (
        read_regular_file_bytes(reference.root / "inputs/baseline-config.yaml") != baseline_payload
        or read_regular_file_bytes(reference.root / "inputs/challenger-config.yaml")
        != challenger_payload
    ):
        raise ValueError("benchmark plan configuration copies differ from their pins")
    _validate_planned_configs(
        cohort=cohort,
        baseline_config=baseline_config,
        challenger_config=challenger_config,
        baseline_effort=config.baseline.effort,
        challenger_effort=config.challenger.effort,
    )
    expected_authorization = _plan_authorization(
        config=config,
        baseline_config=baseline_config,
        challenger_config=challenger_config,
    )
    if (
        _read_json_object(reference.root / "authorization/arm-configs.json")
        != expected_authorization
    ):
        raise ValueError("benchmark plan authorization receipt fails deterministic replay")
    expected_summary = {
        "schemaVersion": 2,
        "runId": config.run.runId,
        "status": "planned",
        "documents": len(cohort.cases),
        "knownDefects": sum(not row.specification.expectedClean for row in cohort.cases),
        "cleanControls": sum(row.specification.expectedClean for row in cohort.cases),
        "expectedIssues": sum(len(row.specification.issues) for row in cohort.cases),
        "cohortCommitSha256": config.cohortRun.commitSha256,
        "promptSha256": cohort.config.prompt.sha256,
        "gatePolicySha256": sha256_bytes(
            canonical_json_bytes(cohort.config.gates.model_dump(mode="json"))
        ),
        "baselineEffort": config.baseline.effort,
        "baselineRunId": baseline_config.run.run_id,
        "challengerEffort": config.challenger.effort,
        "challengerRunId": challenger_config.run.run_id,
        "providerCallsMade": 0,
        "candidateBytesModified": 0,
        "trainingRecordsPublished": False,
    }
    if _read_json_object(reference.root / "summary.json") != expected_summary:
        raise ValueError("benchmark plan summary fails deterministic replay")
    return LoadedBenchmarkPlan(
        reference=reference,
        config=config,
        cohort=cohort,
        baseline_config=baseline_config,
        challenger_config=challenger_config,
        baseline_config_payload=baseline_payload,
        challenger_config_payload=challenger_payload,
    )


def build_benchmark_plan(*, project_root: Path, config_path: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    try:
        config_path.relative_to(project_root)
    except ValueError as error:
        raise ValueError("benchmark plan config must be inside project_root") from error
    config = cast(BenchmarkPlanConfig, _load_config(config_path, BenchmarkPlanConfig))
    cohort = _load_cohort(project_root, config.cohortRun)
    baseline_config, baseline_payload = _load_certification_config_pin(
        project_root, config.baseline.config, label="planned baseline configuration"
    )
    challenger_config, challenger_payload = _load_certification_config_pin(
        project_root, config.challenger.config, label="planned challenger configuration"
    )
    _validate_planned_configs(
        cohort=cohort,
        baseline_config=baseline_config,
        challenger_config=challenger_config,
        baseline_effort=config.baseline.effort,
        challenger_effort=config.challenger.effort,
    )
    authorization = _plan_authorization(
        config=config,
        baseline_config=baseline_config,
        challenger_config=challenger_config,
    )
    transaction = {
        "schemaVersion": 2,
        "runId": config.run.runId,
        "configSha256": sha256_file(config_path),
        "resolvedConfigSha256": sha256_bytes(canonical_json_bytes(config.model_dump(mode="json"))),
        "cohortRun": config.cohortRun.model_dump(mode="json"),
        "baseline": config.baseline.model_dump(mode="json"),
        "challenger": config.challenger.model_dump(mode="json"),
        "certificationImplementation": certification_implementation_contract(),
        "certificationRuntime": _certification_runtime_contract(),
        "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
    }
    transaction_sha256 = sha256_bytes(canonical_json_bytes(transaction))
    summary = {
        "schemaVersion": 2,
        "runId": config.run.runId,
        "status": "planned",
        "documents": len(cohort.cases),
        "knownDefects": sum(not row.specification.expectedClean for row in cohort.cases),
        "cleanControls": sum(row.specification.expectedClean for row in cohort.cases),
        "expectedIssues": sum(len(row.specification.issues) for row in cohort.cases),
        "cohortCommitSha256": config.cohortRun.commitSha256,
        "promptSha256": cohort.config.prompt.sha256,
        "gatePolicySha256": sha256_bytes(
            canonical_json_bytes(cohort.config.gates.model_dump(mode="json"))
        ),
        "baselineEffort": config.baseline.effort,
        "baselineRunId": baseline_config.run.run_id,
        "challengerEffort": config.challenger.effort,
        "challengerRunId": challenger_config.run.run_id,
        "providerCallsMade": 0,
        "candidateBytesModified": 0,
        "trainingRecordsPublished": False,
    }
    artifacts = {
        "config.json": read_regular_file_bytes(config_path),
        "authorization/arm-configs.json": canonical_json_bytes(authorization) + b"\n",
        "inputs/baseline-config.yaml": baseline_payload,
        "inputs/challenger-config.yaml": challenger_payload,
        "provenance/transaction.json": canonical_json_bytes(transaction) + b"\n",
        "summary.json": canonical_json_bytes(summary) + b"\n",
    }
    staged = StagedArtifactRun(
        output_parent=_resolve_inside(
            project_root, config.run.outputDir, label="benchmark plan output directory"
        ),
        run_name=config.run.runId,
        transaction_sha256=transaction_sha256,
    )
    for relative, payload in sorted(artifacts.items()):
        staged.publish_bytes(relative, payload)
    committed = staged.commit(
        expected_artifacts=artifacts,
        metadata={
            "kind": "raw_text_certification_benchmark_plan",
            "documents": len(cohort.cases),
            "provider_calls_made": 0,
            "candidate_bytes_modified": 0,
            "training_records_published": False,
        },
    )
    commit_sha256 = sha256_file(staged.final_root / "_COMMIT.json")
    replayed = _load_plan(
        project_root,
        RunPin(
            path=staged.final_root.relative_to(project_root).as_posix(),
            commitSha256=commit_sha256,
            transactionSha256=transaction_sha256,
        ),
    )
    if replayed.config != config:
        raise ValueError("committed benchmark plan differs after full replay")
    return {
        "runId": config.run.runId,
        "created": committed.created,
        "output": staged.final_root.relative_to(project_root).as_posix(),
        "commitSha256": commit_sha256,
        "transactionSha256": transaction_sha256,
        "summary": summary,
    }


def _normalized_arm_config(config: SynthesisRawTextCertificationConfig) -> dict[str, Any]:
    value = config.model_dump(mode="json")
    cast(dict[str, Any], value["run"])["run_id"] = "<arm>"
    cast(dict[str, Any], value["provider"])["reasoning_effort"] = "<effort>"
    return value


def _validate_evaluation_config(
    *,
    config: SynthesisRawTextCertificationConfig,
    cohort: LoadedBenchmarkCohort,
    effort: BenchmarkEffort,
) -> None:
    provider = config.provider
    workflow = config.workflow
    input_pin = config.input_run
    generation = provider.generation_settings
    if provider.kind != "openrouter":
        raise ValueError(f"{effort} configuration is not an OpenRouter evaluation")
    max_price = provider.max_price
    if (
        "audit_contract_version" not in config.model_fields_set
        or config.audit_contract_version != 2
        or config.environment_file != ".env"
        or config.input_case_contract_filename != "source-contract.json"
        or config.prompt.path != cohort.config.prompt.path
        or config.prompt.sha256 != cohort.config.prompt.sha256
        or input_pin.path != cohort.reference.pin.path
        or input_pin.commit_sha256 != cohort.reference.pin.commitSha256
        or input_pin.transaction_sha256 != cohort.reference.pin.transactionSha256
        or tuple(config.case_ids) != tuple(row.specification.documentId for row in cohort.cases)
        or provider.model != "z-ai/glm-5.3-flash"
        or provider.reasoning_effort != effort
        or provider.request_timeout_seconds != 180
        or provider.allow_fallbacks
        or provider.provider_only is None
        or len(provider.provider_only) != 1
        or provider.provider_only[0] not in _ISOLATED_BENCHMARK_ROUTES
        or provider.provider_order is not None
        or provider.provider_sort is not None
        or max_price is None
        or max_price.prompt > 0.15
        or max_price.completion > 0.5
        or generation is None
        or generation.temperature != 0
        or generation.top_p is not None
        or generation.text_verbosity is not None
        or workflow.documents != len(cohort.cases)
        or workflow.max_concurrent_documents != 1
        or workflow.semantic_audit_passes != 1
        or not workflow.evaluation_only
        or not workflow.require_complete_dimension_coverage
        or workflow.require_unanimous_clean
        or workflow.max_provider_route_rounds != 4
        or workflow.retry_initial_delay_seconds != 5
        or workflow.retry_delay_multiplier != 2
        or workflow.retry_max_delay_seconds != 20
        or workflow.retry_jitter_seconds != 0
    ):
        raise ValueError(f"{effort} configuration differs from the isolated A/B contract")


def _case_has_one_success_after_zero_cost_retryable_prefix(
    case: ValidatedCertificationCase,
) -> bool:
    stages = case.stages
    if not 1 <= len(stages) <= 4:
        return False
    accepted = stages[-1]
    if (
        accepted.errorType is not None
        or accepted.hostError is not None
        or accepted.modelOutput is None
        or accepted.usage.requests != 1
    ):
        return False
    for stage in stages[:-1]:
        usage = stage.usage
        if (
            stage.errorType is None
            or not stage.retryableRouteError
            or stage.routeError is None
            or stage.hostError is not None
            or stage.modelOutput is not None
            or usage.requests != 0
            or usage.inputTokens != 0
            or usage.outputTokens != 0
            or usage.reasoningTokens != 0
            or usage.visibleOutputTokens != 0
            or usage.cacheReadTokens != 0
            or usage.cacheWriteTokens != 0
            or usage.estimatedCostUsd != 0
            or usage.providerReportedCostUsd not in {None, Decimal(0)}
            or usage.providerResponseIds
            or usage.downstreamProviders
            or usage.finishReasons
        ):
            return False
    return case.result.usage.requests == 1


def _validate_arm(
    *,
    project_root: Path,
    pin: RunPin,
    plan: LoadedBenchmarkPlan,
    role: Literal["baseline", "challenger"],
) -> ValidatedCertificationRun:
    cohort = plan.cohort
    specification = plan.config.baseline if role == "baseline" else plan.config.challenger
    expected_config = plan.baseline_config if role == "baseline" else plan.challenger_config
    effort = specification.effort
    reference = _validate_run_pin(project_root, pin, label=f"{role} benchmark arm")
    run = load_validated_certification_run(reference.root)
    _validate_authorized_config(
        config=run.config,
        template=expected_config,
        plan=plan,
        effort=effort,
    )
    authority = run.config.benchmark_plan
    if authority is None:
        raise ValueError(f"{role} benchmark arm lacks its plan authority")
    implementation = certification_implementation_contract()
    runtime = _certification_runtime_contract()
    observed_runtime = run.transaction.get("runtime")
    if (
        run.root.name != expected_config.run.run_id
        or run.root.parent.relative_to(project_root).as_posix() != expected_config.run.output_dir
        or run.transaction.get("benchmarkPlan") != authority.model_dump(mode="json")
        or run.transaction.get("implementationSha256") != implementation["implementationSha256"]
        or run.transaction.get("dependencyImplementationSha256")
        != implementation["dependencyImplementationSha256"]
        or not isinstance(observed_runtime, dict)
        or observed_runtime.get("pydanticAiVersion") != runtime["pydanticAiVersion"]
        or observed_runtime.get("openaiVersion") != runtime["openaiVersion"]
        or any(
            not _case_has_one_success_after_zero_cost_retryable_prefix(case) for case in run.cases
        )
    ):
        raise ValueError(f"{role} benchmark arm differs from its pre-registered plan")
    for source, observed in zip(cohort.cases, run.cases, strict=True):
        source_case_root = cohort.reference.root / "cases" / source.specification.documentId
        observed_case_root = run.root / "cases" / observed.document_id
        if (
            observed.document_id != source.specification.documentId
            or observed.source != source.source
            or observed.candidate != source.candidate
            or observed.source_label != source.source_label
            or observed.target_label != source.target_label
            or observed.contract != source.contract
            or any(
                read_regular_file_bytes(observed_case_root / observed_name)
                != read_regular_file_bytes(source_case_root / source_name)
                for observed_name, source_name in (
                    ("source.txt", "source.txt"),
                    ("final.txt", "final.txt"),
                    ("source-label.json", "source-label.json"),
                    ("target-label.json", "target-label.json"),
                    ("source-contract.json", "source-contract.json"),
                )
            )
        ):
            raise ValueError(f"{role} benchmark arm input bytes differ from the cohort")
    return run


def _committed_arm_pin(
    *,
    project_root: Path,
    config: SynthesisRawTextCertificationConfig,
) -> RunPin:
    root = _resolve_inside(
        project_root,
        f"{config.run.output_dir}/{config.run.run_id}",
        label="benchmark arm output",
    )
    transaction = _read_json_object(root / "provenance/transaction.json")
    return RunPin(
        path=root.relative_to(project_root).as_posix(),
        commitSha256=sha256_file(root / "_COMMIT.json"),
        transactionSha256=sha256_bytes(canonical_json_bytes(transaction)),
    )


def launch_benchmark(*, project_root: Path, config_path: Path) -> dict[str, Any]:
    """Run both pre-registered arms only after validating every authority byte."""

    project_root = project_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    try:
        config_path.relative_to(project_root)
    except ValueError as error:
        raise ValueError("benchmark launch config must be inside project_root") from error
    config = cast(BenchmarkLaunchConfig, _load_config(config_path, BenchmarkLaunchConfig))
    plan = _load_plan(project_root, config.planRun)
    baseline_config, _baseline_payload = _load_certification_config_pin(
        project_root, config.baselineConfig, label="authorized baseline configuration"
    )
    challenger_config, _challenger_payload = _load_certification_config_pin(
        project_root, config.challengerConfig, label="authorized challenger configuration"
    )
    _validate_authorized_config(
        config=baseline_config,
        template=plan.baseline_config,
        plan=plan,
        effort=plan.config.baseline.effort,
    )
    _validate_authorized_config(
        config=challenger_config,
        template=plan.challenger_config,
        plan=plan,
        effort=plan.config.challenger.effort,
    )
    if _normalized_arm_config(baseline_config) != _normalized_arm_config(challenger_config):
        raise ValueError("authorized benchmark arms differ beyond run and effort")

    baseline_config_path = _resolve_inside(
        project_root, config.baselineConfig.path, label="authorized baseline configuration"
    )
    challenger_config_path = _resolve_inside(
        project_root, config.challengerConfig.path, label="authorized challenger configuration"
    )
    baseline_summary = run_raw_text_certification(
        project_root=project_root,
        config_path=baseline_config_path,
        config=baseline_config,
    )
    baseline_pin = _committed_arm_pin(project_root=project_root, config=baseline_config)
    _validate_arm(
        project_root=project_root,
        pin=baseline_pin,
        plan=plan,
        role="baseline",
    )
    challenger_summary = run_raw_text_certification(
        project_root=project_root,
        config_path=challenger_config_path,
        config=challenger_config,
    )
    challenger_pin = _committed_arm_pin(project_root=project_root, config=challenger_config)
    _validate_arm(
        project_root=project_root,
        pin=challenger_pin,
        plan=plan,
        role="challenger",
    )
    return {
        "task": config.task,
        "planRun": config.planRun.model_dump(mode="json"),
        "baselineEffort": plan.config.baseline.effort,
        "baselineRun": baseline_pin.model_dump(mode="json"),
        "baselineSummary": baseline_summary,
        "challengerEffort": plan.config.challenger.effort,
        "challengerRun": challenger_pin.model_dump(mode="json"),
        "challengerSummary": challenger_summary,
        "candidateBytesModified": 0,
        "trainingRecordsPublished": False,
    }


def _predicted_dimensions(case: ValidatedCertificationCase) -> set[AuditDimension]:
    return {_FINDING_DIMENSIONS[finding.findingKind] for finding in case.replay.findings}


def _finding_matches_issue(
    issue: ExpectedIssue,
    case: ValidatedCertificationCase,
    finding_index: int,
) -> bool:
    finding = case.replay.findings[finding_index]
    accepted_dimensions = set(issue.acceptableDimensions)
    cited_ids = {row.lineId for row in finding.evidence}
    return _FINDING_DIMENSIONS[finding.findingKind] in accepted_dimensions and all(
        cited_ids.intersection(group) for group in issue.requiredEvidenceGroups
    )


def _match_expected_issues(
    issues: tuple[ExpectedIssue, ...],
    case: ValidatedCertificationCase,
) -> dict[int, int]:
    """Return a deterministic maximum one-to-one issue-to-finding assignment."""

    finding_owner: dict[int, int] = {}

    def assign(issue_index: int, visited: set[int]) -> bool:
        for finding_index in range(len(case.replay.findings)):
            if finding_index in visited or not _finding_matches_issue(
                issues[issue_index], case, finding_index
            ):
                continue
            visited.add(finding_index)
            prior_issue = finding_owner.get(finding_index)
            if prior_issue is None or assign(prior_issue, visited):
                finding_owner[finding_index] = issue_index
                return True
        return False

    for issue_index in range(len(issues)):
        assign(issue_index, set())
    return {issue_index: finding_index for finding_index, issue_index in finding_owner.items()}


def _case_cost(case: ValidatedCertificationCase) -> Decimal:
    value = case.result.usage.providerReportedCostUsd
    if value is None:
        raise ValueError("benchmark arm lacks provider-reported cost")
    return value


def _arm_rows(
    cohort: LoadedBenchmarkCohort,
    run: ValidatedCertificationRun,
    *,
    role: Literal["baseline", "challenger"],
    effort: BenchmarkEffort,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = run.case_by_id()
    case_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    for source in cohort.cases:
        specification = source.specification
        case = by_id[specification.documentId]
        predicted_clean = case.result.status == "evaluated_clean"
        if case.result.status not in {"evaluated_clean", "needs_review"}:
            raise ValueError(f"{role} arm has a non-classifying outcome")
        predicted_dimensions = sorted(_predicted_dimensions(case))
        issue_matches = _match_expected_issues(specification.issues, case)
        gold_issue_caught = specification.expectedClean or bool(issue_matches)
        case_rows.append(
            {
                "role": role,
                "effort": effort,
                "documentId": specification.documentId,
                "expectedClean": specification.expectedClean,
                "predictedClean": predicted_clean,
                "correct": (
                    predicted_clean
                    if specification.expectedClean
                    else (not predicted_clean and gold_issue_caught)
                ),
                "goldIssueCaught": gold_issue_caught,
                "status": case.result.status,
                "findings": len(case.replay.findings),
                "predictedDimensions": predicted_dimensions,
                "inputTokens": case.result.usage.inputTokens,
                "reasoningTokens": case.result.usage.reasoningTokens,
                "visibleOutputTokens": case.result.usage.visibleOutputTokens,
                "providerReportedCostUsd": str(_case_cost(case)),
            }
        )
        for issue_index, issue in enumerate(specification.issues):
            matched_finding_index = issue_matches.get(issue_index)
            matched_dimension = (
                _FINDING_DIMENSIONS[case.replay.findings[matched_finding_index].findingKind]
                if matched_finding_index is not None
                else None
            )
            issue_rows.append(
                {
                    "role": role,
                    "effort": effort,
                    "documentId": specification.documentId,
                    "issueId": issue.issueId,
                    "acceptableDimensions": list(issue.acceptableDimensions),
                    "requiredEvidenceGroups": [
                        list(group) for group in issue.requiredEvidenceGroups
                    ],
                    "matched": issue_index in issue_matches,
                    "findingIndex": matched_finding_index,
                    "matchedFindingDimension": matched_dimension,
                }
            )
    return case_rows, issue_rows


def _counterfactual_cascade(
    *,
    baseline_cases: list[dict[str, Any]],
    challenger_cases: list[dict[str, Any]],
    baseline_issues: list[dict[str, Any]],
    challenger_issues: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select the challenger only after a baseline-clean result and quantify the cascade."""

    def by_document(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
        indexed = {cast(str, row["documentId"]): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError(f"counterfactual {label} arm repeats a document")
        return indexed

    def by_issue(
        rows: list[dict[str, Any]], *, label: str
    ) -> dict[tuple[str, str], dict[str, Any]]:
        indexed = {(cast(str, row["documentId"]), cast(str, row["issueId"])): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError(f"counterfactual {label} arm repeats an issue")
        return indexed

    baseline_by_id = by_document(baseline_cases, label="baseline")
    challenger_by_id = by_document(challenger_cases, label="challenger")
    baseline_issue_by_key = by_issue(baseline_issues, label="baseline")
    challenger_issue_by_key = by_issue(challenger_issues, label="challenger")
    if set(baseline_by_id) != set(challenger_by_id) or set(baseline_issue_by_key) != set(
        challenger_issue_by_key
    ):
        raise ValueError("counterfactual arm scope differs")

    case_rows: list[dict[str, Any]] = []
    cost = Decimal(0)
    findings = 0
    for baseline_case in baseline_cases:
        document_id = cast(str, baseline_case["documentId"])
        challenger_case = challenger_by_id[document_id]
        if challenger_case["expectedClean"] is not baseline_case["expectedClean"]:
            raise ValueError("counterfactual arm ground truth differs")
        baseline_clean = bool(baseline_case["predictedClean"])
        selected_role: Literal["baseline", "challenger"] = (
            "challenger" if baseline_clean else "baseline"
        )
        selected = challenger_case if baseline_clean else baseline_case
        predicted_clean = baseline_clean and bool(challenger_case["predictedClean"])
        expected_clean = bool(baseline_case["expectedClean"])
        gold_issue_caught = bool(selected["goldIssueCaught"])
        correct = predicted_clean if expected_clean else (not predicted_clean and gold_issue_caught)
        cost += Decimal(cast(str, baseline_case["providerReportedCostUsd"]))
        if baseline_clean:
            cost += Decimal(cast(str, challenger_case["providerReportedCostUsd"]))
        findings += int(selected["findings"])
        case_rows.append(
            {
                "documentId": document_id,
                "expectedClean": expected_clean,
                "predictedClean": predicted_clean,
                "correct": correct,
                "goldIssueCaught": gold_issue_caught,
                "selectedRole": selected_role,
                "challengerWouldRun": baseline_clean,
            }
        )

    issue_rows: list[dict[str, Any]] = []
    for key, baseline_issue in baseline_issue_by_key.items():
        challenger_issue = challenger_issue_by_key[key]
        document_id, _issue_id = key
        if (
            baseline_issue["acceptableDimensions"] != challenger_issue["acceptableDimensions"]
            or baseline_issue["requiredEvidenceGroups"]
            != challenger_issue["requiredEvidenceGroups"]
        ):
            raise ValueError("counterfactual issue ground truth differs")
        selected_role = (
            "challenger" if bool(baseline_by_id[document_id]["predictedClean"]) else "baseline"
        )
        selected = challenger_issue if selected_role == "challenger" else baseline_issue
        issue_rows.append(
            {
                **selected,
                "role": "cascade",
                "selectedRole": selected_role,
                "challengerOnlyMatchOnBaselineRejectedCase": (
                    selected_role == "baseline"
                    and not bool(baseline_issue["matched"])
                    and bool(challenger_issue["matched"])
                ),
            }
        )
    issues_matched = sum(bool(row["matched"]) for row in issue_rows)
    issue_recall = issues_matched / len(issue_rows) if issue_rows else 1.0
    finding_precision = issues_matched / findings if findings else (1.0 if not issue_rows else 0.0)
    return (
        {
            "correct": sum(bool(row["correct"]) for row in case_rows),
            "challengerCalls": sum(bool(row["challengerWouldRun"]) for row in case_rows),
            "issuesMatched": issues_matched,
            "issues": len(issue_rows),
            "issueRecall": issue_recall,
            "findings": findings,
            "findingPrecision": finding_precision,
            "providerReportedCostUsd": str(cost),
            "challengerOnlyMatchesOnBaselineRejectedCases": sum(
                bool(row["challengerOnlyMatchOnBaselineRejectedCase"]) for row in issue_rows
            ),
            "cases": case_rows,
        },
        issue_rows,
    )


def _decimal_ratio(numerator: Decimal, denominator: Decimal, *, label: str) -> Decimal:
    if denominator <= 0:
        raise ValueError(f"benchmark {label} denominator is not positive")
    return numerator / denominator


def _report(summary: dict[str, Any], case_rows: list[dict[str, Any]]) -> bytes:
    baseline_effort = summary["baselineEffort"].upper()
    challenger_effort = summary["challengerEffort"].upper()
    rows = [
        "# Contract-v2 certification benchmark",
        "",
        f"- Gate: **{'PASS' if summary['promotionGatePassed'] else 'BLOCKED'}**",
        f"- Documents: **{summary['documents']}** "
        f"({summary['knownDefects']} known defects, {summary['cleanControls']} controls)",
        f"- Baseline ({baseline_effort}) accuracy: "
        f"**{summary['baseline']['correct']}/{summary['documents']}**",
        f"- Challenger ({challenger_effort}) accuracy: "
        f"**{summary['challenger']['correct']}/{summary['documents']}**",
        f"- Counterfactual baseline→challenger cascade accuracy: "
        f"**{summary['counterfactualCascade']['correct']}/{summary['documents']}**",
        f"- Challenger issue-evidence recall: **{summary['challenger']['issueRecall']:.3f}**",
        f"- Challenger finding precision: **{summary['challenger']['findingPrecision']:.3f}**",
        f"- Cascade issue-evidence recall: "
        f"**{summary['counterfactualCascade']['issueRecall']:.3f}**",
        f"- Challenger-only issue matches hidden by baseline-reject short-circuiting: "
        f"**{summary['counterfactualCascade']['challengerOnlyMatchesOnBaselineRejectedCases']}**",
        f"- Combined measured arm cost: **${summary['combinedArmCostUsd']}**",
        f"- Counterfactual cascade cost: "
        f"**${summary['counterfactualCascade']['providerReportedCostUsd']}**",
        f"- Challenger/baseline median reasoning-token ratio: "
        f"**{summary['medianReasoningRatio']}**",
        f"- Challenger/baseline median cost ratio: **{summary['medianCostRatio']}**",
        "",
        "| Role | Effort | Document | Expected | Predicted | Findings | Reasoning | Cost USD |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in case_rows:
        rows.append(
            f"| {row['role']} | {row['effort']} | `{row['documentId']}` | "
            f"{'clean' if row['expectedClean'] else 'defect'} | "
            f"{'clean' if row['predictedClean'] else 'defect'} | {row['findings']} | "
            f"{row['reasoningTokens']} | {row['providerReportedCostUsd']} |"
        )
    rows.extend(
        (
            "",
            "This is an evaluation artifact, not a training publication. The cascade row is a",
            "counterfactual over two independently completed arms; no candidate bytes were "
            "changed.",
            "",
        )
    )
    return "\n".join(rows).encode()


def score_benchmark(*, project_root: Path, config_path: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    try:
        config_path.relative_to(project_root)
    except ValueError as error:
        raise ValueError("benchmark score config must be inside project_root") from error
    config = cast(BenchmarkScoreConfig, _load_config(config_path, BenchmarkScoreConfig))
    plan = _load_plan(project_root, config.planRun)
    cohort = plan.cohort
    prompt_path = _resolve_inside(
        project_root, cohort.config.prompt.path, label="benchmark auditor prompt"
    )
    if sha256_file(prompt_path) != cohort.config.prompt.sha256:
        raise ValueError("benchmark auditor prompt hash differs")
    baseline = _validate_arm(
        project_root=project_root,
        pin=config.baselineRun,
        plan=plan,
        role="baseline",
    )
    challenger = _validate_arm(
        project_root=project_root,
        pin=config.challengerRun,
        plan=plan,
        role="challenger",
    )
    validate_certification_run_stage_identities(
        {f"baseline:{case.document_id}": case.stages for case in baseline.cases}
        | {f"challenger:{case.document_id}": case.stages for case in challenger.cases}
    )
    if _normalized_arm_config(baseline.config) != _normalized_arm_config(challenger.config):
        raise ValueError("benchmark arms differ beyond run identity and reasoning effort")
    baseline_cases, baseline_issues = _arm_rows(
        cohort,
        baseline,
        role="baseline",
        effort=plan.config.baseline.effort,
    )
    challenger_cases, challenger_issues = _arm_rows(
        cohort,
        challenger,
        role="challenger",
        effort=plan.config.challenger.effort,
    )
    all_case_rows = [*baseline_cases, *challenger_cases]
    all_issue_rows = [*baseline_issues, *challenger_issues]

    def arm_summary(
        case_rows: list[dict[str, Any]], issue_rows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        defects = [row for row in case_rows if not row["expectedClean"]]
        controls = [row for row in case_rows if row["expectedClean"]]
        matched_issues = sum(bool(row["matched"]) for row in issue_rows)
        findings = sum(int(row["findings"]) for row in case_rows)
        costs = [Decimal(cast(str, row["providerReportedCostUsd"])) for row in case_rows]
        return {
            "correct": sum(bool(row["correct"]) for row in case_rows),
            "defectsCaught": sum(
                not bool(row["predictedClean"]) and bool(row["goldIssueCaught"]) for row in defects
            ),
            "defects": len(defects),
            "controlsClean": sum(bool(row["predictedClean"]) for row in controls),
            "controls": len(controls),
            "issuesMatched": matched_issues,
            "issues": len(issue_rows),
            "issueRecall": matched_issues / len(issue_rows),
            "findings": findings,
            "findingPrecision": matched_issues / findings if findings else 0.0,
            "providerReportedCostUsd": str(sum(costs, Decimal(0))),
            "medianProviderReportedCostUsd": str(median(costs)),
            "medianReasoningTokens": str(
                median([int(row["reasoningTokens"]) for row in case_rows])
            ),
            "inputTokens": sum(int(row["inputTokens"]) for row in case_rows),
            "reasoningTokens": sum(int(row["reasoningTokens"]) for row in case_rows),
            "visibleOutputTokens": sum(int(row["visibleOutputTokens"]) for row in case_rows),
        }

    baseline_summary = arm_summary(baseline_cases, baseline_issues)
    challenger_summary = arm_summary(challenger_cases, challenger_issues)
    cascade_summary, cascade_issue_rows = _counterfactual_cascade(
        baseline_cases=baseline_cases,
        challenger_cases=challenger_cases,
        baseline_issues=baseline_issues,
        challenger_issues=challenger_issues,
    )
    expected_dimensions = {
        issue.primaryDimension for source in cohort.cases for issue in source.specification.issues
    }
    observed_challenger_dimensions = {
        cast(AuditDimension, row["matchedFindingDimension"])
        for row in challenger_issues
        if row["matchedFindingDimension"] is not None
    }
    baseline_cost = Decimal(cast(str, baseline_summary["providerReportedCostUsd"]))
    challenger_cost = Decimal(cast(str, challenger_summary["providerReportedCostUsd"]))
    combined_cost = baseline_cost + challenger_cost
    median_reasoning_ratio = _decimal_ratio(
        Decimal(cast(str, challenger_summary["medianReasoningTokens"])),
        Decimal(cast(str, baseline_summary["medianReasoningTokens"])),
        label="median reasoning",
    )
    median_cost_ratio = _decimal_ratio(
        Decimal(cast(str, challenger_summary["medianProviderReportedCostUsd"])),
        Decimal(cast(str, baseline_summary["medianProviderReportedCostUsd"])),
        label="median cost",
    )
    gates = {
        "allChallengerDefectsCaught": challenger_summary["defectsCaught"]
        == challenger_summary["defects"],
        "allChallengerControlsClean": challenger_summary["controlsClean"]
        == challenger_summary["controls"],
        "perfectCounterfactualCascade": cascade_summary["correct"] == len(cohort.cases),
        "everyExpectedDimensionObserved": expected_dimensions <= observed_challenger_dimensions,
        "minimumChallengerIssueRecall": Decimal(str(challenger_summary["issueRecall"]))
        >= cohort.config.gates.minimumChallengerIssueRecall,
        "minimumChallengerFindingPrecision": Decimal(str(challenger_summary["findingPrecision"]))
        >= cohort.config.gates.minimumChallengerFindingPrecision,
        "minimumCascadeIssueRecall": Decimal(str(cascade_summary["issueRecall"]))
        >= cohort.config.gates.minimumCascadeIssueRecall,
        "minimumCascadeFindingPrecision": Decimal(str(cascade_summary["findingPrecision"]))
        >= cohort.config.gates.minimumCascadeFindingPrecision,
        "combinedArmCostWithinLimit": combined_cost
        <= cohort.config.gates.maximumCombinedArmCostUsd,
        "medianReasoningRatioValid": median_reasoning_ratio
        >= cohort.config.gates.minimumMedianReasoningRatio,
        "medianCostRatioWithinLimit": median_cost_ratio
        <= cohort.config.gates.maximumMedianCostRatio,
    }
    promotion_passed = all(gates.values())
    summary = {
        "schemaVersion": 2,
        "runId": config.run.runId,
        "status": "complete",
        "promotionGatePassed": promotion_passed,
        "recommendation": (
            "targeted_production_validation" if promotion_passed else "blocked_do_not_promote"
        ),
        "documents": len(cohort.cases),
        "knownDefects": sum(not row.specification.expectedClean for row in cohort.cases),
        "cleanControls": sum(row.specification.expectedClean for row in cohort.cases),
        "baselineEffort": plan.config.baseline.effort,
        "baseline": baseline_summary,
        "challengerEffort": plan.config.challenger.effort,
        "challenger": challenger_summary,
        "counterfactualCascade": cascade_summary,
        "expectedDimensions": sorted(expected_dimensions),
        "observedChallengerDimensions": sorted(observed_challenger_dimensions),
        "combinedArmCostUsd": str(combined_cost),
        "medianReasoningRatio": str(median_reasoning_ratio),
        "medianCostRatio": str(median_cost_ratio),
        "gates": gates,
        "candidateBytesModified": 0,
        "trainingRecordsPublished": False,
    }
    transaction = {
        "schemaVersion": 2,
        "runId": config.run.runId,
        "configSha256": sha256_file(config_path),
        "resolvedConfigSha256": sha256_bytes(canonical_json_bytes(config.model_dump(mode="json"))),
        "planRun": config.planRun.model_dump(mode="json"),
        "prompt": cohort.config.prompt.model_dump(mode="json"),
        "gates": cohort.config.gates.model_dump(mode="json"),
        "baselineRun": config.baselineRun.model_dump(mode="json"),
        "challengerRun": config.challengerRun.model_dump(mode="json"),
        "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
    }
    transaction_sha256 = sha256_bytes(canonical_json_bytes(transaction))
    output_parent = _resolve_inside(
        project_root, config.run.outputDir, label="benchmark score output directory"
    )
    staged = StagedArtifactRun(
        output_parent=output_parent,
        run_name=config.run.runId,
        transaction_sha256=transaction_sha256,
    )
    artifacts = {
        "config.json": read_regular_file_bytes(config_path),
        "provenance/transaction.json": canonical_json_bytes(transaction) + b"\n",
        "summary.json": canonical_json_bytes(summary) + b"\n",
        "case-results.json": canonical_json_bytes(all_case_rows) + b"\n",
        "issue-results.json": canonical_json_bytes([*all_issue_rows, *cascade_issue_rows]) + b"\n",
        "REPORT.md": _report(summary, all_case_rows),
    }
    for relative, payload in sorted(artifacts.items()):
        staged.publish_bytes(relative, payload)
    committed = staged.commit(
        expected_artifacts=artifacts,
        metadata={
            "kind": "raw_text_certification_benchmark_score",
            "documents": len(cohort.cases),
            "promotion_gate_passed": promotion_passed,
            "candidate_bytes_modified": 0,
            "training_records_published": False,
        },
    )
    return {
        "runId": config.run.runId,
        "created": committed.created,
        "output": staged.final_root.relative_to(project_root).as_posix(),
        "commitSha256": sha256_file(staged.final_root / "_COMMIT.json"),
        "transactionSha256": transaction_sha256,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "deterministic", "plan", "launch", "score"):
        command = commands.add_parser(name)
        command.add_argument("--project-root", type=Path, default=Path.cwd())
        command.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    operations = {
        "build": build_cohort,
        "deterministic": run_deterministic_benchmark,
        "plan": build_benchmark_plan,
        "launch": launch_benchmark,
        "score": score_benchmark,
    }
    operation = operations[arguments.command]
    result = operation(project_root=arguments.project_root, config_path=arguments.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
