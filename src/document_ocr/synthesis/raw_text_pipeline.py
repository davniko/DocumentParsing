"""Production orchestration for rendered, independently certified raw-OCR synthesis.

The component stages remain separately committed and independently resumable.  This orchestrator
builds their dynamic configs from one strict top-level contract, pins every hand-off to the
upstream commit and transaction, shards provider work within the stage limits, and publishes only
when the complete original cohort has passed a fresh read-only certification.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.config import (
    OpenRouterRewriteProviderConfig,
    RawTextCertifiedPublicationSourceConfig,
    SynthesisRawTextCertificationConfig,
    SynthesisRawTextCertifiedCorrectionConfig,
    SynthesisRawTextCertifiedPublicationConfig,
    SynthesisRawTextInventoryBatchConfig,
    SynthesisRawTextPipelineConfig,
    load_synthesis_raw_text_certification_config,
    load_synthesis_raw_text_certified_correction_config,
    load_synthesis_raw_text_inventory_batch_config,
    load_synthesis_raw_text_pipeline_config,
)
from document_ocr.synthesis.raw_text_certification import (
    CertificationCaseResult,
    run_raw_text_certification,
)
from document_ocr.synthesis.raw_text_certification_artifacts import (
    load_validated_certification_run,
)
from document_ocr.synthesis.raw_text_certification_host import (
    complete_deterministic_repair_set,
)
from document_ocr.synthesis.raw_text_certification_invariants import (
    CertificationInvariantAudit,
)
from document_ocr.synthesis.raw_text_certified_correction import (
    CorrectionCaseResult,
    _deterministic_correction_case,
    _refine_carrier_occurrence_contract,
    run_raw_text_certified_correction,
)
from document_ocr.synthesis.raw_text_certified_publication import (
    run_raw_text_certified_publication,
)
from document_ocr.synthesis.raw_text_hybrid_probe import _artifact_inventory
from document_ocr.synthesis.raw_text_inventory_probe import (
    InventoryProbeCaseResult,
    preflight_raw_text_inventory_batch,
    run_raw_text_inventory_batch,
)
from document_ocr.synthesis.raw_text_rewrite_probe import _resolve_pinned_file
from document_ocr.synthesis.run_safety import (
    StagedArtifactRun,
    StagedCommitReceipt,
)
from document_ocr.training.config import resolve_config_path

_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)
_STRICT_MODEL = ConfigDict(extra="forbid", frozen=True, strict=True)
_DOCUMENT_ID = Annotated[str, StringConstraints(pattern=r"^doc_[0-9a-f]{64}$")]
_SHA256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_COHORT_STOP_REASON = (
    "complete-cohort processing stopped after another case exhausted its bounded attempts"
)
_REPEATED_CANDIDATE_REASON = "correction repeated an earlier candidate byte-for-byte"
_CORRECTION_ROUND_LIMIT_REASON = (
    "independent audit still found actionable defects after the configured correction-round limit"
)
_V3_CERTIFICATION_QUARANTINE_REASON = (
    "contract-v3 certification quarantined a semantic, unowned, or non-repairable defect; "
    "it grants no correction authority"
)


class RawTextPipelineBlockedError(RuntimeError):
    """The bounded workflow ended without authority to publish the complete cohort."""


@dataclass(frozen=True, slots=True)
class _RunReference:
    path: str
    root: Path
    commit_sha256: str
    transaction_sha256: str

    def config_value(self) -> dict[str, str]:
        return {
            "path": self.path,
            "commit_sha256": self.commit_sha256,
            "transaction_sha256": self.transaction_sha256,
        }

    def lineage_value(self) -> dict[str, str]:
        return {
            "path": self.path,
            "commitSha256": self.commit_sha256,
            "transactionSha256": self.transaction_sha256,
        }


class _LineageRunReference(BaseModel):
    model_config = _STRICT_MODEL

    path: str
    commitSha256: _SHA256
    transactionSha256: _SHA256


class _InventoryHistoryRow(BaseModel):
    model_config = _STRICT_MODEL

    stage: Literal["inventory"]
    round: int
    status: Literal[
        "training_ready",
        "needs_review",
        "call_failed",
        "compiler_blocked",
    ]
    run: _LineageRunReference
    candidateSha256: _SHA256


class _InventoryResumeLineageRow(BaseModel):
    model_config = _STRICT_MODEL

    documentId: _DOCUMENT_ID
    status: Literal["blocked"]
    inventoryRound: int | None
    inventoryRun: _LineageRunReference | None
    sourceTextSha256: _SHA256 | None
    currentCandidateSha256: _SHA256 | None
    correctionRounds: Literal[0]
    finalCertificationRun: None
    blockingReason: str
    history: tuple[_InventoryHistoryRow, ...]


class _InventoryResumeSubrunRow(BaseModel):
    model_config = _STRICT_MODEL

    stage: Literal["inventory", "inventory_resume_parent"]
    round: int
    shard: Literal[1]
    run: _LineageRunReference
    documentIds: tuple[_DOCUMENT_ID, ...]
    summarySha256: _SHA256


class _CertificationHistoryRow(BaseModel):
    model_config = _STRICT_MODEL

    stage: Literal["certification"]
    attempt: Annotated[int, Field(ge=1)]
    status: Literal["certified", "needs_review", "call_failed"]
    semanticFindings: Annotated[int, Field(ge=0)]
    hostAuditPassed: bool
    run: _LineageRunReference


class _CorrectionHistoryRow(BaseModel):
    model_config = _STRICT_MODEL

    stage: Literal["correction"]
    round: Annotated[int, Field(ge=1)]
    attempt: Annotated[int, Field(ge=1)] | None = None
    status: Literal[
        "unchanged_certified",
        "correction_candidate",
        "needs_review",
        "call_failed",
    ]
    run: _LineageRunReference
    candidateSha256: _SHA256


_PipelineHistoryRow = Annotated[
    _InventoryHistoryRow | _CertificationHistoryRow | _CorrectionHistoryRow,
    Field(discriminator="stage"),
]


class _PipelineResumeLineageRow(BaseModel):
    model_config = _STRICT_MODEL

    documentId: _DOCUMENT_ID
    status: Literal["certified", "blocked"]
    inventoryRound: Annotated[int, Field(ge=1)]
    inventoryRun: _LineageRunReference
    sourceTextSha256: _SHA256
    currentCandidateSha256: _SHA256
    correctionRounds: Annotated[int, Field(ge=0)]
    finalCertificationRun: _LineageRunReference | None
    blockingReason: str | None
    history: tuple[_PipelineHistoryRow, ...]


class _PipelineResumeSubrunRow(BaseModel):
    model_config = _STRICT_MODEL

    stage: Literal[
        "inventory",
        "inventory_resume_parent",
        "pipeline_resume_parent",
        "certification",
        "correction",
        "publication",
    ]
    round: Annotated[int, Field(ge=0)]
    shard: Annotated[int, Field(ge=1)]
    run: _LineageRunReference
    documentIds: tuple[_DOCUMENT_ID, ...]
    summarySha256: _SHA256


_INVENTORY_INPUT_IDENTITY_FILENAMES = (
    "source.txt",
    "source-label.json",
    "target-label.json",
    "inventory.json",
    "deterministic-edits.json",
    "model-slots.json",
    "compound-slots.json",
    "editor-payload.json",
)


@dataclass(frozen=True, slots=True)
class _InventoryInputIdentity:
    artifact_sha256: tuple[tuple[str, str], ...]


@dataclass(slots=True)
class _CaseState:
    document_id: str
    inventory_run: _RunReference
    inventory_round: int
    candidate_run: _RunReference
    source_text_sha256: str
    current_candidate_sha256: str
    contract_filename: Literal["contract.json", "source-contract.json"] = "contract.json"
    certification_attempts: int = 0
    local_certification_attempts: int = 0
    correction_rounds: int = 0
    correction_attempts: int = 0
    local_correction_attempts: int = 0
    pending_correction_run: _RunReference | None = None
    candidate_hashes: set[str] = field(default_factory=set)
    final_certification_run: _RunReference | None = None
    history: list[dict[str, JsonValue]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _InventoryResumeState:
    parent_run: _RunReference
    document_ids: tuple[str, ...]
    states_by_id: dict[str, _CaseState]
    histories_by_id: dict[str, list[dict[str, JsonValue]]]
    remaining: tuple[str, ...]
    prior_rounds: int
    lineage_sha256: str
    input_identities_by_id: dict[str, _InventoryInputIdentity]
    inventory_contract_sha256: str
    inventory_transaction_contract_sha256: str


@dataclass(frozen=True, slots=True)
class _PipelineResumeState:
    parent_run: _RunReference
    document_ids: tuple[str, ...]
    states_by_id: dict[str, _CaseState]
    histories_by_id: dict[str, list[dict[str, JsonValue]]]
    lineage_sha256: str
    certified_sources: dict[_RunReference, list[str]]
    unresolved: tuple[str, ...]
    terminal_failures: dict[str, str]


def _ensure_under_project_root(project_root: Path, path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"{label} must resolve inside project_root: {resolved}") from error
    return resolved


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(read_regular_file_bytes(path))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_bound_transaction(reference: _RunReference) -> dict[str, Any]:
    value = _read_json_object(reference.root / "provenance/transaction.json")
    if sha256_bytes(canonical_json_bytes(value)) != reference.transaction_sha256:
        raise ValueError(f"committed provenance is not bound to its transaction: {reference.root}")
    return value


def _read_model_rows(path: Path, model: type[BaseModel]) -> tuple[BaseModel, ...]:
    payload = read_regular_file_bytes(path)
    if payload and not payload.endswith(b"\n"):
        raise ValueError(f"JSONL artifact lacks a terminal newline: {path}")
    rows: list[BaseModel] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            raise ValueError(f"JSONL artifact contains a blank row: {path}:{line_number}")
        rows.append(model.model_validate_json(line, strict=True))
    return tuple(rows)


def _read_model_array(path: Path, model: type[BaseModel]) -> tuple[BaseModel, ...]:
    value = json.loads(read_regular_file_bytes(path))
    if not isinstance(value, list):
        raise ValueError(f"expected JSON array: {path}")
    return tuple(model.model_validate_json(canonical_json_bytes(row), strict=True) for row in value)


def _inventory_contract_sha256(config: SynthesisRawTextInventoryBatchConfig) -> str:
    """Hash the behavior contract while excluding only run and cohort scheduling."""

    value = config.model_dump(mode="json")
    value.pop("run")
    value.pop("selection")
    value.pop("cases")
    workflow = cast(dict[str, JsonValue], value["workflow"])
    workflow.pop("documents")
    workflow.pop("max_concurrent_documents")
    return sha256_bytes(canonical_json_bytes(value))


def _inventory_transaction_contract_sha256(value: dict[str, Any]) -> str:
    """Hash immutable inventory inputs/runtime while excluding run and scope identity."""

    normalized = dict(value)
    for key in ("configSha256", "runId", "liveDocumentIds"):
        normalized.pop(key, None)
    runtime = normalized.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("inventory transaction lacks a runtime contract")
    normalized_runtime = dict(runtime)
    normalized_runtime.pop("maxConcurrentDocuments", None)
    normalized["runtime"] = normalized_runtime
    return sha256_bytes(canonical_json_bytes(normalized))


def _inventory_input_identity(
    reference: _RunReference,
    document_id: str,
) -> _InventoryInputIdentity:
    return _InventoryInputIdentity(
        artifact_sha256=tuple(
            (
                filename,
                _case_artifact_sha256(reference, document_id, filename),
            )
            for filename in _INVENTORY_INPUT_IDENTITY_FILENAMES
        )
    )


def _require_inventory_scope(
    config: SynthesisRawTextInventoryBatchConfig,
    document_ids: tuple[str, ...],
    *,
    label: str,
) -> None:
    if (
        not document_ids
        or len(document_ids) != len(set(document_ids))
        or config.workflow.documents != len(document_ids)
    ):
        raise ValueError(f"{label} differs from its configured unique scope")
    if (
        config.selection == "explicit_pinned_document_ids"
        and tuple(row.document_id for row in config.cases) != document_ids
    ):
        raise ValueError(f"{label} differs from its explicit pinned selection")


def _chunks(values: tuple[_CaseState, ...], size: int) -> tuple[tuple[_CaseState, ...], ...]:
    return tuple(values[index : index + size] for index in range(0, len(values), size))


def _publish_generated_config(
    staged: StagedArtifactRun,
    *,
    relative_path: str,
    config: BaseModel,
) -> Path:
    payload = yaml.safe_dump(
        config.model_dump(mode="python"),
        allow_unicode=True,
        sort_keys=False,
    ).encode("utf-8")
    staged.publish_bytes(relative_path, payload)
    path = staged.stage_root / relative_path
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"generated child configuration is not a regular file: {path}")
    return path


def _case_artifact_sha256(
    reference: _RunReference,
    document_id: str,
    filename: str,
) -> str:
    path = reference.root / "cases" / document_id / filename
    return sha256_bytes(read_regular_file_bytes(path))


def _committed_run_reference(
    *,
    project_root: Path,
    run_id: str,
    output_dir: str,
) -> _RunReference:
    output_parent = _ensure_under_project_root(
        project_root,
        resolve_config_path(project_root, output_dir),
        label="pipeline child output directory",
    )
    root = output_parent / run_id
    commit_path = root / "_COMMIT.json"
    receipt = StagedCommitReceipt.model_validate_json(
        read_regular_file_bytes(commit_path), strict=True
    )
    if receipt.run_name != run_id:
        raise ValueError(f"child commit run name differs: {root}")
    StagedArtifactRun(
        output_parent=output_parent,
        run_name=run_id,
        transaction_sha256=receipt.transaction_sha256,
    ).validate_committed_run()
    return _RunReference(
        path=root.relative_to(project_root).as_posix(),
        root=root,
        commit_sha256=sha256_file(commit_path),
        transaction_sha256=receipt.transaction_sha256,
    )


def _pinned_run_reference(
    *,
    project_root: Path,
    path: str,
    commit_sha256: str,
    transaction_sha256: str,
    label: str,
) -> _RunReference:
    root = _ensure_under_project_root(
        project_root,
        resolve_config_path(project_root, path),
        label=label,
    )
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} must be a regular committed directory: {root}")
    commit_path = root / "_COMMIT.json"
    if sha256_file(commit_path) != commit_sha256:
        raise ValueError(f"{label} commit SHA-256 differs: {root}")
    receipt = StagedCommitReceipt.model_validate_json(
        read_regular_file_bytes(commit_path), strict=True
    )
    if receipt.run_name != root.name or receipt.transaction_sha256 != transaction_sha256:
        raise ValueError(f"{label} commit identity differs: {root}")
    StagedArtifactRun(
        output_parent=root.parent,
        run_name=root.name,
        transaction_sha256=transaction_sha256,
    ).validate_committed_run()
    return _RunReference(
        path=root.relative_to(project_root).as_posix(),
        root=root,
        commit_sha256=commit_sha256,
        transaction_sha256=transaction_sha256,
    )


def _validate_inventory_child_run(
    *,
    project_root: Path,
    reference: _RunReference,
    document_ids: tuple[str, ...],
    inventory_contract_sha256: str,
    transaction_contract_sha256: str | None,
    stage_hashes: dict[str, str],
) -> tuple[
    tuple[InventoryProbeCaseResult, ...],
    str,
    dict[str, _InventoryInputIdentity],
]:
    """Validate one committed inventory run and return its exact result projection."""

    config_path = reference.root / "config.yaml"
    child_config_sha256 = sha256_file(config_path)
    child_config = load_synthesis_raw_text_inventory_batch_config(config_path)
    if (
        child_config.run.run_id != reference.root.name
        or _ensure_under_project_root(
            project_root,
            resolve_config_path(project_root, child_config.run.output_dir),
            label="inventory child output directory",
        )
        != reference.root.parent
    ):
        raise ValueError("inventory child configuration differs from its committed run root")
    _require_inventory_scope(
        child_config,
        document_ids,
        label="inventory child run",
    )
    if _inventory_contract_sha256(child_config) != inventory_contract_sha256:
        raise ValueError("inventory child behavior contract differs from its parent")

    transaction = _read_bound_transaction(reference)
    provider = cast(OpenRouterRewriteProviderConfig, child_config.provider)
    expected_implementation_fields = {
        "implementationSha256": stage_hashes["inventory"],
        "inventoryImplementationSha256": stage_hashes["inventoryContract"],
        "hybridBatchImplementationSha256": stage_hashes["hybridBatch"],
        "hybridCompilerImplementationSha256": stage_hashes["hybridRuntime"],
        "containerSemanticsImplementationSha256": stage_hashes["containerSemantics"],
        "providerRuntimeImplementationSha256": stage_hashes["providerRuntime"],
        "rewriteContractImplementationSha256": stage_hashes["rewriteContract"],
    }
    if any(
        transaction.get(key) != expected for key, expected in expected_implementation_fields.items()
    ):
        raise ValueError("inventory child implementation provenance differs")
    if (
        transaction.get("configSha256") != child_config_sha256
        or transaction.get("runId") != child_config.run.run_id
        or transaction.get("liveDocumentIds") != list(document_ids)
        or transaction.get("baseBatchConfigSha256") != child_config.base_batch_config.sha256
        or transaction.get("regressionOracleSha256") != child_config.regression_oracle.sha256
        or transaction.get("templateMutationProfileSha256")
        != (
            child_config.template_mutation_profile.sha256
            if child_config.template_mutation_profile is not None
            else None
        )
        or transaction.get("promptSha256") != child_config.prompt.sha256
        or transaction.get("referenceCommits")
        != {
            "glm": child_config.reference_runs.glm.commit_sha256,
            "luna": child_config.reference_runs.luna.commit_sha256,
        }
    ):
        raise ValueError("inventory child transaction differs from its configuration")
    runtime = transaction.get("runtime")
    if not isinstance(runtime, dict) or (
        runtime.get("model") != provider.model
        or runtime.get("outputMode") != child_config.workflow.output_mode
        or runtime.get("providerOrder") != list(provider.provider_order or ())
        or runtime.get("maxConcurrentDocuments") != child_config.workflow.max_concurrent_documents
        or runtime.get("maxProviderRouteRounds") != child_config.workflow.max_provider_route_rounds
    ):
        raise ValueError("inventory child runtime provenance differs from its configuration")
    child_transaction_contract_sha256 = _inventory_transaction_contract_sha256(transaction)
    if (
        transaction_contract_sha256 is not None
        and child_transaction_contract_sha256 != transaction_contract_sha256
    ):
        raise ValueError("inventory child transaction contract differs from its parent")

    results = cast(
        tuple[InventoryProbeCaseResult, ...],
        _read_model_rows(
            reference.root / "generation/results.jsonl",
            InventoryProbeCaseResult,
        ),
    )
    if tuple(result.documentId for result in results) != document_ids:
        raise ValueError("inventory child results differ from their configured scope")
    compiler_blocked = tuple(
        result.documentId for result in results if result.status == "compiler_blocked"
    )
    if compiler_blocked:
        raise ValueError(
            "inventory resume cannot reuse compiler-blocked cases under an identical "
            "implementation; start a fresh full-cohort run after fixing the compiler: "
            f"{list(compiler_blocked)}"
        )
    status_counts = {
        status: sum(result.status == status for result in results)
        for status in (
            "training_ready",
            "needs_review",
            "call_failed",
            "compiler_blocked",
        )
    }
    summary = _read_json_object(reference.root / "summary.json")
    if (
        summary.get("schemaVersion") != 2
        or summary.get("runId") != child_config.run.run_id
        or summary.get("status") != "complete"
        or summary.get("liveDocuments") != len(document_ids)
        or summary.get("trainingReadyDocuments") != status_counts["training_ready"]
        or summary.get("needsReviewDocuments") != status_counts["needs_review"]
        or summary.get("callFailedDocuments") != status_counts["call_failed"]
        or summary.get("compilerBlockedDocuments") != status_counts["compiler_blocked"]
    ):
        raise ValueError("inventory child summary differs from its exact results")
    identities: dict[str, _InventoryInputIdentity] = {}
    for result in results:
        case_result = InventoryProbeCaseResult.model_validate_json(
            read_regular_file_bytes(reference.root / "cases" / result.documentId / "result.json"),
            strict=True,
        )
        if case_result != result:
            raise ValueError(
                f"inventory aggregate and per-case results differ: {result.documentId}"
            )
        if (
            _case_artifact_sha256(reference, result.documentId, "final.txt")
            != result.outputTextSha256
        ):
            raise ValueError(
                "inventory result candidate identity differs from its committed artifact: "
                f"{result.documentId}"
            )
        identities[result.documentId] = _inventory_input_identity(
            reference,
            result.documentId,
        )
    return results, child_transaction_contract_sha256, identities


def _stage_template_contract_sha256(
    *,
    environment_file: str,
    prompt: BaseModel,
    provider: BaseModel,
    workflow: BaseModel,
    audit_contract_version: Literal[1, 2, 3] | None = None,
    invariant_inputs: BaseModel | None = None,
) -> str:
    """Hash model behavior while excluding only child scope and scheduling."""

    workflow_value = workflow.model_dump(mode="json")
    workflow_value.pop("documents")
    workflow_value.pop("max_concurrent_documents")
    value: dict[str, JsonValue] = {
        "environmentFile": environment_file,
        "prompt": cast(JsonValue, prompt.model_dump(mode="json")),
        "provider": cast(JsonValue, provider.model_dump(mode="json")),
        "workflow": cast(JsonValue, workflow_value),
    }
    if audit_contract_version is not None:
        value["auditContractVersion"] = audit_contract_version
    if invariant_inputs is not None:
        value["invariantInputs"] = cast(JsonValue, invariant_inputs.model_dump(mode="json"))
    return sha256_bytes(canonical_json_bytes(value))


def _provider_order(provider: BaseModel) -> list[str]:
    value = getattr(provider, "provider_order", None)
    return list(value or ())


def _validate_component_config_root(
    *,
    project_root: Path,
    reference: _RunReference,
    run_id: str,
    output_dir: str,
    label: str,
) -> None:
    if run_id != reference.root.name:
        raise ValueError(f"{label} configuration run ID differs from its committed root")
    output_parent = _ensure_under_project_root(
        project_root,
        resolve_config_path(project_root, output_dir),
        label=f"{label} output directory",
    )
    if output_parent != reference.root.parent:
        raise ValueError(f"{label} configuration output differs from its committed root")


def _validate_certification_child_run(
    *,
    project_root: Path,
    reference: _RunReference,
    resolve_reference: Callable[[_LineageRunReference], _RunReference],
    expected_template_sha256: str,
    stage_hashes: dict[str, str],
) -> tuple[
    tuple[CertificationCaseResult, ...],
    _RunReference,
    Literal["contract.json", "source-contract.json"],
]:
    """Validate one committed certification run and its immutable candidate handoff."""

    config_path = reference.root / "config.yaml"
    configured_child = load_synthesis_raw_text_certification_config(config_path)
    certification_run = load_validated_certification_run(
        reference.root,
        project_root=(project_root if configured_child.audit_contract_version == 3 else None),
    )
    child_config = certification_run.config
    _validate_component_config_root(
        project_root=project_root,
        reference=reference,
        run_id=child_config.run.run_id,
        output_dir=child_config.run.output_dir,
        label="pipeline certification child",
    )
    document_ids = tuple(child_config.case_ids)
    if not document_ids or len(document_ids) != len(set(document_ids)):
        raise ValueError("pipeline certification child scope is not unique")
    if (
        _stage_template_contract_sha256(
            environment_file=child_config.environment_file,
            prompt=child_config.prompt,
            provider=child_config.provider,
            workflow=child_config.workflow,
            audit_contract_version=child_config.audit_contract_version,
            invariant_inputs=child_config.invariant_inputs,
        )
        != expected_template_sha256
    ):
        raise ValueError("pipeline certification child behavior contract differs")
    input_run = resolve_reference(
        _LineageRunReference(
            path=child_config.input_run.path,
            commitSha256=child_config.input_run.commit_sha256,
            transactionSha256=child_config.input_run.transaction_sha256,
        )
    )
    contract_filename = child_config.input_case_contract_filename
    transaction = certification_run.transaction
    expected_dependencies = {
        "containerSemantics": stage_hashes["containerSemantics"],
        "providerRuntime": stage_hashes["providerRuntime"],
        "hybridRuntime": stage_hashes["hybridRuntime"],
        "inventory": stage_hashes["inventoryContract"],
        "inventoryRunner": stage_hashes["inventory"],
        "rewriteContract": stage_hashes["rewriteContract"],
        "rewriteRuntime": stage_hashes["rewriteRuntime"],
        "transportCapacity": stage_hashes["transportCapacity"],
        **(
            {
                "certificationHost": stage_hashes["certificationHost"],
                "certificationInvariants": stage_hashes["certificationInvariants"],
                "certificationReferences": stage_hashes["certificationReferences"],
            }
            if child_config.audit_contract_version == 3
            else {}
        ),
    }
    runtime = transaction.get("runtime")
    if (
        transaction.get("schemaVersion") != (3 if child_config.audit_contract_version == 3 else 2)
        or transaction.get("runId") != child_config.run.run_id
        or transaction.get("configSha256") != sha256_file(config_path)
        or transaction.get("resolvedConfigSha256")
        != sha256_bytes(canonical_json_bytes(child_config.model_dump(mode="json")))
        or transaction.get("inputRunCommitSha256") != input_run.commit_sha256
        or transaction.get("inputRunTransactionSha256") != input_run.transaction_sha256
        or transaction.get("inputCaseContractFilename") != contract_filename
        or transaction.get("promptSha256") != child_config.prompt.sha256
        or transaction.get("implementationSha256") != stage_hashes["certification"]
        or transaction.get("dependencyImplementationSha256") != expected_dependencies
        or transaction.get("documentIds") != list(document_ids)
        or not isinstance(runtime, dict)
        or runtime.get("model") != child_config.provider.model
        or runtime.get("providerOrder") != _provider_order(child_config.provider)
        or runtime.get("maxConcurrentDocuments") != child_config.workflow.max_concurrent_documents
        or runtime.get("semanticAuditPasses") != child_config.workflow.semantic_audit_passes
        or runtime.get("auditContractVersion") != child_config.audit_contract_version
        or runtime.get("certificationMode") != "production"
        or runtime.get("reasoningEfforts") != ["low", "high"]
    ):
        raise ValueError("pipeline certification child provenance differs")

    results = tuple(row.result for row in certification_run.cases)
    if tuple(row.documentId for row in results) != document_ids:
        raise ValueError("pipeline certification child results differ from its scope")
    summary = certification_run.summary
    status_counts = {
        status: sum(row.status == status for row in results)
        for status in ("certified", "evaluated_clean", "needs_review", "call_failed")
    }
    if (
        summary.get("schemaVersion") != (3 if child_config.audit_contract_version == 3 else 2)
        or summary.get("runId") != child_config.run.run_id
        or summary.get("status") != "complete"
        or summary.get("documents") != len(results)
        or summary.get("certifiedDocuments") != status_counts["certified"]
        or summary.get("evaluatedCleanDocuments") != status_counts["evaluated_clean"]
        or summary.get("needsReviewDocuments") != status_counts["needs_review"]
        or summary.get("callFailedDocuments") != status_counts["call_failed"]
        or summary.get("semanticFindings") != sum(row.semanticFindings for row in results)
        or (
            child_config.audit_contract_version == 3
            and summary.get("deterministicFindings")
            != sum(row.deterministicFindings for row in results)
        )
        or summary.get("candidateBytesModified") != 0
        or summary.get("trainingRecordsPublished") is not False
    ):
        raise ValueError("pipeline certification child summary differs from its results")
    for result in results:
        source_sha256 = _case_artifact_sha256(reference, result.documentId, "source.txt")
        input_sha256 = _case_artifact_sha256(reference, result.documentId, "input-candidate.txt")
        final_sha256 = _case_artifact_sha256(reference, result.documentId, "final.txt")
        if (
            source_sha256 != result.sourceTextSha256
            or input_sha256 != result.inputCandidateSha256
            or final_sha256 != result.finalTextSha256
            or input_sha256 != final_sha256
            or result.candidateImmutable is not True
        ):
            raise ValueError(
                f"pipeline certification child candidate identity differs: {result.documentId}"
            )
    return results, input_run, contract_filename


def _validate_correction_child_run(
    *,
    project_root: Path,
    reference: _RunReference,
    resolve_reference: Callable[[_LineageRunReference], _RunReference],
    expected_template_sha256: str,
    stage_hashes: dict[str, str],
) -> tuple[tuple[CorrectionCaseResult, ...], _RunReference]:
    """Validate one committed correction run and its certification handoff."""

    config_path = reference.root / "config.yaml"
    child_config = load_synthesis_raw_text_certified_correction_config(config_path)
    _validate_component_config_root(
        project_root=project_root,
        reference=reference,
        run_id=child_config.run.run_id,
        output_dir=child_config.run.output_dir,
        label="pipeline correction child",
    )
    document_ids = tuple(child_config.case_ids)
    if not document_ids or len(document_ids) != len(set(document_ids)):
        raise ValueError("pipeline correction child scope is not unique")
    if (
        _stage_template_contract_sha256(
            environment_file=child_config.environment_file,
            prompt=child_config.prompt,
            provider=child_config.provider,
            workflow=child_config.workflow,
        )
        != expected_template_sha256
    ):
        raise ValueError("pipeline correction child behavior contract differs")
    certification_run = resolve_reference(
        _LineageRunReference(
            path=child_config.certification_run.path,
            commitSha256=child_config.certification_run.commit_sha256,
            transactionSha256=child_config.certification_run.transaction_sha256,
        )
    )
    required_contract = child_config.workflow.required_audit_contract_version
    validated_certification = load_validated_certification_run(
        certification_run.root,
        project_root=project_root if required_contract == 3 else None,
    )
    if (
        validated_certification.config.audit_contract_version != required_contract
        or validated_certification.config.workflow.evaluation_only
        or validated_certification.config.workflow.semantic_audit_passes != 2
    ):
        raise ValueError("pipeline correction source is not its required production contract")
    transaction = _read_bound_transaction(reference)
    expected_dependencies = (
        {
            "certification": stage_hashes["certification"],
            "certificationArtifacts": stage_hashes["certificationArtifacts"],
            "certificationHost": stage_hashes["certificationHost"],
            "certificationInvariants": stage_hashes["certificationInvariants"],
            "certificationReferences": stage_hashes["certificationReferences"],
            "rewriteRuntime": stage_hashes["rewriteRuntime"],
        }
        if required_contract == 3
        else {
            "certification": stage_hashes["certification"],
            "certificationArtifacts": stage_hashes["certificationArtifacts"],
            "providerRuntime": stage_hashes["providerRuntime"],
            "hybridRuntime": stage_hashes["hybridRuntime"],
            "inventory": stage_hashes["inventoryContract"],
            "inventoryRunner": stage_hashes["inventory"],
            "rewriteContract": stage_hashes["rewriteContract"],
            "rewriteRuntime": stage_hashes["rewriteRuntime"],
        }
    )
    runtime = transaction.get("runtime")
    common_transaction_mismatch = (
        transaction.get("schemaVersion") != (3 if required_contract == 3 else 2)
        or transaction.get("runId") != child_config.run.run_id
        or transaction.get("configSha256") != sha256_file(config_path)
        or transaction.get("resolvedConfigSha256")
        != sha256_bytes(canonical_json_bytes(child_config.model_dump(mode="json")))
        or transaction.get("certificationRunCommitSha256") != certification_run.commit_sha256
        or transaction.get("certificationRunTransactionSha256")
        != certification_run.transaction_sha256
        or transaction.get("certificationConfigSha256")
        != sha256_file(certification_run.root / "config.yaml")
        or transaction.get("certificationResolvedConfigSha256")
        != sha256_bytes(
            canonical_json_bytes(validated_certification.config.model_dump(mode="json"))
        )
        or transaction.get("certificationTransactionArtifactSha256")
        != sha256_file(certification_run.root / "provenance/transaction.json")
        or transaction.get("certificationSummarySha256")
        != sha256_file(certification_run.root / "summary.json")
        or transaction.get("certificationAuditContractVersion") != required_contract
        or transaction.get("certificationMode") != "production"
        or transaction.get("implementationSha256") != stage_hashes["correction"]
        or transaction.get("dependencyImplementationSha256") != expected_dependencies
        or transaction.get("documentIds") != list(document_ids)
        or not isinstance(runtime, dict)
    )
    legacy_runtime_mismatch = required_contract == 2 and (
        transaction.get("promptSha256") != child_config.prompt.sha256
        or not isinstance(runtime, dict)
        or runtime.get("model") != child_config.provider.model
        or runtime.get("providerOrder") != _provider_order(child_config.provider)
        or runtime.get("maxConcurrentDocuments") != child_config.workflow.max_concurrent_documents
        or runtime.get("maxSuccessfulModelResponsesPerDocument")
        != child_config.workflow.max_successful_model_responses_per_document
    )
    deterministic_runtime_mismatch = required_contract == 3 and (
        transaction.get("correctionMode") != "deterministic_exact_fragment_v1"
        or transaction.get("invariantReferenceReceiptSha256")
        != validated_certification.transaction.get("invariantReferenceReceiptSha256")
        or transaction.get("capacityConfigSha256")
        != validated_certification.transaction.get("capacityConfigSha256")
        or not isinstance(runtime, dict)
        or runtime
        != {
            "providerRequests": 0,
            "maxConcurrentDocuments": child_config.workflow.max_concurrent_documents,
        }
    )
    if common_transaction_mismatch or legacy_runtime_mismatch or deterministic_runtime_mismatch:
        raise ValueError("pipeline correction child provenance differs")

    results = cast(
        tuple[CorrectionCaseResult, ...],
        _read_model_rows(
            reference.root / "generation/results.jsonl",
            CorrectionCaseResult,
        ),
    )
    if tuple(row.documentId for row in results) != document_ids:
        raise ValueError("pipeline correction child results differ from its scope")
    summary = _read_json_object(reference.root / "summary.json")
    status_counts = {
        status: sum(row.status == status for row in results)
        for status in (
            "unchanged_certified",
            "correction_candidate",
            "needs_review",
            "call_failed",
        )
    }
    if (
        summary.get("schemaVersion") != (3 if required_contract == 3 else 1)
        or summary.get("runId") != child_config.run.run_id
        or summary.get("status") != "complete"
        or summary.get("documents") != len(results)
        or summary.get("unchangedCertifiedDocuments") != status_counts["unchanged_certified"]
        or summary.get("correctionCandidateDocuments") != status_counts["correction_candidate"]
        or summary.get("needsReviewDocuments") != status_counts["needs_review"]
        or summary.get("callFailedDocuments") != status_counts["call_failed"]
        or summary.get("sourceFindings") != sum(row.sourceFindings for row in results)
        or (
            required_contract == 3
            and (
                summary.get("semanticFindings") != sum(row.semanticFindings for row in results)
                or summary.get("deterministicFindings")
                != sum(row.deterministicFindings for row in results)
                or summary.get("appliedDeterministicRepairs")
                != sum(row.appliedDeterministicRepairs for row in results)
                or summary.get("requests") != 0
                or summary.get("providerAttempts") != 0
            )
        )
        or summary.get("citedLines") != sum(row.citedLines for row in results)
        or summary.get("changedLines") != sum(row.changedLines for row in results)
        or summary.get("trainingRecordsPublished") is not False
    ):
        raise ValueError("pipeline correction child summary differs from its results")
    certified_cases = validated_certification.case_by_id()
    for result in results:
        per_case = CorrectionCaseResult.model_validate_json(
            read_regular_file_bytes(reference.root / "cases" / result.documentId / "result.json"),
            strict=True,
        )
        if per_case != result:
            raise ValueError(
                f"pipeline correction aggregate and per-case results differ: {result.documentId}"
            )
        if (
            _case_artifact_sha256(reference, result.documentId, "source.txt")
            != result.sourceTextSha256
            or _case_artifact_sha256(reference, result.documentId, "input-candidate.txt")
            != result.inputCandidateSha256
            or _case_artifact_sha256(reference, result.documentId, "final.txt")
            != result.finalTextSha256
        ):
            raise ValueError(
                f"pipeline correction child candidate identity differs: {result.documentId}"
            )
        for filename in ("source-label.json", "target-label.json"):
            if _read_json_object(
                reference.root / "cases" / result.documentId / filename
            ) != _read_json_object(certification_run.root / "cases" / result.documentId / filename):
                raise ValueError(
                    f"pipeline correction child contract handoff differs: {result.documentId}"
                )
        source = read_regular_file_bytes(
            certification_run.root / "cases" / result.documentId / "source.txt"
        ).decode("utf-8")
        source_contract = _read_json_object(
            certification_run.root / "cases" / result.documentId / "source-contract.json"
        )
        expected_contract = (
            source_contract
            if required_contract == 3
            else _refine_carrier_occurrence_contract(
                source=source,
                contract=source_contract,
            )
        )
        if (
            _read_json_object(reference.root / "cases" / result.documentId / "source-contract.json")
            != expected_contract
        ):
            raise ValueError(
                f"pipeline correction child refined contract differs: {result.documentId}"
            )
        if required_contract == 3:
            certified_invariant = certified_cases[result.documentId].invariant_case
            if certified_invariant is None:
                raise ValueError("pipeline contract-v3 source lost its invariant artifacts")
            expected = _deterministic_correction_case(
                certified=certified_cases[result.documentId],
                certification_run=validated_certification,
            )
            case_root = reference.root / "cases" / result.documentId
            if (
                result != expected.result
                or read_regular_file_bytes(case_root / "final.txt")
                != expected.final.encode("utf-8")
                or json.loads(
                    read_regular_file_bytes(case_root / "authorized-deterministic-repairs.json")
                )
                != [row.model_dump(mode="json") for row in expected.authorized_repairs]
                or json.loads(
                    read_regular_file_bytes(case_root / "applied-deterministic-repairs.json")
                )
                != [row.model_dump(mode="json") for row in expected.applied_repairs]
                or _read_json_object(case_root / "pre-invariant-envelope.json")
                != certified_invariant.envelope.model_dump(mode="json")
                or _read_json_object(case_root / "pre-invariant-audit.json")
                != certified_invariant.audit.model_dump(mode="json")
                or _read_json_object(case_root / "invariant-envelope.json")
                != expected.final_invariant.envelope.model_dump(mode="json")
                or _read_json_object(case_root / "invariant-audit.json")
                != expected.final_invariant.audit.model_dump(mode="json")
                or json.loads(read_regular_file_bytes(case_root / "stages.json")) != []
            ):
                raise ValueError(
                    f"pipeline deterministic correction replay differs: {result.documentId}"
                )
            attempted_envelope = case_root / "attempted-invariant-envelope.json"
            attempted_audit = case_root / "attempted-invariant-audit.json"
            if expected.attempted_invariant is None:
                if attempted_envelope.exists() or attempted_audit.exists():
                    raise ValueError(
                        "pipeline correction has an unexpected attempted audit: "
                        f"{result.documentId}"
                    )
            elif _read_json_object(
                attempted_envelope
            ) != expected.attempted_invariant.envelope.model_dump(mode="json") or _read_json_object(
                attempted_audit
            ) != expected.attempted_invariant.audit.model_dump(mode="json"):
                raise ValueError(
                    f"pipeline correction attempted audit differs: {result.documentId}"
                )
    return results, certification_run


def _stage_implementation_hashes() -> dict[str, str]:
    return {
        name: sha256_file(Path(__file__).with_name(filename))
        for name, filename in (
            ("config", "config.py"),
            ("containerSemantics", "container_semantics.py"),
            ("inventory", "raw_text_inventory_probe.py"),
            ("inventoryContract", "raw_text_inventory.py"),
            ("hybridBatch", "raw_text_hybrid_batch.py"),
            ("hybridRuntime", "raw_text_hybrid_probe.py"),
            ("certification", "raw_text_certification.py"),
            ("certificationArtifacts", "raw_text_certification_artifacts.py"),
            ("certificationHost", "raw_text_certification_host.py"),
            ("certificationInvariants", "raw_text_certification_invariants.py"),
            ("certificationReferences", "raw_text_certification_references.py"),
            ("correction", "raw_text_certified_correction.py"),
            ("publication", "raw_text_certified_publication.py"),
            ("providerRuntime", "linguistic_probe_runtime.py"),
            ("rewriteContract", "raw_text_rewrite_cycle_probe.py"),
            ("rewriteRuntime", "raw_text_rewrite_probe.py"),
            ("transportCapacity", "transport_capacity.py"),
        )
    }


def _target_schema_implementation_sha256() -> str:
    return sha256_file(Path(__file__).parents[1] / "label_schemas/bill_of_lading_v5.py")


def _load_inventory_resume_state(
    *,
    project_root: Path,
    config: SynthesisRawTextPipelineConfig,
    initial_inventory: SynthesisRawTextInventoryBatchConfig,
    _visited: frozenset[tuple[str, str, str]] = frozenset(),
) -> _InventoryResumeState | None:
    """Rehydrate only an inventory-blocked parent under the identical stage contract."""

    resume_config = config.inventory_resume_run
    if resume_config is None:
        if initial_inventory.workflow.documents != config.workflow.documents:
            raise ValueError("pipeline inventory count differs from workflow.documents")
        return None
    parent_run = _pinned_run_reference(
        project_root=project_root,
        path=resume_config.path,
        commit_sha256=resume_config.commit_sha256,
        transaction_sha256=resume_config.transaction_sha256,
        label="pipeline inventory resume run",
    )
    parent_identity = (
        parent_run.path,
        parent_run.commit_sha256,
        parent_run.transaction_sha256,
    )
    if parent_identity in _visited:
        raise ValueError("pipeline inventory resume ancestry contains a cycle")
    visited = _visited | {parent_identity}
    current_root = _ensure_under_project_root(
        project_root,
        resolve_config_path(project_root, config.run.output_dir) / config.run.run_id,
        label="pipeline output root",
    )
    if parent_run.root == current_root:
        raise ValueError("pipeline inventory resume run cannot be the current run")

    summary = _read_json_object(parent_run.root / "summary.json")
    if (
        summary.get("schemaVersion") != config.schema_version
        or summary.get("status") != "blocked"
        or summary.get("documents") != config.workflow.documents
        or summary.get("certifiedDocuments") != 0
        or summary.get("correctionRounds") != 0
        or summary.get("trainingRecordsPublished") is not False
        or summary.get("publicationRun") is not None
    ):
        raise ValueError(
            "pipeline inventory resume source must be blocked during inventory, before "
            "certification or correction"
        )

    parent_transaction = _read_bound_transaction(parent_run)
    parent_config_path = parent_run.root / "config.yaml"
    parent_config_sha256 = sha256_file(parent_config_path)
    parent_config = load_synthesis_raw_text_pipeline_config(parent_config_path)
    _validate_component_config_root(
        project_root=project_root,
        reference=parent_run,
        run_id=parent_config.run.run_id,
        output_dir=parent_config.run.output_dir,
        label="pipeline continuation parent",
    )
    parent_inventory_path = parent_run.root / "inputs/inventory-config.yaml"
    parent_inventory_sha256 = sha256_file(parent_inventory_path)
    parent_inventory = load_synthesis_raw_text_inventory_batch_config(parent_inventory_path)
    if (
        parent_config.run.run_id != parent_run.root.name
        or parent_config.workflow.documents != config.workflow.documents
        or parent_config.schema_version != config.schema_version
        or parent_config.task != config.task
        or parent_transaction.get("schemaVersion") != parent_config.schema_version
        or parent_transaction.get("configSha256") != parent_config_sha256
        or parent_transaction.get("runId") != parent_config.run.run_id
        or parent_transaction.get("inventoryConfigSha256") != parent_inventory_sha256
        or parent_config.inventory_config.sha256 != parent_inventory_sha256
    ):
        raise ValueError("pipeline inventory resume parent configuration provenance differs")
    inventory_contract_sha256 = _inventory_contract_sha256(parent_inventory)
    ancestor_resume = (
        _load_inventory_resume_state(
            project_root=project_root,
            config=parent_config,
            initial_inventory=parent_inventory,
            _visited=visited,
        )
        if parent_config.inventory_resume_run is not None
        else None
    )
    if (
        ancestor_resume is not None
        and ancestor_resume.inventory_contract_sha256 != inventory_contract_sha256
    ):
        raise ValueError("pipeline inventory resume contract differs from its root ancestor")
    if _inventory_contract_sha256(initial_inventory) != inventory_contract_sha256:
        raise ValueError(
            "pipeline inventory resume config differs from the parent inventory contract"
        )

    parent_stage_hashes = parent_transaction.get("stageImplementationSha256")
    if not isinstance(parent_stage_hashes, dict):
        raise ValueError("pipeline inventory resume source lacks stage implementation hashes")
    current_stage_hashes = _stage_implementation_hashes()
    inventory_dependencies = (
        "containerSemantics",
        "inventory",
        "inventoryContract",
        "hybridBatch",
        "hybridRuntime",
        "providerRuntime",
        "rewriteContract",
        "rewriteRuntime",
    )
    mismatched_dependencies = tuple(
        name
        for name in inventory_dependencies
        if parent_stage_hashes.get(name) != current_stage_hashes[name]
    )
    if mismatched_dependencies:
        raise ValueError(
            "pipeline inventory resume implementation differs for: "
            + ", ".join(mismatched_dependencies)
        )
    if (
        parent_transaction.get("targetSchemaImplementationSha256")
        != _target_schema_implementation_sha256()
    ):
        raise ValueError("pipeline inventory resume target schema implementation differs")

    lineage_path = parent_run.root / "lineage/cases.jsonl"
    lineage_payload = read_regular_file_bytes(lineage_path)
    lineage_sha256 = sha256_bytes(lineage_payload)
    if summary.get("caseLineageSha256") != lineage_sha256:
        raise ValueError("pipeline inventory resume case lineage differs from its summary")
    rows = cast(
        tuple[_InventoryResumeLineageRow, ...],
        _read_model_rows(lineage_path, _InventoryResumeLineageRow),
    )
    document_ids = tuple(row.documentId for row in rows)
    if len(rows) != config.workflow.documents or len(document_ids) != len(set(document_ids)):
        raise ValueError("pipeline inventory resume lineage differs from the unique cohort")

    reference_cache: dict[tuple[str, str, str], _RunReference] = {}

    def resolve_reference(value: _LineageRunReference) -> _RunReference:
        key = (value.path, value.commitSha256, value.transactionSha256)
        reference = reference_cache.get(key)
        if reference is None:
            reference = _pinned_run_reference(
                project_root=project_root,
                path=value.path,
                commit_sha256=value.commitSha256,
                transaction_sha256=value.transactionSha256,
                label="pipeline inventory resume child run",
            )
            reference_cache[key] = reference
        return reference

    history_by_round: dict[
        int,
        list[tuple[str, _InventoryHistoryRow]],
    ] = defaultdict(list)
    histories_by_id: dict[str, list[dict[str, JsonValue]]] = {}
    for row in rows:
        if not row.history:
            raise ValueError(
                f"pipeline inventory resume row has no inventory history: {row.documentId}"
            )
        rounds = tuple(history.round for history in row.history)
        if any(round_number < 1 for round_number in rounds) or tuple(sorted(set(rounds))) != rounds:
            raise ValueError(
                f"pipeline inventory resume rounds are not strictly ordered: {row.documentId}"
            )
        histories_by_id[row.documentId] = [
            cast(dict[str, JsonValue], history.model_dump(mode="json")) for history in row.history
        ]
        for history in row.history:
            history_by_round[history.round].append((row.documentId, history))

    if not history_by_round:
        raise ValueError("pipeline inventory resume lineage has no inventory rounds")
    prior_rounds = max(history_by_round)
    if tuple(sorted(history_by_round)) != tuple(range(1, prior_rounds + 1)):
        raise ValueError("pipeline inventory resume lineage has a non-contiguous round topology")

    current_stage_hashes = _stage_implementation_hashes()
    expected_scope = document_ids
    validated_by_round: dict[
        int,
        tuple[
            _RunReference,
            tuple[str, ...],
            tuple[InventoryProbeCaseResult, ...],
        ],
    ] = {}
    input_identities_by_id: dict[str, _InventoryInputIdentity] = {}
    inventory_transaction_contract_sha256: str | None = None
    seen_run_identities: set[tuple[str, str, str]] = set()
    for round_number in range(1, prior_rounds + 1):
        round_history = history_by_round[round_number]
        round_ids = tuple(document_id for document_id, _history in round_history)
        if round_ids != expected_scope:
            raise ValueError(
                "pipeline inventory resume history scope differs from prior non-ready cases"
            )
        run_values = {history.run for _document_id, history in round_history}
        if len(run_values) != 1:
            raise ValueError("pipeline inventory resume round references multiple child runs")
        run_value = next(iter(run_values))
        run_identity = (
            run_value.path,
            run_value.commitSha256,
            run_value.transactionSha256,
        )
        if run_identity in seen_run_identities:
            raise ValueError("pipeline inventory resume reuses one child run across rounds")
        seen_run_identities.add(run_identity)
        reference = resolve_reference(run_value)
        (
            results,
            child_transaction_contract_sha256,
            child_identities,
        ) = _validate_inventory_child_run(
            project_root=project_root,
            reference=reference,
            document_ids=round_ids,
            inventory_contract_sha256=inventory_contract_sha256,
            transaction_contract_sha256=inventory_transaction_contract_sha256,
            stage_hashes=current_stage_hashes,
        )
        if inventory_transaction_contract_sha256 is None:
            inventory_transaction_contract_sha256 = child_transaction_contract_sha256
        for (document_id, history), result in zip(
            round_history,
            results,
            strict=True,
        ):
            if (
                result.documentId != document_id
                or result.status != history.status
                or result.outputTextSha256 != history.candidateSha256
            ):
                raise ValueError(
                    "pipeline inventory resume history differs from its child result: "
                    f"{document_id}"
                )
            identity = child_identities[document_id]
            prior_identity = input_identities_by_id.setdefault(document_id, identity)
            if prior_identity != identity:
                raise ValueError(
                    "pipeline inventory resume input identity changed across attempts: "
                    f"{document_id}"
                )
        validated_by_round[round_number] = (reference, round_ids, results)
        expected_scope = tuple(
            result.documentId for result in results if result.status != "training_ready"
        )

    if inventory_transaction_contract_sha256 is None:
        raise AssertionError("inventory transaction contract was not established")
    if ancestor_resume is not None:
        if (
            ancestor_resume.document_ids != document_ids
            or ancestor_resume.inventory_contract_sha256 != inventory_contract_sha256
            or ancestor_resume.inventory_transaction_contract_sha256
            != inventory_transaction_contract_sha256
            or ancestor_resume.prior_rounds >= prior_rounds
            or ancestor_resume.input_identities_by_id != input_identities_by_id
        ):
            raise ValueError("pipeline inventory resume ancestry contract differs")
        for document_id in document_ids:
            ancestor_history = ancestor_resume.histories_by_id[document_id]
            current_history = histories_by_id[document_id]
            if current_history[: len(ancestor_history)] != ancestor_history:
                raise ValueError(
                    "pipeline inventory resume ancestry is not an exact history prefix: "
                    f"{document_id}"
                )
            if document_id in ancestor_resume.states_by_id and len(current_history) != len(
                ancestor_history
            ):
                raise ValueError(
                    f"pipeline inventory resume retried an ancestor-ready case: {document_id}"
                )

    subruns = cast(
        tuple[_InventoryResumeSubrunRow, ...],
        _read_model_array(
            parent_run.root / "lineage/subruns.json",
            _InventoryResumeSubrunRow,
        ),
    )
    if not subruns or summary.get("componentRuns") != len(subruns):
        raise ValueError("pipeline inventory resume subrun count differs from its summary")
    resume_markers = tuple(row for row in subruns if row.stage == "inventory_resume_parent")
    inventory_subruns = tuple(row for row in subruns if row.stage == "inventory")
    if len(inventory_subruns) != parent_config.workflow.max_inventory_rounds:
        raise ValueError(
            "pipeline inventory resume parent did not exhaust its configured local rounds"
        )
    parent_resume = parent_config.inventory_resume_run
    transaction_resume = parent_transaction.get("inventoryResumeRun")
    if parent_resume is None:
        if (
            resume_markers
            or transaction_resume is not None
            or parent_transaction.get("inventoryResumeLineageSha256") is not None
            or subruns != inventory_subruns
            or tuple(row.round for row in inventory_subruns) != tuple(range(1, prior_rounds + 1))
        ):
            raise ValueError("pipeline inventory resume parent subrun topology differs")
    else:
        expected_parent_resume = {
            "path": parent_resume.path,
            "commitSha256": parent_resume.commit_sha256,
            "transactionSha256": parent_resume.transaction_sha256,
        }
        recorded_parent_resume = _read_json_object(
            parent_run.root / "inputs/inventory-resume-run.json"
        )
        if ancestor_resume is None:
            raise ValueError("pipeline chained inventory resume lacks a validated ancestor")
        if (
            len(resume_markers) != 1
            or subruns[0] != resume_markers[0]
            or transaction_resume != expected_parent_resume
            or recorded_parent_resume != expected_parent_resume
            or resume_markers[0].run.model_dump(mode="json") != expected_parent_resume
            or resume_markers[0].documentIds != document_ids
            or resume_markers[0].round != ancestor_resume.prior_rounds
            or tuple(row.round for row in inventory_subruns)
            != tuple(
                range(
                    resume_markers[0].round + 1,
                    resume_markers[0].round + 1 + len(inventory_subruns),
                )
            )
            or inventory_subruns[-1].round != prior_rounds
        ):
            raise ValueError("pipeline chained inventory resume topology differs")
        ancestor = resolve_reference(resume_markers[0].run)
        ancestor_lineage_sha256 = sha256_file(ancestor.root / "lineage/cases.jsonl")
        if (
            ancestor_resume.parent_run != ancestor
            or resume_markers[0].summarySha256 != sha256_file(ancestor.root / "summary.json")
            or parent_transaction.get("inventoryResumeLineageSha256") != ancestor_lineage_sha256
        ):
            raise ValueError("pipeline chained inventory resume provenance differs")

    for subrun in inventory_subruns:
        validated = validated_by_round.get(subrun.round)
        if validated is None:
            raise ValueError("pipeline inventory subrun references an unknown round")
        reference, round_ids, _results = validated
        if (
            subrun.run.model_dump(mode="json") != reference.lineage_value()
            or subrun.documentIds != round_ids
            or subrun.summarySha256 != sha256_file(reference.root / "summary.json")
        ):
            raise ValueError("pipeline inventory subrun differs from its child round")

    states_by_id: dict[str, _CaseState] = {}
    for row in rows:
        state_fields = (
            row.inventoryRound,
            row.inventoryRun,
            row.sourceTextSha256,
            row.currentCandidateSha256,
        )
        if row.inventoryRun is None:
            if any(value is not None for value in state_fields):
                raise ValueError(
                    f"pipeline inventory resume blocked row has partial state: {row.documentId}"
                )
            if row.history[-1].status == "training_ready":
                raise ValueError(f"pipeline inventory resume lost a ready state: {row.documentId}")
            continue
        if (
            row.inventoryRound is None
            or row.sourceTextSha256 is None
            or row.currentCandidateSha256 is None
            or row.history[-1].status != "training_ready"
            or row.inventoryRound != row.history[-1].round
            or row.inventoryRun != row.history[-1].run
            or row.currentCandidateSha256 != row.history[-1].candidateSha256
        ):
            raise ValueError(
                f"pipeline inventory resume ready-state topology differs: {row.documentId}"
            )
        reference = resolve_reference(row.inventoryRun)
        source_sha256 = _case_artifact_sha256(reference, row.documentId, "source.txt")
        candidate_sha256 = _case_artifact_sha256(reference, row.documentId, "final.txt")
        read_regular_file_bytes(reference.root / "cases" / row.documentId / "contract.json")
        if source_sha256 != row.sourceTextSha256 or candidate_sha256 != row.currentCandidateSha256:
            raise ValueError(
                f"pipeline inventory resume candidate handoff differs: {row.documentId}"
            )
        states_by_id[row.documentId] = _CaseState(
            document_id=row.documentId,
            inventory_run=reference,
            inventory_round=row.inventoryRound,
            candidate_run=reference,
            source_text_sha256=source_sha256,
            current_candidate_sha256=candidate_sha256,
            candidate_hashes={candidate_sha256},
            history=list(histories_by_id[row.documentId]),
        )

    remaining = tuple(
        document_id for document_id in document_ids if document_id not in states_by_id
    )
    if (
        remaining != expected_scope
        or len(states_by_id) != summary.get("inventoryReadyDocuments")
        or summary.get("primaryBlockingDocuments") != len(remaining)
        or summary.get("blockedDocuments") != len(document_ids)
    ):
        raise ValueError("pipeline inventory resume outcome differs from its summary")
    configured_ids = tuple(row.document_id for row in initial_inventory.cases)
    if (
        initial_inventory.selection != "explicit_pinned_document_ids"
        or initial_inventory.workflow.documents != len(remaining)
        or configured_ids != remaining
    ):
        raise ValueError(
            "pipeline inventory resume config must select exactly the unresolved documents "
            "in cohort order"
        )
    if not remaining:
        raise ValueError("pipeline inventory resume source has no unresolved documents")
    return _InventoryResumeState(
        parent_run=parent_run,
        document_ids=document_ids,
        states_by_id=states_by_id,
        histories_by_id=histories_by_id,
        remaining=remaining,
        prior_rounds=prior_rounds,
        lineage_sha256=lineage_sha256,
        input_identities_by_id=input_identities_by_id,
        inventory_contract_sha256=inventory_contract_sha256,
        inventory_transaction_contract_sha256=(inventory_transaction_contract_sha256),
    )


def _load_pipeline_resume_state(
    *,
    project_root: Path,
    config: SynthesisRawTextPipelineConfig,
    initial_inventory: SynthesisRawTextInventoryBatchConfig,
    _visited: frozenset[tuple[str, str, str]] = frozenset(),
) -> _PipelineResumeState | None:
    """Replay and validate one complete blocked pipeline before continuing it."""

    resume_config = config.pipeline_resume_run
    if resume_config is None:
        return None
    parent_run = _pinned_run_reference(
        project_root=project_root,
        path=resume_config.path,
        commit_sha256=resume_config.commit_sha256,
        transaction_sha256=resume_config.transaction_sha256,
        label="pipeline continuation run",
    )
    parent_identity = (
        parent_run.path,
        parent_run.commit_sha256,
        parent_run.transaction_sha256,
    )
    if parent_identity in _visited:
        raise ValueError("pipeline continuation ancestry contains a cycle")
    visited = _visited | {parent_identity}
    current_root = _ensure_under_project_root(
        project_root,
        resolve_config_path(project_root, config.run.output_dir) / config.run.run_id,
        label="pipeline output root",
    )
    if parent_run.root == current_root:
        raise ValueError("pipeline continuation run cannot be the current run")

    summary = _read_json_object(parent_run.root / "summary.json")
    if (
        summary.get("schemaVersion") != config.schema_version
        or summary.get("status") != "blocked"
        or summary.get("documents") != config.workflow.documents
        or summary.get("inventoryReadyDocuments") != config.workflow.documents
        or summary.get("trainingRecordsPublished") is not False
        or summary.get("publicationRun") is not None
    ):
        raise ValueError(
            "pipeline continuation source must be blocked after complete inventory and "
            "before publication"
        )

    parent_transaction = _read_bound_transaction(parent_run)
    parent_config_path = parent_run.root / "config.yaml"
    parent_config_sha256 = sha256_file(parent_config_path)
    parent_config = load_synthesis_raw_text_pipeline_config(parent_config_path)
    parent_inventory_path = parent_run.root / "inputs/inventory-config.yaml"
    parent_inventory_sha256 = sha256_file(parent_inventory_path)
    parent_inventory = load_synthesis_raw_text_inventory_batch_config(parent_inventory_path)
    if (
        parent_config.run.run_id != parent_run.root.name
        or parent_config.workflow.documents != config.workflow.documents
        or parent_transaction.get("schemaVersion") != parent_config.schema_version
        or parent_config.schema_version != config.schema_version
        or parent_config.task != config.task
        or parent_transaction.get("configSha256") != parent_config_sha256
        or parent_transaction.get("runId") != parent_config.run.run_id
        or parent_transaction.get("inventoryConfigSha256") != parent_inventory_sha256
        or parent_config.inventory_config.sha256 != parent_inventory_sha256
        or config.inventory_config.sha256 != parent_inventory_sha256
    ):
        raise ValueError("pipeline continuation parent configuration provenance differs")
    if _inventory_contract_sha256(initial_inventory) != _inventory_contract_sha256(
        parent_inventory
    ):
        raise ValueError("pipeline continuation inventory behavior contract differs")

    current_certification_contract = _stage_template_contract_sha256(
        environment_file=config.certification.environment_file,
        prompt=config.certification.prompt,
        provider=config.certification.provider,
        workflow=config.certification.workflow,
        audit_contract_version=config.certification.audit_contract_version,
        invariant_inputs=config.certification.invariant_inputs,
    )
    parent_certification_contract = _stage_template_contract_sha256(
        environment_file=parent_config.certification.environment_file,
        prompt=parent_config.certification.prompt,
        provider=parent_config.certification.provider,
        workflow=parent_config.certification.workflow,
        audit_contract_version=parent_config.certification.audit_contract_version,
        invariant_inputs=parent_config.certification.invariant_inputs,
    )
    current_correction_contract = _stage_template_contract_sha256(
        environment_file=config.correction.environment_file,
        prompt=config.correction.prompt,
        provider=config.correction.provider,
        workflow=config.correction.workflow,
    )
    parent_correction_contract = _stage_template_contract_sha256(
        environment_file=parent_config.correction.environment_file,
        prompt=parent_config.correction.prompt,
        provider=parent_config.correction.provider,
        workflow=parent_config.correction.workflow,
    )
    current_workflow = config.workflow.model_dump(mode="json")
    current_workflow.pop("max_inventory_rounds")
    parent_workflow = parent_config.workflow.model_dump(mode="json")
    parent_workflow.pop("max_inventory_rounds")
    if (
        current_certification_contract != parent_certification_contract
        or current_correction_contract != parent_correction_contract
        or config.correction.max_rounds_per_document
        != parent_config.correction.max_rounds_per_document
        or config.publication != parent_config.publication
        or current_workflow != parent_workflow
    ):
        raise ValueError("pipeline continuation stage behavior contract differs")

    parent_stage_hashes = parent_transaction.get("stageImplementationSha256")
    if not isinstance(parent_stage_hashes, dict):
        raise ValueError("pipeline continuation source lacks stage implementation hashes")
    current_stage_hashes = _stage_implementation_hashes()
    substantive_stages = tuple(name for name in current_stage_hashes if name != "config")
    mismatched_stages = tuple(
        name
        for name in substantive_stages
        if parent_stage_hashes.get(name) != current_stage_hashes[name]
    )
    if mismatched_stages:
        raise ValueError(
            "pipeline continuation implementation differs for: " + ", ".join(mismatched_stages)
        )
    if (
        parent_transaction.get("targetSchemaImplementationSha256")
        != _target_schema_implementation_sha256()
    ):
        raise ValueError("pipeline continuation target schema implementation differs")

    ancestor_pipeline = (
        _load_pipeline_resume_state(
            project_root=project_root,
            config=parent_config,
            initial_inventory=parent_inventory,
            _visited=visited,
        )
        if parent_config.pipeline_resume_run is not None
        else None
    )
    ancestor_inventory = (
        _load_inventory_resume_state(
            project_root=project_root,
            config=parent_config,
            initial_inventory=parent_inventory,
            _visited=visited,
        )
        if parent_config.inventory_resume_run is not None
        else None
    )

    lineage_path = parent_run.root / "lineage/cases.jsonl"
    lineage_payload = read_regular_file_bytes(lineage_path)
    lineage_sha256 = sha256_bytes(lineage_payload)
    if summary.get("caseLineageSha256") != lineage_sha256:
        raise ValueError("pipeline continuation case lineage differs from its summary")
    rows = cast(
        tuple[_PipelineResumeLineageRow, ...],
        _read_model_rows(lineage_path, _PipelineResumeLineageRow),
    )
    document_ids = tuple(row.documentId for row in rows)
    if len(rows) != config.workflow.documents or len(document_ids) != len(set(document_ids)):
        raise ValueError("pipeline continuation lineage differs from the unique cohort")

    reference_cache: dict[tuple[str, str, str], _RunReference] = {}

    def resolve_reference(value: _LineageRunReference) -> _RunReference:
        key = (value.path, value.commitSha256, value.transactionSha256)
        reference = reference_cache.get(key)
        if reference is None:
            reference = _pinned_run_reference(
                project_root=project_root,
                path=value.path,
                commit_sha256=value.commitSha256,
                transaction_sha256=value.transactionSha256,
                label="pipeline continuation component run",
            )
            reference_cache[key] = reference
        return reference

    histories_by_id = {
        row.documentId: [
            cast(
                dict[str, JsonValue],
                event.model_dump(mode="json", exclude_none=True),
            )
            for event in row.history
        ]
        for row in rows
    }
    if ancestor_pipeline is not None:
        if ancestor_pipeline.document_ids != document_ids:
            raise ValueError("pipeline continuation ancestry cohort differs")
        prefix_histories = ancestor_pipeline.histories_by_id
    elif ancestor_inventory is not None:
        if ancestor_inventory.document_ids != document_ids:
            raise ValueError("pipeline continuation inventory ancestry cohort differs")
        prefix_histories = ancestor_inventory.histories_by_id
    else:
        prefix_histories = {document_id: [] for document_id in document_ids}
    for row in rows:
        prefix = prefix_histories[row.documentId]
        current = histories_by_id[row.documentId]
        if current[: len(prefix)] != prefix:
            raise ValueError(
                f"pipeline continuation ancestry is not an exact history prefix: {row.documentId}"
            )
        if (
            ancestor_pipeline is not None
            and ancestor_pipeline.states_by_id[row.documentId].final_certification_run is not None
            and current != prefix
        ):
            raise ValueError(
                f"pipeline continuation retried an already certified case: {row.documentId}"
            )

    subruns = cast(
        tuple[_PipelineResumeSubrunRow, ...],
        _read_model_array(
            parent_run.root / "lineage/subruns.json",
            _PipelineResumeSubrunRow,
        ),
    )
    if summary.get("componentRuns") != len(subruns):
        raise ValueError("pipeline continuation subrun count differs from its summary")
    if any(row.stage == "publication" for row in subruns):
        raise ValueError("blocked pipeline continuation source contains a publication run")
    marker_stages = {"inventory_resume_parent", "pipeline_resume_parent"}
    markers = tuple(row for row in subruns if row.stage in marker_stages)
    expected_marker: _PipelineResumeSubrunRow | None = None
    if ancestor_pipeline is not None:
        expected_parent = ancestor_pipeline.parent_run
        expected_value = expected_parent.lineage_value()
        recorded = _read_json_object(parent_run.root / "inputs/pipeline-resume-run.json")
        if (
            len(markers) != 1
            or not subruns
            or markers[0] != subruns[0]
            or markers[0].stage != "pipeline_resume_parent"
            or markers[0].round != 0
            or markers[0].run.model_dump(mode="json") != expected_value
            or markers[0].documentIds != document_ids
            or markers[0].summarySha256 != sha256_file(expected_parent.root / "summary.json")
            or parent_transaction.get("pipelineResumeRun") != expected_value
            or parent_transaction.get("pipelineResumeLineageSha256")
            != ancestor_pipeline.lineage_sha256
            or recorded != expected_value
            or parent_transaction.get("inventoryResumeRun") is not None
            or parent_transaction.get("inventoryResumeLineageSha256") is not None
        ):
            raise ValueError("pipeline continuation parent marker provenance differs")
        expected_marker = markers[0]
    elif ancestor_inventory is not None:
        expected_parent = ancestor_inventory.parent_run
        expected_value = expected_parent.lineage_value()
        recorded = _read_json_object(parent_run.root / "inputs/inventory-resume-run.json")
        if (
            len(markers) != 1
            or not subruns
            or markers[0] != subruns[0]
            or markers[0].stage != "inventory_resume_parent"
            or markers[0].round != ancestor_inventory.prior_rounds
            or markers[0].run.model_dump(mode="json") != expected_value
            or markers[0].documentIds != document_ids
            or markers[0].summarySha256 != sha256_file(expected_parent.root / "summary.json")
            or parent_transaction.get("inventoryResumeRun") != expected_value
            or parent_transaction.get("inventoryResumeLineageSha256")
            != ancestor_inventory.lineage_sha256
            or recorded != expected_value
            or parent_transaction.get("pipelineResumeRun") is not None
            or parent_transaction.get("pipelineResumeLineageSha256") is not None
        ):
            raise ValueError("pipeline continuation inventory marker provenance differs")
        expected_marker = markers[0]
    elif (
        markers
        or parent_transaction.get("inventoryResumeRun") is not None
        or parent_transaction.get("inventoryResumeLineageSha256") is not None
        or parent_transaction.get("pipelineResumeRun") is not None
        or parent_transaction.get("pipelineResumeLineageSha256") is not None
    ):
        raise ValueError("pipeline continuation base run has unexpected resume provenance")

    inventory_events_by_round: dict[int, list[tuple[str, _InventoryHistoryRow]]] = defaultdict(list)
    appended_events_by_run: dict[tuple[str, str, str], list[tuple[str, _PipelineHistoryRow]]] = (
        defaultdict(list)
    )
    for row in rows:
        saw_non_inventory = False
        prefix_length = len(prefix_histories[row.documentId])
        for index, event in enumerate(row.history):
            if isinstance(event, _InventoryHistoryRow):
                if saw_non_inventory:
                    raise ValueError(
                        "pipeline continuation inventory history follows a terminal stage: "
                        f"{row.documentId}"
                    )
                inventory_events_by_round[event.round].append((row.documentId, event))
            else:
                saw_non_inventory = True
            if index >= prefix_length:
                run_value = event.run
                key = (
                    run_value.path,
                    run_value.commitSha256,
                    run_value.transactionSha256,
                )
                appended_events_by_run[key].append((row.documentId, event))

    if not inventory_events_by_round:
        raise ValueError("pipeline continuation lineage has no inventory history")
    inventory_rounds = tuple(sorted(inventory_events_by_round))
    if inventory_rounds != tuple(range(1, inventory_rounds[-1] + 1)):
        raise ValueError("pipeline continuation inventory rounds are not contiguous")
    inventory_contract_sha256 = _inventory_contract_sha256(parent_inventory)
    inventory_transaction_contract_sha256: str | None = None
    inventory_input_identities: dict[str, _InventoryInputIdentity] = {}
    states_by_id: dict[str, _CaseState] = {}
    expected_scope = document_ids
    for round_number in inventory_rounds:
        round_events = inventory_events_by_round[round_number]
        round_ids = tuple(document_id for document_id, _event in round_events)
        if round_ids != expected_scope:
            raise ValueError(
                "pipeline continuation inventory scope differs from prior non-ready cases"
            )
        run_values = {event.run for _document_id, event in round_events}
        if len(run_values) != 1:
            raise ValueError("pipeline continuation inventory round has multiple child runs")
        reference = resolve_reference(next(iter(run_values)))
        (
            inventory_results,
            child_transaction_contract_sha256,
            child_identities,
        ) = _validate_inventory_child_run(
            project_root=project_root,
            reference=reference,
            document_ids=round_ids,
            inventory_contract_sha256=inventory_contract_sha256,
            transaction_contract_sha256=inventory_transaction_contract_sha256,
            stage_hashes=current_stage_hashes,
        )
        if inventory_transaction_contract_sha256 is None:
            inventory_transaction_contract_sha256 = child_transaction_contract_sha256
        for (document_id, event), result in zip(round_events, inventory_results, strict=True):
            if (
                result.documentId != document_id
                or result.status != event.status
                or result.outputTextSha256 != event.candidateSha256
            ):
                raise ValueError(
                    "pipeline continuation inventory history differs from its child result: "
                    f"{document_id}"
                )
            identity = child_identities[document_id]
            prior_identity = inventory_input_identities.setdefault(document_id, identity)
            if prior_identity != identity:
                raise ValueError(
                    f"pipeline continuation inventory input identity changed: {document_id}"
                )
            if result.status == "training_ready":
                if document_id in states_by_id:
                    raise ValueError(
                        f"pipeline continuation inventory retried a ready case: {document_id}"
                    )
                source_sha256 = _case_artifact_sha256(reference, document_id, "source.txt")
                candidate_sha256 = _case_artifact_sha256(reference, document_id, "final.txt")
                states_by_id[document_id] = _CaseState(
                    document_id=document_id,
                    inventory_run=reference,
                    inventory_round=round_number,
                    candidate_run=reference,
                    source_text_sha256=source_sha256,
                    current_candidate_sha256=candidate_sha256,
                    candidate_hashes={candidate_sha256},
                )
        expected_scope = tuple(
            result.documentId for result in inventory_results if result.status != "training_ready"
        )
    if expected_scope or len(states_by_id) != len(document_ids):
        raise ValueError("pipeline continuation source does not have a ready inventory cohort")
    if inventory_transaction_contract_sha256 is None:
        raise AssertionError("pipeline continuation inventory contract was not established")
    if ancestor_inventory is not None and (
        ancestor_inventory.inventory_contract_sha256 != inventory_contract_sha256
        or ancestor_inventory.inventory_transaction_contract_sha256
        != inventory_transaction_contract_sha256
        or any(
            inventory_input_identities[document_id]
            != ancestor_inventory.input_identities_by_id[document_id]
            for document_id in document_ids
        )
    ):
        raise ValueError("pipeline continuation inventory ancestry contract differs")

    certification_events: dict[tuple[str, str, str], list[tuple[str, _CertificationHistoryRow]]] = (
        defaultdict(list)
    )
    correction_events: dict[tuple[str, str, str], list[tuple[str, _CorrectionHistoryRow]]] = (
        defaultdict(list)
    )
    for row in rows:
        for event in row.history:
            run_value = event.run
            key = (
                run_value.path,
                run_value.commitSha256,
                run_value.transactionSha256,
            )
            if isinstance(event, _CertificationHistoryRow):
                certification_events[key].append((row.documentId, event))
            elif isinstance(event, _CorrectionHistoryRow):
                correction_events[key].append((row.documentId, event))

    certification_results: dict[
        tuple[str, str, str],
        tuple[
            dict[str, CertificationCaseResult],
            _RunReference,
            Literal["contract.json", "source-contract.json"],
        ],
    ] = {}
    for cert_key, cert_events in certification_events.items():
        cert_reference = resolve_reference(cert_events[0][1].run)
        (
            cert_child_results,
            cert_input_run,
            cert_contract_filename,
        ) = _validate_certification_child_run(
            project_root=project_root,
            reference=cert_reference,
            resolve_reference=resolve_reference,
            expected_template_sha256=current_certification_contract,
            stage_hashes=current_stage_hashes,
        )
        cert_event_ids = tuple(document_id for document_id, _event in cert_events)
        if tuple(result.documentId for result in cert_child_results) != cert_event_ids:
            raise ValueError("pipeline continuation certification history differs from child scope")
        certification_results[cert_key] = (
            {result.documentId: result for result in cert_child_results},
            cert_input_run,
            cert_contract_filename,
        )

    correction_results: dict[
        tuple[str, str, str],
        tuple[dict[str, CorrectionCaseResult], _RunReference],
    ] = {}
    for correction_key, correction_run_events in correction_events.items():
        correction_reference = resolve_reference(correction_run_events[0][1].run)
        correction_child_results, correction_certification_run = _validate_correction_child_run(
            project_root=project_root,
            reference=correction_reference,
            resolve_reference=resolve_reference,
            expected_template_sha256=current_correction_contract,
            stage_hashes=current_stage_hashes,
        )
        correction_event_ids = tuple(document_id for document_id, _event in correction_run_events)
        if tuple(result.documentId for result in correction_child_results) != correction_event_ids:
            raise ValueError("pipeline continuation correction history differs from child scope")
        correction_results[correction_key] = (
            {result.documentId: result for result in correction_child_results},
            correction_certification_run,
        )

    certified_sources: dict[_RunReference, list[str]] = defaultdict(list)
    terminal_failures: dict[str, str] = {}
    for row in rows:
        state = states_by_id[row.documentId]
        state.history = list(histories_by_id[row.documentId])
        prefix_length = len(prefix_histories[row.documentId])
        derived_terminal_reason: str | None = None
        for event_index, event in enumerate(row.history):
            if isinstance(event, _InventoryHistoryRow):
                continue
            is_local_event = event_index >= prefix_length
            run_value = event.run
            key = (
                run_value.path,
                run_value.commitSha256,
                run_value.transactionSha256,
            )
            reference = resolve_reference(run_value)
            if isinstance(event, _CertificationHistoryRow):
                if state.final_certification_run is not None:
                    raise ValueError(
                        "pipeline continuation history follows final certification: "
                        f"{row.documentId}"
                    )
                if state.pending_correction_run is not None:
                    raise ValueError(
                        "pipeline continuation recertified before resolving an audit: "
                        f"{row.documentId}"
                    )
                cert_result_map, input_run, contract_filename = certification_results[key]
                cert_transition_result = cert_result_map[row.documentId]
                if (
                    input_run != state.candidate_run
                    or contract_filename != state.contract_filename
                    or cert_transition_result.sourceTextSha256 != state.source_text_sha256
                    or cert_transition_result.inputCandidateSha256 != state.current_candidate_sha256
                    or cert_transition_result.finalTextSha256 != state.current_candidate_sha256
                    or event.attempt != state.certification_attempts + 1
                    or event.status != cert_transition_result.status
                    or event.semanticFindings != cert_transition_result.semanticFindings
                    or event.hostAuditPassed != cert_transition_result.hostAudit.passed
                ):
                    raise ValueError(
                        f"pipeline continuation certification transition differs: {row.documentId}"
                    )
                if is_local_event:
                    state.local_certification_attempts += 1
                    if (
                        state.local_certification_attempts
                        > parent_config.certification.max_attempts_per_candidate
                    ):
                        raise ValueError(
                            "pipeline continuation parent exceeded its local "
                            "certification-attempt bound: "
                            f"{row.documentId}"
                        )
                for filename in ("source-label.json", "target-label.json"):
                    if _read_json_object(
                        reference.root / "cases" / row.documentId / filename
                    ) != _read_json_object(
                        state.candidate_run.root / "cases" / row.documentId / filename
                    ):
                        raise ValueError(
                            "pipeline continuation certification label handoff differs: "
                            f"{row.documentId}"
                        )
                if _read_json_object(
                    reference.root / "cases" / row.documentId / "source-contract.json"
                ) != _read_json_object(
                    state.candidate_run.root / "cases" / row.documentId / state.contract_filename
                ):
                    raise ValueError(
                        "pipeline continuation certification contract handoff differs: "
                        f"{row.documentId}"
                    )
                state.certification_attempts += 1
                if cert_transition_result.status == "certified":
                    state.final_certification_run = reference
                    certified_sources[reference].append(row.documentId)
                else:
                    invariant_audit = (
                        _certification_invariant_audit(reference, row.documentId)
                        if cert_transition_result.auditContractVersion == 3
                        else None
                    )
                    terminal_reason = (
                        _v3_certification_terminal_reason(
                            result=cert_transition_result,
                            invariant_audit=invariant_audit,
                            correction_rounds=state.correction_rounds,
                            correction_round_limit=(
                                parent_config.correction.max_rounds_per_document
                            ),
                        )
                        if invariant_audit is not None
                        else None
                    )
                    if terminal_reason is not None:
                        derived_terminal_reason = terminal_reason
                    elif _certification_is_actionable(
                        cert_transition_result,
                        invariant_audit,
                    ):
                        state.pending_correction_run = reference
            else:
                correction_result_map, certification_run = correction_results[key]
                correction_transition_result = correction_result_map[row.documentId]
                expected_attempt = state.correction_attempts + 1
                if (
                    state.final_certification_run is not None
                    or state.pending_correction_run is None
                    or state.correction_rounds >= parent_config.correction.max_rounds_per_document
                    or certification_run != state.pending_correction_run
                    or correction_transition_result.sourceTextSha256 != state.source_text_sha256
                    or correction_transition_result.inputCandidateSha256
                    != state.current_candidate_sha256
                    or event.round != state.correction_rounds + 1
                    or event.attempt not in (None, expected_attempt)
                    or event.status != correction_transition_result.status
                    or event.candidateSha256 != correction_transition_result.finalTextSha256
                ):
                    raise ValueError(
                        f"pipeline continuation correction transition differs: {row.documentId}"
                    )
                if is_local_event:
                    state.local_correction_attempts += 1
                    if (
                        state.local_correction_attempts
                        > parent_config.correction.max_attempts_per_round
                    ):
                        raise ValueError(
                            "pipeline continuation parent exceeded its local "
                            "correction-attempt bound: "
                            f"{row.documentId}"
                        )
                state.correction_attempts = expected_attempt
                if (
                    correction_transition_result.status == "correction_candidate"
                    and correction_transition_result.requiresRecertification
                ):
                    if correction_transition_result.finalTextSha256 in state.candidate_hashes:
                        if is_local_event and (
                            event_index != len(row.history) - 1
                            or row.blockingReason != _REPEATED_CANDIDATE_REASON
                        ):
                            raise ValueError(
                                "pipeline continuation parent continued after a repeated "
                                f"candidate: {row.documentId}"
                            )
                        derived_terminal_reason = _REPEATED_CANDIDATE_REASON
                        continue
                    state.candidate_hashes.add(correction_transition_result.finalTextSha256)
                    state.candidate_run = reference
                    state.current_candidate_sha256 = correction_transition_result.finalTextSha256
                    state.contract_filename = "source-contract.json"
                    state.certification_attempts = 0
                    state.correction_rounds += 1
                    if state.correction_rounds > parent_config.correction.max_rounds_per_document:
                        raise ValueError(
                            "pipeline continuation parent exceeded its "
                            "correction-round bound: "
                            f"{row.documentId}"
                        )
                    state.correction_attempts = 0
                    state.local_certification_attempts = 0
                    state.local_correction_attempts = 0
                    state.pending_correction_run = None
                else:
                    terminal_reason = _v3_correction_terminal_reason(correction_transition_result)
                    if terminal_reason is not None:
                        derived_terminal_reason = terminal_reason

        expected_final = (
            resolve_reference(row.finalCertificationRun)
            if row.finalCertificationRun is not None
            else None
        )
        if (
            row.inventoryRound != state.inventory_round
            or resolve_reference(row.inventoryRun) != state.inventory_run
            or row.sourceTextSha256 != state.source_text_sha256
            or row.currentCandidateSha256 != state.current_candidate_sha256
            or row.correctionRounds != state.correction_rounds
            or expected_final != state.final_certification_run
            or (row.status == "certified") != (state.final_certification_run is not None)
            or (row.blockingReason is None) != (row.status == "certified")
        ):
            raise ValueError(
                "pipeline continuation terminal case state differs from its history: "
                f"{row.documentId}"
            )
        if derived_terminal_reason is not None:
            if row.blockingReason != derived_terminal_reason:
                raise ValueError(
                    "pipeline continuation terminal reason differs from its deterministic "
                    f"state: {row.documentId}"
                )
            terminal_failures[row.documentId] = derived_terminal_reason

    local_subruns = tuple(row for row in subruns if row is not expected_marker)
    local_inventory_subruns = tuple(row for row in local_subruns if row.stage == "inventory")
    if ancestor_pipeline is not None:
        if local_inventory_subruns:
            raise ValueError("pipeline continuation parent reran inventory after a pipeline resume")
    else:
        prior_inventory_rounds = (
            ancestor_inventory.prior_rounds if ancestor_inventory is not None else 0
        )
        if (
            not local_inventory_subruns
            or len(local_inventory_subruns) > parent_config.workflow.max_inventory_rounds
            or local_subruns[: len(local_inventory_subruns)] != local_inventory_subruns
            or tuple(row.round for row in local_inventory_subruns)
            != tuple(
                range(
                    prior_inventory_rounds + 1,
                    prior_inventory_rounds + 1 + len(local_inventory_subruns),
                )
            )
            or any(row.shard != 1 for row in local_inventory_subruns)
        ):
            raise ValueError("pipeline continuation parent inventory subrun topology differs")
        first_inventory_reference = resolve_reference(local_inventory_subruns[0].run)
        expected_first_inventory_root = _ensure_under_project_root(
            project_root,
            resolve_config_path(project_root, parent_inventory.run.output_dir)
            / parent_inventory.run.run_id,
            label="pipeline continuation initial inventory root",
        )
        if first_inventory_reference.root != expected_first_inventory_root:
            raise ValueError(
                "pipeline continuation parent did not begin with its pinned inventory run"
            )
    local_subrun_keys: set[tuple[str, str, str]] = set()
    for subrun in local_subruns:
        key = (
            subrun.run.path,
            subrun.run.commitSha256,
            subrun.run.transactionSha256,
        )
        if key in local_subrun_keys:
            raise ValueError("pipeline continuation repeats a local component run")
        local_subrun_keys.add(key)
        local_events = appended_events_by_run.get(key)
        if not local_events:
            raise ValueError("pipeline continuation subrun has no appended case history")
        first_event = local_events[0][1]
        event_stage = first_event.stage
        if (
            subrun.stage != event_stage
            or subrun.documentIds != tuple(document_id for document_id, _event in local_events)
            or any(event.stage != event_stage for _document_id, event in local_events)
        ):
            raise ValueError("pipeline continuation subrun differs from appended history")
        if (
            event_stage == "certification"
            and len(subrun.documentIds) > parent_config.certification.shard_size
        ) or (
            event_stage == "correction"
            and len(subrun.documentIds) > parent_config.correction.shard_size
        ):
            raise ValueError("pipeline continuation parent component scope exceeds its shard bound")
        reference = resolve_reference(subrun.run)
        if subrun.summarySha256 != sha256_file(reference.root / "summary.json"):
            raise ValueError("pipeline continuation subrun summary identity differs")
        if isinstance(first_event, _InventoryHistoryRow):
            if not all(
                isinstance(event, _InventoryHistoryRow) for _document_id, event in local_events
            ):
                raise ValueError("pipeline continuation subrun mixes history stages")
            rounds = {
                event.round
                for _document_id, event in local_events
                if isinstance(event, _InventoryHistoryRow)
            }
            expected_round = next(iter(rounds)) if len(rounds) == 1 else -1
        elif isinstance(first_event, _CertificationHistoryRow):
            if not all(
                isinstance(event, _CertificationHistoryRow) for _document_id, event in local_events
            ):
                raise ValueError("pipeline continuation subrun mixes history stages")
            attempts = {
                event.attempt
                for _document_id, event in local_events
                if isinstance(event, _CertificationHistoryRow)
            }
            expected_round = next(iter(attempts)) if len(attempts) == 1 else -1
        else:
            if not all(
                isinstance(event, _CorrectionHistoryRow) for _document_id, event in local_events
            ):
                raise ValueError("pipeline continuation subrun mixes history stages")
            rounds = {
                event.round
                for _document_id, event in local_events
                if isinstance(event, _CorrectionHistoryRow)
            }
            expected_round = next(iter(rounds)) if len(rounds) == 1 else -1
        if subrun.round != expected_round:
            raise ValueError("pipeline continuation subrun round differs from case history")
        child_config_path = reference.root / "config.yaml"
        if event_stage == "inventory" and (reference.root.name == parent_inventory.run.run_id):
            recorded_config_path = parent_inventory_path
        else:
            recorded_config_path = (
                parent_run.root / "generated-configs" / f"{reference.root.name}.yaml"
            )
        if sha256_file(recorded_config_path) != sha256_file(child_config_path):
            raise ValueError("pipeline continuation generated child config differs")
    if local_subrun_keys != set(appended_events_by_run):
        raise ValueError("pipeline continuation appended history lacks a local subrun")

    certified = sum(state.final_certification_run is not None for state in states_by_id.values())
    blocked = len(document_ids) - certified
    primary_blocking = sum(row.blockingReason not in (None, _COHORT_STOP_REASON) for row in rows)
    if (
        summary.get("certifiedDocuments") != certified
        or summary.get("correctionRounds")
        != sum(state.correction_rounds for state in states_by_id.values())
        or summary.get("primaryBlockingDocuments") != primary_blocking
        or summary.get("blockedDocuments") != blocked
        or blocked <= 0
    ):
        raise ValueError("pipeline continuation summary differs from replayed state")

    for state in states_by_id.values():
        state.local_certification_attempts = 0
        state.local_correction_attempts = 0
    unresolved = tuple(
        document_id
        for document_id in document_ids
        if states_by_id[document_id].final_certification_run is None
    )
    return _PipelineResumeState(
        parent_run=parent_run,
        document_ids=document_ids,
        states_by_id=states_by_id,
        histories_by_id=histories_by_id,
        lineage_sha256=lineage_sha256,
        certified_sources=dict(certified_sources),
        unresolved=unresolved,
        terminal_failures=terminal_failures,
    )


def _record_subrun(
    records: list[dict[str, JsonValue]],
    *,
    stage: str,
    round_number: int,
    shard_number: int,
    reference: _RunReference,
    document_ids: tuple[str, ...],
) -> None:
    records.append(
        {
            "stage": stage,
            "round": round_number,
            "shard": shard_number,
            "run": cast(JsonValue, reference.lineage_value()),
            "documentIds": cast(JsonValue, list(document_ids)),
            "summarySha256": sha256_file(reference.root / "summary.json"),
        }
    )


def _validate_pipeline_paths(*, project_root: Path, config: SynthesisRawTextPipelineConfig) -> None:
    _ensure_under_project_root(
        project_root,
        resolve_config_path(project_root, config.run.output_dir),
        label="pipeline output directory",
    )


def _validate_pipeline_config_file(
    *, config_path: Path, config: SynthesisRawTextPipelineConfig
) -> None:
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("pipeline configuration must be a regular file")
    loaded_config = load_synthesis_raw_text_pipeline_config(config_path)
    if loaded_config != config:
        raise ValueError("pipeline configuration object differs from config_path")
    required_contract = 3 if loaded_config.schema_version == 2 else 2
    if (
        "audit_contract_version" not in loaded_config.certification.model_fields_set
        or loaded_config.certification.audit_contract_version != required_contract
        or "required_audit_contract_version"
        not in loaded_config.correction.workflow.model_fields_set
        or loaded_config.correction.workflow.required_audit_contract_version != required_contract
        or "required_audit_contract_version" not in loaded_config.publication.model_fields_set
        or loaded_config.publication.required_audit_contract_version != required_contract
    ):
        raise ValueError(
            f"pipeline schema v{loaded_config.schema_version} requires explicit matching "
            f"contract-v{required_contract} certification, correction, and publication"
        )


def _load_inventory_config(
    *,
    project_root: Path,
    config: SynthesisRawTextPipelineConfig,
) -> tuple[Path, SynthesisRawTextInventoryBatchConfig]:
    inventory_path = _resolve_pinned_file(
        project_root,
        config.inventory_config.path,
        config.inventory_config.sha256,
        label="raw-text pipeline inventory config",
    )
    inventory = load_synthesis_raw_text_inventory_batch_config(inventory_path)
    if (
        config.inventory_resume_run is None
        and config.pipeline_resume_run is None
        and inventory.workflow.documents != config.workflow.documents
    ):
        raise ValueError("pipeline inventory count differs from workflow.documents")
    inventory_output = _ensure_under_project_root(
        project_root,
        resolve_config_path(project_root, inventory.run.output_dir),
        label="inventory output directory",
    )
    pipeline_output = _ensure_under_project_root(
        project_root,
        resolve_config_path(project_root, config.run.output_dir),
        label="pipeline output directory",
    )
    if inventory_output / inventory.run.run_id == pipeline_output / config.run.run_id:
        raise ValueError("pipeline and inventory runs must use distinct artifact roots")
    generated_prefixes = (
        f"{config.run.run_id}-inventory-r",
        f"{config.run.run_id}-cert-",
        f"{config.run.run_id}-correct-",
    )
    if inventory_output == pipeline_output and (
        inventory.run.run_id == f"{config.run.run_id}-publication"
        or inventory.run.run_id.startswith(generated_prefixes)
    ):
        raise ValueError("inventory run collides with a generated pipeline child run")
    for label, pinned in (
        ("pipeline certification prompt", config.certification.prompt),
        ("pipeline correction prompt", config.correction.prompt),
    ):
        _resolve_pinned_file(
            project_root,
            pinned.path,
            pinned.sha256,
            label=label,
        )
    return inventory_path, inventory


def preflight_raw_text_pipeline(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisRawTextPipelineConfig,
) -> dict[str, JsonValue]:
    """Validate the full static contract and compile the inventory cohort without a provider."""

    project_root = project_root.resolve(strict=True)
    _validate_pipeline_config_file(config_path=config_path, config=config)
    _validate_pipeline_paths(project_root=project_root, config=config)
    inventory_path, inventory = _load_inventory_config(
        project_root=project_root,
        config=config,
    )
    pipeline_resume = _load_pipeline_resume_state(
        project_root=project_root,
        config=config,
        initial_inventory=inventory,
    )
    inventory_resume = (
        _load_inventory_resume_state(
            project_root=project_root,
            config=config,
            initial_inventory=inventory,
        )
        if pipeline_resume is None
        else None
    )
    if pipeline_resume is None:
        inventory_result = preflight_raw_text_inventory_batch(
            project_root=project_root,
            config_path=inventory_path,
            config=inventory,
        )
        if inventory_result.get("compiledDocuments") != inventory.workflow.documents:
            raise ValueError("pipeline preflight did not compile its complete inventory scope")
        resumed_inventory_documents = (
            len(inventory_resume.states_by_id) if inventory_resume is not None else 0
        )
        if resumed_inventory_documents + inventory.workflow.documents != config.workflow.documents:
            raise ValueError("pipeline preflight scopes do not cover the complete cohort")
        compiled_inventory_documents = inventory.workflow.documents
    else:
        resumed_inventory_documents = config.workflow.documents
        compiled_inventory_documents = 0
    return {
        "schemaVersion": config.schema_version,
        "runId": config.run.run_id,
        "status": "preflight_complete",
        "documents": config.workflow.documents,
        "resumedInventoryDocuments": resumed_inventory_documents,
        "compiledInventoryDocuments": compiled_inventory_documents,
        "resumedPipelineDocuments": (
            config.workflow.documents if pipeline_resume is not None else 0
        ),
        "resumedCertifiedDocuments": (
            config.workflow.documents - len(pipeline_resume.unresolved)
            if pipeline_resume is not None
            else 0
        ),
        "unresolvedPipelineDocuments": (
            len(pipeline_resume.unresolved) if pipeline_resume is not None else 0
        ),
        "terminalPipelineDocuments": (
            len(pipeline_resume.terminal_failures) if pipeline_resume is not None else 0
        ),
        "retryablePipelineDocuments": (
            len(pipeline_resume.unresolved) - len(pipeline_resume.terminal_failures)
            if pipeline_resume is not None
            else 0
        ),
        "certificationShardSize": config.certification.shard_size,
        "correctionShardSize": config.correction.shard_size,
        "providerSecretsLoaded": False,
        "providerRequests": 0,
    }


def _retry_inventory_config(
    *,
    base: SynthesisRawTextInventoryBatchConfig,
    run_id: str,
    output_dir: str,
    document_ids: tuple[str, ...],
) -> SynthesisRawTextInventoryBatchConfig:
    value = base.model_dump(mode="python")
    value["run"] = {"run_id": run_id, "output_dir": output_dir}
    value["selection"] = "explicit_pinned_document_ids"
    value["cases"] = [{"document_id": document_id} for document_id in document_ids]
    workflow = cast(dict[str, Any], value["workflow"])
    workflow["documents"] = len(document_ids)
    workflow["max_concurrent_documents"] = min(
        cast(int, workflow["max_concurrent_documents"]), len(document_ids)
    )
    return SynthesisRawTextInventoryBatchConfig.model_validate(value, strict=True)


def _certification_config(
    *,
    pipeline: SynthesisRawTextPipelineConfig,
    run_id: str,
    source: _RunReference,
    contract_filename: Literal["contract.json", "source-contract.json"],
    document_ids: tuple[str, ...],
) -> SynthesisRawTextCertificationConfig:
    workflow = pipeline.certification.workflow.model_dump(mode="python")
    workflow["documents"] = len(document_ids)
    workflow["max_concurrent_documents"] = min(
        cast(int, workflow["max_concurrent_documents"]), len(document_ids)
    )
    return SynthesisRawTextCertificationConfig.model_validate(
        {
            "schema_version": 3 if pipeline.certification.audit_contract_version == 3 else 2,
            "task": (
                "bill_of_lading_synthetic_raw_text_certification_v3"
                if pipeline.certification.audit_contract_version == 3
                else "bill_of_lading_synthetic_raw_text_certification_v2"
            ),
            "audit_contract_version": pipeline.certification.audit_contract_version,
            "invariant_inputs": (
                pipeline.certification.invariant_inputs.model_dump(mode="python")
                if pipeline.certification.invariant_inputs is not None
                else None
            ),
            "environment_file": pipeline.certification.environment_file,
            "run": {"run_id": run_id, "output_dir": pipeline.run.output_dir},
            "input_run": source.config_value(),
            "input_case_contract_filename": contract_filename,
            "case_ids": list(document_ids),
            "prompt": pipeline.certification.prompt.model_dump(mode="python"),
            "provider": pipeline.certification.provider.model_dump(mode="python"),
            "workflow": workflow,
        },
        strict=True,
    )


def _correction_config(
    *,
    pipeline: SynthesisRawTextPipelineConfig,
    run_id: str,
    certification: _RunReference,
    document_ids: tuple[str, ...],
) -> SynthesisRawTextCertifiedCorrectionConfig:
    workflow = pipeline.correction.workflow.model_dump(mode="python")
    workflow["documents"] = len(document_ids)
    workflow["max_concurrent_documents"] = min(
        cast(int, workflow["max_concurrent_documents"]), len(document_ids)
    )
    workflow["required_audit_contract_version"] = pipeline.certification.audit_contract_version
    return SynthesisRawTextCertifiedCorrectionConfig.model_validate(
        {
            "schema_version": 1,
            "task": "bill_of_lading_synthetic_raw_text_certified_correction_v1",
            "environment_file": pipeline.correction.environment_file,
            "run": {"run_id": run_id, "output_dir": pipeline.run.output_dir},
            "certification_run": certification.config_value(),
            "case_ids": list(document_ids),
            "prompt": pipeline.correction.prompt.model_dump(mode="python"),
            "provider": pipeline.correction.provider.model_dump(mode="python"),
            "workflow": workflow,
        },
        strict=True,
    )


def _publication_config(
    *,
    pipeline: SynthesisRawTextPipelineConfig,
    run_id: str,
    certified: dict[_RunReference, list[str]],
) -> SynthesisRawTextCertifiedPublicationConfig:
    sources = tuple(
        RawTextCertifiedPublicationSourceConfig.model_validate(
            {
                "run": reference.config_value(),
                "certified_documents": len(document_ids),
                "certified_document_ids_sha256": sha256_bytes(
                    canonical_json_bytes(sorted(document_ids))
                ),
            },
            strict=True,
        )
        for reference, document_ids in certified.items()
        if document_ids
    )
    return SynthesisRawTextCertifiedPublicationConfig.model_validate(
        {
            "schema_version": 1,
            "task": "bill_of_lading_synthetic_raw_text_certified_publication_v1",
            "run": {"run_id": run_id, "output_dir": pipeline.run.output_dir},
            "certification_sources": [row.model_dump(mode="json") for row in sources],
            "workflow": pipeline.publication.model_dump(mode="json"),
        },
        strict=True,
    )


def _certification_invariant_audit(
    reference: _RunReference,
    document_id: str,
) -> CertificationInvariantAudit:
    return CertificationInvariantAudit.model_validate_json(
        read_regular_file_bytes(reference.root / "cases" / document_id / "invariant-audit.json"),
        strict=True,
    )


def _certification_is_actionable(
    result: CertificationCaseResult,
    invariant_audit: CertificationInvariantAudit | None = None,
) -> bool:
    if result.status != "needs_review" or result.certificationMode != "production":
        return False
    if result.auditContractVersion == 2:
        if invariant_audit is not None:
            raise ValueError("contract-v2 actionability received contract-v3 invariants")
        return result.semanticFindings > 0 or not result.hostAudit.passed
    if invariant_audit is None:
        raise ValueError("contract-v3 actionability lacks its deterministic invariant audit")
    repairs = complete_deterministic_repair_set(invariant_audit)
    return (
        result.semanticFindings == 0
        and result.deterministicFindings == len(invariant_audit.findings)
        and bool(repairs)
    )


def _v3_certification_terminal_reason(
    *,
    result: CertificationCaseResult,
    invariant_audit: CertificationInvariantAudit,
    correction_rounds: int,
    correction_round_limit: int,
) -> str | None:
    """Return a stable terminal cause only when identical replay cannot make progress."""

    if result.auditContractVersion != 3 or result.status != "needs_review":
        return None
    if not _certification_is_actionable(result, invariant_audit):
        return _V3_CERTIFICATION_QUARANTINE_REASON
    if correction_rounds >= correction_round_limit:
        return _CORRECTION_ROUND_LIMIT_REASON
    return None


def _v3_correction_terminal_reason(result: CorrectionCaseResult) -> str | None:
    if result.correctionContractVersion != 3 or (
        result.status == "correction_candidate" and result.requiresRecertification
    ):
        return None
    return f"deterministic correction could not prove a complete atomic repair: {result.status}"


def _case_lineage(
    *,
    document_ids: tuple[str, ...],
    states_by_id: dict[str, _CaseState],
    inventory_histories: dict[str, list[dict[str, JsonValue]]],
    failures: dict[str, str],
) -> bytes:
    rows: list[dict[str, JsonValue]] = []
    for document_id in document_ids:
        state = states_by_id.get(document_id)
        rows.append(
            {
                "documentId": document_id,
                "status": "certified" if state and state.final_certification_run else "blocked",
                "inventoryRound": state.inventory_round if state is not None else None,
                "inventoryRun": (
                    cast(JsonValue, state.inventory_run.lineage_value())
                    if state is not None
                    else None
                ),
                "sourceTextSha256": (state.source_text_sha256 if state is not None else None),
                "currentCandidateSha256": (
                    state.current_candidate_sha256 if state is not None else None
                ),
                "correctionRounds": state.correction_rounds if state is not None else 0,
                "finalCertificationRun": (
                    cast(JsonValue, state.final_certification_run.lineage_value())
                    if state is not None and state.final_certification_run is not None
                    else None
                ),
                "blockingReason": failures.get(document_id),
                "history": cast(
                    JsonValue,
                    state.history
                    if state is not None
                    else inventory_histories.get(document_id, []),
                ),
            }
        )
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _pipeline_report(
    *,
    status: Literal["complete", "blocked"],
    documents: int,
    certified: int,
    subruns: int,
    failures: dict[str, str],
    publication: _RunReference | None,
) -> str:
    rows = [
        "# Synthetic raw-text production pipeline",
        "",
        "## Outcome",
        "",
        f"- Status: **{status}**.",
        f"- Independently certified: **{certified}/{documents}**.",
        f"- Committed component runs: **{subruns}**.",
        f"- Complete-cohort publication: **{'yes' if publication is not None else 'no'}**.",
    ]
    if publication is not None:
        rows.append(f"- Publication run: `{publication.path}`.")
    if failures:
        rows.extend(("", "## Blocking cases", ""))
        rows.extend(f"- `{document_id}`: {reason}" for document_id, reason in failures.items())
    rows.extend(
        (
            "",
            "Every component is an immutable staged run. Publication is gated on the exact "
            "original cohort, and corrected candidates are never accepted without a fresh "
            "independent certification.",
            "",
        )
    )
    return "\n".join(rows)


def _finish_pipeline(
    *,
    staged: StagedArtifactRun,
    config: SynthesisRawTextPipelineConfig,
    transaction: dict[str, JsonValue],
    document_ids: tuple[str, ...],
    states: tuple[_CaseState, ...],
    inventory_histories: dict[str, list[dict[str, JsonValue]]],
    subruns: list[dict[str, JsonValue]],
    failures: dict[str, str],
    publication: _RunReference | None,
) -> dict[str, JsonValue]:
    if len(document_ids) != config.workflow.documents or len(document_ids) != len(
        set(document_ids)
    ):
        raise RuntimeError("pipeline finalization scope is not the configured unique cohort")
    states_by_id = {state.document_id: state for state in states}
    if len(states_by_id) != len(states) or not set(states_by_id) <= set(document_ids):
        raise RuntimeError("pipeline state identities differ from the original cohort")
    if not set(failures) <= set(document_ids):
        raise RuntimeError("pipeline failures contain a case outside the original cohort")
    certified = sum(state.final_certification_run is not None for state in states)
    if publication is not None:
        if failures or certified != len(document_ids) or len(states) != len(document_ids):
            raise RuntimeError("pipeline cannot publish an incomplete or failed cohort")
        status: Literal["complete", "blocked"] = "complete"
        all_failures: dict[str, str] = {}
    else:
        if not failures:
            raise RuntimeError("blocked pipeline finalization requires an explicit cause")
        status = "blocked"
        all_failures = dict(failures)
        for document_id in document_ids:
            state = states_by_id.get(document_id)
            if state is None or state.final_certification_run is None:
                all_failures.setdefault(
                    document_id,
                    _COHORT_STOP_REASON,
                )
    lineage = _case_lineage(
        document_ids=document_ids,
        states_by_id=states_by_id,
        inventory_histories=inventory_histories,
        failures=all_failures,
    )
    staged.publish_bytes("lineage/cases.jsonl", lineage)
    staged.publish_json("lineage/subruns.json", subruns)
    staged.publish_json("provenance/transaction.json", transaction)
    summary: dict[str, JsonValue] = {
        "schemaVersion": config.schema_version,
        "runId": config.run.run_id,
        "status": status,
        "documents": config.workflow.documents,
        "inventoryReadyDocuments": len(states),
        "certifiedDocuments": certified,
        "correctionRounds": sum(state.correction_rounds for state in states),
        "componentRuns": len(subruns),
        "primaryBlockingDocuments": len(failures),
        "blockedDocuments": len(all_failures),
        "publicationRun": (
            cast(JsonValue, publication.lineage_value()) if publication is not None else None
        ),
        "trainingRecordsPublished": publication is not None,
        "caseLineageSha256": sha256_bytes(lineage),
    }
    staged.publish_json("summary.json", summary)
    staged.publish_bytes(
        "REPORT.md",
        _pipeline_report(
            status=status,
            documents=config.workflow.documents,
            certified=certified,
            subruns=len(subruns),
            failures=all_failures,
            publication=publication,
        ).encode(),
    )
    staged.commit(
        expected_artifacts=_artifact_inventory(staged.stage_root),
        metadata={
            "schemaVersion": config.schema_version,
            "status": status,
            "documents": config.workflow.documents,
            "certifiedDocuments": certified,
            "trainingRecordsPublished": publication is not None,
        },
    )
    output = dict(summary)
    output["artifactRoot"] = str(staged.final_root)
    output["commitSha256"] = sha256_file(staged.final_root / "_COMMIT.json")
    return output


def _raise_blocked(result: dict[str, JsonValue]) -> None:
    if result.get("status") == "blocked":
        raise RawTextPipelineBlockedError(
            "raw-text pipeline exhausted its bounded attempts without complete publication; "
            f"see {result.get('artifactRoot')}"
        )


def run_raw_text_pipeline(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisRawTextPipelineConfig,
) -> dict[str, JsonValue]:
    """Run the complete fail-closed raw-text synthesis and publication workflow."""

    project_root = project_root.resolve(strict=True)
    _validate_pipeline_config_file(config_path=config_path, config=config)
    _validate_pipeline_paths(project_root=project_root, config=config)
    output_parent = _ensure_under_project_root(
        project_root,
        resolve_config_path(project_root, config.run.output_dir),
        label="pipeline output directory",
    )
    inventory_path, initial_inventory = _load_inventory_config(
        project_root=project_root,
        config=config,
    )
    pipeline_resume = _load_pipeline_resume_state(
        project_root=project_root,
        config=config,
        initial_inventory=initial_inventory,
    )
    inventory_resume = (
        _load_inventory_resume_state(
            project_root=project_root,
            config=config,
            initial_inventory=initial_inventory,
        )
        if pipeline_resume is None
        else None
    )
    stage_implementation_sha256 = _stage_implementation_hashes()
    transaction: dict[str, JsonValue] = {
        "schemaVersion": config.schema_version,
        "runId": config.run.run_id,
        "configSha256": sha256_file(config_path),
        "inventoryConfigSha256": config.inventory_config.sha256,
        "inventoryResumeRun": (
            cast(JsonValue, inventory_resume.parent_run.lineage_value())
            if inventory_resume is not None
            else None
        ),
        "inventoryResumeLineageSha256": (
            inventory_resume.lineage_sha256 if inventory_resume is not None else None
        ),
        "pipelineResumeRun": (
            cast(JsonValue, pipeline_resume.parent_run.lineage_value())
            if pipeline_resume is not None
            else None
        ),
        "pipelineResumeLineageSha256": (
            pipeline_resume.lineage_sha256 if pipeline_resume is not None else None
        ),
        "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
        "stageImplementationSha256": cast(JsonValue, stage_implementation_sha256),
        "targetSchemaImplementationSha256": _target_schema_implementation_sha256(),
    }
    staged = StagedArtifactRun(
        output_parent=output_parent,
        run_name=config.run.run_id,
        transaction_sha256=sha256_bytes(canonical_json_bytes(transaction)),
    )
    if staged.completed:
        result = cast(dict[str, JsonValue], _read_json_object(staged.final_root / "summary.json"))
        result["artifactRoot"] = str(staged.final_root)
        result["commitSha256"] = sha256_file(staged.final_root / "_COMMIT.json")
        _raise_blocked(result)
        return result
    staged.recover_interrupted_temporary_files()
    staged.publish_bytes("config.yaml", read_regular_file_bytes(config_path))

    staged.publish_bytes("inputs/inventory-config.yaml", read_regular_file_bytes(inventory_path))
    if inventory_resume is not None:
        staged.publish_json(
            "inputs/inventory-resume-run.json",
            inventory_resume.parent_run.lineage_value(),
        )
    if pipeline_resume is not None:
        staged.publish_json(
            "inputs/pipeline-resume-run.json",
            pipeline_resume.parent_run.lineage_value(),
        )
    if pipeline_resume is None:
        preflight_raw_text_pipeline(
            project_root=project_root,
            config_path=config_path,
            config=config,
        )

    subruns: list[dict[str, JsonValue]] = []
    states_by_id: dict[str, _CaseState]
    inventory_histories: dict[str, list[dict[str, JsonValue]]]
    ordered_ids: tuple[str, ...]
    remaining: tuple[str, ...]
    inventory_input_identities: dict[str, _InventoryInputIdentity]
    inventory_transaction_contract_sha256: str | None
    if pipeline_resume is not None:
        states_by_id = dict(pipeline_resume.states_by_id)
        inventory_histories = defaultdict(
            list,
            {
                document_id: list(history)
                for document_id, history in pipeline_resume.histories_by_id.items()
            },
        )
        ordered_ids = pipeline_resume.document_ids
        remaining = tuple()
        prior_inventory_rounds = max(state.inventory_round for state in states_by_id.values())
        inventory_input_identities = {}
        inventory_contract_sha256 = _inventory_contract_sha256(initial_inventory)
        inventory_transaction_contract_sha256 = None
        _record_subrun(
            subruns,
            stage="pipeline_resume_parent",
            round_number=0,
            shard_number=1,
            reference=pipeline_resume.parent_run,
            document_ids=ordered_ids,
        )
    elif inventory_resume is None:
        states_by_id = {}
        inventory_histories = defaultdict(list)
        ordered_ids = ()
        remaining = ()
        prior_inventory_rounds = 0
        inventory_input_identities = {}
        inventory_contract_sha256 = _inventory_contract_sha256(initial_inventory)
        inventory_transaction_contract_sha256 = None
    else:
        states_by_id = dict(inventory_resume.states_by_id)
        inventory_histories = defaultdict(
            list,
            {
                document_id: list(history)
                for document_id, history in inventory_resume.histories_by_id.items()
            },
        )
        ordered_ids = inventory_resume.document_ids
        remaining = inventory_resume.remaining
        prior_inventory_rounds = inventory_resume.prior_rounds
        inventory_input_identities = dict(inventory_resume.input_identities_by_id)
        inventory_contract_sha256 = inventory_resume.inventory_contract_sha256
        inventory_transaction_contract_sha256 = (
            inventory_resume.inventory_transaction_contract_sha256
        )
        _record_subrun(
            subruns,
            stage="inventory_resume_parent",
            round_number=prior_inventory_rounds,
            shard_number=1,
            reference=inventory_resume.parent_run,
            document_ids=ordered_ids,
        )
    inventory_config = initial_inventory
    inventory_round_limit = (
        0 if pipeline_resume is not None else config.workflow.max_inventory_rounds
    )
    for local_inventory_round in range(1, inventory_round_limit + 1):
        inventory_round = prior_inventory_rounds + local_inventory_round
        if local_inventory_round > 1:
            run_id = f"{config.run.run_id}-inventory-r{inventory_round:02d}"
            inventory_config = _retry_inventory_config(
                base=initial_inventory,
                run_id=run_id,
                output_dir=config.run.output_dir,
                document_ids=remaining,
            )
            generated_path = _publish_generated_config(
                staged,
                relative_path=f"generated-configs/{run_id}.yaml",
                config=inventory_config,
            )
            preflight_raw_text_inventory_batch(
                project_root=project_root,
                config_path=generated_path,
                config=inventory_config,
            )
            active_path = generated_path
        else:
            active_path = inventory_path
        run_raw_text_inventory_batch(
            project_root=project_root,
            config_path=active_path,
            config=inventory_config,
        )
        reference = _committed_run_reference(
            project_root=project_root,
            run_id=inventory_config.run.run_id,
            output_dir=inventory_config.run.output_dir,
        )
        if inventory_resume is None and local_inventory_round == 1:
            expected_ids = (
                tuple(row.document_id for row in inventory_config.cases)
                if inventory_config.selection == "explicit_pinned_document_ids"
                else tuple(
                    row.documentId
                    for row in cast(
                        tuple[InventoryProbeCaseResult, ...],
                        _read_model_rows(
                            reference.root / "generation/results.jsonl",
                            InventoryProbeCaseResult,
                        ),
                    )
                )
            )
        else:
            expected_ids = remaining
        (
            inventory_results,
            child_transaction_contract_sha256,
            child_input_identities,
        ) = _validate_inventory_child_run(
            project_root=project_root,
            reference=reference,
            document_ids=expected_ids,
            inventory_contract_sha256=inventory_contract_sha256,
            transaction_contract_sha256=inventory_transaction_contract_sha256,
            stage_hashes=stage_implementation_sha256,
        )
        if inventory_transaction_contract_sha256 is None:
            inventory_transaction_contract_sha256 = child_transaction_contract_sha256
        for document_id, identity in child_input_identities.items():
            prior_identity = inventory_input_identities.setdefault(
                document_id,
                identity,
            )
            if prior_identity != identity:
                raise ValueError(f"inventory input identity changed across attempts: {document_id}")
        row_ids = tuple(row.documentId for row in inventory_results)
        if inventory_resume is None and local_inventory_round == 1:
            ordered_ids = row_ids
        _record_subrun(
            subruns,
            stage="inventory",
            round_number=inventory_round,
            shard_number=1,
            reference=reference,
            document_ids=row_ids,
        )
        for inventory_result in inventory_results:
            inventory_histories[inventory_result.documentId].append(
                {
                    "stage": "inventory",
                    "round": inventory_round,
                    "status": inventory_result.status,
                    "run": cast(JsonValue, reference.lineage_value()),
                    "candidateSha256": inventory_result.outputTextSha256,
                }
            )
            if inventory_result.status != "training_ready":
                continue
            if inventory_result.documentId in states_by_id:
                raise ValueError(
                    f"inventory retries repeated a ready case: {inventory_result.documentId}"
                )
            source_text_sha256 = _case_artifact_sha256(
                reference, inventory_result.documentId, "source.txt"
            )
            candidate_sha256 = _case_artifact_sha256(
                reference, inventory_result.documentId, "final.txt"
            )
            if candidate_sha256 != inventory_result.outputTextSha256:
                raise ValueError(
                    "inventory result candidate identity differs from its committed artifact: "
                    f"{inventory_result.documentId}"
                )
            state = _CaseState(
                document_id=inventory_result.documentId,
                inventory_run=reference,
                inventory_round=inventory_round,
                candidate_run=reference,
                source_text_sha256=source_text_sha256,
                current_candidate_sha256=candidate_sha256,
                candidate_hashes={inventory_result.outputTextSha256},
                history=list(inventory_histories[inventory_result.documentId]),
            )
            states_by_id[inventory_result.documentId] = state
        remaining = tuple(
            document_id for document_id in ordered_ids if document_id not in states_by_id
        )
        if not remaining:
            break

    ordered_states = tuple(states_by_id[row] for row in ordered_ids if row in states_by_id)
    if remaining:
        inventory_failures = {
            document_id: (
                "inventory did not produce a training-ready candidate within "
                f"{config.workflow.max_inventory_rounds} round(s)"
            )
            for document_id in remaining
        }
        result = _finish_pipeline(
            staged=staged,
            config=config,
            transaction=transaction,
            document_ids=ordered_ids,
            states=ordered_states,
            inventory_histories=inventory_histories,
            subruns=subruns,
            failures=inventory_failures,
            publication=None,
        )
        _raise_blocked(result)
        raise AssertionError("blocked pipeline did not raise")

    certified_sources: dict[_RunReference, list[str]] = defaultdict(list)
    if pipeline_resume is not None:
        for reference, certified_document_ids in pipeline_resume.certified_sources.items():
            certified_sources[reference].extend(certified_document_ids)
    terminal_failures: dict[str, str] = (
        dict(pipeline_resume.terminal_failures) if pipeline_resume is not None else {}
    )
    pending: dict[str, _CaseState] = {
        state.document_id: state
        for state in ordered_states
        if state.final_certification_run is None and state.document_id not in terminal_failures
    }
    certification_sequence = 0
    correction_sequence = 0
    while pending:
        before_signature = tuple(
            (
                state.document_id,
                state.candidate_run.path,
                state.certification_attempts,
                state.local_certification_attempts,
                state.correction_rounds,
                state.correction_attempts,
                state.local_correction_attempts,
                (
                    state.pending_correction_run.path
                    if state.pending_correction_run is not None
                    else None
                ),
            )
            for state in pending.values()
        )
        certification_groups: dict[tuple[_RunReference, str, int], list[_CaseState]] = defaultdict(
            list
        )
        actionable: list[tuple[_CaseState, _RunReference]] = []
        for state in tuple(pending.values()):
            if state.pending_correction_run is not None:
                if state.correction_rounds >= config.correction.max_rounds_per_document:
                    terminal_failures[state.document_id] = (
                        "independent audit still found actionable defects after the "
                        "configured correction-round limit"
                    )
                    del pending[state.document_id]
                else:
                    actionable.append((state, state.pending_correction_run))
                continue
            certification_groups[
                (state.candidate_run, state.contract_filename, state.certification_attempts)
            ].append(state)
        for (candidate_run, contract_filename, _attempt), grouped in certification_groups.items():
            for shard_number, shard in enumerate(
                _chunks(tuple(grouped), config.certification.shard_size), start=1
            ):
                certification_sequence += 1
                run_id = f"{config.run.run_id}-cert-{certification_sequence:04d}"
                document_ids = tuple(state.document_id for state in shard)
                certification_config = _certification_config(
                    pipeline=config,
                    run_id=run_id,
                    source=candidate_run,
                    contract_filename=cast(
                        Literal["contract.json", "source-contract.json"], contract_filename
                    ),
                    document_ids=document_ids,
                )
                child_path = _publish_generated_config(
                    staged,
                    relative_path=f"generated-configs/{run_id}.yaml",
                    config=certification_config,
                )
                for state in shard:
                    state.certification_attempts += 1
                    state.local_certification_attempts += 1
                run_raw_text_certification(
                    project_root=project_root,
                    config_path=child_path,
                    config=certification_config,
                )
                reference = _committed_run_reference(
                    project_root=project_root,
                    run_id=run_id,
                    output_dir=config.run.output_dir,
                )
                _record_subrun(
                    subruns,
                    stage="certification",
                    round_number=max(state.certification_attempts for state in shard),
                    shard_number=shard_number,
                    reference=reference,
                    document_ids=document_ids,
                )
                certification_results = cast(
                    tuple[CertificationCaseResult, ...],
                    _read_model_rows(
                        reference.root / "generation/results.jsonl", CertificationCaseResult
                    ),
                )
                if tuple(row.documentId for row in certification_results) != document_ids:
                    raise ValueError("certification results differ from their configured scope")
                for state, certification_result in zip(shard, certification_results, strict=True):
                    source_sha256 = _case_artifact_sha256(
                        reference, state.document_id, "source.txt"
                    )
                    input_sha256 = _case_artifact_sha256(
                        reference, state.document_id, "input-candidate.txt"
                    )
                    final_sha256 = _case_artifact_sha256(reference, state.document_id, "final.txt")
                    if (
                        source_sha256 != state.source_text_sha256
                        or input_sha256 != state.current_candidate_sha256
                        or final_sha256 != state.current_candidate_sha256
                        or certification_result.sourceTextSha256 != source_sha256
                        or certification_result.inputCandidateSha256 != input_sha256
                        or certification_result.finalTextSha256 != final_sha256
                    ):
                        raise ValueError(
                            "certification result identity differs from its candidate handoff: "
                            f"{state.document_id}"
                        )
                    state.history.append(
                        {
                            "stage": "certification",
                            "attempt": state.certification_attempts,
                            "status": certification_result.status,
                            "semanticFindings": certification_result.semanticFindings,
                            "hostAuditPassed": certification_result.hostAudit.passed,
                            "run": cast(JsonValue, reference.lineage_value()),
                        }
                    )
                    if certification_result.status == "certified":
                        state.final_certification_run = reference
                        certified_sources[reference].append(state.document_id)
                        del pending[state.document_id]
                    else:
                        invariant_audit = (
                            _certification_invariant_audit(reference, state.document_id)
                            if certification_result.auditContractVersion == 3
                            else None
                        )
                        terminal_reason = (
                            _v3_certification_terminal_reason(
                                result=certification_result,
                                invariant_audit=invariant_audit,
                                correction_rounds=state.correction_rounds,
                                correction_round_limit=(config.correction.max_rounds_per_document),
                            )
                            if invariant_audit is not None
                            else None
                        )
                        if terminal_reason is not None:
                            terminal_failures[state.document_id] = terminal_reason
                            del pending[state.document_id]
                        elif _certification_is_actionable(
                            certification_result,
                            invariant_audit,
                        ):
                            state.pending_correction_run = reference
                            actionable.append((state, reference))
                        elif (
                            state.local_certification_attempts
                            >= config.certification.max_attempts_per_candidate
                        ):
                            terminal_failures[state.document_id] = (
                                "independent certification did not return a valid actionable "
                                "audit within the configured attempt limit"
                            )
                            del pending[state.document_id]

        correction_groups: dict[tuple[_RunReference, int], list[_CaseState]] = defaultdict(list)
        for state, certification_run in actionable:
            if state.document_id not in pending:
                continue
            correction_groups[(certification_run, state.correction_rounds + 1)].append(state)
        for (certification_run, correction_round), grouped in correction_groups.items():
            for shard_number, shard in enumerate(
                _chunks(tuple(grouped), config.correction.shard_size), start=1
            ):
                correction_sequence += 1
                run_id = f"{config.run.run_id}-correct-{correction_sequence:04d}"
                document_ids = tuple(state.document_id for state in shard)
                correction_config = _correction_config(
                    pipeline=config,
                    run_id=run_id,
                    certification=certification_run,
                    document_ids=document_ids,
                )
                child_path = _publish_generated_config(
                    staged,
                    relative_path=f"generated-configs/{run_id}.yaml",
                    config=correction_config,
                )
                run_raw_text_certified_correction(
                    project_root=project_root,
                    config_path=child_path,
                    config=correction_config,
                )
                reference = _committed_run_reference(
                    project_root=project_root,
                    run_id=run_id,
                    output_dir=config.run.output_dir,
                )
                _record_subrun(
                    subruns,
                    stage="correction",
                    round_number=correction_round,
                    shard_number=shard_number,
                    reference=reference,
                    document_ids=document_ids,
                )
                correction_results = cast(
                    tuple[CorrectionCaseResult, ...],
                    _read_model_rows(
                        reference.root / "generation/results.jsonl", CorrectionCaseResult
                    ),
                )
                if tuple(row.documentId for row in correction_results) != document_ids:
                    raise ValueError("correction results differ from their configured scope")
                for state, correction_result in zip(shard, correction_results, strict=True):
                    source_sha256 = _case_artifact_sha256(
                        reference, state.document_id, "source.txt"
                    )
                    input_sha256 = _case_artifact_sha256(
                        reference, state.document_id, "input-candidate.txt"
                    )
                    final_sha256 = _case_artifact_sha256(reference, state.document_id, "final.txt")
                    if (
                        source_sha256 != state.source_text_sha256
                        or input_sha256 != state.current_candidate_sha256
                        or correction_result.sourceTextSha256 != source_sha256
                        or correction_result.inputCandidateSha256 != input_sha256
                        or correction_result.finalTextSha256 != final_sha256
                    ):
                        raise ValueError(
                            "correction result identity differs from its candidate handoff: "
                            f"{state.document_id}"
                        )
                    state.correction_attempts += 1
                    state.local_correction_attempts += 1
                    state.history.append(
                        {
                            "stage": "correction",
                            "round": correction_round,
                            "attempt": state.correction_attempts,
                            "status": correction_result.status,
                            "run": cast(JsonValue, reference.lineage_value()),
                            "candidateSha256": correction_result.finalTextSha256,
                        }
                    )
                    if (
                        correction_result.status != "correction_candidate"
                        or not correction_result.requiresRecertification
                    ):
                        terminal_reason = _v3_correction_terminal_reason(correction_result)
                        if terminal_reason is not None:
                            terminal_failures[state.document_id] = terminal_reason
                            del pending[state.document_id]
                        elif (
                            state.local_correction_attempts
                            >= config.correction.max_attempts_per_round
                        ):
                            terminal_failures[state.document_id] = (
                                "exact-evidence correction did not produce a locally valid "
                                "candidate within the configured attempt limit: "
                                f"{correction_result.status}"
                            )
                            del pending[state.document_id]
                        continue
                    if correction_result.finalTextSha256 in state.candidate_hashes:
                        terminal_failures[state.document_id] = _REPEATED_CANDIDATE_REASON
                        del pending[state.document_id]
                        continue
                    state.candidate_hashes.add(correction_result.finalTextSha256)
                    state.candidate_run = reference
                    state.current_candidate_sha256 = correction_result.finalTextSha256
                    state.contract_filename = "source-contract.json"
                    state.certification_attempts = 0
                    state.local_certification_attempts = 0
                    state.correction_rounds += 1
                    state.correction_attempts = 0
                    state.local_correction_attempts = 0
                    state.pending_correction_run = None
        after_signature = tuple(
            (
                state.document_id,
                state.candidate_run.path,
                state.certification_attempts,
                state.local_certification_attempts,
                state.correction_rounds,
                state.correction_attempts,
                state.local_correction_attempts,
                (
                    state.pending_correction_run.path
                    if state.pending_correction_run is not None
                    else None
                ),
            )
            for state in pending.values()
        )
        if pending and after_signature == before_signature:
            raise RuntimeError("raw-text pipeline state machine made no bounded progress")

    if terminal_failures:
        result = _finish_pipeline(
            staged=staged,
            config=config,
            transaction=transaction,
            document_ids=ordered_ids,
            states=ordered_states,
            inventory_histories=inventory_histories,
            subruns=subruns,
            failures=terminal_failures,
            publication=None,
        )
        _raise_blocked(result)
        raise AssertionError("blocked pipeline did not raise")

    certified_ids = {
        document_id for document_ids in certified_sources.values() for document_id in document_ids
    }
    if certified_ids != set(ordered_ids):
        raise RuntimeError("certification union differs from the original inventory cohort")
    publication_run_id = f"{config.run.run_id}-publication"
    publication_config = _publication_config(
        pipeline=config,
        run_id=publication_run_id,
        certified=certified_sources,
    )
    publication_path = _publish_generated_config(
        staged,
        relative_path=f"generated-configs/{publication_run_id}.yaml",
        config=publication_config,
    )
    publication_summary = run_raw_text_certified_publication(
        project_root=project_root,
        config_path=publication_path,
        config=publication_config,
    )
    if (
        publication_summary.get("documents") != config.workflow.documents
        or publication_summary.get("certifiedDocuments") != config.workflow.documents
        or publication_summary.get("trainingRecordsPublished") is not True
    ):
        raise RuntimeError("publication did not commit the exact complete certified cohort")
    publication_reference = _committed_run_reference(
        project_root=project_root,
        run_id=publication_run_id,
        output_dir=config.run.output_dir,
    )
    _record_subrun(
        subruns,
        stage="publication",
        round_number=1,
        shard_number=1,
        reference=publication_reference,
        document_ids=ordered_ids,
    )
    result = _finish_pipeline(
        staged=staged,
        config=config,
        transaction=transaction,
        document_ids=ordered_ids,
        states=ordered_states,
        inventory_histories=inventory_histories,
        subruns=subruns,
        failures={},
        publication=publication_reference,
    )
    return result
