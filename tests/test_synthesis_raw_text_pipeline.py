from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import pytest
import yaml

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.config import (
    SynthesisRawTextCertificationConfig,
    SynthesisRawTextCertifiedCorrectionConfig,
    SynthesisRawTextInventoryBatchConfig,
    SynthesisRawTextPipelineConfig,
    load_synthesis_raw_text_certification_config,
    load_synthesis_raw_text_certified_correction_config,
    load_synthesis_raw_text_certified_publication_config,
    load_synthesis_raw_text_inventory_batch_config,
    load_synthesis_raw_text_pipeline_config,
)
from document_ocr.synthesis.linguistic_probe_runtime import LinguisticUsageReceipt
from document_ocr.synthesis.raw_text_certification import (
    CertificationCaseResult,
    CertificationHostAudit,
)
from document_ocr.synthesis.raw_text_certified_correction import CorrectionCaseResult
from document_ocr.synthesis.raw_text_hybrid_probe import _artifact_inventory
from document_ocr.synthesis.raw_text_inventory_probe import InventoryProbeCaseResult
from document_ocr.synthesis.raw_text_pipeline import (
    RawTextPipelineBlockedError,
    run_raw_text_pipeline,
)
from document_ocr.synthesis.run_safety import StagedArtifactRun

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DOCUMENT_IDS = tuple(f"doc_{value * 64}" for value in ("1", "2", "3"))


def _empty_usage() -> LinguisticUsageReceipt:
    return LinguisticUsageReceipt(
        requests=0,
        providerResponseIds=(),
        finishReasons=(),
        inputTokens=0,
        cacheReadTokens=0,
        cacheWriteTokens=0,
        outputTokens=0,
        reasoningTokens=0,
        visibleOutputTokens=0,
        estimatedCostUsd=Decimal(0),
    )


def _write_pipeline_fixture(
    tmp_path: Path,
) -> tuple[Path, SynthesisRawTextPipelineConfig]:
    source_inventory = load_synthesis_raw_text_inventory_batch_config(
        _PROJECT_ROOT
        / "configs/synthesis/"
        "mpci_bl_raw_text_inventory100_additional_maritime_v55_pipeline_glm53.yaml"
    )
    inventory_value = source_inventory.model_dump(mode="python")
    inventory_value["run"] = {"run_id": "inventory-test", "output_dir": "runs"}
    inventory_value["selection"] = "explicit_pinned_document_ids"
    inventory_value["cases"] = [
        {"document_id": document_id} for document_id in _DOCUMENT_IDS
    ]
    inventory_workflow = inventory_value["workflow"]
    inventory_workflow["documents"] = 3
    inventory_workflow["max_concurrent_documents"] = 3
    inventory = SynthesisRawTextInventoryBatchConfig.model_validate(
        inventory_value, strict=True
    )
    inventory_path = tmp_path / "inventory.yaml"
    inventory_path.write_text(
        yaml.safe_dump(inventory.model_dump(mode="python"), sort_keys=False),
        encoding="utf-8",
    )

    certification_prompt = tmp_path / "certification.md"
    correction_prompt = tmp_path / "correction.md"
    certification_prompt.write_text("read-only audit", encoding="utf-8")
    correction_prompt.write_text("exact-line correction", encoding="utf-8")

    source_pipeline = load_synthesis_raw_text_pipeline_config(
        _PROJECT_ROOT
        / "configs/synthesis/mpci_bl_raw_text_pipeline100_additional_maritime_v1_glm53.yaml"
    )
    pipeline_value = source_pipeline.model_dump(mode="python")
    pipeline_value["run"] = {"run_id": "pipeline-test", "output_dir": "runs"}
    pipeline_value["inventory_config"] = {
        "path": "inventory.yaml",
        "sha256": sha256_bytes(inventory_path.read_bytes()),
    }
    pipeline_value["certification"]["shard_size"] = 2
    pipeline_value["certification"]["prompt"] = {
        "path": "certification.md",
        "sha256": sha256_bytes(certification_prompt.read_bytes()),
    }
    pipeline_value["certification"]["workflow"]["documents"] = 2
    pipeline_value["certification"]["workflow"]["max_concurrent_documents"] = 2
    pipeline_value["correction"]["shard_size"] = 2
    pipeline_value["correction"]["prompt"] = {
        "path": "correction.md",
        "sha256": sha256_bytes(correction_prompt.read_bytes()),
    }
    pipeline_value["correction"]["workflow"]["documents"] = 2
    pipeline_value["correction"]["workflow"]["max_concurrent_documents"] = 2
    pipeline_value["publication"]["documents"] = 3
    pipeline_value["workflow"]["documents"] = 3
    pipeline = SynthesisRawTextPipelineConfig.model_validate(pipeline_value, strict=True)
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        yaml.safe_dump(pipeline.model_dump(mode="python"), sort_keys=False),
        encoding="utf-8",
    )
    return pipeline_path, pipeline


def _commit_fake_run(
    *,
    project_root: Path,
    config: Any,
    rows: tuple[Any, ...],
    summary: dict[str, Any],
    case_texts: dict[str, tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    inventory_rows = bool(rows) and all(
        isinstance(row, InventoryProbeCaseResult) for row in rows
    )
    certification_rows = isinstance(config, SynthesisRawTextCertificationConfig)
    correction_rows = isinstance(config, SynthesisRawTextCertifiedCorrectionConfig)
    if inventory_rows:
        summary = {
            **summary,
            "schemaVersion": 2,
            "runId": config.run.run_id,
            "status": "complete",
            "liveDocuments": len(rows),
            "trainingReadyDocuments": sum(
                row.status == "training_ready" for row in rows
            ),
            "needsReviewDocuments": sum(row.status == "needs_review" for row in rows),
            "callFailedDocuments": sum(row.status == "call_failed" for row in rows),
            "compilerBlockedDocuments": sum(
                row.status == "compiler_blocked" for row in rows
            ),
        }
    elif certification_rows:
        summary = {
            **summary,
            "schemaVersion": 2,
            "runId": config.run.run_id,
            "status": "complete",
            "documents": len(rows),
            "certifiedDocuments": sum(row.status == "certified" for row in rows),
            "needsReviewDocuments": sum(row.status == "needs_review" for row in rows),
            "callFailedDocuments": sum(row.status == "call_failed" for row in rows),
            "semanticFindings": sum(row.semanticFindings for row in rows),
            "candidateBytesModified": 0,
            "trainingRecordsPublished": False,
        }
    elif correction_rows:
        summary = {
            **summary,
            "schemaVersion": 1,
            "runId": config.run.run_id,
            "status": "complete",
            "documents": len(rows),
            "unchangedCertifiedDocuments": sum(
                row.status == "unchanged_certified" for row in rows
            ),
            "correctionCandidateDocuments": sum(
                row.status == "correction_candidate" for row in rows
            ),
            "needsReviewDocuments": sum(row.status == "needs_review" for row in rows),
            "callFailedDocuments": sum(row.status == "call_failed" for row in rows),
            "sourceFindings": sum(row.sourceFindings for row in rows),
            "citedLines": sum(row.citedLines for row in rows),
            "changedLines": sum(row.changedLines for row in rows),
            "trainingRecordsPublished": False,
        }
    config_payload = yaml.safe_dump(
        config.model_dump(mode="python"),
        sort_keys=False,
    ).encode()
    if inventory_rows:
        import document_ocr.synthesis.raw_text_pipeline as pipeline_module

        stage_hashes = pipeline_module._stage_implementation_hashes()
        transaction = {
            "schemaVersion": 2,
            "runId": config.run.run_id,
            "configSha256": sha256_bytes(config_payload),
            "baseBatchConfigSha256": config.base_batch_config.sha256,
            "regressionOracleSha256": config.regression_oracle.sha256,
            "templateMutationProfileSha256": (
                config.template_mutation_profile.sha256
                if config.template_mutation_profile is not None
                else None
            ),
            "promptSha256": config.prompt.sha256,
            "referenceCommits": {
                "glm": config.reference_runs.glm.commit_sha256,
                "luna": config.reference_runs.luna.commit_sha256,
            },
            "implementationSha256": stage_hashes["inventory"],
            "inventoryImplementationSha256": stage_hashes["inventoryContract"],
            "hybridBatchImplementationSha256": stage_hashes["hybridBatch"],
            "hybridCompilerImplementationSha256": stage_hashes["hybridRuntime"],
            "providerRuntimeImplementationSha256": stage_hashes["providerRuntime"],
            "rewriteContractImplementationSha256": stage_hashes["rewriteContract"],
            "linguisticPlanSha256": "a" * 64,
            "sourceCorpusSha256": "b" * 64,
            "targetSha256": "c" * 64,
            "liveDocumentIds": [row.documentId for row in rows],
            "runtime": {
                "maxConcurrentDocuments": config.workflow.max_concurrent_documents,
                "maxProviderRouteRounds": config.workflow.max_provider_route_rounds,
                "model": config.provider.model,
                "openaiVersion": "test",
                "outputMode": config.workflow.output_mode,
                "providerOrder": list(config.provider.provider_order),
                "pydanticAiVersion": "test",
            },
        }
        transaction_sha256 = sha256_bytes(canonical_json_bytes(transaction))
    elif certification_rows:
        import document_ocr.synthesis.raw_text_pipeline as pipeline_module

        stage_hashes = pipeline_module._stage_implementation_hashes()
        transaction = {
            "schemaVersion": 2,
            "runId": config.run.run_id,
            "configSha256": sha256_bytes(config_payload),
            "inputRunCommitSha256": config.input_run.commit_sha256,
            "inputRunTransactionSha256": config.input_run.transaction_sha256,
            "inputCaseContractFilename": config.input_case_contract_filename,
            "promptSha256": config.prompt.sha256,
            "implementationSha256": stage_hashes["certification"],
            "dependencyImplementationSha256": {
                "providerRuntime": stage_hashes["providerRuntime"],
                "hybridRuntime": stage_hashes["hybridRuntime"],
                "inventory": stage_hashes["inventoryContract"],
                "inventoryRunner": stage_hashes["inventory"],
                "rewriteContract": stage_hashes["rewriteContract"],
                "rewriteRuntime": stage_hashes["rewriteRuntime"],
            },
            "documentIds": list(config.case_ids),
            "runtime": {
                "pydanticAiVersion": "test",
                "openaiVersion": "test",
                "model": config.provider.model,
                "providerOrder": list(config.provider.provider_order or ()),
                "maxConcurrentDocuments": config.workflow.max_concurrent_documents,
                "semanticAuditPasses": config.workflow.semantic_audit_passes,
            },
        }
        transaction_sha256 = sha256_bytes(canonical_json_bytes(transaction))
    elif correction_rows:
        import document_ocr.synthesis.raw_text_pipeline as pipeline_module

        stage_hashes = pipeline_module._stage_implementation_hashes()
        transaction = {
            "schemaVersion": 1,
            "runId": config.run.run_id,
            "configSha256": sha256_bytes(config_payload),
            "certificationRunCommitSha256": config.certification_run.commit_sha256,
            "certificationRunTransactionSha256": (
                config.certification_run.transaction_sha256
            ),
            "promptSha256": config.prompt.sha256,
            "implementationSha256": stage_hashes["correction"],
            "dependencyImplementationSha256": {
                "certification": stage_hashes["certification"],
                "providerRuntime": stage_hashes["providerRuntime"],
                "hybridRuntime": stage_hashes["hybridRuntime"],
                "inventory": stage_hashes["inventoryContract"],
                "inventoryRunner": stage_hashes["inventory"],
                "rewriteContract": stage_hashes["rewriteContract"],
                "rewriteRuntime": stage_hashes["rewriteRuntime"],
            },
            "documentIds": list(config.case_ids),
            "runtime": {
                "pydanticAiVersion": "test",
                "openaiVersion": "test",
                "model": config.provider.model,
                "providerOrder": list(config.provider.provider_order or ()),
                "maxConcurrentDocuments": config.workflow.max_concurrent_documents,
                "maxSuccessfulModelResponsesPerDocument": (
                    config.workflow.max_successful_model_responses_per_document
                ),
            },
        }
        transaction_sha256 = sha256_bytes(canonical_json_bytes(transaction))
    else:
        transaction = None
        transaction_sha256 = sha256_bytes(
            canonical_json_bytes(config.model_dump(mode="json"))
        )
    staged = StagedArtifactRun(
        output_parent=project_root / config.run.output_dir,
        run_name=config.run.run_id,
        transaction_sha256=transaction_sha256,
    )
    staged.publish_bytes("config.yaml", config_payload)
    if transaction is not None:
        staged.publish_json("provenance/transaction.json", transaction)
    staged.publish_json("summary.json", summary)
    staged.publish_bytes(
        "generation/results.jsonl",
        b"".join(canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in rows),
    )
    for document_id, (source, input_candidate, final) in (case_texts or {}).items():
        prefix = f"cases/{document_id}"
        staged.publish_bytes(f"{prefix}/source.txt", source.encode())
        staged.publish_bytes(f"{prefix}/input-candidate.txt", input_candidate.encode())
        staged.publish_bytes(f"{prefix}/final.txt", final.encode())
        fake_contract = {
            "documentId": document_id,
            "targetValueOccurrenceRequirements": [],
            "changedLeaves": [],
        }
        staged.publish_json(f"{prefix}/contract.json", fake_contract)
        if certification_rows or correction_rows:
            staged.publish_json(f"{prefix}/source-contract.json", fake_contract)
        if certification_rows or correction_rows:
            for filename in ("source-label.json", "target-label.json"):
                staged.publish_json(
                    f"{prefix}/{filename}",
                    {"artifact": filename, "documentId": document_id},
                )
        if inventory_rows:
            for filename in (
                "source-label.json",
                "target-label.json",
                "inventory.json",
                "deterministic-edits.json",
                "model-slots.json",
                "compound-slots.json",
                "editor-payload.json",
            ):
                staged.publish_json(
                    f"{prefix}/{filename}",
                    {"artifact": filename, "documentId": document_id},
                )
    for row in rows:
        staged.publish_json(
            f"cases/{row.documentId}/result.json",
            row.model_dump(mode="json"),
        )
    staged.commit(
        expected_artifacts=_artifact_inventory(staged.stage_root),
        metadata={"schemaVersion": 1, "documents": len(rows)},
    )
    return {
        **summary,
        "artifactRoot": str(staged.final_root),
    }


def _inventory_result(document_id: str) -> InventoryProbeCaseResult:
    return InventoryProbeCaseResult(
        documentId=document_id,
        status="training_ready",
        reason="all inventory gates passed",
        sourceLines=1,
        modelLines=1,
        compilerWorkItems=1,
        inventoryCandidates=1,
        deterministicInventoryEdits=0,
        hostRewriteAuditPassed=True,
        legacyResidualCandidates=0,
        outputTextSha256=sha256_bytes(_initial_candidate(document_id).encode()),
        usage=_empty_usage(),
    )


def _source_text(document_id: str) -> str:
    return f"source:{document_id}"


def _initial_candidate(document_id: str) -> str:
    return f"inventory:{document_id}"


def _corrected_candidate(document_id: str) -> str:
    return f"corrected:{document_id}"


def _certification_result(
    document_id: str,
    *,
    status: Literal["certified", "needs_review", "call_failed"],
    source: str,
    candidate: str,
) -> CertificationCaseResult:
    actionable = status == "needs_review"
    return CertificationCaseResult(
        documentId=document_id,
        status=status,
        reason=f"test {status}",
        auditPasses=1 if status != "call_failed" else 0,
        semanticFindings=1 if actionable else 0,
        candidateImmutable=True,
        hostAudit=CertificationHostAudit(
            passed=True,
            findings=(),
            sourceLines=1,
            outputLines=1,
            targetLiteralRequirements=0,
            targetOccurrenceRequirements=0,
        ),
        sourceTextSha256=sha256_bytes(source.encode()),
        inputCandidateSha256=sha256_bytes(candidate.encode()),
        finalTextSha256=sha256_bytes(candidate.encode()),
        usage=_empty_usage(),
    )


def _correction_result(
    document_id: str, *, source: str, current: str, final: str
) -> CorrectionCaseResult:
    return CorrectionCaseResult(
        documentId=document_id,
        status="correction_candidate",
        reason="local gates passed",
        citedLines=1,
        changedLines=1,
        sourceFindings=1,
        locallyResolvedFindings=1,
        requiresRecertification=True,
        sourceTextSha256=sha256_bytes(source.encode()),
        inputCandidateSha256=sha256_bytes(current.encode()),
        finalTextSha256=sha256_bytes(final.encode()),
        usage=_empty_usage(),
    )


def test_pipeline_shards_retries_corrects_recertifies_and_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import document_ocr.synthesis.raw_text_pipeline as pipeline_module

    config_path, config = _write_pipeline_fixture(tmp_path)
    certification_attempts: dict[str, int] = {}
    certification_configs = []
    correction_configs = []
    publication_configs = []

    monkeypatch.setattr(
        pipeline_module,
        "preflight_raw_text_pipeline",
        lambda **_kwargs: {"status": "preflight_complete"},
    )

    def run_inventory(*, project_root: Path, config: Any, **_kwargs: Any) -> dict[str, Any]:
        rows = tuple(_inventory_result(document_id) for document_id in _DOCUMENT_IDS)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=rows,
            summary={
                "status": "complete",
                "documents": 3,
                "trainingReadyDocuments": 3,
            },
            case_texts={
                document_id: (
                    _source_text(document_id),
                    _initial_candidate(document_id),
                    _initial_candidate(document_id),
                )
                for document_id in _DOCUMENT_IDS
            },
        )

    def run_certification(
        *, project_root: Path, config: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        certification_configs.append(config)
        results = []
        case_texts: dict[str, tuple[str, str, str]] = {}
        input_root = project_root / config.input_run.path
        for document_id in config.case_ids:
            certification_attempts[document_id] = certification_attempts.get(document_id, 0) + 1
            status: Literal["certified", "needs_review", "call_failed"]
            if document_id == _DOCUMENT_IDS[1] and certification_attempts[document_id] == 1:
                status = "needs_review"
            elif document_id == _DOCUMENT_IDS[2] and certification_attempts[document_id] == 1:
                status = "call_failed"
            else:
                status = "certified"
            source = (input_root / "cases" / document_id / "source.txt").read_text()
            candidate = (input_root / "cases" / document_id / "final.txt").read_text()
            results.append(
                _certification_result(
                    document_id,
                    status=status,
                    source=source,
                    candidate=candidate,
                )
            )
            case_texts[document_id] = (source, candidate, candidate)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=tuple(results),
            summary={"status": "complete", "documents": len(results)},
            case_texts=case_texts,
        )

    def run_correction(
        *, project_root: Path, config: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        correction_configs.append(config)
        input_root = project_root / config.certification_run.path
        case_texts: dict[str, tuple[str, str, str]] = {}
        rows = []
        for document_id in config.case_ids:
            source = (input_root / "cases" / document_id / "source.txt").read_text()
            current = (input_root / "cases" / document_id / "final.txt").read_text()
            final = _corrected_candidate(document_id)
            rows.append(
                _correction_result(
                    document_id,
                    source=source,
                    current=current,
                    final=final,
                )
            )
            case_texts[document_id] = (source, current, final)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=tuple(rows),
            summary={"status": "complete", "documents": len(rows)},
            case_texts=case_texts,
        )

    def run_publication(
        *, project_root: Path, config: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        publication_configs.append(config)
        result = _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=(),
            summary={
                "status": "complete",
                "documents": 3,
                "certifiedDocuments": 3,
                "trainingRecordsPublished": True,
            },
        )
        return result

    monkeypatch.setattr(pipeline_module, "run_raw_text_inventory_batch", run_inventory)
    monkeypatch.setattr(pipeline_module, "run_raw_text_certification", run_certification)
    monkeypatch.setattr(pipeline_module, "run_raw_text_certified_correction", run_correction)
    monkeypatch.setattr(pipeline_module, "run_raw_text_certified_publication", run_publication)

    result = run_raw_text_pipeline(
        project_root=tmp_path,
        config_path=config_path,
        config=config,
    )

    assert result["status"] == "complete"
    assert result["certifiedDocuments"] == 3
    assert result["trainingRecordsPublished"] is True
    assert result["correctionRounds"] == 1
    assert len(correction_configs) == 1
    assert correction_configs[0].case_ids == (_DOCUMENT_IDS[1],)
    corrected_certification = next(
        row for row in certification_configs if row.case_ids == (_DOCUMENT_IDS[1],)
    )
    assert corrected_certification.input_case_contract_filename == "source-contract.json"
    assert certification_attempts[_DOCUMENT_IDS[2]] == 2
    assert len(publication_configs) == 1
    assert sum(
        source.certified_documents
        for source in publication_configs[0].certification_sources
    ) == 3
    lineage = (Path(str(result["artifactRoot"])) / "lineage/cases.jsonl").read_text()
    assert '"stage":"correction"' in lineage
    assert '"stage":"certification"' in lineage
    generated_root = Path(str(result["artifactRoot"])) / "generated-configs"
    generated_paths = tuple(sorted(generated_root.iterdir()))
    assert generated_paths
    assert all(path.suffix == ".yaml" for path in generated_paths)
    for path in generated_paths:
        if "-correct-" in path.name:
            load_synthesis_raw_text_certified_correction_config(path)
        elif path.name.endswith("-publication.yaml"):
            load_synthesis_raw_text_certified_publication_config(path)
        else:
            load_synthesis_raw_text_certification_config(path)


def test_pipeline_commits_blocked_lineage_and_never_publishes_partial_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import document_ocr.synthesis.raw_text_pipeline as pipeline_module

    config_path, config = _write_pipeline_fixture(tmp_path)
    publication_called = False

    monkeypatch.setattr(
        pipeline_module,
        "preflight_raw_text_pipeline",
        lambda **_kwargs: {"status": "preflight_complete"},
    )

    def run_inventory(*, project_root: Path, config: Any, **_kwargs: Any) -> dict[str, Any]:
        rows = tuple(_inventory_result(document_id) for document_id in _DOCUMENT_IDS)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=rows,
            summary={"status": "complete", "documents": 3},
            case_texts={
                document_id: (
                    _source_text(document_id),
                    _initial_candidate(document_id),
                    _initial_candidate(document_id),
                )
                for document_id in _DOCUMENT_IDS
            },
        )

    def run_certification(
        *, project_root: Path, config: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        input_root = project_root / config.input_run.path
        case_texts: dict[str, tuple[str, str, str]] = {}
        rows = []
        for document_id in config.case_ids:
            source = (input_root / "cases" / document_id / "source.txt").read_text()
            candidate = (input_root / "cases" / document_id / "final.txt").read_text()
            rows.append(
                _certification_result(
                    document_id,
                    status="call_failed",
                    source=source,
                    candidate=candidate,
                )
            )
            case_texts[document_id] = (source, candidate, candidate)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=tuple(rows),
            summary={"status": "complete", "documents": len(rows)},
            case_texts=case_texts,
        )

    def run_publication(**_kwargs: Any) -> dict[str, Any]:
        nonlocal publication_called
        publication_called = True
        raise AssertionError("partial cohorts must not reach publication")

    monkeypatch.setattr(pipeline_module, "run_raw_text_inventory_batch", run_inventory)
    monkeypatch.setattr(pipeline_module, "run_raw_text_certification", run_certification)
    monkeypatch.setattr(pipeline_module, "run_raw_text_certified_publication", run_publication)

    with pytest.raises(RawTextPipelineBlockedError, match="without complete publication"):
        run_raw_text_pipeline(
            project_root=tmp_path,
            config_path=config_path,
            config=config,
        )

    assert publication_called is False
    summary = json.loads((tmp_path / "runs/pipeline-test/summary.json").read_text())
    assert summary["status"] == "blocked"
    assert summary["certifiedDocuments"] == 0
    assert summary["blockedDocuments"] == 3
    assert summary["trainingRecordsPublished"] is False


def test_pipeline_retries_only_unready_inventory_cases_and_records_complete_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import document_ocr.synthesis.raw_text_pipeline as pipeline_module

    config_path, config = _write_pipeline_fixture(tmp_path)
    inventory_scopes: list[tuple[str, ...]] = []
    certification_called = False

    monkeypatch.setattr(
        pipeline_module,
        "preflight_raw_text_pipeline",
        lambda **_kwargs: {"status": "preflight_complete"},
    )
    monkeypatch.setattr(
        pipeline_module,
        "preflight_raw_text_inventory_batch",
        lambda **_kwargs: {"status": "preflight_complete"},
    )

    def run_inventory(*, project_root: Path, config: Any, **_kwargs: Any) -> dict[str, Any]:
        document_ids = (
            tuple(row.document_id for row in config.cases)
            if config.selection == "explicit_pinned_document_ids"
            else _DOCUMENT_IDS
        )
        inventory_scopes.append(document_ids)
        rows = tuple(
            _inventory_result(document_id).model_copy(
                update={
                    "status": (
                        "needs_review" if document_id == _DOCUMENT_IDS[2] else "training_ready"
                    ),
                    "reason": "test inventory outcome",
                }
            )
            for document_id in document_ids
        )
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=rows,
            summary={"status": "complete", "documents": len(rows)},
            case_texts={
                document_id: (
                    _source_text(document_id),
                    _initial_candidate(document_id),
                    _initial_candidate(document_id),
                )
                for document_id in document_ids
            },
        )

    def run_certification(**_kwargs: Any) -> dict[str, Any]:
        nonlocal certification_called
        certification_called = True
        raise AssertionError("an incomplete inventory cohort must not be certified")

    monkeypatch.setattr(pipeline_module, "run_raw_text_inventory_batch", run_inventory)
    monkeypatch.setattr(pipeline_module, "run_raw_text_certification", run_certification)

    with pytest.raises(RawTextPipelineBlockedError, match="without complete publication"):
        run_raw_text_pipeline(
            project_root=tmp_path,
            config_path=config_path,
            config=config,
        )

    assert inventory_scopes == [
        _DOCUMENT_IDS,
        (_DOCUMENT_IDS[2],),
        (_DOCUMENT_IDS[2],),
    ]
    assert certification_called is False
    root = tmp_path / "runs/pipeline-test"
    summary = json.loads((root / "summary.json").read_text())
    assert summary["inventoryReadyDocuments"] == 2
    assert summary["primaryBlockingDocuments"] == 1
    assert summary["blockedDocuments"] == 3
    lineage = tuple(
        json.loads(line)
        for line in (root / "lineage/cases.jsonl").read_text().splitlines()
    )
    assert tuple(row["documentId"] for row in lineage) == _DOCUMENT_IDS
    assert len(lineage[2]["history"]) == 3
    assert "inventory did not produce" in lineage[2]["blockingReason"]
    retry_configs = tuple(sorted((root / "generated-configs").glob("*inventory*.yaml")))
    assert len(retry_configs) == 2
    assert all(
        load_synthesis_raw_text_inventory_batch_config(path).workflow.documents == 1
        for path in retry_configs
    )


def test_pipeline_inventory_resume_restores_ready_cases_and_retries_only_missing_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import document_ocr.synthesis.raw_text_pipeline as pipeline_module

    real_pipeline_preflight = pipeline_module.preflight_raw_text_pipeline
    config_path, config = _write_pipeline_fixture(tmp_path)
    initial_inventory_scopes: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        pipeline_module,
        "preflight_raw_text_pipeline",
        lambda **_kwargs: {"status": "preflight_complete"},
    )
    monkeypatch.setattr(
        pipeline_module,
        "preflight_raw_text_inventory_batch",
        lambda *, config, **_kwargs: {
            "status": "preflight_complete",
            "compiledDocuments": config.workflow.documents,
        },
    )

    def run_blocked_inventory(
        *, project_root: Path, config: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        document_ids = (
            tuple(row.document_id for row in config.cases)
            if config.selection == "explicit_pinned_document_ids"
            else _DOCUMENT_IDS
        )
        initial_inventory_scopes.append(document_ids)
        rows = tuple(
            _inventory_result(document_id).model_copy(
                update={
                    "status": (
                        "call_failed"
                        if document_id == _DOCUMENT_IDS[2]
                        else "training_ready"
                    ),
                    "reason": "test inventory outcome",
                }
            )
            for document_id in document_ids
        )
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=rows,
            summary={"status": "complete", "documents": len(rows)},
            case_texts={
                document_id: (
                    _source_text(document_id),
                    _initial_candidate(document_id),
                    _initial_candidate(document_id),
                )
                for document_id in document_ids
            },
        )

    monkeypatch.setattr(
        pipeline_module, "run_raw_text_inventory_batch", run_blocked_inventory
    )
    with pytest.raises(RawTextPipelineBlockedError, match="without complete publication"):
        run_raw_text_pipeline(
            project_root=tmp_path,
            config_path=config_path,
            config=config,
        )
    assert initial_inventory_scopes == [
        _DOCUMENT_IDS,
        (_DOCUMENT_IDS[2],),
        (_DOCUMENT_IDS[2],),
    ]

    parent_root = tmp_path / "runs/pipeline-test"
    parent_commit = json.loads((parent_root / "_COMMIT.json").read_text())
    inventory_value = load_synthesis_raw_text_inventory_batch_config(
        tmp_path / "inventory.yaml"
    ).model_dump(mode="python")
    inventory_value["run"] = {
        "run_id": "inventory-resume-test",
        "output_dir": "runs",
    }
    inventory_value["selection"] = "explicit_pinned_document_ids"
    inventory_value["cases"] = [{"document_id": _DOCUMENT_IDS[2]}]
    inventory_value["workflow"]["documents"] = 1
    inventory_value["workflow"]["max_concurrent_documents"] = 1
    resume_inventory = SynthesisRawTextInventoryBatchConfig.model_validate(
        inventory_value, strict=True
    )
    resume_inventory_path = tmp_path / "inventory-resume.yaml"
    resume_inventory_path.write_text(
        yaml.safe_dump(resume_inventory.model_dump(mode="python"), sort_keys=False),
        encoding="utf-8",
    )

    pipeline_value = config.model_dump(mode="python")
    pipeline_value["run"] = {"run_id": "pipeline-resumed", "output_dir": "runs"}
    pipeline_value["inventory_config"] = {
        "path": "inventory-resume.yaml",
        "sha256": sha256_bytes(resume_inventory_path.read_bytes()),
    }
    pipeline_value["inventory_resume_run"] = {
        "path": "runs/pipeline-test",
        "commit_sha256": sha256_bytes((parent_root / "_COMMIT.json").read_bytes()),
        "transaction_sha256": parent_commit["transaction_sha256"],
    }
    resume_config = SynthesisRawTextPipelineConfig.model_validate(
        pipeline_value, strict=True
    )
    resume_config_path = tmp_path / "pipeline-resumed.yaml"
    resume_config_path.write_text(
        yaml.safe_dump(resume_config.model_dump(mode="python"), sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pipeline_module,
        "preflight_raw_text_pipeline",
        real_pipeline_preflight,
    )
    preflight = real_pipeline_preflight(
        project_root=tmp_path,
        config_path=resume_config_path,
        config=resume_config,
    )
    assert preflight["documents"] == 3
    assert preflight["resumedInventoryDocuments"] == 2
    assert preflight["compiledInventoryDocuments"] == 1
    assert preflight["providerSecretsLoaded"] is False
    assert preflight["providerRequests"] == 0

    resumed_inventory_scopes: list[tuple[str, ...]] = []
    certification_scopes: list[tuple[str, ...]] = []
    publication_configs = []

    def run_resumed_inventory(
        *, project_root: Path, config: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        document_ids = tuple(row.document_id for row in config.cases)
        resumed_inventory_scopes.append(document_ids)
        rows = tuple(_inventory_result(document_id) for document_id in document_ids)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=rows,
            summary={"status": "complete", "documents": len(rows)},
            case_texts={
                document_id: (
                    _source_text(document_id),
                    _initial_candidate(document_id),
                    _initial_candidate(document_id),
                )
                for document_id in document_ids
            },
        )

    def run_certification(
        *, project_root: Path, config: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        certification_scopes.append(config.case_ids)
        input_root = project_root / config.input_run.path
        rows = []
        case_texts: dict[str, tuple[str, str, str]] = {}
        for document_id in config.case_ids:
            source = (input_root / "cases" / document_id / "source.txt").read_text()
            candidate = (input_root / "cases" / document_id / "final.txt").read_text()
            rows.append(
                _certification_result(
                    document_id,
                    status="certified",
                    source=source,
                    candidate=candidate,
                )
            )
            case_texts[document_id] = (source, candidate, candidate)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=tuple(rows),
            summary={"status": "complete", "documents": len(rows)},
            case_texts=case_texts,
        )

    def run_publication(
        *, project_root: Path, config: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        publication_configs.append(config)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=(),
            summary={
                "status": "complete",
                "documents": 3,
                "certifiedDocuments": 3,
                "trainingRecordsPublished": True,
            },
        )

    monkeypatch.setattr(
        pipeline_module, "run_raw_text_inventory_batch", run_resumed_inventory
    )
    monkeypatch.setattr(
        pipeline_module, "run_raw_text_certification", run_certification
    )
    monkeypatch.setattr(
        pipeline_module, "run_raw_text_certified_publication", run_publication
    )

    result = run_raw_text_pipeline(
        project_root=tmp_path,
        config_path=resume_config_path,
        config=resume_config,
    )

    assert result["status"] == "complete"
    assert result["inventoryReadyDocuments"] == 3
    assert result["certifiedDocuments"] == 3
    assert resumed_inventory_scopes == [(_DOCUMENT_IDS[2],)]
    assert sorted(certification_scopes) == [
        (_DOCUMENT_IDS[0], _DOCUMENT_IDS[1]),
        (_DOCUMENT_IDS[2],),
    ]
    assert len(publication_configs) == 1
    assert sum(
        source.certified_documents
        for source in publication_configs[0].certification_sources
    ) == 3
    resume_root = Path(str(result["artifactRoot"]))
    resume_input = json.loads(
        (resume_root / "inputs/inventory-resume-run.json").read_text()
    )
    assert resume_input["path"] == "runs/pipeline-test"
    lineage = tuple(
        json.loads(line)
        for line in (resume_root / "lineage/cases.jsonl").read_text().splitlines()
    )
    assert [row["inventoryRound"] for row in lineage] == [1, 1, 4]

    stage_hashes = pipeline_module._stage_implementation_hashes()
    monkeypatch.setattr(
        pipeline_module,
        "_stage_implementation_hashes",
        lambda: {**stage_hashes, "inventory": "0" * 64},
    )
    with pytest.raises(ValueError, match="implementation differs for: inventory"):
        real_pipeline_preflight(
            project_root=tmp_path,
            config_path=resume_config_path,
            config=resume_config,
        )

    monkeypatch.setattr(
        pipeline_module,
        "_stage_implementation_hashes",
        lambda: stage_hashes,
    )
    child_root = tmp_path / lineage[0]["history"][0]["run"]["path"]
    (child_root / "cases" / _DOCUMENT_IDS[0] / "result.json").unlink()
    with pytest.raises(RuntimeError, match="committed artifact inventory differs"):
        real_pipeline_preflight(
            project_root=tmp_path,
            config_path=resume_config_path,
            config=resume_config,
        )


def test_inventory_resume_explicitly_rejects_compiler_blocked_children(
    tmp_path: Path,
) -> None:
    import document_ocr.synthesis.raw_text_pipeline as pipeline_module

    _pipeline_path, _pipeline = _write_pipeline_fixture(tmp_path)
    value = load_synthesis_raw_text_inventory_batch_config(
        tmp_path / "inventory.yaml"
    ).model_dump(mode="python")
    value["run"] = {"run_id": "inventory-compiler-blocked", "output_dir": "runs"}
    value["cases"] = [{"document_id": _DOCUMENT_IDS[0]}]
    value["workflow"]["documents"] = 1
    value["workflow"]["max_concurrent_documents"] = 1
    inventory = SynthesisRawTextInventoryBatchConfig.model_validate(value, strict=True)
    result = _inventory_result(_DOCUMENT_IDS[0]).model_copy(
        update={
            "status": "compiler_blocked",
            "reason": "compiler rejected the source contract",
            "hostRewriteAuditPassed": False,
            "fullDocumentAudit": None,
            "compilerErrorType": "ValueError",
            "compilerErrorMessage": "unowned source fact",
        }
    )
    receipt = _commit_fake_run(
        project_root=tmp_path,
        config=inventory,
        rows=(result,),
        summary={"status": "complete", "documents": 1},
        case_texts={
            _DOCUMENT_IDS[0]: (
                _source_text(_DOCUMENT_IDS[0]),
                _initial_candidate(_DOCUMENT_IDS[0]),
                _source_text(_DOCUMENT_IDS[0]),
            )
        },
    )
    reference = pipeline_module._committed_run_reference(
        project_root=tmp_path,
        run_id=inventory.run.run_id,
        output_dir=inventory.run.output_dir,
    )

    with pytest.raises(ValueError, match="cannot reuse compiler-blocked cases"):
        pipeline_module._validate_inventory_child_run(
            project_root=tmp_path,
            reference=reference,
            document_ids=(_DOCUMENT_IDS[0],),
            inventory_contract_sha256=pipeline_module._inventory_contract_sha256(
                inventory
            ),
            transaction_contract_sha256=None,
            stage_hashes=pipeline_module._stage_implementation_hashes(),
        )

    assert receipt["status"] == "complete"


def test_pipeline_resume_replays_exact_state_and_chains_only_unresolved_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import document_ocr.synthesis.raw_text_pipeline as pipeline_module

    real_pipeline_preflight = pipeline_module.preflight_raw_text_pipeline
    config_path, config = _write_pipeline_fixture(tmp_path)
    phase = "parent"
    inventory_calls = 0
    certification_scopes: list[tuple[str, tuple[str, ...]]] = []
    correction_scopes: list[tuple[str, tuple[str, ...]]] = []
    final_resume_correction_calls = 0
    publication_configs = []

    monkeypatch.setattr(
        pipeline_module,
        "preflight_raw_text_pipeline",
        lambda **_kwargs: {"status": "preflight_complete"},
    )

    def run_inventory(
        *, project_root: Path, config: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        nonlocal inventory_calls
        inventory_calls += 1
        rows = tuple(_inventory_result(document_id) for document_id in _DOCUMENT_IDS)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=rows,
            summary={"status": "complete", "documents": 3},
            case_texts={
                document_id: (
                    _source_text(document_id),
                    _initial_candidate(document_id),
                    _initial_candidate(document_id),
                )
                for document_id in _DOCUMENT_IDS
            },
        )

    def run_certification(
        *, project_root: Path, config: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        certification_scopes.append((phase, config.case_ids))
        input_root = project_root / config.input_run.path
        results = []
        case_texts: dict[str, tuple[str, str, str]] = {}
        for document_id in config.case_ids:
            source = (input_root / "cases" / document_id / "source.txt").read_text()
            candidate = (input_root / "cases" / document_id / "final.txt").read_text()
            status: Literal["certified", "needs_review", "call_failed"]
            if (
                phase == "parent"
                and document_id != _DOCUMENT_IDS[0]
                and candidate.startswith("inventory:")
            ):
                status = "needs_review"
            else:
                status = "certified"
            results.append(
                _certification_result(
                    document_id,
                    status=status,
                    source=source,
                    candidate=candidate,
                )
            )
            case_texts[document_id] = (source, candidate, candidate)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=tuple(results),
            summary={"status": "complete", "documents": len(results)},
            case_texts=case_texts,
        )

    def run_correction(
        *, project_root: Path, config: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        nonlocal final_resume_correction_calls
        correction_scopes.append((phase, config.case_ids))
        input_root = project_root / config.certification_run.path
        rows = []
        case_texts: dict[str, tuple[str, str, str]] = {}
        for document_id in config.case_ids:
            source = (input_root / "cases" / document_id / "source.txt").read_text()
            current = (input_root / "cases" / document_id / "final.txt").read_text()
            succeeds = phase == "parent" and document_id == _DOCUMENT_IDS[1]
            if phase == "resume-final":
                final_resume_correction_calls += 1
                succeeds = final_resume_correction_calls == 2
            final = _corrected_candidate(document_id) if succeeds else current
            result = _correction_result(
                document_id,
                source=source,
                current=current,
                final=final,
            )
            if not succeeds:
                result = result.model_copy(
                    update={
                        "status": "call_failed",
                        "reason": "test transient route exhaustion",
                        "changedLines": 0,
                        "locallyResolvedFindings": 0,
                        "requiresRecertification": False,
                    }
                )
            rows.append(result)
            case_texts[document_id] = (source, current, final)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=tuple(rows),
            summary={"status": "complete", "documents": len(rows)},
            case_texts=case_texts,
        )

    def run_publication(
        *, project_root: Path, config: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        publication_configs.append(config)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=(),
            summary={
                "status": "complete",
                "documents": 3,
                "certifiedDocuments": 3,
                "trainingRecordsPublished": True,
            },
        )

    monkeypatch.setattr(pipeline_module, "run_raw_text_inventory_batch", run_inventory)
    monkeypatch.setattr(pipeline_module, "run_raw_text_certification", run_certification)
    monkeypatch.setattr(
        pipeline_module, "run_raw_text_certified_correction", run_correction
    )
    monkeypatch.setattr(
        pipeline_module, "run_raw_text_certified_publication", run_publication
    )

    with pytest.raises(RawTextPipelineBlockedError, match="without complete publication"):
        run_raw_text_pipeline(
            project_root=tmp_path,
            config_path=config_path,
            config=config,
        )
    assert inventory_calls == 1
    parent_root = tmp_path / "runs/pipeline-test"
    parent_summary = json.loads((parent_root / "summary.json").read_text())
    assert parent_summary["certifiedDocuments"] == 2
    assert parent_summary["primaryBlockingDocuments"] == 1
    assert parent_summary["blockedDocuments"] == 1
    parent_lineage = {
        row["documentId"]: row
        for row in (
            json.loads(line)
            for line in (parent_root / "lineage/cases.jsonl").read_text().splitlines()
        )
    }
    assert parent_lineage[_DOCUMENT_IDS[1]]["status"] == "certified"
    assert parent_lineage[_DOCUMENT_IDS[2]]["status"] == "blocked"
    assert "configured attempt limit" in parent_lineage[_DOCUMENT_IDS[2]][
        "blockingReason"
    ]

    def resume_config(
        *, parent_root: Path, run_id: str, correction_attempts: int
    ) -> tuple[Path, SynthesisRawTextPipelineConfig]:
        parent_receipt = json.loads((parent_root / "_COMMIT.json").read_text())
        value = config.model_dump(mode="python")
        value["run"] = {"run_id": run_id, "output_dir": "runs"}
        value["inventory_resume_run"] = None
        value["pipeline_resume_run"] = {
            "path": parent_root.relative_to(tmp_path).as_posix(),
            "commit_sha256": sha256_bytes((parent_root / "_COMMIT.json").read_bytes()),
            "transaction_sha256": parent_receipt["transaction_sha256"],
        }
        value["correction"]["max_attempts_per_round"] = correction_attempts
        resumed = SynthesisRawTextPipelineConfig.model_validate(value, strict=True)
        path = tmp_path / f"{run_id}.yaml"
        path.write_text(
            yaml.safe_dump(resumed.model_dump(mode="python"), sort_keys=False),
            encoding="utf-8",
        )
        return path, resumed

    phase = "resume-blocked"
    first_resume_path, first_resume = resume_config(
        parent_root=parent_root,
        run_id="pipeline-resume-one",
        correction_attempts=1,
    )
    monkeypatch.setattr(
        pipeline_module,
        "preflight_raw_text_pipeline",
        real_pipeline_preflight,
    )
    first_preflight = real_pipeline_preflight(
        project_root=tmp_path,
        config_path=first_resume_path,
        config=first_resume,
    )
    assert first_preflight["resumedPipelineDocuments"] == 3
    assert first_preflight["resumedCertifiedDocuments"] == 2
    assert first_preflight["unresolvedPipelineDocuments"] == 1
    assert first_preflight["compiledInventoryDocuments"] == 0
    assert first_preflight["providerRequests"] == 0
    with pytest.raises(RawTextPipelineBlockedError, match="without complete publication"):
        run_raw_text_pipeline(
            project_root=tmp_path,
            config_path=first_resume_path,
            config=first_resume,
        )
    assert inventory_calls == 1
    first_resume_root = tmp_path / "runs/pipeline-resume-one"

    phase = "resume-final"
    final_resume_path, final_resume = resume_config(
        parent_root=first_resume_root,
        run_id="pipeline-resume-two",
        correction_attempts=2,
    )
    final_preflight = real_pipeline_preflight(
        project_root=tmp_path,
        config_path=final_resume_path,
        config=final_resume,
    )
    assert final_preflight["resumedCertifiedDocuments"] == 2
    assert final_preflight["unresolvedPipelineDocuments"] == 1
    result = run_raw_text_pipeline(
        project_root=tmp_path,
        config_path=final_resume_path,
        config=final_resume,
    )

    assert result["status"] == "complete"
    assert result["certifiedDocuments"] == 3
    assert result["correctionRounds"] == 2
    assert result["trainingRecordsPublished"] is True
    assert inventory_calls == 1
    assert [scope for item_phase, scope in certification_scopes if item_phase != "parent"] == [
        (_DOCUMENT_IDS[2],),
    ]
    assert [scope for item_phase, scope in correction_scopes if item_phase == "resume-blocked"] == [
        (_DOCUMENT_IDS[2],)
    ]
    assert [scope for item_phase, scope in correction_scopes if item_phase == "resume-final"] == [
        (_DOCUMENT_IDS[2],),
        (_DOCUMENT_IDS[2],),
    ]
    assert len(publication_configs) == 1
    assert sum(
        source.certified_documents
        for source in publication_configs[0].certification_sources
    ) == 3
    final_lineage = tuple(
        json.loads(line)
        for line in (
            Path(str(result["artifactRoot"])) / "lineage/cases.jsonl"
        ).read_text().splitlines()
    )
    assert final_lineage[0]["history"][-1]["stage"] == "certification"
    assert final_lineage[1]["history"][-1]["stage"] == "certification"
    assert [
        row["attempt"]
        for row in final_lineage[2]["history"]
        if row["stage"] == "correction"
    ] == [1, 2, 3, 4]

    parent_certification_root = next(
        tmp_path / row["run"]["path"]
        for row in json.loads((parent_root / "lineage/subruns.json").read_text())
        if row["stage"] == "certification"
    )
    (parent_certification_root / "cases" / _DOCUMENT_IDS[0] / "result.json").unlink()
    with pytest.raises(RuntimeError, match="committed artifact inventory differs"):
        real_pipeline_preflight(
            project_root=tmp_path,
            config_path=final_resume_path,
            config=final_resume,
        )


def test_pipeline_config_rejects_a_publication_count_mismatch(tmp_path: Path) -> None:
    _config_path, config = _write_pipeline_fixture(tmp_path)
    value = config.model_dump(mode="python")
    value["publication"]["documents"] = 2

    with pytest.raises(ValueError, match="publication count differs"):
        SynthesisRawTextPipelineConfig.model_validate(value, strict=True)


def test_pipeline_config_rejects_two_resume_authorities(tmp_path: Path) -> None:
    _config_path, config = _write_pipeline_fixture(tmp_path)
    value = config.model_dump(mode="python")
    reference = {
        "path": "runs/prior",
        "commit_sha256": "a" * 64,
        "transaction_sha256": "b" * 64,
    }
    value["inventory_resume_run"] = reference
    value["pipeline_resume_run"] = reference

    with pytest.raises(ValueError, match="mutually exclusive"):
        SynthesisRawTextPipelineConfig.model_validate(value, strict=True)
