"""CLI for the non-generative, lossless synthesis foundation stage."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json, read_regular_file_bytes
from document_ocr.synthesis.config import (
    load_synthesis_cargo_language_probe_config,
    load_synthesis_controlled_pilot_config,
    load_synthesis_dangerous_goods_analysis_config,
    load_synthesis_dangerous_goods_plan_config,
    load_synthesis_dangerous_goods_registry_config,
    load_synthesis_deterministic_smoke_config,
    load_synthesis_foundation_config,
    load_synthesis_linguistic_completion_config,
    load_synthesis_linguistic_probe_analysis_config,
    load_synthesis_package_compatibility_catalog_config,
    load_synthesis_party_identity_probe_config,
    load_synthesis_party_structure_benchmark_config,
    load_synthesis_preparation_config,
    load_synthesis_raw_text_certification_config,
    load_synthesis_raw_text_certified_correction_config,
    load_synthesis_raw_text_certified_publication_config,
    load_synthesis_raw_text_hybrid_batch_config,
    load_synthesis_raw_text_hybrid_probe_config,
    load_synthesis_raw_text_inventory_batch_config,
    load_synthesis_raw_text_inventory_probe_config,
    load_synthesis_raw_text_pipeline_config,
    load_synthesis_raw_text_rewrite_cycle_probe_config,
    load_synthesis_raw_text_rewrite_probe_config,
    load_synthesis_route_scenario_pilot_config,
    load_synthesis_semantic_completion_config,
    load_synthesis_semantic_plan_config,
    load_synthesis_structured_baseline_config,
)
from document_ocr.synthesis.pipeline import prepare_synthesis_foundation
from document_ocr.synthesis.preparation import prepare_synthesis_corpus

_EXIT_INCOMPLETE = 5


def main() -> None:
    parser = argparse.ArgumentParser(prog="document-kie-synthesis")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "validate-config",
        "prepare-foundation",
        "validate-preparation-config",
        "prepare-corpus",
        "validate-deterministic-smoke-config",
        "run-deterministic-smoke",
        "validate-structured-baseline-config",
        "run-structured-baseline",
        "validate-route-scenario-pilot-config",
        "run-route-scenario-pilot",
        "validate-party-structure-benchmark-config",
        "run-party-structure-benchmark",
        "validate-controlled-pilot-config",
        "run-controlled-pilot",
        "validate-semantic-plan-config",
        "run-semantic-plan",
        "validate-dangerous-goods-registry-config",
        "build-dangerous-goods-registry",
        "validate-dangerous-goods-analysis-config",
        "run-dangerous-goods-analysis",
        "validate-dangerous-goods-plan-config",
        "run-dangerous-goods-plan",
        "validate-semantic-completion-config",
        "run-semantic-completion",
        "validate-package-compatibility-catalog-config",
        "run-package-compatibility-catalog",
        "validate-party-identity-probe-config",
        "run-party-identity-probe",
        "validate-cargo-language-probe-config",
        "run-cargo-language-probe",
        "validate-linguistic-probe-analysis-config",
        "run-linguistic-probe-analysis",
        "validate-linguistic-completion-config",
        "run-linguistic-completion",
        "validate-raw-text-rewrite-probe-config",
        "run-raw-text-rewrite-probe",
        "validate-raw-text-rewrite-cycle-probe-config",
        "run-raw-text-rewrite-cycle-probe",
        "validate-raw-text-hybrid-probe-config",
        "run-raw-text-hybrid-probe",
        "validate-raw-text-hybrid-batch-config",
        "run-raw-text-hybrid-batch",
        "validate-raw-text-inventory-probe-config",
        "run-raw-text-inventory-probe",
        "validate-raw-text-inventory-batch-config",
        "preflight-raw-text-inventory-batch",
        "run-raw-text-inventory-batch",
        "validate-raw-text-certification-config",
        "run-raw-text-certification",
        "validate-raw-text-certified-correction-config",
        "run-raw-text-certified-correction",
        "validate-raw-text-certified-publication-config",
        "run-raw-text-certified-publication",
        "validate-raw-text-pipeline-config",
        "preflight-raw-text-pipeline",
        "run-raw-text-pipeline",
        "validate-template-compilation-config",
        "preflight-template-compilation",
        "compile-raw-text-templates",
        "validate-template-catalog-join-config",
        "join-template-compilation-catalogs",
        "validate-compiled-descendant-eda-config",
        "analyze-compiled-descendant-run",
    ):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--project-root", type=Path, default=Path.cwd())
    readiness = commands.add_parser("audit-template-compilation-readiness")
    readiness.add_argument("--config", required=True, type=Path)
    readiness.add_argument("--project-root", type=Path, default=Path.cwd())
    readiness.add_argument("--checkpoint-root", action="append", type=Path, default=[])
    readiness.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    exit_code = 0
    try:
        project_root = arguments.project_root.resolve(strict=True)
        if arguments.config.is_symlink():
            raise ValueError("configuration must be a real file")
        config_path = arguments.config.resolve(strict=True)
        if not config_path.is_file():
            raise ValueError("configuration must be a real file")
        if arguments.command in {
            "validate-template-compilation-config",
            "preflight-template-compilation",
            "compile-raw-text-templates",
            "audit-template-compilation-readiness",
        }:
            from document_ocr.synthesis.template_compiler.pipeline import load_config

            template_compilation_config = load_config(config_path)
        elif arguments.command in {
            "validate-template-catalog-join-config",
            "join-template-compilation-catalogs",
        }:
            from document_ocr.synthesis.template_compiler.catalog_join import load_join_config

            template_catalog_join_config = load_join_config(config_path)
        elif arguments.command in {
            "validate-compiled-descendant-eda-config",
            "analyze-compiled-descendant-run",
        }:
            from document_ocr.synthesis.template_compiler.descendant_eda import load_eda_config

            descendant_eda_config = load_eda_config(config_path)
        elif arguments.command in {
            "validate-raw-text-pipeline-config",
            "preflight-raw-text-pipeline",
            "run-raw-text-pipeline",
        }:
            raw_text_pipeline_config = load_synthesis_raw_text_pipeline_config(config_path)
        elif arguments.command in {
            "validate-raw-text-certified-publication-config",
            "run-raw-text-certified-publication",
        }:
            raw_text_certified_publication_config = (
                load_synthesis_raw_text_certified_publication_config(config_path)
            )
        elif arguments.command in {
            "validate-raw-text-certified-correction-config",
            "run-raw-text-certified-correction",
        }:
            raw_text_certified_correction_config = (
                load_synthesis_raw_text_certified_correction_config(config_path)
            )
        elif arguments.command in {
            "validate-raw-text-certification-config",
            "run-raw-text-certification",
        }:
            raw_text_certification_config = load_synthesis_raw_text_certification_config(
                config_path
            )
        elif arguments.command in {
            "validate-raw-text-inventory-batch-config",
            "preflight-raw-text-inventory-batch",
            "run-raw-text-inventory-batch",
        }:
            raw_text_inventory_batch_config = load_synthesis_raw_text_inventory_batch_config(
                config_path
            )
        elif arguments.command in {
            "validate-raw-text-inventory-probe-config",
            "run-raw-text-inventory-probe",
        }:
            raw_text_inventory_config = load_synthesis_raw_text_inventory_probe_config(config_path)
        elif arguments.command in {
            "validate-raw-text-hybrid-batch-config",
            "run-raw-text-hybrid-batch",
        }:
            raw_text_hybrid_batch_config = load_synthesis_raw_text_hybrid_batch_config(config_path)
        elif arguments.command in {
            "validate-raw-text-hybrid-probe-config",
            "run-raw-text-hybrid-probe",
        }:
            raw_text_hybrid_config = load_synthesis_raw_text_hybrid_probe_config(config_path)
        elif arguments.command in {
            "validate-raw-text-rewrite-cycle-probe-config",
            "run-raw-text-rewrite-cycle-probe",
        }:
            raw_text_rewrite_cycle_config = load_synthesis_raw_text_rewrite_cycle_probe_config(
                config_path
            )
        elif arguments.command in {
            "validate-raw-text-rewrite-probe-config",
            "run-raw-text-rewrite-probe",
        }:
            raw_text_rewrite_config = load_synthesis_raw_text_rewrite_probe_config(config_path)
        elif arguments.command in {
            "validate-linguistic-completion-config",
            "run-linguistic-completion",
        }:
            linguistic_completion_config = load_synthesis_linguistic_completion_config(config_path)
        elif arguments.command in {
            "validate-linguistic-probe-analysis-config",
            "run-linguistic-probe-analysis",
        }:
            linguistic_probe_analysis_config = load_synthesis_linguistic_probe_analysis_config(
                config_path
            )
        elif arguments.command in {
            "validate-package-compatibility-catalog-config",
            "run-package-compatibility-catalog",
        }:
            package_compatibility_config = load_synthesis_package_compatibility_catalog_config(
                config_path
            )
        elif arguments.command in {
            "validate-cargo-language-probe-config",
            "run-cargo-language-probe",
        }:
            cargo_language_config = load_synthesis_cargo_language_probe_config(config_path)
        elif arguments.command in {
            "validate-party-identity-probe-config",
            "run-party-identity-probe",
        }:
            party_identity_config = load_synthesis_party_identity_probe_config(config_path)
        elif arguments.command in {
            "validate-semantic-completion-config",
            "run-semantic-completion",
        }:
            semantic_completion_config = load_synthesis_semantic_completion_config(config_path)
        elif arguments.command in {
            "validate-dangerous-goods-plan-config",
            "run-dangerous-goods-plan",
        }:
            dangerous_goods_plan_config = load_synthesis_dangerous_goods_plan_config(config_path)
        elif arguments.command in {
            "validate-dangerous-goods-analysis-config",
            "run-dangerous-goods-analysis",
        }:
            dangerous_goods_analysis_config = load_synthesis_dangerous_goods_analysis_config(
                config_path
            )
        elif arguments.command in {
            "validate-dangerous-goods-registry-config",
            "build-dangerous-goods-registry",
        }:
            dangerous_goods_registry_config = load_synthesis_dangerous_goods_registry_config(
                config_path
            )
        elif arguments.command in {
            "validate-semantic-plan-config",
            "run-semantic-plan",
        }:
            semantic_plan_config = load_synthesis_semantic_plan_config(config_path)
        elif arguments.command in {
            "validate-controlled-pilot-config",
            "run-controlled-pilot",
        }:
            controlled_config = load_synthesis_controlled_pilot_config(config_path)
        elif arguments.command in {
            "validate-party-structure-benchmark-config",
            "run-party-structure-benchmark",
        }:
            party_benchmark_config = load_synthesis_party_structure_benchmark_config(config_path)
        elif arguments.command in {
            "validate-route-scenario-pilot-config",
            "run-route-scenario-pilot",
        }:
            route_scenario_config = load_synthesis_route_scenario_pilot_config(config_path)
        elif arguments.command in {
            "validate-structured-baseline-config",
            "run-structured-baseline",
        }:
            structured_config = load_synthesis_structured_baseline_config(config_path)
        elif arguments.command in {
            "validate-deterministic-smoke-config",
            "run-deterministic-smoke",
        }:
            deterministic_config = load_synthesis_deterministic_smoke_config(config_path)
        elif arguments.command in {"validate-preparation-config", "prepare-corpus"}:
            preparation_config = load_synthesis_preparation_config(config_path)
        else:
            foundation_config = load_synthesis_foundation_config(config_path)
        if arguments.command == "validate-compiled-descendant-eda-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_name": descendant_eda_config.run_name,
                "documents": descendant_eda_config.expected_documents,
                "manual_reviews": descendant_eda_config.expected_manual_reviews,
                "descendant_run": descendant_eda_config.descendant_run.path,
            }
        elif arguments.command == "analyze-compiled-descendant-run":
            from document_ocr.synthesis.template_compiler.descendant_eda import (
                analyze_descendant_run,
            )

            artifact_root = analyze_descendant_run(
                project_root=project_root,
                config_path=config_path,
            )
            summary = json.loads(read_regular_file_bytes(artifact_root / "summary.json"))
            if not isinstance(summary, dict):
                raise ValueError("compiled descendant EDA summary must be an object")
            result = {
                "command": arguments.command,
                "status": "complete",
                "artifact": str(artifact_root),
                "documents": summary["documents"],
                "passed_documents": summary["passedDocuments"],
                "decision": summary["decision"],
                "scale_ready": summary["scaleReady"],
            }
        elif arguments.command == "validate-template-catalog-join-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_name": template_catalog_join_config.run_name,
                "source_runs": len(template_catalog_join_config.sources),
                "expected_source_outcomes": (
                    template_catalog_join_config.expected_source_outcomes
                ),
                "expected_certified_templates": (
                    template_catalog_join_config.expected_certified_templates
                ),
            }
        elif arguments.command == "join-template-compilation-catalogs":
            from document_ocr.synthesis.template_compiler.catalog_join import (
                join_template_catalogs,
            )

            artifact_root = join_template_catalogs(
                project_root=project_root,
                config_path=config_path,
            )
            summary = json.loads(read_regular_file_bytes(artifact_root / "summary.json"))
            if not isinstance(summary, dict):
                raise ValueError("template catalog join summary must be an object")
            result = {
                "command": arguments.command,
                "status": "complete",
                "artifact": str(artifact_root),
                "source_outcomes": summary["sourceOutcomes"],
                "certified_templates": summary["certifiedTemplates"],
                "review_required": summary["reviewRequired"],
            }
        elif arguments.command == "audit-template-compilation-readiness":
            from document_ocr.synthesis.template_compiler.readiness import (
                preflight_corpus_readiness,
            )

            if arguments.output.is_symlink():
                raise ValueError("readiness output must not be a symbolic link")
            output = (
                arguments.output
                if arguments.output.is_absolute()
                else project_root / arguments.output
            )
            output_parent = output.parent.resolve(strict=True)
            try:
                output_parent.relative_to(project_root)
            except ValueError as error:
                raise ValueError("readiness output escapes the project root") from error
            output = output_parent / output.name
            readiness_report = preflight_corpus_readiness(
                project_root=project_root,
                config_path=config_path,
                checkpoint_roots=tuple(arguments.checkpoint_root),
            )
            atomic_publish_json(output, readiness_report)
            result = {
                "command": arguments.command,
                "status": "complete",
                "artifact": str(output),
                "offline_gate_passed": readiness_report["offlineGatePassed"],
                "documents": readiness_report["deterministicallyPreflightedDocuments"],
                "checkpoint_replay": readiness_report["checkpointReplay"],
            }
        elif arguments.command == "validate-template-compilation-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_name": template_compilation_config.run_name,
                "documents": template_compilation_config.workflow.documents,
                "model": template_compilation_config.compiler_provider.model,
                "max_concurrent_documents": (
                    template_compilation_config.workflow.max_concurrent_documents
                ),
                "max_concurrent_requests": (
                    template_compilation_config.workflow.max_concurrent_requests
                ),
            }
        elif arguments.command == "preflight-template-compilation":
            from document_ocr.synthesis.template_compiler.pipeline import preflight_extraction

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": preflight_extraction(
                    project_root=project_root,
                    config_path=config_path,
                ),
            }
        elif arguments.command == "compile-raw-text-templates":
            from document_ocr.synthesis.template_compiler.pipeline import run_extraction

            artifact_root = asyncio.run(
                run_extraction(
                    project_root=project_root,
                    config_path=config_path,
                )
            )
            summary = json.loads(read_regular_file_bytes(artifact_root / "summary.json"))
            if not isinstance(summary, dict):
                raise ValueError("template compilation summary must be an object")
            acceptance_gate_passed = summary.get("acceptanceGatePassed")
            if not isinstance(acceptance_gate_passed, bool):
                raise ValueError(
                    "template compilation summary acceptanceGatePassed must be a boolean"
                )
            status_counts = summary.get("statusCounts")
            if not isinstance(status_counts, dict):
                raise ValueError("template compilation summary statusCounts must be an object")
            exit_code = 0 if acceptance_gate_passed else _EXIT_INCOMPLETE
            result = {
                "command": arguments.command,
                "status": "complete" if acceptance_gate_passed else "incomplete",
                "artifact": str(artifact_root),
                "acceptance_gate_passed": acceptance_gate_passed,
                "status_counts": status_counts,
            }
        elif arguments.command == "validate-raw-text-pipeline-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_name": raw_text_pipeline_config.run_name,
                "documents": raw_text_pipeline_config.workflow.documents,
                "template_run": raw_text_pipeline_config.inputs.template_run.path,
                "max_requests_per_document": (
                    raw_text_pipeline_config.workflow.max_requests_per_document
                ),
                "publishes_training_records": (
                    raw_text_pipeline_config.workflow.publish_training_records
                ),
            }
        elif arguments.command == "preflight-raw-text-pipeline":
            from document_ocr.synthesis.raw_text_pipeline import preflight_raw_text_pipeline

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": preflight_raw_text_pipeline(
                    project_root=project_root,
                    config_path=config_path,
                    config=raw_text_pipeline_config,
                ),
            }
        elif arguments.command == "run-raw-text-pipeline":
            from document_ocr.synthesis.raw_text_pipeline import run_raw_text_pipeline

            pipeline_result = run_raw_text_pipeline(
                project_root=project_root,
                config_path=config_path,
                config=raw_text_pipeline_config,
            )
            pipeline_status = pipeline_result.get("status")
            if pipeline_status not in {"passed", "failed"}:
                raise ValueError("raw-text pipeline summary status must be passed or failed")
            exit_code = 0 if pipeline_status == "passed" else _EXIT_INCOMPLETE
            result = {
                "command": arguments.command,
                "status": "complete" if pipeline_status == "passed" else "incomplete",
                "result": pipeline_result,
            }
        elif arguments.command == "validate-raw-text-certified-publication-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": raw_text_certified_publication_config.run.run_id,
                "documents": raw_text_certified_publication_config.workflow.documents,
                "certification_sources": len(
                    raw_text_certified_publication_config.certification_sources
                ),
            }
        elif arguments.command == "validate-raw-text-certified-correction-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": raw_text_certified_correction_config.run.run_id,
                "documents": raw_text_certified_correction_config.workflow.documents,
                "model": raw_text_certified_correction_config.provider.model,
                "provider_routes": (
                    list(raw_text_certified_correction_config.provider.provider_order or ())
                    if raw_text_certified_correction_config.provider.kind == "openrouter"
                    else []
                ),
            }
        elif arguments.command == "validate-raw-text-certification-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": raw_text_certification_config.run.run_id,
                "documents": raw_text_certification_config.workflow.documents,
                "semantic_audit_passes": (
                    raw_text_certification_config.workflow.semantic_audit_passes
                ),
                "model": raw_text_certification_config.provider.model,
                "provider_routes": (
                    list(raw_text_certification_config.provider.provider_order or ())
                    if raw_text_certification_config.provider.kind == "openrouter"
                    else []
                ),
            }
        elif arguments.command == "validate-raw-text-inventory-probe-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": raw_text_inventory_config.run.run_id,
                "regression_documents": (raw_text_inventory_config.workflow.regression_documents),
                "live_documents": len(raw_text_inventory_config.cases),
                "model": raw_text_inventory_config.provider.model,
                "output_mode": raw_text_inventory_config.workflow.output_mode,
                "max_model_requests_per_document": (
                    raw_text_inventory_config.workflow.max_model_requests_per_document
                ),
            }
        elif arguments.command == "validate-raw-text-hybrid-batch-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": raw_text_hybrid_batch_config.run.run_id,
                "cases": raw_text_hybrid_batch_config.inputs.synthetic_targets.records,
                "max_concurrent_cases": (
                    raw_text_hybrid_batch_config.workflow.max_concurrent_cases
                ),
                "editor_model": raw_text_hybrid_batch_config.providers.editor.model,
                "reviewer_model": raw_text_hybrid_batch_config.providers.reviewer.model,
            }
        elif arguments.command == "validate-raw-text-hybrid-probe-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": raw_text_hybrid_config.run.run_id,
                "audit_documents": raw_text_hybrid_config.workflow.audit_documents,
                "model_probe_documents": len(raw_text_hybrid_config.cases),
                "editor_model": raw_text_hybrid_config.providers.editor.model,
                "reviewer_model": raw_text_hybrid_config.providers.reviewer.model,
            }
        elif arguments.command == "validate-raw-text-rewrite-cycle-probe-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": raw_text_rewrite_cycle_config.run.run_id,
                "cases": len(raw_text_rewrite_cycle_config.cases),
                "editor_model": raw_text_rewrite_cycle_config.providers.editor.model,
                "editor_reasoning_effort": (
                    raw_text_rewrite_cycle_config.providers.editor.reasoning_effort
                ),
                "reviewer_model": raw_text_rewrite_cycle_config.providers.reviewer.model,
                "reviewer_reasoning_effort": (
                    raw_text_rewrite_cycle_config.providers.reviewer.reasoning_effort
                ),
                "max_correction_cycles": (
                    raw_text_rewrite_cycle_config.workflow.max_correction_cycles
                ),
            }
        elif arguments.command == "validate-raw-text-rewrite-probe-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": raw_text_rewrite_config.run.run_id,
                "cases": len(raw_text_rewrite_config.cases),
                "model": raw_text_rewrite_config.provider.model,
                "reasoning_effort": raw_text_rewrite_config.provider.reasoning_effort,
                "max_concurrent_requests": (
                    raw_text_rewrite_config.workflow.max_concurrent_requests
                ),
                "max_concurrent_cases": (raw_text_rewrite_config.workflow.max_concurrent_cases),
            }
        elif arguments.command == "validate-linguistic-completion-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": linguistic_completion_config.run.run_id,
                "records": linguistic_completion_config.inputs.completion_plans.records,
                "model": linguistic_completion_config.provider.model,
                "reasoning_effort": linguistic_completion_config.provider.reasoning_effort,
                "max_concurrent_requests": (
                    linguistic_completion_config.workflow.max_concurrent_requests
                ),
                "max_concurrent_documents": (
                    linguistic_completion_config.workflow.max_concurrent_documents
                ),
                "generation_settings": (
                    linguistic_completion_config.provider.generation_settings.model_dump(
                        mode="json", exclude_none=True
                    )
                    if linguistic_completion_config.provider.generation_settings is not None
                    else None
                ),
            }
        elif arguments.command == "validate-linguistic-probe-analysis-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": linguistic_probe_analysis_config.run.run_id,
                "party_cases": linguistic_probe_analysis_config.party.results.records,
                "cargo_cases": linguistic_probe_analysis_config.cargo.results.records,
            }
        elif arguments.command == "validate-package-compatibility-catalog-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": package_compatibility_config.run.run_id,
                "records": package_compatibility_config.inputs.dangerous_goods_plans.records,
                "model": package_compatibility_config.provider.model,
                "reasoning_effort": package_compatibility_config.provider.reasoning_effort,
            }
        elif arguments.command == "validate-cargo-language-probe-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": cargo_language_config.run.run_id,
                "cases": len(cargo_language_config.cases),
                "model": cargo_language_config.provider.model,
                "reasoning_effort": cargo_language_config.provider.reasoning_effort,
                "structured_output_retries": (
                    cargo_language_config.workflow.structured_output_retries
                ),
            }
        elif arguments.command == "validate-party-identity-probe-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": party_identity_config.run.run_id,
                "cases": len(party_identity_config.cases),
                "model": party_identity_config.provider.model,
                "reasoning_effort": party_identity_config.provider.reasoning_effort,
                "structured_output_retries": (
                    party_identity_config.workflow.structured_output_retries
                ),
            }
        elif arguments.command == "validate-semantic-completion-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": semantic_completion_config.run.run_id,
                "records": semantic_completion_config.inputs.dangerous_goods_plans.records,
                "target_task": semantic_completion_config.target_task,
            }
        elif arguments.command == "validate-dangerous-goods-plan-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": dangerous_goods_plan_config.run.run_id,
                "records": dangerous_goods_plan_config.inputs.semantic_plans.records,
            }
        elif arguments.command == "validate-dangerous-goods-analysis-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": dangerous_goods_analysis_config.run.run_id,
                "hmt_records": dangerous_goods_analysis_config.registry.hmt_records.records,
                "ecics_links": dangerous_goods_analysis_config.registry.ecics_links.records,
            }
        elif arguments.command == "validate-dangerous-goods-registry-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": dangerous_goods_registry_config.run.run_id,
            }
        elif arguments.command == "validate-semantic-plan-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": semantic_plan_config.run.run_id,
                "requested_documents": (semantic_plan_config.inputs.structured_selection.records),
                "dangerous_goods_policy": (semantic_plan_config.generation.dangerous_goods_policy),
            }
        elif arguments.command == "validate-controlled-pilot-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": controlled_config.run.run_id,
                "requested_documents": controlled_config.selection.requested_documents,
                "vessel_name_method": controlled_config.generation.transport.vessel_name_method,
                "voyage_number_method": (
                    controlled_config.generation.transport.voyage_number_method
                ),
            }
        elif arguments.command == "validate-party-structure-benchmark-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": party_benchmark_config.run.run_id,
                "scope": party_benchmark_config.modeling.scope,
                "candidates": list(party_benchmark_config.modeling.candidates),
            }
        elif arguments.command == "validate-route-scenario-pilot-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": route_scenario_config.run.run_id,
                "requested_documents": route_scenario_config.selection.requested_documents,
                "direct_routes_only": route_scenario_config.generation.direct_routes_only,
            }
        elif arguments.command == "validate-structured-baseline-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": structured_config.run.run_id,
                "requested_documents": structured_config.selection.requested_documents,
                "benchmark_candidates": list(structured_config.modeling.candidates),
            }
        elif arguments.command == "validate-deterministic-smoke-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": deterministic_config.run.run_id,
            }
        elif arguments.command == "validate-preparation-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": preparation_config.run.run_id,
            }
        elif arguments.command == "validate-config":
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": foundation_config.run.run_id,
            }
        elif arguments.command == "build-dangerous-goods-registry":
            from document_ocr.synthesis.dangerous_goods_pipeline import (
                run_dangerous_goods_registry_build,
            )

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_dangerous_goods_registry_build(
                    project_root=project_root,
                    config_path=config_path,
                    config=dangerous_goods_registry_config,
                ),
            }
        elif arguments.command == "run-raw-text-certified-publication":
            from document_ocr.synthesis.raw_text_certified_publication import (
                run_raw_text_certified_publication,
            )

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_raw_text_certified_publication(
                    project_root=project_root,
                    config_path=config_path,
                    config=raw_text_certified_publication_config,
                ),
            }
        elif arguments.command == "run-raw-text-certified-correction":
            from document_ocr.synthesis.raw_text_certified_correction import (
                run_raw_text_certified_correction,
            )

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_raw_text_certified_correction(
                    project_root=project_root,
                    config_path=config_path,
                    config=raw_text_certified_correction_config,
                ),
            }
        elif arguments.command == "run-raw-text-certification":
            from document_ocr.synthesis.raw_text_certification import (
                run_raw_text_certification,
            )

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_raw_text_certification(
                    project_root=project_root,
                    config_path=config_path,
                    config=raw_text_certification_config,
                ),
            }
        elif arguments.command == "run-raw-text-inventory-batch":
            from document_ocr.synthesis.raw_text_inventory_probe import (
                run_raw_text_inventory_batch,
            )

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_raw_text_inventory_batch(
                    project_root=project_root,
                    config_path=config_path,
                    config=raw_text_inventory_batch_config,
                ),
            }
        elif arguments.command == "preflight-raw-text-inventory-batch":
            from document_ocr.synthesis.raw_text_inventory_probe import (
                preflight_raw_text_inventory_batch,
            )

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": preflight_raw_text_inventory_batch(
                    project_root=project_root,
                    config_path=config_path,
                    config=raw_text_inventory_batch_config,
                ),
            }
        elif arguments.command == "validate-raw-text-inventory-batch-config":
            provider = raw_text_inventory_batch_config.provider
            result = {
                "command": arguments.command,
                "status": "valid",
                "run_id": raw_text_inventory_batch_config.run.run_id,
                "documents": raw_text_inventory_batch_config.workflow.documents,
                "provider_routes": list(
                    (provider.provider_order or ()) if provider.kind == "openrouter" else ()
                ),
            }
        elif arguments.command == "run-raw-text-inventory-probe":
            from document_ocr.synthesis.raw_text_inventory_probe import (
                run_raw_text_inventory_probe,
            )

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_raw_text_inventory_probe(
                    project_root=project_root,
                    config_path=config_path,
                    config=raw_text_inventory_config,
                ),
            }
        elif arguments.command == "run-raw-text-hybrid-batch":
            from document_ocr.synthesis.raw_text_hybrid_batch import run_raw_text_hybrid_batch

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_raw_text_hybrid_batch(
                    project_root=project_root,
                    config_path=config_path,
                    config=raw_text_hybrid_batch_config,
                ),
            }
        elif arguments.command == "run-raw-text-hybrid-probe":
            from document_ocr.synthesis.raw_text_hybrid_probe import run_raw_text_hybrid_probe

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_raw_text_hybrid_probe(
                    project_root=project_root,
                    config_path=config_path,
                    config=raw_text_hybrid_config,
                ),
            }
        elif arguments.command == "run-raw-text-rewrite-cycle-probe":
            from document_ocr.synthesis.raw_text_rewrite_cycle_probe import (
                run_raw_text_rewrite_cycle_probe,
            )

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_raw_text_rewrite_cycle_probe(
                    project_root=project_root,
                    config_path=config_path,
                    config=raw_text_rewrite_cycle_config,
                ),
            }
        elif arguments.command == "run-raw-text-rewrite-probe":
            from document_ocr.synthesis.raw_text_rewrite_probe import (
                run_raw_text_rewrite_probe,
            )

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_raw_text_rewrite_probe(
                    project_root=project_root,
                    config_path=config_path,
                    config=raw_text_rewrite_config,
                ),
            }
        elif arguments.command == "run-linguistic-completion":
            from document_ocr.synthesis.linguistic_completion_pipeline import (
                run_linguistic_completion,
            )

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_linguistic_completion(
                    project_root=project_root,
                    config_path=config_path,
                    config=linguistic_completion_config,
                ),
            }
        elif arguments.command == "run-package-compatibility-catalog":
            from document_ocr.synthesis.package_compatibility_catalog import (
                run_package_compatibility_catalog,
            )

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_package_compatibility_catalog(
                    project_root=project_root,
                    config_path=config_path,
                    config=package_compatibility_config,
                ),
            }
        elif arguments.command == "run-linguistic-probe-analysis":
            from document_ocr.synthesis.linguistic_probe_analysis import (
                run_linguistic_probe_analysis,
            )

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_linguistic_probe_analysis(
                    project_root=project_root,
                    config_path=config_path,
                    config=linguistic_probe_analysis_config,
                ),
            }
        elif arguments.command == "run-party-identity-probe":
            from document_ocr.synthesis.party_identity_probe import run_party_identity_probe

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_party_identity_probe(
                    project_root=project_root,
                    config_path=config_path,
                    config=party_identity_config,
                ),
            }
        elif arguments.command == "run-dangerous-goods-analysis":
            from document_ocr.synthesis.dangerous_goods_analysis import (
                run_dangerous_goods_analysis,
            )

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_dangerous_goods_analysis(
                    project_root=project_root,
                    config_path=config_path,
                    config=dangerous_goods_analysis_config,
                ),
            }
        elif arguments.command == "run-cargo-language-probe":
            from document_ocr.synthesis.cargo_language_probe import run_cargo_language_probe

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_cargo_language_probe(
                    project_root=project_root,
                    config_path=config_path,
                    config=cargo_language_config,
                ),
            }
        elif arguments.command == "run-semantic-completion":
            from document_ocr.synthesis.semantic_completion_pipeline import (
                run_semantic_completion,
            )

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_semantic_completion(
                    project_root=project_root,
                    config_path=config_path,
                    config=semantic_completion_config,
                ),
            }
        elif arguments.command == "run-dangerous-goods-plan":
            from document_ocr.synthesis.dangerous_goods_plan_pipeline import (
                run_dangerous_goods_plan,
            )

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_dangerous_goods_plan(
                    project_root=project_root,
                    config_path=config_path,
                    config=dangerous_goods_plan_config,
                ),
            }
        elif arguments.command == "run-semantic-plan":
            from document_ocr.synthesis.semantic_plan_pipeline import (
                run_semantic_plan_pipeline,
            )

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_semantic_plan_pipeline(
                    project_root=project_root,
                    config_path=config_path,
                    config=semantic_plan_config,
                ),
            }
        elif arguments.command == "run-controlled-pilot":
            from document_ocr.synthesis.controlled_generation_pipeline import (
                run_controlled_pilot,
            )

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_controlled_pilot(
                    project_root=project_root,
                    config_path=config_path,
                    config=controlled_config,
                ),
            }
        elif arguments.command == "run-party-structure-benchmark":
            from document_ocr.synthesis.scenario_benchmark_pipeline import (
                run_party_structure_benchmark_pipeline,
            )

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_party_structure_benchmark_pipeline(
                    project_root=project_root,
                    config_path=config_path,
                    config=party_benchmark_config,
                ),
            }
        elif arguments.command == "run-route-scenario-pilot":
            from document_ocr.synthesis.route_scenario_pipeline import run_route_scenario_pilot

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_route_scenario_pilot(
                    project_root=project_root,
                    config_path=config_path,
                    config=route_scenario_config,
                ),
            }
        elif arguments.command == "run-structured-baseline":
            from document_ocr.synthesis.structured_generation import run_structured_baseline

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_structured_baseline(
                    project_root=project_root,
                    config_path=config_path,
                    config=structured_config,
                ),
            }
        elif arguments.command == "run-deterministic-smoke":
            from document_ocr.synthesis.generation import run_deterministic_smoke

            result = {
                "command": arguments.command,
                "status": "complete",
                "result": run_deterministic_smoke(
                    project_root=project_root,
                    config_path=config_path,
                    config=deterministic_config,
                ),
            }
        elif arguments.command == "prepare-corpus":
            result = {
                "command": arguments.command,
                "status": "complete",
                "result": prepare_synthesis_corpus(
                    project_root=project_root,
                    config_path=config_path,
                    config=preparation_config,
                ),
            }
        else:
            result = {
                "command": arguments.command,
                "status": "complete",
                "result": prepare_synthesis_foundation(
                    project_root=project_root,
                    config_path=config_path,
                    config=foundation_config,
                ),
            }
    except KeyboardInterrupt:
        print(json.dumps({"status": "error", "error_type": "KeyboardInterrupt"}))
        raise SystemExit(130) from None
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "diagnostic": str(error),
                },
                sort_keys=True,
            )
        )
        raise SystemExit(4) from None
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
