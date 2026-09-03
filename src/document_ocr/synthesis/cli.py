"""CLI for the non-generative, lossless synthesis foundation stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    load_synthesis_route_scenario_pilot_config,
    load_synthesis_semantic_completion_config,
    load_synthesis_semantic_plan_config,
    load_synthesis_structured_baseline_config,
)
from document_ocr.synthesis.pipeline import prepare_synthesis_foundation
from document_ocr.synthesis.preparation import prepare_synthesis_corpus


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
    ):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    try:
        project_root = arguments.project_root.resolve(strict=True)
        if arguments.config.is_symlink():
            raise ValueError("configuration must be a real file")
        config_path = arguments.config.resolve(strict=True)
        if not config_path.is_file():
            raise ValueError("configuration must be a real file")
        if arguments.command in {
            "validate-linguistic-completion-config",
            "run-linguistic-completion",
        }:
            linguistic_completion_config = load_synthesis_linguistic_completion_config(
                config_path
            )
        elif arguments.command in {
            "validate-linguistic-probe-analysis-config",
            "run-linguistic-probe-analysis",
        }:
            linguistic_probe_analysis_config = (
                load_synthesis_linguistic_probe_analysis_config(config_path)
            )
        elif arguments.command in {
            "validate-package-compatibility-catalog-config",
            "run-package-compatibility-catalog",
        }:
            package_compatibility_config = (
                load_synthesis_package_compatibility_catalog_config(config_path)
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
        if arguments.command == "validate-linguistic-completion-config":
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
    raise SystemExit(0)


if __name__ == "__main__":
    main()
