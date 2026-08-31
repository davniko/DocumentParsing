"""CLI for the non-generative, lossless synthesis foundation stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from document_ocr.synthesis.config import (
    load_synthesis_controlled_pilot_config,
    load_synthesis_deterministic_smoke_config,
    load_synthesis_foundation_config,
    load_synthesis_party_structure_benchmark_config,
    load_synthesis_preparation_config,
    load_synthesis_route_scenario_pilot_config,
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
        if arguments.command == "validate-controlled-pilot-config":
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
