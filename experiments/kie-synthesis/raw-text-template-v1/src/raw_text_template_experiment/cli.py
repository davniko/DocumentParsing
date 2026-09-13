from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from document_ocr.atomic import atomic_write_json

from .pipeline import (
    audit_corpus_carrier_resolution,
    preflight_extraction,
    prepare_selection,
    publish_offline_audit,
    run_extraction,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Carrier-bound raw-text template experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    selection = subparsers.add_parser("selection", help="materialize the deterministic selection")
    selection.add_argument("--config", type=Path, required=True)
    selection.add_argument("--output", type=Path)
    preflight = subparsers.add_parser(
        "preflight", help="validate every selected source before provider spend"
    )
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--output", type=Path)
    carrier_audit = subparsers.add_parser(
        "audit-carriers", help="classify carrier-bound eligibility across the pinned corpus"
    )
    carrier_audit.add_argument("--config", type=Path, required=True)
    carrier_audit.add_argument("--output", type=Path)
    offline_audit = subparsers.add_parser(
        "offline-audit", help="commit the corpus audit and both deterministic preflights"
    )
    offline_audit.add_argument("--development-config", type=Path, required=True)
    offline_audit.add_argument("--transfer-config", type=Path, required=True)
    offline_audit.add_argument("--run-name", required=True)
    extraction = subparsers.add_parser("extract", help="run agent-assisted template extraction")
    extraction.add_argument("--config", type=Path, required=True)
    analysis = subparsers.add_parser("analyze-failures", help="analyze a committed extraction run")
    analysis.add_argument("--run-dir", type=Path, required=True)
    analysis.add_argument("--output-parent", type=Path, required=True)
    analysis.add_argument("--run-name", required=True)
    efficiency = subparsers.add_parser(
        "audit-efficiency",
        help="publish a zero-provider-call cost and contract-efficiency audit",
    )
    efficiency.add_argument("--config", type=Path, required=True)
    descendant_preflight = subparsers.add_parser(
        "preflight-descendants",
        help="validate target/template compatibility and the one-call descendant plan",
    )
    descendant_preflight.add_argument("--config", type=Path, required=True)
    descendant_preflight.add_argument("--output", type=Path)
    descendant_run = subparsers.add_parser(
        "render-descendants",
        help="render the 30 carrier-bound compiled-template descendants",
    )
    descendant_run.add_argument("--config", type=Path, required=True)
    descendant_analysis = subparsers.add_parser(
        "analyze-descendants",
        help="publish EDA for a committed 30-document descendant run",
    )
    descendant_analysis.add_argument("--run-dir", type=Path, required=True)
    descendant_analysis.add_argument("--baseline-dir", type=Path, required=True)
    descendant_analysis.add_argument("--manual-review", type=Path, required=True)
    descendant_analysis.add_argument("--output-parent", type=Path, required=True)
    descendant_analysis.add_argument("--run-name", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.command == "selection":
        manifest = prepare_selection(arguments.config)
        if arguments.output is not None:
            atomic_write_json(arguments.output, manifest.model_dump(mode="json"))
            print(arguments.output)
        else:
            print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
        return
    if arguments.command == "preflight":
        report = preflight_extraction(arguments.config)
        if arguments.output is not None:
            atomic_write_json(arguments.output, report)
            print(arguments.output)
        else:
            print(json.dumps(report, indent=2, sort_keys=True))
        return
    if arguments.command == "audit-carriers":
        report = audit_corpus_carrier_resolution(arguments.config)
        if arguments.output is not None:
            atomic_write_json(arguments.output, report)
            print(arguments.output)
        else:
            print(json.dumps(report, indent=2, sort_keys=True))
        return
    if arguments.command == "offline-audit":
        print(
            publish_offline_audit(
                development_config_path=arguments.development_config,
                transfer_config_path=arguments.transfer_config,
                run_name=arguments.run_name,
            )
        )
        return
    if arguments.command == "extract":
        asyncio.run(run_extraction(arguments.config))
        return
    if arguments.command == "analyze-failures":
        from .analysis import analyze_failed_extraction

        print(
            analyze_failed_extraction(
                run_dir=arguments.run_dir,
                output_parent=arguments.output_parent,
                run_name=arguments.run_name,
            )
        )
        return
    if arguments.command == "audit-efficiency":
        from .efficiency import publish_efficiency_audit

        print(publish_efficiency_audit(arguments.config))
        return
    if arguments.command == "preflight-descendants":
        from .descendant import preflight_descendants

        report = preflight_descendants(arguments.config)
        if arguments.output is not None:
            atomic_write_json(arguments.output, report)
            print(arguments.output)
        else:
            print(json.dumps(report, indent=2, sort_keys=True))
        return
    if arguments.command == "render-descendants":
        from .descendant import run_descendants

        print(asyncio.run(run_descendants(arguments.config)))
        return
    if arguments.command == "analyze-descendants":
        from .descendant_analysis import analyze_descendants

        print(
            analyze_descendants(
                run_dir=arguments.run_dir,
                baseline_dir=arguments.baseline_dir,
                manual_review_path=arguments.manual_review,
                output_parent=arguments.output_parent,
                run_name=arguments.run_name,
            )
        )
        return
    raise AssertionError(f"unhandled command: {arguments.command}")


if __name__ == "__main__":
    main()
