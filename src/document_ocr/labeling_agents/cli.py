"""CLI for PydanticAI OCR-conditioned labeling and independent review."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.labeling_agents.config import load_agent_labeling_config
from document_ocr.labeling_agents.orchestrator import (
    LabelingProgress,
    publish_agent_run,
    run_agent_labeling,
)
from document_ocr.labeling_agents.provider import PydanticAgentGateway
from document_ocr.labeling_agents.work_items import (
    agent_run_paths,
    inventory_payload,
    inventory_work_items,
    prepare_agent_run,
    select_work_items,
)

_EXIT_SUCCESS = 0
_EXIT_USAGE = 2
_EXIT_CONFIG = 3
_EXIT_OPERATION = 4
_EXIT_INTERRUPTED = 130


class _UsageError(ValueError):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="document-kie-label-agents",
        description="Inventory, prepare, preflight, run, or publish a PydanticAI labeling run.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("inventory", "validate all source work items and report the inventory digest"),
        ("prepare", "freeze the deterministic selection, prompts, config, and work items"),
        ("preflight", "validate provider configuration and any configured Ollama server"),
        ("run", "run/resume extraction plus mandatory independent review"),
        ("publish", "publish a complete run's training records, usage, and manifest"),
        (
            "publish-interrupted",
            "publish committed outcomes plus all persisted usage from an interrupted run",
        ),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, type=Path, metavar="PATH")
    return parser


def _emit(stream: Any, value: dict[str, Any]) -> None:
    stream.write((canonical_json_bytes(value) + b"\n").decode("utf-8"))
    stream.flush()


def _exception_leaf_details(error: BaseException) -> tuple[dict[str, str], ...]:
    """Flatten concurrent failures so the actual task error reaches operators."""

    details: list[dict[str, str]] = []

    def visit(current: BaseException) -> None:
        if isinstance(current, BaseExceptionGroup):
            for child in current.exceptions:
                visit(child)
            return
        details.append(
            {
                "error_type": type(current).__name__,
                "message": str(current).strip() or type(current).__name__,
            }
        )

    visit(error)
    return tuple(details)


def _progress(value: LabelingProgress) -> None:
    _emit(
        sys.stderr,
        {
            "excluded_documents": value.excluded_documents,
            "needs_review_documents": value.needs_review_documents,
            "phase": "label_and_review",
            "processed_documents": value.processed_documents,
            "remaining_documents": value.total_documents - value.processed_documents,
            "status": "progress",
            "total_documents": value.total_documents,
            "validated_documents": value.validated_documents,
        },
    )


def _gateway(config: Any, project_root: Path) -> PydanticAgentGateway:
    paths = agent_run_paths(config)
    prompts = {
        "extractor": (paths.prompt_root / "extractor.md").read_text(encoding="utf-8"),
        "reviewer": (paths.prompt_root / "reviewer.md").read_text(encoding="utf-8"),
        "document": (paths.prompt_root / "document-layout.md").read_text(encoding="utf-8"),
    }
    return PydanticAgentGateway(
        config,
        project_root=project_root,
        extractor_prompt=prompts["extractor"],
        reviewer_prompt=prompts["reviewer"],
        document_prompt=prompts["document"],
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
    except _UsageError as error:
        _emit(
            sys.stderr,
            {"error_type": type(error).__name__, "message": str(error), "status": "error"},
        )
        return _EXIT_USAGE
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else _EXIT_USAGE

    command = cast(str, arguments.command)
    config_path = cast(Path, arguments.config)
    project_root = Path.cwd().resolve(strict=True)
    try:
        config = load_agent_labeling_config(config_path)
    except Exception as error:
        _emit(
            sys.stderr,
            {
                "command": command,
                "error_type": "AgentLabelingConfigError",
                "message": f"agent labeling config could not be loaded: {config_path}: {error}",
                "status": "error",
            },
        )
        return _EXIT_CONFIG

    try:
        rows = inventory_work_items(config)
        actual_inventory = sha256_bytes(inventory_payload(rows))
        if command == "inventory":
            result: dict[str, Any] = {
                "documents": len(rows),
                "pages": sum(row.item.source.documentPageCount for row in rows),
                "inventory_sha256": actual_inventory,
                "matches_config": actual_inventory == config.source.expected_inventory_sha256,
            }
        elif command == "prepare":
            selected = prepare_agent_run(config, project_root=project_root)
            result = {
                "inventory_sha256": actual_inventory,
                "selected_documents": len(selected),
                "selected_pages": sum(row.item.source.documentPageCount for row in selected),
            }
        elif command == "preflight":
            prepare_agent_run(config, project_root=project_root)
            gateway = _gateway(config, project_root)
            result = {"ollama": asyncio.run(gateway.preflight_ollama())}
        elif command == "run":
            prepare_agent_run(config, project_root=project_root)
            gateway = _gateway(config, project_root)
            result = asyncio.run(
                run_agent_labeling(
                    config,
                    project_root=project_root,
                    gateway=gateway,
                    progress=_progress,
                )
            )
        elif command == "publish":
            selected = select_work_items(config, rows)
            manifest_path = publish_agent_run(config, selected)
            result = {
                "manifest_path": str(manifest_path),
                "selected_documents": len(selected),
            }
        elif command == "publish-interrupted":
            frozen_selection = select_work_items(config, rows)
            paths = agent_run_paths(config)
            selected = tuple(
                row
                for row in frozen_selection
                if (paths.state_root / "outcomes" / f"{row.item.source.documentId}.json").is_file()
            )
            if not selected:
                raise RuntimeError("interrupted run has no committed document outcomes")
            if len(selected) == len(frozen_selection):
                raise RuntimeError(
                    "all frozen documents have outcomes; use the complete publish command"
                )
            manifest_path = publish_agent_run(
                config,
                selected,
                frozen_selection=frozen_selection,
            )
            result = {
                "manifest_path": str(manifest_path),
                "selected_documents": len(selected),
                "frozen_selection_documents": len(frozen_selection),
            }
        else:
            raise RuntimeError(f"unhandled command: {command}")
    except KeyboardInterrupt:
        _emit(
            sys.stderr,
            {
                "command": command,
                "error_type": "KeyboardInterrupt",
                "message": "operation interrupted",
                "status": "error",
            },
        )
        return _EXIT_INTERRUPTED
    except Exception as error:
        payload: dict[str, Any] = {
            "command": command,
            "error_type": type(error).__name__,
            "message": str(error).strip() or type(error).__name__,
            "status": "error",
        }
        if isinstance(error, BaseExceptionGroup):
            payload["causes"] = _exception_leaf_details(error)
        _emit(sys.stderr, payload)
        return _EXIT_OPERATION

    _emit(sys.stdout, {"command": command, **result, "status": "complete"})
    return _EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
