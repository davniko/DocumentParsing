"""Structured OCR commands integrated beneath ``document-ocr paddle``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from document_ocr.paddle_ocr.config import load_config


def add_commands(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="paddle_command", required=True)
    for name, help_text in (
        ("validate-config", "validate structured OCR settings without loading models"),
        ("prepare-models", "download pinned model snapshots without running inference"),
        ("inventory", "hash and inspect all configured PDFs without running OCR"),
        ("run", "extract or resume native and structured OCR artifacts"),
        ("status", "verify page commits, native artifacts and final publication"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, required=True)
        if name == "run":
            command.add_argument("--project-root", type=Path, default=Path.cwd())


def _emit(value: dict[str, Any], *, progress: bool = False) -> None:
    print(
        json.dumps(value, ensure_ascii=False, allow_nan=False),
        file=sys.stderr if progress else sys.stdout,
        flush=True,
    )


def execute(arguments: argparse.Namespace) -> int:
    try:
        config = load_config(arguments.config)
        result: dict[str, Any]
        match arguments.paddle_command:
            case "validate-config":
                result = {"status": "valid", "backend": config.backend, "run_id": config.run.run_id}
            case "prepare-models":
                from document_ocr.paddle_ocr.backend import prepare_models

                result = {"status": "ready", "models": prepare_models(config.paddle)}
            case "inventory":
                from document_ocr.paddle_ocr.pipeline import prepare_inventory, run_root

                inventory = prepare_inventory(config)
                result = {k: v for k, v in inventory.items() if k != "documents"}
                result.update(
                    status="ready", inventory_path=str(run_root(config) / "inventory.json")
                )
            case "run":
                from document_ocr.paddle_ocr.pipeline import run

                result = run(
                    config,
                    arguments.project_root.resolve(),
                    lambda value: _emit(value, progress=True),
                )
            case "status":
                from document_ocr.paddle_ocr.pipeline import status

                result = status(config)
            case _:
                raise ValueError("unknown PaddleOCR command")
        _emit(result)
        return 5 if result["status"] == "incomplete" else 0
    except KeyboardInterrupt:
        _emit({"status": "interrupted", "message": "completed page commits are preserved"})
        return 130
    except Exception as exc:
        _emit({"status": "error", "error_type": type(exc).__name__, "message": str(exc)})
        return 4
