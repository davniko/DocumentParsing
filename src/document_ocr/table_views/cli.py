"""CLI for the immutable GLM-OCR table-recognition auxiliary view."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

from document_ocr.hashing import canonical_json_bytes
from document_ocr.table_views.config import load_table_view_config
from document_ocr.table_views.join import publish_joined_dataset, verify_joined_dataset
from document_ocr.table_views.pipeline import (
    TableProgress,
    prepare_table_inputs,
    publish_table_dataset,
    run_table_view,
    table_status,
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
        prog="document-ocr-table-view",
        description="Prepare, run, publish, join, or verify page-aligned GLM-OCR table views.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("prepare", "freeze and validate page-aligned table-view inputs"),
        ("run", "run/resume table recognition and publish the complete view"),
        ("publish", "republish a complete committed table-view dataset"),
        ("status", "inspect successful, retryable, exhausted, and pending pages"),
        ("join", "clone configured dataset splits and attach the auxiliary table view"),
        ("verify-join", "verify source preservation and table-view integrity in the clone"),
    ):
        subparser = commands.add_parser(command, help=help_text)
        subparser.add_argument("--config", required=True, type=Path, metavar="PATH")
    return parser


def _emit(stream: Any, value: dict[str, Any]) -> None:
    stream.write((canonical_json_bytes(value) + b"\n").decode("utf-8"))
    stream.flush()


def _progress(value: TableProgress) -> None:
    _emit(
        sys.stderr,
        {
            "elapsed_seconds": round(value.elapsed_seconds, 3),
            "eta_seconds": (
                None if value.eta_seconds is None else round(value.eta_seconds, 3)
            ),
            "failed_pages_this_invocation": value.failed_pages_this_invocation,
            "phase": "table_recognition",
            "processed_pages": value.processed_pages,
            "remaining_pages": value.remaining_pages,
            "status": "progress",
            "successful_pages": value.successful_pages,
            "throughput_pages_per_hour": round(
                value.throughput_pages_per_second * 3600.0, 3
            ),
            "throughput_pages_per_second": round(
                value.throughput_pages_per_second, 6
            ),
            "total_pages": value.total_pages,
        },
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
    try:
        config = load_table_view_config(config_path)
    except Exception as error:
        _emit(
            sys.stderr,
            {
                "command": command,
                "error_type": "TableViewConfigError",
                "message": (
                    f"table-view configuration could not be loaded: {config_path}: {error}"
                ),
                "status": "error",
            },
        )
        return _EXIT_CONFIG

    try:
        if command == "prepare":
            pages = prepare_table_inputs(config)
            result: dict[str, Any] = {
                "documents": config.source.expected_documents,
                "pages": len(pages),
            }
        elif command == "run":
            result = asyncio.run(run_table_view(config, progress=_progress))
        elif command == "publish":
            manifest_path = publish_table_dataset(config)
            result = {"manifest_path": str(manifest_path), **table_status(config)}
        elif command == "status":
            result = table_status(config)
        elif command == "join":
            manifest_path = publish_joined_dataset(config)
            result = {"manifest_path": str(manifest_path), **verify_joined_dataset(config)}
        elif command == "verify-join":
            result = verify_joined_dataset(config)
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
        _emit(
            sys.stderr,
            {
                "command": command,
                "error_type": type(error).__name__,
                "message": str(error).strip() or type(error).__name__,
                "status": "error",
            },
        )
        return _EXIT_OPERATION

    _emit(sys.stdout, {"command": command, **result, "status": "complete"})
    return _EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
