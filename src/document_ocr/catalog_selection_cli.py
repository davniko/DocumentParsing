"""CLI for immutable catalog-derived full extraction surfaces."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

from document_ocr.catalog_selection import (
    materialize_catalog_selection,
    plan_catalog_selection,
    verify_catalog_selection,
)
from document_ocr.config import load_catalog_selection_config
from document_ocr.hashing import canonical_json_bytes

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
        prog="document-ocr-catalog-selection",
        description="Plan, materialize, or verify a full catalog-derived OCR surface.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("plan", "verify inputs and print the deterministic selection digest"),
        ("materialize", "hard-link and commit the configured extraction surface"),
        ("verify", "verify the catalog, exclusions, manifest, links, and PDFs"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, type=Path, metavar="PATH")
    return parser


def _emit(stream: Any, value: dict[str, Any]) -> None:
    stream.write((canonical_json_bytes(value) + b"\n").decode("utf-8"))
    stream.flush()


def _progress(phase: str, completed: int, total: int) -> None:
    if phase == "verify_selection_pdf" and completed % 50 != 0 and completed != total:
        return
    _emit(
        sys.stderr,
        {"completed": completed, "phase": phase, "status": "progress", "total": total},
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
    except _UsageError as error:
        _emit(sys.stderr, {"error_type": type(error).__name__, "message": str(error)})
        return _EXIT_USAGE
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else _EXIT_USAGE

    command = cast(str, arguments.command)
    config_path = cast(Path, arguments.config)
    try:
        config = load_catalog_selection_config(config_path)
    except Exception:
        _emit(
            sys.stderr,
            {
                "command": command,
                "error_type": "CatalogSelectionConfigError",
                "message": f"catalog selection config could not be loaded: {config_path}",
                "status": "error",
            },
        )
        return _EXIT_CONFIG
    try:
        if command == "plan":
            planned = plan_catalog_selection(config, progress=_progress)
            _emit(
                sys.stdout,
                {
                    "command": command,
                    "manifest_sha256": planned.manifest_sha256,
                    "statistics": planned.statistics.model_dump(mode="json"),
                    "status": "complete",
                },
            )
            return _EXIT_SUCCESS
        if command == "materialize":
            result = materialize_catalog_selection(config, progress=_progress)
        elif command == "verify":
            result = verify_catalog_selection(config, progress=_progress)
        else:
            raise RuntimeError(f"unhandled command: {command}")
    except KeyboardInterrupt:
        _emit(
            sys.stderr,
            {"command": command, "error_type": "KeyboardInterrupt", "status": "error"},
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
    _emit(
        sys.stdout,
        {
            "command": command,
            "created": result.created,
            "manifest_sha256": result.commit.manifest_sha256,
            "selection_root": str(result.root),
            "statistics": result.commit.statistics.model_dump(mode="json"),
            "status": "complete",
        },
    )
    return _EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
