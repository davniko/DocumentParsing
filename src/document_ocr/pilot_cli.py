"""Command-line interface for deterministic classification-catalog pilots."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

from document_ocr.config import load_pilot_config
from document_ocr.hashing import canonical_json_bytes
from document_ocr.pilot import materialize_pilot, plan_pilot, verify_pilot

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


def _build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="document-ocr-pilot",
        description="Plan, materialize, or verify a quality-filtered local OCR pilot.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("plan", "verify the catalog and print the deterministic pilot digest"),
        ("materialize", "hard-link and commit the configured pilot"),
        ("verify", "offline-verify the catalog selection, manifest, and PDFs"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, type=Path, metavar="PATH")
    return parser


def _emit(stream: Any, value: dict[str, Any]) -> None:
    stream.write((canonical_json_bytes(value) + b"\n").decode("utf-8"))
    stream.flush()


def _progress(phase: str, completed: int, total: int) -> None:
    if phase == "verify_pilot_pdf" and completed % 50 != 0 and completed != total:
        return
    _emit(
        sys.stderr,
        {
            "completed": completed,
            "phase": phase,
            "status": "progress",
            "total": total,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        arguments = parser.parse_args(argv)
    except _UsageError as error:
        _emit(
            sys.stderr,
            {
                "command": None,
                "error_type": type(error).__name__,
                "message": str(error),
                "status": "error",
            },
        )
        return _EXIT_USAGE
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else _EXIT_USAGE

    command = cast(str, arguments.command)
    config_path = cast(Path, arguments.config)
    try:
        config = load_pilot_config(config_path)
    except Exception:
        _emit(
            sys.stderr,
            {
                "command": command,
                "error_type": "PilotConfigError",
                "message": f"pilot configuration could not be loaded: {config_path}",
                "status": "error",
            },
        )
        return _EXIT_CONFIG

    try:
        if command == "plan":
            planned = plan_pilot(config, progress=_progress)
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
            result = materialize_pilot(config, progress=_progress)
        elif command == "verify":
            result = verify_pilot(config, progress=_progress)
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

    _emit(
        sys.stdout,
        {
            "command": command,
            "created": result.created,
            "manifest_sha256": result.commit.manifest_sha256,
            "pilot_root": str(result.root),
            "statistics": result.commit.statistics.model_dump(mode="json"),
            "status": "complete",
        },
    )
    return _EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
