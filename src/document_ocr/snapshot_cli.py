"""Command-line interface for immutable local S3 source snapshots."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

from document_ocr.config import load_snapshot_config
from document_ocr.hashing import canonical_json_bytes
from document_ocr.snapshot import materialize_snapshot, verify_snapshot

_EXIT_SUCCESS = 0
_EXIT_USAGE = 2
_EXIT_CONFIG = 3
_EXIT_OPERATION = 4
_EXIT_INTERRUPTED = 130
_SECRET_ENVIRONMENT_VARIABLES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
)


class _UsageError(ValueError):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="document-ocr-snapshot",
        description="Materialize or verify a content-pinned local S3 PDF snapshot.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("materialize", "conditionally download and commit the configured snapshot"),
        ("verify", "verify every local snapshot artifact without contacting S3"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, type=Path, metavar="PATH")
    return parser


def _emit(stream: Any, value: dict[str, Any]) -> None:
    stream.write((canonical_json_bytes(value) + b"\n").decode("utf-8"))
    stream.flush()


def _safe_message(error: Exception) -> str:
    message = str(error).strip() or type(error).__name__
    secrets = {value for name in _SECRET_ENVIRONMENT_VARIABLES if (value := os.environ.get(name))}
    for secret in sorted(secrets, key=len, reverse=True):
        message = message.replace(secret, "[REDACTED]")
    return message


def _progress(completed: int, total: int, completed_bytes: int, reused: int) -> None:
    if completed % 100 != 0 and completed != total:
        return
    _emit(
        sys.stderr,
        {
            "processed_bytes": completed_bytes,
            "processed_documents": completed,
            "reused_documents": reused,
            "status": "progress",
            "total_documents": total,
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
        config = load_snapshot_config(config_path)
    except Exception as error:
        _emit(
            sys.stderr,
            {
                "command": command,
                "error_type": type(error).__name__,
                "message": f"snapshot configuration could not be loaded: {config_path}",
                "status": "error",
            },
        )
        return _EXIT_CONFIG

    try:
        if command == "materialize":
            result = materialize_snapshot(config, progress=_progress)
        elif command == "verify":
            result = verify_snapshot(config, progress=_progress)
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
                "message": _safe_message(error),
                "status": "error",
            },
        )
        return _EXIT_OPERATION

    commit = result.commit
    _emit(
        sys.stdout,
        {
            "actual_pages": commit.actual_pages,
            "by_document_type": {
                key: value.model_dump(mode="json") for key, value in commit.by_document_type.items()
            },
            "command": command,
            "created": result.created,
            "document_count": commit.document_count,
            "extraction_document_count": commit.extraction_document_count,
            "extraction_pages": commit.extraction_pages,
            "manifest_sha256": commit.manifest_sha256,
            "quarantined_document_count": commit.quarantined_document_count,
            "selection_sha256": commit.selection_sha256,
            "selection_audit": commit.selection_audit.model_dump(mode="json"),
            "snapshot_root": str(result.root),
            "source_bytes": commit.source_bytes,
            "status": "complete",
        },
    )
    return _EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
