"""Command-line interface for immutable combined local corpora."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

from document_ocr.config import load_corpus_config
from document_ocr.corpus import materialize_corpus, plan_corpus, verify_corpus
from document_ocr.hashing import canonical_json_bytes

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
        prog="document-ocr-corpus",
        description="Materialize or verify a hard-linked local PDF corpus.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("plan", "verify sources and print the deterministic corpus manifest digest"),
        ("materialize", "assemble and commit the configured corpus"),
        ("verify", "verify the source snapshots, manifest, links, and PDFs"),
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


def _progress(phase: str, completed: int, total: int) -> None:
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
        config = load_corpus_config(config_path)
    except Exception:
        _emit(
            sys.stderr,
            {
                "command": command,
                "error_type": "CorpusConfigError",
                "message": f"corpus configuration could not be loaded: {config_path}",
                "status": "error",
            },
        )
        return _EXIT_CONFIG

    try:
        if command == "plan":
            manifest_sha256, document_count = plan_corpus(config, progress=_progress)
            _emit(
                sys.stdout,
                {
                    "command": command,
                    "document_count": document_count,
                    "document_type": config.document_type,
                    "manifest_sha256": manifest_sha256,
                    "status": "complete",
                },
            )
            return _EXIT_SUCCESS
        if command == "materialize":
            result = materialize_corpus(config, progress=_progress)
        elif command == "verify":
            result = verify_corpus(config, progress=_progress)
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
            "command": command,
            "corpus_root": str(result.root),
            "created": result.created,
            "deduplicated_alias_count": commit.deduplicated_alias_count,
            "document_count": commit.document_count,
            "document_page_count": commit.document_page_count,
            "document_type": commit.document_type,
            "logical_source_bytes": commit.logical_source_bytes,
            "manifest_sha256": commit.manifest_sha256,
            "source_alias_count": commit.source_alias_count,
            "status": "complete",
        },
    )
    return _EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
