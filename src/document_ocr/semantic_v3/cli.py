"""CLI for semantic-v3 readiness audits and immutable target transforms."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

from document_ocr.hashing import canonical_json_bytes
from document_ocr.semantic_v3.category_review import publish_category_assignments
from document_ocr.semantic_v3.config import load_semantic_v3_config
from document_ocr.semantic_v3.transform import (
    audit_semantic_v3_readiness,
    publish_semantic_v3_dataset,
)

_EXIT_SUCCESS = 0
_EXIT_USAGE = 2
_EXIT_CONFIG = 3
_EXIT_OPERATION = 4


class _UsageError(ValueError):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="document-kie-semantic-v3",
        description="Audit or publish a reversible relation-explicit semantic-v3 dataset.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("audit", "inventory category and cargo-relation readiness without changing data"),
        ("transform", "publish a clone using frozen reviewed category registries"),
    ):
        subparser = commands.add_parser(command, help=help_text)
        subparser.add_argument("--config", required=True, type=Path, metavar="PATH")
    categories = commands.add_parser(
        "categories",
        help="publish exact source-key assignments from a frozen categorical review",
    )
    categories.add_argument("--config", required=True, type=Path, metavar="PATH")
    categories.add_argument(
        "--review-config", required=True, type=Path, metavar="PATH"
    )
    return parser


def _emit(stream: Any, value: dict[str, Any]) -> None:
    stream.write((canonical_json_bytes(value) + b"\n").decode("utf-8"))
    stream.flush()


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
        config = load_semantic_v3_config(config_path)
    except Exception as error:
        _emit(
            sys.stderr,
            {
                "command": command,
                "error_type": "SemanticV3ConfigError",
                "message": str(error).strip() or type(error).__name__,
                "status": "error",
            },
        )
        return _EXIT_CONFIG
    try:
        if command == "audit":
            manifest = audit_semantic_v3_readiness(config)
        elif command == "categories":
            manifest = publish_category_assignments(
                config, cast(Path, arguments.review_config)
            )
        else:
            manifest = publish_semantic_v3_dataset(config)
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
        {"command": command, "manifest_path": str(manifest), "status": "complete"},
    )
    return _EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
