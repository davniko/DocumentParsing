"""CLI for projecting quality-filtered OCR into immutable labeling inputs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn

from document_ocr.hashing import canonical_json_bytes
from document_ocr.labeling_agents.source_projection import project_quality_filter_work_items

_EXIT_SUCCESS = 0
_EXIT_USAGE = 2
_EXIT_OPERATION = 4
_EXIT_INTERRUPTED = 130


class _UsageError(ValueError):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="document-kie-label-source",
        description="Project a published whole-document OCR quality view into label work items.",
    )
    parser.add_argument("--quality-filter-root", required=True, type=Path, metavar="PATH")
    parser.add_argument("--quality-filter-manifest-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path, metavar="PATH")
    return parser


def _emit(stream: Any, value: dict[str, Any]) -> None:
    stream.write((canonical_json_bytes(value) + b"\n").decode("utf-8"))
    stream.flush()


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

    try:
        manifest, documents, pages, manifest_sha256 = project_quality_filter_work_items(
            quality_filter_root=arguments.quality_filter_root,
            quality_filter_manifest_sha256=arguments.quality_filter_manifest_sha256,
            output_root=arguments.output_root,
        )
    except KeyboardInterrupt:
        _emit(
            sys.stderr,
            {
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
                "error_type": type(error).__name__,
                "message": str(error).strip() or type(error).__name__,
                "status": "error",
            },
        )
        return _EXIT_OPERATION

    _emit(
        sys.stdout,
        {
            "documents": documents,
            "manifest_path": str(manifest),
            "manifest_sha256": manifest_sha256,
            "pages": pages,
            "status": "complete",
        },
    )
    return _EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
