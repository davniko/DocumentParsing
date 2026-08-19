"""Command-line entry point for audited OCR quality filtering."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn

from document_ocr.hashing import canonical_json_bytes
from document_ocr.quality_filter import filter_local_ocr_run

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
        prog="document-ocr-quality-filter",
        description=(
            "Validate and publish a document-level quality projection of a retained local OCR run."
        ),
    )
    parser.add_argument("--run-dir", required=True, type=Path, metavar="PATH")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path, metavar="PATH")
    parser.add_argument(
        "--artifact-verification-workers",
        type=int,
        default=8,
        metavar="N",
        help="parallel SHA-256 verification workers (default: 8)",
    )
    return parser


def _emit(stream: Any, value: dict[str, Any]) -> None:
    stream.write((canonical_json_bytes(value) + b"\n").decode("utf-8"))
    stream.flush()


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
    try:
        arguments = _parser().parse_args(argv)
        if arguments.artifact_verification_workers <= 0:
            raise _UsageError("--artifact-verification-workers must be positive")
    except _UsageError as error:
        _emit(
            sys.stderr,
            {
                "error_type": type(error).__name__,
                "message": str(error),
                "status": "error",
            },
        )
        return _EXIT_USAGE
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else _EXIT_USAGE

    try:
        result = filter_local_ocr_run(
            run_dir=arguments.run_dir,
            run_id=arguments.run_id,
            output_dir=arguments.output_dir,
            artifact_verification_workers=arguments.artifact_verification_workers,
            progress=_progress,
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

    documents = result.summary["documents"]
    pages = result.summary["pages"]
    _emit(
        sys.stdout,
        {
            "created": result.created,
            "eligible_documents": documents["eligible"],
            "eligible_pages": pages["eligible"],
            "excluded_documents": documents["excluded_unique"],
            "manifest_sha256": result.manifest_sha256,
            "output_dir": str(result.output_dir),
            "status": "complete",
        },
    )
    return _EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
