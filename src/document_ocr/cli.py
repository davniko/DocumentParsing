"""Operator command line interface for the document OCR pipeline."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

from document_ocr.config import PipelineConfig, load_config
from document_ocr.hashing import canonical_json_bytes, canonical_json_sha256

if TYPE_CHECKING:
    from document_ocr.benchmark import PublishedBenchmarkReport
    from document_ocr.pipeline import PipelineResult
    from document_ocr.sources import FrozenSourceInventory

_EXIT_SUCCESS = 0
_EXIT_USAGE = 2
_EXIT_CONFIG = 3
_EXIT_OPERATION = 4
_EXIT_INCOMPLETE = 5
_EXIT_INTERRUPTED = 130

_COMMON_SECRET_ENVIRONMENT_VARIABLES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
)


class _UsageError(ValueError):
    """Command line arguments do not satisfy the CLI contract."""


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(message)


@dataclass(frozen=True, slots=True)
class _CommandFailure(Exception):
    exit_code: int
    error_type: str
    diagnostic: str


async def prepare_inventory(config: PipelineConfig) -> FrozenSourceInventory:
    """Load inventory machinery only for the inventory command."""

    from document_ocr.pipeline import prepare_inventory as prepare

    return await prepare(config)


async def run_pipeline(*, project_root: Path, config: PipelineConfig) -> PipelineResult:
    """Load the extraction stack only when an extraction is requested."""

    from document_ocr.pipeline import run_pipeline as run

    return await run(project_root=project_root, config=config)


async def require_run_complete(config: PipelineConfig) -> dict[str, int]:
    """Load ledger machinery only for the status command."""

    from document_ocr.pipeline import require_run_complete as require_complete

    return await require_complete(config)


async def run_benchmark(
    *,
    config: PipelineConfig,
    raster_manifest_path: Path,
    report_path: Path,
) -> PublishedBenchmarkReport:
    """Load the benchmark and HTTP client only for an explicit benchmark command."""

    from document_ocr.benchmark import BenchmarkError
    from document_ocr.benchmark import run_benchmark as benchmark
    from document_ocr.client import VllmOcrClient

    sweep = config.benchmark
    if sweep is None:
        raise BenchmarkError("pipeline config has no benchmark sweep")
    async with VllmOcrClient(
        config.vllm,
        max_connections=max(point.max_inflight_pages_global for point in sweep.points),
    ) as client:
        return await benchmark(
            config=config,
            raster_manifest_path=raster_manifest_path,
            report_path=report_path,
            client=client,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="document-ocr",
        description="Prepare, run, and inspect immutable GLM-OCR extraction runs.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser(
        "validate-config",
        help="strictly validate a pipeline YAML configuration",
    )
    validate.add_argument("--config", required=True, type=Path, metavar="PATH")

    inventory = commands.add_parser(
        "inventory",
        aliases=["prepare-inventory"],
        help="create or verify the immutable source inventory",
    )
    inventory.add_argument("--config", required=True, type=Path, metavar="PATH")

    run = commands.add_parser("run", help="run or resume page-level OCR extraction")
    run.add_argument("--config", required=True, type=Path, metavar="PATH")
    run.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        metavar="PATH",
        help="repository root used for immutable runtime provenance (default: current directory)",
    )

    status = commands.add_parser(
        "status",
        help="verify that a run ledger is complete without contacting vLLM",
    )
    status.add_argument("--config", required=True, type=Path, metavar="PATH")

    benchmark = commands.add_parser(
        "benchmark",
        help="benchmark a running vLLM server on an immutable raster JSONL manifest",
    )
    benchmark.add_argument("--config", required=True, type=Path, metavar="PATH")
    benchmark.add_argument("--raster-manifest", required=True, type=Path, metavar="JSONL")
    benchmark.add_argument("--report", required=True, type=Path, metavar="JSON")
    return parser


def _secret_values(config: PipelineConfig | None) -> tuple[str, ...]:
    names = list(_COMMON_SECRET_ENVIRONMENT_VARIABLES)
    if config is not None:
        names.append(config.vllm.api_key_env)
    values = {value for name in names if (value := os.environ.get(name))}
    return tuple(sorted(values, key=len, reverse=True))


def _safe_diagnostic(error: Exception, config: PipelineConfig | None) -> str:
    message = str(error).strip() or type(error).__name__
    for secret in _secret_values(config):
        message = message.replace(secret, "[REDACTED]")
    return message


def _emit(stream: Any, value: dict[str, Any]) -> None:
    stream.write((canonical_json_bytes(value) + b"\n").decode("utf-8"))


def _load_command_config(path: Path) -> PipelineConfig:
    try:
        return load_config(path)
    except Exception as error:
        raise _CommandFailure(
            exit_code=_EXIT_CONFIG,
            error_type=type(error).__name__,
            diagnostic=f"configuration could not be loaded or validated: {path}",
        ) from error


def _resolve_project_root(path: Path, config: PipelineConfig) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise _CommandFailure(
            exit_code=_EXIT_OPERATION,
            error_type=type(error).__name__,
            diagnostic=_safe_diagnostic(error, config),
        ) from error
    if not resolved.is_dir():
        raise _CommandFailure(
            exit_code=_EXIT_OPERATION,
            error_type="NotADirectoryError",
            diagnostic=f"project root is not a directory: {resolved}",
        )
    return resolved


async def _dispatch(arguments: argparse.Namespace) -> dict[str, Any]:
    command = cast(str, arguments.command)
    config_path = cast(Path, arguments.config)
    config = _load_command_config(config_path)

    try:
        if command == "validate-config":
            resolved_config = config.model_dump(mode="json")
            return {
                "command": "validate-config",
                "config_sha256": canonical_json_sha256(resolved_config),
                "run_id": config.run.run_id,
                "schema_version": config.schema_version,
                "source_type": config.source.type,
                "status": "valid",
            }

        if command in {"inventory", "prepare-inventory"}:
            inventory = await prepare_inventory(config)
            return {
                "command": "prepare-inventory",
                "document_count": len(inventory.objects),
                "inventory_path": str(inventory.path),
                "inventory_sha256": inventory.sha256,
                "run_id": config.run.run_id,
                "status": "ready",
            }

        if command == "run":
            project_root = _resolve_project_root(cast(Path, arguments.project_root), config)
            result = await run_pipeline(project_root=project_root, config=config)
            server = None
            if result.server_info is not None:
                server = {
                    "container_image": result.server_info.container_image,
                    "contract_sha256": result.server_info.contract_sha256,
                    "generation_config": result.server_info.generation_config,
                    "gpu_memory_utilization": (result.server_info.gpu_memory_utilization),
                    "image_limit_per_prompt": (result.server_info.image_limit_per_prompt),
                    "max_model_len": result.server_info.max_model_len,
                    "max_num_seqs": result.server_info.max_num_seqs,
                    "model_repository": result.server_info.model_repository,
                    "model_revision": result.server_info.model_revision,
                    "models": list(result.server_info.models),
                    "num_speculative_tokens": (result.server_info.num_speculative_tokens),
                    "served_model_name": result.server_info.served_model_name,
                    "speculative_method": result.server_info.speculative_method,
                    "version": result.server_info.version,
                }
            return {
                "command": "run",
                "dataset_manifest": str(result.dataset_manifest),
                "resumed_complete_run": result.resumed_complete_run,
                "run_id": result.run_id,
                "server": server,
                "status": "complete",
                "summary": result.summary,
            }

        if command == "status":
            summary = await require_run_complete(config)
            return {
                "command": "status",
                "complete": True,
                "run_id": config.run.run_id,
                "status": "complete",
                "summary": summary,
            }

        if command == "benchmark":
            published = await run_benchmark(
                config=config,
                raster_manifest_path=cast(Path, arguments.raster_manifest),
                report_path=cast(Path, arguments.report),
            )
            return {
                "benchmark_identity_sha256": published.report["benchmark_identity_sha256"],
                "command": "benchmark",
                "created": published.created,
                "report_path": str(published.path),
                "report_sha256": published.sha256,
                "status": "complete",
            }

        raise RuntimeError(f"unhandled command: {command}")
    except _CommandFailure:
        raise
    except Exception as error:
        from document_ocr.ledger import IncompleteRunError

        if isinstance(error, IncompleteRunError):
            raise _CommandFailure(
                exit_code=_EXIT_INCOMPLETE,
                error_type=type(error).__name__,
                diagnostic=_safe_diagnostic(error, config),
            ) from error
        raise _CommandFailure(
            exit_code=_EXIT_OPERATION,
            error_type=type(error).__name__,
            diagnostic=_safe_diagnostic(error, config),
        ) from error


def main(argv: Sequence[str] | None = None) -> int:
    """Run one CLI command and return a stable process exit code."""

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
    try:
        payload = asyncio.run(_dispatch(arguments))
    except _CommandFailure as error:
        _emit(
            sys.stderr,
            {
                "command": command,
                "error_type": error.error_type,
                "message": error.diagnostic,
                "status": "error",
            },
        )
        return error.exit_code
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

    _emit(sys.stdout, payload)
    return _EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
