"""Operator CLI for validation, dataset preparation, and explicit KIE training."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

from document_ocr.hashing import canonical_json_bytes, canonical_json_sha256, sha256_file
from document_ocr.training.config import (
    TrainingConfig,
    load_dataset_partition_config,
    load_training_config,
)
from document_ocr.training.data import inspect_dataset, prepare_datasets
from document_ocr.training.prompting import load_prompt
from document_ocr.training.tasks import load_training_task

_EXIT_SUCCESS = 0
_EXIT_USAGE = 2
_EXIT_CONFIG = 3
_EXIT_OPERATION = 4
_EXIT_INTERRUPTED = 130


class _UsageError(ValueError):
    """CLI arguments do not satisfy the command contract."""


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(message)


@dataclass(frozen=True, slots=True)
class _CommandFailure(Exception):
    exit_code: int
    error_type: str
    diagnostic: str


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="document-kie-train",
        description="Validate, prepare, and explicitly train OCR-conditioned KIE adapters.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    split_command = commands.add_parser(
        "split-dataset",
        help="publish deterministic train/validation JSONLs before training",
    )
    split_command.add_argument("--config", required=True, type=Path, metavar="PATH")
    split_command.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        metavar="PATH",
        help="repository root used to resolve relative paths (default: current directory)",
    )
    for name, help_text in (
        (
            "validate-config",
            "validate strict YAML and its prompt without loading a tokenizer/model",
        ),
        ("inspect-dataset", "verify all hashed JSONL records and target schemas without a model"),
        ("prepare-dataset", "load only the tokenizer and build the exact cached tokenized dataset"),
        ("train", "load the model and execute the configured PEFT training run"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, type=Path, metavar="PATH")
        command.add_argument(
            "--project-root",
            type=Path,
            default=Path.cwd(),
            metavar="PATH",
            help="repository root used to resolve relative paths (default: current directory)",
        )
    return parser


def _load_config(path: Path) -> tuple[Path, TrainingConfig]:
    if path.is_symlink():
        raise _CommandFailure(
            _EXIT_CONFIG,
            "ValueError",
            f"configuration path must not be a symbolic link: {path}",
        )
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError("configuration path is not a regular file")
        return resolved, load_training_config(resolved)
    except Exception as error:
        raise _CommandFailure(
            _EXIT_CONFIG,
            type(error).__name__,
            f"configuration could not be loaded or validated: {path}: {error}",
        ) from error


def _regular_config_path(path: Path, description: str) -> Path:
    if path.is_symlink():
        raise _CommandFailure(
            _EXIT_CONFIG,
            "ValueError",
            f"{description} path must not be a symbolic link: {path}",
        )
    try:
        resolved = path.resolve(strict=True)
    except Exception as error:
        raise _CommandFailure(
            _EXIT_CONFIG,
            type(error).__name__,
            f"{description} could not be resolved: {path}: {error}",
        ) from error
    if not resolved.is_file():
        raise _CommandFailure(
            _EXIT_CONFIG,
            "ValueError",
            f"{description} path is not a regular file: {resolved}",
        )
    return resolved


def _project_root(path: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise _CommandFailure(_EXIT_OPERATION, type(error).__name__, str(error)) from error
    if not resolved.is_dir():
        raise _CommandFailure(
            _EXIT_OPERATION, "NotADirectoryError", f"project root is not a directory: {resolved}"
        )
    return resolved


def _redact(message: str, config: TrainingConfig | None) -> str:
    if config is None or config.model.token_env is None:
        return message
    token = os.environ.get(config.model.token_env)
    return message.replace(token, "[REDACTED]") if token else message


def _dispatch(arguments: argparse.Namespace) -> dict[str, Any]:
    command = cast(str, arguments.command)
    project_root = _project_root(cast(Path, arguments.project_root))
    if command == "split-dataset":
        config_path = _regular_config_path(
            cast(Path, arguments.config), "dataset partition configuration"
        )
        try:
            from document_ocr.training.splitting import partition_dataset

            split_config = load_dataset_partition_config(config_path)
            manifest = partition_dataset(project_root=project_root, config=split_config)
            return {
                "command": command,
                "status": "complete",
                "config_path": str(config_path),
                "config_file_sha256": sha256_file(config_path),
                "partition_manifest": manifest,
            }
        except Exception as error:
            raise _CommandFailure(
                _EXIT_OPERATION,
                type(error).__name__,
                str(error).strip() or type(error).__name__,
            ) from error

    config_path, config = _load_config(cast(Path, arguments.config))
    try:
        task = load_training_task(project_root, config)
        prompt = load_prompt(project_root, config.prompt, task)
        config_identity = canonical_json_sha256(config.model_dump(mode="json"))
        common = {
            "command": command,
            "status": "valid" if command == "validate-config" else "complete",
            "run_id": config.run.run_id,
            "task": config.task,
            "config_path": str(config_path),
            "config_file_sha256": sha256_file(config_path),
            "resolved_config_sha256": config_identity,
            "prompt_sha256": prompt.sha256,
            "prompt_template_sha256": prompt.template_sha256,
            "output_schema_sha256": prompt.output_schema_sha256,
        }
        if command == "validate-config":
            return common

        if command == "inspect-dataset":
            started = time.perf_counter()
            _, inspection = inspect_dataset(
                project_root=project_root,
                config=config,
                prompt=prompt,
                task=task,
            )
            duration = time.perf_counter() - started
            return {
                **common,
                "duration_seconds": duration,
                "records_per_second": inspection.total_records / duration,
                "dataset": inspection.to_dict(),
            }

        if command == "prepare-dataset":
            from document_ocr.training.runtime import load_tokenizer

            started = time.perf_counter()
            tokenizer = load_tokenizer(config)
            prepared = prepare_datasets(
                project_root=project_root,
                config=config,
                prompt=prompt,
                task=task,
                tokenizer=tokenizer,
            )
            return {
                **common,
                "duration_seconds": time.perf_counter() - started,
                "dataset": prepared.report(),
            }

        if command == "train":
            from document_ocr.training.runtime import run_training

            manifest = run_training(
                project_root=project_root,
                config_path=config_path,
                config=config,
                prompt=prompt,
                task=task,
            )
            return {**common, "training_manifest": manifest}
        raise AssertionError(f"unhandled command: {command}")
    except _CommandFailure:
        raise
    except Exception as error:
        raise _CommandFailure(
            _EXIT_OPERATION,
            type(error).__name__,
            _redact(str(error).strip() or type(error).__name__, config),
        ) from error


def _emit(stream: Any, value: dict[str, Any]) -> None:
    stream.write(canonical_json_bytes(value).decode("utf-8") + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run one training-pipeline command with machine-readable output."""

    parser = _parser()
    try:
        arguments = parser.parse_args(argv)
        result = _dispatch(arguments)
    except _UsageError as error:
        _emit(
            sys.stderr,
            {"status": "error", "error_type": type(error).__name__, "diagnostic": str(error)},
        )
        return _EXIT_USAGE
    except _CommandFailure as error:
        _emit(
            sys.stderr,
            {
                "status": "error",
                "error_type": error.error_type,
                "diagnostic": error.diagnostic,
            },
        )
        return error.exit_code
    except KeyboardInterrupt:
        _emit(
            sys.stderr,
            {
                "status": "error",
                "error_type": "KeyboardInterrupt",
                "diagnostic": "operation interrupted",
            },
        )
        return _EXIT_INTERRUPTED
    _emit(sys.stdout, result)
    return _EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
