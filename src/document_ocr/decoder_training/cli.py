"""Command-line boundary for decoder-only configuration, inspection, and training."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, cast

from document_ocr.decoder_training.config import load_decoder_training_config
from document_ocr.decoder_training.data import (
    ChatTokenizer,
    inspect_decoder_dataset,
    load_task_and_prompt,
    project_for_trl,
)
from document_ocr.hashing import canonical_json_sha256, sha256_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="document-kie-decoder")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-config", "inspect-dataset", "inspect-tokens", "train"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def _dispatch(arguments: argparse.Namespace) -> dict[str, Any]:
    project_root = arguments.project_root.resolve(strict=True)
    if not project_root.is_dir():
        raise ValueError("project root is not a directory")
    config_path = _regular_file(arguments.config, "configuration")
    config = load_decoder_training_config(config_path)
    task, prompt = load_task_and_prompt(project_root, config)
    common = {
        "command": arguments.command,
        "run_id": config.run.run_id,
        "method": config.method,
        "config_file_sha256": sha256_file(config_path),
        "resolved_config_sha256": canonical_json_sha256(config.model_dump(mode="json")),
        "prompt_sha256": prompt.sha256,
    }
    if arguments.command == "validate-config":
        return {**common, "status": "valid"}
    if arguments.command in {"inspect-dataset", "inspect-tokens"}:
        started = time.perf_counter()
        records, inspection = inspect_decoder_dataset(
            project_root=project_root, config=config, task=task, prompt=prompt
        )
        if arguments.command == "inspect-dataset":
            elapsed = time.perf_counter() - started
            return {
                **common,
                "status": "complete",
                "elapsed_seconds": elapsed,
                "records_per_second": inspection.source_records / elapsed,
                "dataset": inspection.to_dict(),
            }
        token = None
        if config.model.token_env is not None:
            token = os.environ.get(config.model.token_env)
            if not token:
                raise ValueError(
                    f"required model token environment variable is empty: "
                    f"{config.model.token_env}"
                )
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError("token inspection requires the decoder environment") from error
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.name_or_path,
            revision=config.model.revision,
            token=token,
            trust_remote_code=config.model.trust_remote_code,
            local_files_only=config.model.local_files_only,
            use_fast=True,
        )
        _, token_inspection = project_for_trl(
            records, tokenizer=cast(ChatTokenizer, tokenizer), config=config
        )
        elapsed = time.perf_counter() - started
        return {
            **common,
            "status": "complete",
            "elapsed_seconds": elapsed,
            "records_per_second": inspection.source_records / elapsed,
            "dataset": inspection.to_dict(),
            "tokens": token_inspection.to_dict(),
        }
    from document_ocr.decoder_training.runtime import run_training

    return {
        **common,
        "status": "complete",
        "result": run_training(project_root=project_root, config_path=config_path, config=config),
    }


def main() -> None:
    try:
        result = _dispatch(_parser().parse_args())
    except KeyboardInterrupt:
        print(json.dumps({"status": "error", "error_type": "KeyboardInterrupt"}))
        raise SystemExit(130) from None
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "diagnostic": str(error).strip() or type(error).__name__,
                },
                sort_keys=True,
            )
        )
        raise SystemExit(4) from None
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    raise SystemExit(0)


if __name__ == "__main__":
    main()
