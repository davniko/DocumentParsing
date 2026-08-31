#!/usr/bin/env python3
"""Run the pinned vessel/voyage structural SDV benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from document_ocr.synthesis.transport_identity_benchmark_pipeline import (
    load_transport_identity_benchmark_config,
    run_transport_identity_benchmark_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    project_root = arguments.project_root.resolve(strict=True)
    config_path = arguments.config.resolve(strict=True)
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("configuration must be a regular file")
    config = load_transport_identity_benchmark_config(config_path)
    result = run_transport_identity_benchmark_pipeline(
        project_root=project_root,
        config_path=config_path,
        config=config,
    )
    print(json.dumps({"status": "complete", "result": result}, sort_keys=True))


if __name__ == "__main__":
    main()
