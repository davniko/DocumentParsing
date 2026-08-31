#!/usr/bin/env python3
"""Run the pinned GPU generator-method experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from document_ocr.synthesis.generator_method_probe import (
    load_generator_method_probe_config,
    run_generator_method_probe,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    arguments = parser.parse_args()
    project_root = arguments.project_root.resolve(strict=True)
    config_path = arguments.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config_path = config_path.resolve(strict=True)
    config = load_generator_method_probe_config(config_path)
    result = run_generator_method_probe(
        project_root=project_root,
        config_path=config_path,
        config=config,
    )
    print(json.dumps({"status": "complete", "result": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
