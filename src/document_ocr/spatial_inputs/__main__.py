"""Build an input-only dataset variant from a pinned, completed Paddle run."""

import argparse
import json
from pathlib import Path

import yaml

from document_ocr.spatial_inputs.dataset import SpatialDatasetConfig, build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    config = SpatialDatasetConfig.model_validate(yaml.safe_load(args.config.read_text()))
    print(json.dumps(build(config, args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
