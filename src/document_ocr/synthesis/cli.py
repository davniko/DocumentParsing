"""CLI for the non-generative, lossless synthesis foundation stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from document_ocr.synthesis.config import load_synthesis_foundation_config
from document_ocr.synthesis.pipeline import prepare_synthesis_foundation


def main() -> None:
    parser = argparse.ArgumentParser(prog="document-kie-synthesis")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-config", "prepare-foundation"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    try:
        project_root = arguments.project_root.resolve(strict=True)
        if arguments.config.is_symlink():
            raise ValueError("configuration must be a real file")
        config_path = arguments.config.resolve(strict=True)
        if not config_path.is_file():
            raise ValueError("configuration must be a real file")
        config = load_synthesis_foundation_config(config_path)
        if arguments.command == "validate-config":
            result = {"command": arguments.command, "status": "valid", "run_id": config.run.run_id}
        else:
            result = {
                "command": arguments.command,
                "status": "complete",
                "result": prepare_synthesis_foundation(
                    project_root=project_root, config_path=config_path, config=config
                ),
            }
    except KeyboardInterrupt:
        print(json.dumps({"status": "error", "error_type": "KeyboardInterrupt"}))
        raise SystemExit(130) from None
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "diagnostic": str(error),
                },
                sort_keys=True,
            )
        )
        raise SystemExit(4) from None
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    raise SystemExit(0)


if __name__ == "__main__":
    main()
