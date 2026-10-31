"""Run explicitly selected inspection batches with fresh processes per stage.

This orchestrator records holds; it never signs acceptance, changes training data,
or retries failed proposals silently. Each batch keeps native and replay evidence.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from document_ocr.atomic import atomic_publish_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batches", nargs="+", type=int, required=True)
    args = parser.parse_args()
    for number in args.batches:
        directory = args.root / f"batch-{number:02}"
        config = directory / "prepared-config.json"
        if not config.exists():
            command = [
                sys.executable,
                "scripts/synthesis/prepare_multirow_batch.py",
                "--output",
                str(directory),
                "--run",
            ]
            subprocess.run(command, check=True)
        if not config.exists():
            print(json.dumps({"batch": number, "status": "PREPARATION_HELD"}), flush=True)
            continue
        subprocess.run(
            [
                sys.executable,
                "scripts/synthesis/run_package_alias_batch.py",
                "--config",
                str(config),
                "--prepared",
            ],
            check=True,
        )
        latest = max((directory / "stage-runs").iterdir(), key=lambda p: p.name)
        report = json.loads((latest / "summary.json").read_text())
        atomic_publish_json(
            directory / "latest-inspection.json",
            {
                "status": "AWAITING_MAIN_AGENT_REVIEW",
                "summary": str(latest / "summary.json"),
                "geometryCounts": report["stages"]["geometry"]["counts"],
            },
        )
        print(
            json.dumps(
                {
                    "batch": number,
                    "status": "INSPECTED_NOT_CERTIFIED",
                    "geometry": report["stages"]["geometry"]["counts"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
