"""Replay selected inspection batches against the final production authorities."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", required=True)
    args = parser.parse_args()
    for number in args.batches:
        directory = args.root / f"batch-{number:02}" / "production-replay"
        config = directory / "config.json"
        if not config.exists():
            raise ValueError("explicit final replay configuration is missing: " + str(config))
        subprocess.run(
            [
                sys.executable,
                "scripts/synthesis/check_package_alias_candidates.py",
                "--config",
                str(config),
                "--output",
                str(directory / "geometry"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
