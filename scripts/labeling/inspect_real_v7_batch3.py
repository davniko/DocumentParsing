"""Read-only review view: original line numbers and complete candidate labels."""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "artifacts/kie-labeling/direct-real-batch003-20261004"


def read(path):
    return json.loads(path.read_text())


def baseline(row):
    path = BATCH / "runs" / row["name"] / "refine/target.json"
    if not path.exists():
        assert row["name"] in ("003-cd075e92", "078-4704110c"), "Unexpected missing final target"
        path = BATCH / "runs" / row["name"] / "refine/reviewed-draft.json"
    return path, read(path)


def show(numbers):
    selection = read(BATCH / "selection.json")
    for row in selection["documents"]:
        if row["index"] not in numbers:
            continue
        print("\nDOCUMENT", row["name"], row["split"], "PDF", row["pdf"])
        for n, line in enumerate((ROOT / row["input"] / "ocr.txt").read_text().splitlines(), 1):
            if len(line) > 300 and line.startswith(
                (
                    "The Merchant",
                    "This shipment is subject to compliance",
                    "The contract evidenced",
                    "Received by the Carrier",
                    "RECEIVED by the Carrier",
                    "The Merchant(s)",
                    "If this shipment is in violation",
                )
            ):
                print(
                    n,
                    "[LONG LEGAL LINE; original retained]",
                    line[:70],
                    "URLS",
                    re.findall(r"(?:https?://|www\.)\S+", line),
                )
            elif line.strip():
                print(n, line)
        path, target = baseline(row)
        print("TARGET", path)
        for k, v in target["documentPatch"].items():
            if v is not None:
                print(k, json.dumps(v, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("numbers", type=int, nargs="+")
    show(parser.parse_args().numbers)
