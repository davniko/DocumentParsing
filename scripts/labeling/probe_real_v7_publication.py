"""Exercise publication rejection gates against an isolated real-data copy."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
import time
from pathlib import Path

from publish_real_v7_batch2 import ROOT, apply_decisions, read, validate_rows

from document_ocr.atomic import atomic_publish_json, atomic_write_bytes, atomic_write_json
from document_ocr.hashing import sha256_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/curated/mpci-bl-real-v7-reviewed-r2")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/kie-labeling/direct-real-batch002-20261004/manual-review/publication-negative-controls.json"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        nargs="+",
        help="Explicit manifests for a standalone publication instead of dataset/batches/*.",
    )
    args = parser.parse_args()
    started = time.monotonic()
    dataset = ROOT / args.dataset
    manifests = (
        [ROOT / p for p in args.manifest]
        if args.manifest
        else sorted((dataset / "batches").glob("*/manifest.json"))
    )
    assert manifests, "A publication manifest is required"
    receipts = {r["documentId"]: r for p in manifests for r in read(p)["documents"]}
    results = {}
    with tempfile.TemporaryDirectory(prefix="mpci-publication-probe-") as temporary:
        clone = Path(temporary) / "dataset"
        shutil.copytree(dataset, clone)
        results["positiveControl"] = validate_rows(clone, receipts)
        path = clone / "train.jsonl"
        original = path.read_bytes()
        rows = [json.loads(line) for line in original.splitlines()]
        record = rows[0]
        sample = clone / "samples" / record["documentId"]
        original_label = (sample / "labels.json").read_bytes()
        original_ocr = (sample / "ocr.txt").read_bytes()

        def expect_rejected(name, changed_rows):
            atomic_write_bytes(
                path,
                b"".join(
                    (json.dumps(row, ensure_ascii=False) + "\n").encode() for row in changed_rows
                ),
            )
            try:
                validate_rows(clone, receipts)
            except AssertionError:
                results[name] = "rejected"
            else:
                raise AssertionError(f"Gate accepted mutation: {name}")
            atomic_write_bytes(path, original)
            atomic_write_bytes(sample / "labels.json", original_label)
            atomic_write_bytes(sample / "ocr.txt", original_ocr)

        changed = copy.deepcopy(rows)
        changed[0]["target"]["documentPatch"]["billOfLadingNumber"] = "UNRECORDED"
        atomic_write_json(sample / "labels.json", changed[0]["target"])
        expect_rejected("matchingJsonlAndSampleLabelStillNeedsReceipt", changed)

        changed = copy.deepcopy(rows)
        changed[0]["joinedRawText"] += "\nUNRECORDED INPUT CHANGE"
        changed[0]["joinedRawTextSha256"] = sha256_bytes(changed[0]["joinedRawText"].encode())
        atomic_write_bytes(sample / "ocr.txt", changed[0]["joinedRawText"].encode())
        expect_rejected("matchingInputAndHashStillNeedsOriginalSource", changed)
        expect_rejected("duplicateDocument", [*rows, rows[0]])
        expect_rejected("missingDocument", rows[1:])

        for name, changed in (
            (
                "staleBeforeValue",
                [dict(path=["billOfLadingNumber"], before="WRONG", after="OTHER", reason="test")],
            ),
            ("addCannotOverwrite", [dict(path=["billOfLadingNumber"], add="OTHER", reason="test")]),
        ):
            try:
                apply_decisions({"documentPatch": {"billOfLadingNumber": "ORIGINAL"}}, changed)
            except AssertionError:
                results[name] = "rejected"
            else:
                raise AssertionError(name)
        results["restoredPositiveControl"] = validate_rows(clone, receipts)
    results["elapsedSeconds"] = time.monotonic() - started
    results["scope"] = (
        "Publication/provenance integrity; semantic acceptance uses recorded source review."
    )
    destination = ROOT / args.output
    atomic_publish_json(destination, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
