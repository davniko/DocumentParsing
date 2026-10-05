"""Publish the 25 explicitly approved mixed-cohort exclusions, preserving R6."""

import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import shutil
import tempfile
import time

from filter_real_v7_starting_dataset import (
    ANALYSIS, ROOT, digest, read, tree_hashes, validate_records, write_json,
)

SOURCE = ROOT / "data/curated/mpci-bl-real-v7-reviewed-r6-single-goods"
DESTINATION = SOURCE.with_name("mpci-bl-real-v7-reviewed-r7-input-filtered")
REVIEW = ANALYSIS / "mixed_no_container_review"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    before = tree_hashes(SOURCE)
    prior = read(SOURCE / "filter-manifest.json")
    assert before == prior["files"] | {"filter-manifest.json": digest(SOURCE / "filter-manifest.json")}
    assert before == read(REVIEW / "dataset-before-sha256.json")
    inventory = {r["documentId"]: r for r in read(ANALYSIS / "inventory.json")}
    reviewed = {r["sample"]: r for r in read(REVIEW / "inventory.json")}
    judgments = read(REVIEW / "adjudications.json")
    approved = {
        j["sample"]: j["category"] for j in judgments
        if j["recommendation"].startswith("filter_")
        or j["sample"] == "003/068-df5d72cd"
    }
    assert len(approved) == 25
    for sample in approved:
        row = reviewed[sample]
        assert digest(ROOT / row["pdf"]) == row["pdfSha256"]
    originals = {}
    patches = {}
    for split in ("train", "validation"):
        for line in (SOURCE / f"{split}.jsonl").read_bytes().splitlines(keepends=True):
            record = json.loads(line)
            key = record["documentId"]
            assert key not in originals
            originals[key] = (split, line)
            patches[key] = record["target"]["documentPatch"]
    validate_records(SOURCE, originals)
    removed = {key: approved[inventory[key]["sample"]] for key in originals if inventory[key]["sample"] in approved}
    retained = {key: row for key, row in originals.items() if key not in removed}
    assert len(originals) == 360 and len(removed) == 25 and len(retained) == 335
    assert all(len(patches[k]["goodsItemDetails"]) == 1 and patches[k]["goodsItemDetails"][0].get("description") for k in retained)
    counts = dict(Counter(split for split, _ in retained.values()))
    assert counts == {"train": 298, "validation": 37}
    manifest = {
        "date": "2026-10-05", "schemaVersion": "7.0.0",
        "sourceDataset": str(SOURCE.relative_to(ROOT)), "dataset": str(DESTINATION.relative_to(ROOT)),
        "before": dict(Counter(s for s, _ in originals.values())), "after": counts,
        "excludedCount": len(removed), "retainedCount": len(retained),
        "excludedReasons": dict(Counter(removed.values())),
        "excluded": [{"documentId": k, "sample": inventory[k]["sample"], "split": originals[k][0], "reason": reason} for k, reason in removed.items()],
        "retained": [{"documentId": k, "sample": inventory[k]["sample"], "split": value[0]} for k, value in retained.items()],
        "sourceFiles": before,
        "auditInputs": {str(p.relative_to(ROOT)): digest(p) for p in (REVIEW / "inventory.json", REVIEW / "adjudications.json")},
        "policy": "User-approved 24 input-quality exclusions plus explicit trailer sample 003/068-df5d72cd. No rare fields removed; no OCR, label, split or retained-order changes.",
        "labelEdits": 0, "ocrEdits": 0, "apiCalls": 0, "apiCostUsd": 0,
    }
    if not args.apply:
        print(json.dumps({k: manifest[k] for k in ("before", "after", "excludedCount", "excludedReasons")}, indent=2))
        return
    stage = Path(tempfile.mkdtemp(prefix=".real-v7-mixed-filter-", dir=SOURCE.parent))
    shutil.copytree(SOURCE / "batches", stage / "batches")
    shutil.copy2(SOURCE / "schema.json", stage / "schema.json")
    (stage / "samples").mkdir()
    for key in retained:
        shutil.copytree(SOURCE / "samples" / key, stage / "samples" / key)
    for split in ("train", "validation"):
        with (stage / f"{split}.jsonl").open("wb") as stream:
            for s, line in retained.values():
                if s == split:
                    stream.write(line)
    validate_records(stage, retained)
    for name, sha in tree_hashes(stage).items():
        if name not in ("train.jsonl", "validation.jsonl"):
            assert before[name] == sha
    assert tree_hashes(SOURCE) == before
    (stage / "README.md").write_text(
        "# Reviewed real baseline R7 — input-quality filtered\n\n"
        "**335 documents: 298 training / 37 validation.** Published 2026-10-05.\n\n"
        "Applies the 24 approved mixed-cohort input-quality exclusions and the explicit "
        "trailer exclusion `003/068-df5d72cd` to R6. All retained records have exactly one "
        "described goods entry. Multiple containers remain in scope. Retained OCR, targets, "
        "row order and split assignments are byte-preserved. No fields were removed.\n\n"
        "`filter-manifest.json` is authoritative for membership and exclusions. `batches/` "
        "contains unchanged historical receipts, including records no longer retained. "
        "Current described schema: `batches/004/label-schema.json`.\n\n"
        "R6 is preserved unchanged at `../mpci-bl-real-v7-reviewed-r6-single-goods/`; "
        "the original 450-record backup remains at `../mpci-bl-real-v7-reviewed-r4_source_20261005/`. "
        "All exclusions are recoverable. Training configurations have not been redirected.\n\n"
        "Analysis: `docs/analysis/real-v7-starting-dataset-2026-10-04/R7_FIELD_AUDIT_2026-10-05.md`. "
        "This is the selected baseline, not a claim of perfect source transcription or a "
        "new semantic adjudication of all goods identities.\n"
    )
    manifest["files"] = tree_hashes(stage)
    manifest["elapsedSeconds"] = round(time.monotonic() - started, 3)
    manifest["peakRssKiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(stage / "filter-manifest.json", manifest)
    stage.rename(DESTINATION)
    print(json.dumps({k: manifest[k] for k in ("dataset", "after", "excludedReasons", "elapsedSeconds", "peakRssKiB")}, indent=2))


if __name__ == "__main__":
    main()
