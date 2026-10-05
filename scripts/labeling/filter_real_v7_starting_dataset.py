"""Publish the user-approved 2026-10-05 filter without altering source records.

Dry-run by default. --apply creates a complete source backup and a separate
filtered dataset. Existing destinations are never overwritten. Historical
batch receipts are preserved; filter-manifest.json defines current membership.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7Label

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/curated/mpci-bl-real-v7-reviewed-r4"
BACKUP = SOURCE.with_name("mpci-bl-real-v7-reviewed-r4_source_20261005")
DESTINATION = SOURCE.with_name("mpci-bl-real-v7-reviewed-r5-filtered")
ANALYSIS = ROOT / "docs/analysis/real-v7-starting-dataset-2026-10-04"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}


def read(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def reasons_for(row: dict, cohorts: dict) -> list[str]:
    reasons = []
    if row["pdfPages"] > 5:
        reasons.append("pdf_over_5_pages")
    if row["sample"] in cohorts["scope_candidates"]["samples"]:
        reasons.append("scope_or_test_demo")
    if row["sample"] in cohorts["pdf_confirmed_missing_cargo_in_ocr"]["samples"]:
        reasons.append("pdf_confirmed_missing_cargo_in_ocr")
    return reasons


def validate_records(directory: Path, expected: dict[str, tuple[str, bytes]]) -> None:
    seen = set()
    for split in ("train", "validation"):
        for line in (directory / f"{split}.jsonl").read_bytes().splitlines(keepends=True):
            record = json.loads(line)
            key = record["documentId"]
            assert key not in seen, f"Duplicate document: {key}"
            assert expected[key] == (split, line), f"Changed row or split: {key}"
            seen.add(key)
            BillOfLadingExtractionV7Label.model_validate_json(json.dumps(record["target"]))
            sample = directory / "samples" / key
            text = (sample / "ocr.txt").read_text()
            assert text == record["joinedRawText"], f"OCR copy differs: {key}"
            assert hashlib.sha256(text.encode()).hexdigest() == record["joinedRawTextSha256"]
            assert read(sample / "labels.json") == record["target"], f"Label copy differs: {key}"
    assert seen == set(expected), "Wrong document membership"
    assert {p.name for p in (directory / "samples").iterdir()} == seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    if BACKUP.exists() or DESTINATION.exists():
        raise FileExistsError("Source-backup or filtered destination already exists; will not overwrite.")

    source_hashes = tree_hashes(SOURCE)
    assert source_hashes == read(ANALYSIS / "summary.json")["originalDatasetHashes"], "Source changed since audit"
    inventory = read(ANALYSIS / "inventory.json")
    cohorts = read(ANALYSIS / "reviewed_cohorts.json")
    rows = {r["documentId"]: r for r in inventory}
    assert len(rows) == len(inventory)
    source_records = {}
    for split in ("train", "validation"):
        for line in (SOURCE / f"{split}.jsonl").read_bytes().splitlines(keepends=True):
            key = json.loads(line)["documentId"]
            assert key not in source_records
            assert rows[key]["split"] == split
            source_records[key] = (split, line)
    assert set(rows) == set(source_records)
    validate_records(SOURCE, source_records)

    excluded = []
    for key, row in rows.items():
        reasons = reasons_for(row, cohorts)
        if reasons:
            excluded.append({k: row[k] for k in ("documentId", "sample", "split", "pdf", "pdfPages")} | {"reasons": reasons})
    excluded_ids = {r["documentId"] for r in excluded}
    retained = {k: v for k, v in source_records.items() if k not in excluded_ids}
    retained_rows = [rows[k] for k in retained]
    manifest = {
        "date": "2026-10-05", "schemaVersion": "7.0.0",
        "sourceDataset": str(SOURCE.relative_to(ROOT)),
        "sourceBackup": str(BACKUP.relative_to(ROOT)),
        "dataset": str(DESTINATION.relative_to(ROOT)),
        "policy": {
            "maximumPdfPages": 5,
            "removeReviewedScopeAndTestDemoCandidates": True,
            "removePdfConfirmedCargoOcrOmissions": True,
            "descriptionEmptyReviewCandidates": "Retained; not established as the same OCR defect.",
            "duplicateShipments": "Retained pending discussion; no deduplication authorized in this pass.",
            "contentChanges": "None: retained JSONL bytes, labels, OCR, ordering and split assignments are preserved.",
        },
        "auditInputs": {str(p.relative_to(ROOT)): digest(p) for p in
                        (ANALYSIS / "inventory.json", ANALYSIS / "reviewed_cohorts.json", ANALYSIS / "summary.json")},
        "before": dict(Counter(s for s, _ in source_records.values())),
        "after": dict(Counter(s for s, _ in retained.values())),
        "excludedCount": len(excluded), "retainedCount": len(retained),
        "excludedReasons": dict(Counter(reason for r in excluded for reason in r["reasons"])),
        "excluded": excluded,
        "retained": [{"documentId": k, "sample": rows[k]["sample"], "split": v[0]} for k, v in retained.items()],
        "remainingTraits": dict(Counter(f for r in retained_rows for f in r["flags"])),
        "remainingTotals": {k: sum(r[k] for r in retained_rows) for k in ("goodsCount", "containerCount")},
        "remainingSpecialCargoDocuments": {k: sum(bool(r[k]) for r in retained_rows) for k in ("dgGoods", "thermalContainers")},
        "apiCalls": 0, "apiCostUsd": 0,
    }
    if not args.apply:
        print(json.dumps({k: v for k, v in manifest.items() if k not in ("retained", "excluded")}, indent=2))
        return

    # All work is staged; the old source is never deleted or rewritten.
    stage = Path(tempfile.mkdtemp(prefix=".real-v7-filter-20261005-", dir=SOURCE.parent))
    print(f"Staging backup and filtered dataset in {stage}", flush=True)
    backup_stage, destination_stage = stage / "source", stage / "filtered"
    shutil.copytree(SOURCE, backup_stage)
    assert tree_hashes(backup_stage) == source_hashes, "Backup is not an exact source copy"
    write_json(backup_stage / "source-snapshot.json", {
        "source": str(SOURCE.relative_to(ROOT)), "date": "2026-10-05",
        "samples": len(source_records), "splits": manifest["before"], "files": source_hashes,
        "purpose": "Unfiltered source snapshot before page/scope/OCR filtering. Do not use as the cleaned cohort.",
    })
    (backup_stage / "SOURCE_SNAPSHOT.md").write_text(
        "# Unfiltered source dataset backup — 2026-10-05\n\n"
        "450 documents: 400 training / 50 validation. All 924 original files are copied byte-for-byte; "
        "their hashes are recorded in `source-snapshot.json`. This backup intentionally retains long, "
        "scope/test/demo and incomplete-OCR samples. Original PDFs remain at the paths recorded in batch receipts.\n\n"
        "The working filtered cohort is `../mpci-bl-real-v7-reviewed-r5-filtered/`. "
        "Preserve this snapshot unchanged for recovery and provenance.\n"
    )

    destination_stage.mkdir()
    shutil.copytree(SOURCE / "batches", destination_stage / "batches")
    shutil.copy2(SOURCE / "schema.json", destination_stage / "schema.json")
    (destination_stage / "samples").mkdir()
    for key in retained:
        shutil.copytree(SOURCE / "samples" / key, destination_stage / "samples" / key)
    for split in ("train", "validation"):
        with (destination_stage / f"{split}.jsonl").open("wb") as stream:
            for source_split, line in retained.values():
                if source_split == split:
                    stream.write(line)
    validate_records(destination_stage, retained)
    for path, sha in tree_hashes(destination_stage).items():
        if path not in ("train.jsonl", "validation.jsonl"):
            assert source_hashes[path] == sha, f"Copied source content changed: {path}"
    assert tree_hashes(SOURCE) == source_hashes, "Source changed during publication"

    (destination_stage / "README.md").write_text(
        "# Reviewed real Bill-of-Lading dataset — filtered R5\n\n"
        f"{len(retained)} documents: **{manifest['after']['train']} train / {manifest['after']['validation']} validation**. "
        "Schema 7.0.0. Published 2026-10-05.\n\n"
        "The user-approved filter removes PDFs longer than five pages, the five reviewed scope/test/demo "
        "candidates and the three PDF-confirmed cargo OCR omissions. No retained OCR, labels, record order "
        "or split assignments were changed. Nine description-empty cargo records remain review candidates; "
        "duplicate shipments were not removed in this pass.\n\n"
        "`filter-manifest.json` is authoritative for this revision's current membership and exclusions. "
        "`batches/` contains unchanged historical adjudication receipts, including receipts for excluded "
        "documents; their historical aggregate counts/hashes are not current filtered membership. "
        "Only the current JSONL files and `samples/` contain the retained cohort.\n\n"
        f"Complete unfiltered backup: `../{BACKUP.name}/` (450 documents, original files verified by hash). "
        "The earlier R4 publication is also left unchanged. Excluded records remain recoverable in both.\n\n"
        "Current described schema: `batches/004/label-schema.json`. The root `schema.json` remains the original "
        "compatible snapshot. Report: `docs/analysis/real-v7-starting-dataset-2026-10-04/FILTERING_2026-10-05.md`.\n"
    )
    manifest["files"] = tree_hashes(destination_stage)
    manifest["sourceFilesVerified"] = len(source_hashes)
    manifest["elapsedSeconds"] = time.monotonic() - started
    manifest["peakRssKiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(destination_stage / "filter-manifest.json", manifest)

    backup_stage.rename(BACKUP)
    destination_stage.rename(DESTINATION)
    stage.rmdir()  # Only the known, now-empty staging directory is removed.
    print(json.dumps({k: manifest[k] for k in ("sourceBackup", "dataset", "before", "after", "excludedCount",
                                               "retainedCount", "excludedReasons", "elapsedSeconds", "peakRssKiB")}, indent=2))


if __name__ == "__main__":
    main()
