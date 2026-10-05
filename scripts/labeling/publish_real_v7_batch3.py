"""Publish the finite batch003 adjudications as a fresh, fully replayed revision.

No provider calls, input edits, inferred defaults or production labeling changes.
This is a one-time publication operation; an existing destination is an error.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

from adjudicate_real_v7_batch3 import reviewed
from inspect_real_v7_batch3 import BATCH, ROOT, baseline, read
from publish_real_v7_batch2 import apply_decisions, validate_rows

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.hashing import sha256_bytes, sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7Label

PREVIOUS = ROOT / "data/curated/mpci-bl-real-v7-reviewed-r2"
DESTINATION = ROOT / "data/curated/mpci-bl-real-v7-reviewed-r3"


def differences(before, after, path=()):
    """Exact dictionary edits; changed arrays are replaced atomically, never shifted."""
    reason = "Finite source/OCR adjudication; full explanation in this document receipt decisions."
    if isinstance(before, dict) and isinstance(after, dict):
        for key, value in before.items():
            if key not in after:
                yield dict(path=[*path, key], before=value, remove=True, reason=reason)
            else:
                yield from differences(value, after[key], (*path, key))
        for key, value in after.items():
            if key not in before:
                yield dict(path=[*path, key], add=value, reason=reason)
    elif before != after:
        yield dict(path=list(path), before=before, after=after, reason=reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    assert not DESTINATION.exists(), "Existing revision must never be overwritten"
    selection = read(BATCH / "selection.json")["documents"]
    assert len(selection) == 100 and len({r["documentId"] for r in selection}) == 100
    old_manifests = [read(p) for p in sorted((PREVIOUS / "batches").glob("*/manifest.json"))]
    old_receipts = {r["documentId"]: r for m in old_manifests for r in m["documents"]}
    before_validation = validate_rows(PREVIOUS, old_receipts)
    assert before_validation["train"] == 130 and before_validation["validation"] == 20
    for name, digest in old_manifests[-1]["files"].items():
        assert sha256_file(PREVIOUS / name) == digest
    original = {
        r["documentId"]: r
        for r in map(
            json.loads,
            (
                ROOT
                / "artifacts/kie-training/analysis/real-data-baseline-audit-20261001"
                / "trained_real_v6.jsonl"
            )
            .read_text()
            .splitlines(),
        )
    }
    stage = Path(tempfile.mkdtemp(prefix=".mpci-v7-batch003-", dir=PREVIOUS.parent))
    shutil.copytree(
        PREVIOUS,
        stage,
        dirs_exist_ok=True,
        ignore=lambda directory, names: (
            set(names) & {"train.jsonl", "validation.jsonl", "README.md"}
            if Path(directory) == PREVIOUS
            else set()
        ),
    )
    records = {
        s: list(map(json.loads, (PREVIOUS / f"{s}.jsonl").read_text().splitlines()))
        for s in ("train", "validation")
    }
    receipts, decisions, operations, statuses = [], {}, Counter(), Counter()
    for index, row in enumerate(selection, 1):
        assert int(row["name"][:3]) == index
        doc = row["documentId"]
        assert doc not in old_receipts
        source, initial = baseline(row)
        reviewed_target, notes = reviewed(index, initial)
        target = BillOfLadingExtractionV7Label.model_validate_json(
            json.dumps(reviewed_target)
        ).canonical_target()
        changes = list(differences(initial["documentPatch"], target["documentPatch"]))
        assert apply_decisions(initial, changes) == target
        text = (ROOT / row["input"] / "ocr.txt").read_bytes()
        assert sha256_bytes(text) == row["ocrSha256"]
        assert original[doc]["joinedRawText"].encode() == text
        assert original[doc]["auditSplit"] == row["split"]
        assert sha256_file(ROOT / row["pdf"]) == row["pdfSha256"]
        folder = stage / "samples" / doc
        assert not folder.exists()
        atomic_publish_bytes(folder / "ocr.txt", text)
        atomic_publish_json(folder / "labels.json", target)
        records[row["split"]].append(
            dict(
                documentId=doc,
                joinedRawText=text.decode(),
                joinedRawTextSha256=row["ocrSha256"],
                target=target,
            )
        )
        status = read(BATCH / "runs" / row["name"] / "summary.json")["status"]
        statuses[status] += 1
        decisions[row["name"][:3]] = dict(name=row["name"], notes=notes, changes=changes)
        receipts.append(
            dict(
                documentId=doc,
                pilotDocument=row["name"],
                split=row["split"],
                status="manually_adjudicated",
                sourceOcrSha256=row["ocrSha256"],
                pdf=row["pdf"],
                pdfSha256=row["pdfSha256"],
                inputTarget=str(source.relative_to(ROOT)),
                inputTargetSha256=sha256_file(source),
                editFormat="explicit_add_remove_v1",
                changes=changes,
                decisions=notes,
                targetSha256=sha256_file(folder / "labels.json"),
                originalAgentStatus=status,
            )
        )
        operations.update(c["path"][0] for c in changes)
    for split, rows in records.items():
        atomic_publish_bytes(
            stage / f"{split}.jsonl",
            "".join(
                json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows
            ).encode(),
        )
    all_receipts = {**old_receipts, **{r["documentId"]: r for r in receipts}}
    result = validate_rows(stage, all_receipts)
    assert result["train"] == 220 and result["validation"] == 30
    preserved = 0
    for old in PREVIOUS.rglob("*"):
        if old.is_file() and not (
            old.parent == PREVIOUS and old.name in ("train.jsonl", "validation.jsonl", "README.md")
        ):
            assert sha256_file(old) == sha256_file(stage / old.relative_to(PREVIOUS))
            preserved += 1
    # Existing JSONL records stay identical, not merely schema-equivalent.
    for split in records:
        old_lines = (PREVIOUS / f"{split}.jsonl").read_bytes()
        assert (stage / f"{split}.jsonl").read_bytes().startswith(old_lines)
    decision_file = stage / "batches/003/manual-decisions.json"
    atomic_publish_json(decision_file, dict(batch="003", documents=decisions))
    result.update(
        manuallyReviewed=100,
        remainingAdjudications=0,
        originalAgentStatuses=dict(statuses),
        changedDocuments=sum(bool(r["changes"]) for r in receipts),
        unchangedDocuments=sum(not r["changes"] for r in receipts),
        changedAutomaticPasses=sum(
            bool(r["changes"]) and r["originalAgentStatus"] == "reviewed_candidate"
            for r in receipts
        ),
        editOperations=sum(operations.values()),
        editSections=dict(operations),
        preservedHistoricalFiles=preserved,
        priorDatasetValidation=before_validation,
        paidCalls=0,
        estimatedUsd=0,
        elapsedSeconds=time.monotonic() - started,
        peakRssKiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    manifest = dict(
        batch="003",
        samples=100,
        totalDatasetSamples=250,
        splits=dict(Counter(r["split"] for r in receipts)),
        sourceSelection=str((BATCH / "selection.json").relative_to(ROOT)),
        sourceSelectionSha256=sha256_file(BATCH / "selection.json"),
        decisionsSha256=sha256_file(decision_file),
        implementationSha256=sha256_file(Path(__file__)),
        adjudicationImplementationSha256=sha256_file(
            Path(__file__).with_name("adjudicate_real_v7_batch3.py")
        ),
        files={f"{s}.jsonl": sha256_file(stage / f"{s}.jsonl") for s in records},
        documents=receipts,
    )
    atomic_publish_json(stage / "batches/003/manifest.json", manifest)
    atomic_publish_json(stage / "batches/003/validation.json", result)
    atomic_publish_bytes(
        stage / "README.md",
        (
            "# Reviewed real Bill-of-Lading labels — revision 3\n\n"
            "250 source documents: **220 train / 30 validation**. Schema 7.0.0.\n\n"
            "Batch003 adds 100 manually source-reviewed and adjudicated documents "
            "(90 train /10 validation), including all automatic passes and all held candidates. "
            "All 100 are accepted under the OCR-grounded annotation policy. "
            "No document remains held.\n\n"
            "OCR, PDFs and original splits are unchanged. Layout consultation does not import "
            "values absent from OCR. Ambiguous or unsupported optional values are deliberately "
            "absent and documented in each receipt; this includes two malformed container IDs, "
            "missing units, PDF-only fields and unallocated shared quantities.\n\n"
            "The preceding 150 records and their sample files/receipts are unchanged; the complete "
            "earlier revision remains at ../mpci-bl-real-v7-reviewed-r2/.\n\n"
            "Exact replay receipts, per-document adjudication explanations and validation: "
            "batches/003/. Historical aggregate hashes refer to their original revision; "
            "the latest batch manifest records this combined revision.\n\n"
            "Certification here is source-grounded manual adjudication plus mechanical publication "
            "validation, not automatic-agent agreement or a claim that schema validation detects "
            "every semantic mistake. See docs/kie-real-reviewed-dataset-build-2026-10-04.md "
            "for methods and remaining automation limitations.\n"
        ).encode(),
    )
    if args.publish:
        os.rename(stage, DESTINATION)
        assert validate_rows(DESTINATION, all_receipts)["schemasAndExactReplays"] == 250
        print("PUBLISHED", DESTINATION)
    else:
        print("STAGED ONLY", stage)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
