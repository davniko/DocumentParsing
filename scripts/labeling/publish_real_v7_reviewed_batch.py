"""Append an explicitly adjudicated batch without altering prior samples or OCR.

Every selected document needs a hash-bound manual decision, including unchanged
documents. Agent pass status does not authorize publication. The previous dataset
and paid model artifacts are never edited by this operation.
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

from publish_real_v7_batch2 import apply_decisions, validate_rows
from review_real_v7_batch import ROOT, candidate, read

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.hashing import sha256_bytes, sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7Label


def publish(previous: Path, batch: Path, destination: Path, number: str, commit: bool):
    started = time.monotonic()
    assert not destination.exists(), "Existing dataset revisions are immutable"
    assert previous.is_dir() and previous != destination
    selection = read(batch / "selection.json")["documents"]
    decisions_path = batch / "manual-review/decisions.json"
    decisions = read(decisions_path)
    source_notes_path = batch / "manual-review/source-notes.json"
    source_notes = read(source_notes_path)
    expected = {row["name"] for row in selection}
    assert len(expected) == len(selection)
    assert set(decisions) == expected, "Every selected document needs manual adjudication"
    old_manifests = [read(p) for p in sorted((previous / "batches").glob("*/manifest.json"))]
    old_receipts = {r["documentId"]: r for m in old_manifests for r in m["documents"]}
    before_validation = validate_rows(previous, old_receipts)
    for name, digest in old_manifests[-1]["files"].items():
        assert sha256_file(previous / name) == digest
    assert not (previous / "batches" / number).exists()
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
    records = {
        split: list(map(json.loads, (previous / f"{split}.jsonl").read_text().splitlines()))
        for split in ("train", "validation")
    }
    prepared = []
    statuses, sections = Counter(), Counter()
    # Complete all adjudication/schema/source checks before creating a revision.
    for row in selection:
        doc = row["documentId"]
        assert doc not in old_receipts
        decision = decisions[row["name"]]
        assert decision["reviewed"] is True and decision["notes"]
        assert source_notes[f"{row['index']:03}"]
        path, initial, state = candidate(batch, row)
        assert sha256_file(path) == decision["inputTargetSha256"]
        corrected = apply_decisions(initial, decision["changes"])
        target = BillOfLadingExtractionV7Label.model_validate_json(
            json.dumps(corrected)
        ).canonical_target()
        assert corrected == target, "Manual edits must already obey target semantics"
        text = (ROOT / row["input"] / "ocr.txt").read_bytes()
        assert sha256_bytes(text) == row["ocrSha256"]
        assert original[doc]["joinedRawText"].encode() == text
        assert original[doc]["auditSplit"] == row["split"]
        assert sha256_file(ROOT / row["pdf"]) == row["pdfSha256"]
        summary = read(batch / "runs" / row["name"] / "summary.json")
        assert summary["status"] == state["status"]
        statuses[summary["status"]] += 1
        sections.update(c["path"][0] for c in decision["changes"])
        prepared.append((row, path, target, text, summary))
    stage = Path(tempfile.mkdtemp(prefix=f".mpci-v7-batch{number}-", dir=destination.parent))
    shutil.copytree(
        previous,
        stage,
        dirs_exist_ok=True,
        ignore=lambda directory, names: (
            set(names) & {"train.jsonl", "validation.jsonl", "README.md"}
            if Path(directory) == previous
            else set()
        ),
    )
    receipts = []
    for row, path, target, text, summary in prepared:
        doc = row["documentId"]
        decision = decisions[row["name"]]
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
        receipts.append(
            dict(
                documentId=doc,
                pilotDocument=row["name"],
                split=row["split"],
                status="manually_adjudicated",
                sourceOcrSha256=row["ocrSha256"],
                pdf=row["pdf"],
                pdfSha256=row["pdfSha256"],
                inputTarget=str(path.relative_to(ROOT)),
                inputTargetSha256=sha256_file(path),
                editFormat="explicit_add_remove_v1",
                changes=decision["changes"],
                decisions=[source_notes[f"{row['index']:03}"], *decision["notes"]],
                targetSha256=sha256_file(folder / "labels.json"),
                originalAgentStatus=summary["status"],
            )
        )
    for split, rows in records.items():
        atomic_publish_bytes(
            stage / f"{split}.jsonl",
            "".join(
                json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows
            ).encode(),
        )
        assert (
            (stage / f"{split}.jsonl")
            .read_bytes()
            .startswith((previous / f"{split}.jsonl").read_bytes())
        )
    all_receipts = {**old_receipts, **{r["documentId"]: r for r in receipts}}
    result = validate_rows(stage, all_receipts)
    assert result["schemasAndExactReplays"] == len(all_receipts)
    preserved = 0
    for old in previous.rglob("*"):
        if old.is_file() and not (
            old.parent == previous and old.name in ("train.jsonl", "validation.jsonl", "README.md")
        ):
            assert sha256_file(old) == sha256_file(stage / old.relative_to(previous))
            preserved += 1
    batch_dir = stage / "batches" / number
    atomic_publish_json(
        batch_dir / "manual-decisions.json", dict(batch=number, documents=decisions)
    )
    atomic_publish_json(batch_dir / "source-notes.json", source_notes)
    result.update(
        manuallyReviewed=len(selection),
        remainingAdjudications=0,
        originalAgentStatuses=dict(statuses),
        changedDocuments=sum(bool(r["changes"]) for r in receipts),
        unchangedDocuments=sum(not r["changes"] for r in receipts),
        changedAutomaticPasses=sum(
            bool(r["changes"]) and r["originalAgentStatus"] == "reviewed_candidate"
            for r in receipts
        ),
        editOperations=sum(sections.values()),
        editSections=dict(sections),
        preservedHistoricalFiles=preserved,
        priorDatasetValidation=before_validation,
        paidPublicationCalls=0,
        elapsedSeconds=time.monotonic() - started,
        peakRssKiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    atomic_publish_json(
        batch_dir / "manifest.json",
        dict(
            batch=number,
            samples=len(selection),
            totalDatasetSamples=len(all_receipts),
            splits=dict(Counter(r["split"] for r in receipts)),
            sourceSelection=str((batch / "selection.json").relative_to(ROOT)),
            sourceSelectionSha256=sha256_file(batch / "selection.json"),
            decisionsSha256=sha256_file(batch_dir / "manual-decisions.json"),
            sourceNotesSha256=sha256_file(batch_dir / "source-notes.json"),
            implementationSha256=sha256_file(Path(__file__)),
            files={f"{s}.jsonl": sha256_file(stage / f"{s}.jsonl") for s in records},
            documents=receipts,
        ),
    )
    atomic_publish_json(batch_dir / "validation.json", result)
    atomic_publish_json(
        batch_dir / "label-schema.json", BillOfLadingExtractionV7Label.model_json_schema()
    )
    atomic_publish_bytes(
        stage / "README.md",
        (
            f"# Reviewed real Bill-of-Lading labels — through batch {number}\n\n"
            f"{len(all_receipts)} source documents: **{len(records['train'])} train / "
            f"{len(records['validation'])} validation**. Schema 7.0.0.\n\n"
            f"Batch {number} adds {len(selection)} individually source-reviewed documents. "
            "All automatic passes and held candidates receive explicit manual adjudication. "
            "Original OCR, PDFs and train/validation assignments are unchanged. PDF consultation "
            "establishes layout/ownership, never supplies values absent from OCR. Unsupported "
            "or ambiguous optional values are absent, with decisions recorded per document.\n\n"
            f"The previous dataset at `../{previous.name}/` and all its existing records/receipts "
            "are preserved, including user adjudications. Exact replay, source hashes, schema "
            f"and split checks are in `batches/{number}/`. Historical aggregate hashes describe "
            "their original revision; the newest manifest describes this combined dataset.\n\n"
            "Certification means source-grounded manual adjudication and mechanical publication "
            "checks, not automatic agent agreement. See "
            "`docs/kie-real-reviewed-dataset-build-2026-10-04.md` for the audit and scope.\n"
        ).encode(),
    )
    if commit:
        os.rename(stage, destination)
        assert validate_rows(destination, all_receipts)["schemasAndExactReplays"] == len(
            all_receipts
        )
        print("PUBLISHED", destination)
    else:
        print("STAGED ONLY", stage)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--number", required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    assert args.number.isdigit() and len(args.number) == 3
    publish(
        ROOT / args.previous, ROOT / args.batch, ROOT / args.destination, args.number, args.publish
    )
