"""Publish one completely adjudicated full-field batch without mixing projections."""

from __future__ import annotations

import argparse
import json
import os
import resource
import tempfile
import time
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium
from publish_real_v7_batch2 import apply_decisions, validate_rows
from review_real_v7_batch import ROOT, candidate, read

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.hashing import sha256_bytes, sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7Label


def publish(batches, destination, exclusions_path, expected_count, commit=False):
    start = time.monotonic()
    assert not destination.exists()
    exclusions = read(exclusions_path)
    assert set(exclusions) <= {str(b.relative_to(ROOT)) for b in batches}
    manifest = {"documents": []}
    decisions = {}
    notes = {}
    selections = []
    excluded = []
    for batch in batches:
        batch_key = str(batch.relative_to(ROOT))
        selected = read(batch / "selection.json")
        reviewed = read(batch / "manual-review/decisions.json")
        source_notes = read(batch / "manual-review/source-notes.json")
        omitted = exclusions.get(batch_key, {})
        names = {r["name"] for r in selected["documents"]}
        assert len(names) == len(selected["documents"])
        assert not (set(reviewed) & set(omitted))
        assert set(reviewed) | set(omitted) == names, (
            "Every paid selection needs an explicit disposition"
        )
        assert set(source_notes) == {f"{r['index']:03}" for r in selected["documents"]}
        selections.append(
            dict(
                path=str((batch / "selection.json").relative_to(ROOT)),
                sha256=sha256_file(batch / "selection.json"),
            )
        )
        for row in selected["documents"]:
            assert source_notes[f"{row['index']:03}"]
            if row["name"] in omitted:
                assert omitted[row["name"]].strip()
                excluded.append(dict(**row, sourceBatch=batch_key, reason=omitted[row["name"]]))
                continue
            key = f"{batch_key}/{row['name']}"
            assert key not in decisions
            decisions[key] = reviewed[row["name"]]
            notes[key] = source_notes[f"{row['index']:03}"]
            manifest["documents"].append(dict(**row, sourceBatch=batch_key, reviewKey=key))
    assert len(manifest["documents"]) == expected_count
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
    rows = {"train": [], "validation": []}
    receipts = []
    prior = ROOT / "data/curated/mpci-bl-real-v7-reviewed-r8-reduced-fields"
    prior_hashes = {
        str(p.relative_to(prior)): sha256_file(p) for p in prior.rglob("*") if p.is_file()
    }
    # Filtered-out records are not fresh top-up documents either.
    original450 = ROOT / "data/curated/mpci-bl-real-v7-reviewed-r4_source_20261005"
    prior_records = [
        json.loads(line)
        for split in rows
        for line in (original450 / f"{split}.jsonl").read_text().splitlines()
    ]
    ids = {r["documentId"] for r in prior_records}
    hashes = {r["joinedRawTextSha256"] for r in prior_records}
    prepared = []
    for row in manifest["documents"]:
        doc = row["documentId"]
        decision = decisions[row["reviewKey"]]
        assert decision["reviewed"] and decision["notes"] and notes[row["reviewKey"]]
        path, initial, state = candidate(ROOT / row["sourceBatch"], row)
        assert sha256_file(path) == decision["inputTargetSha256"]
        target = apply_decisions(initial, decision["changes"])
        canonical = BillOfLadingExtractionV7Label.model_validate_json(
            json.dumps(target)
        ).canonical_target()
        assert canonical == target
        # Manual additions can change insertion order, not values. Serialize in
        # schema order so the relation list remains the final goods field.
        target = canonical
        text = (ROOT / row["input"] / "ocr.txt").read_bytes()
        assert doc not in ids and row["ocrSha256"] not in hashes
        ids.add(doc)
        hashes.add(row["ocrSha256"])
        assert sha256_bytes(text) == row["ocrSha256"]
        assert original[doc]["joinedRawText"].encode() == text
        assert original[doc]["auditSplit"] == row["split"]
        assert sha256_file(ROOT / row["pdf"]) == row["pdfSha256"]
        with pdfium.PdfDocument(ROOT / row["pdf"]) as pdf:
            row["pages"] = len(pdf)
        assert 1 <= row["pages"] <= 5
        prepared.append((row, path, target, text, state))
    stage = Path(tempfile.mkdtemp(prefix=".reviewed-full-batch-", dir=destination.parent))
    for row, path, target, text, state in prepared:
        doc = row["documentId"]
        folder = stage / "samples" / doc
        decision = decisions[row["reviewKey"]]
        atomic_publish_bytes(folder / "ocr.txt", text)
        atomic_publish_json(folder / "labels.json", target)
        rows[row["split"]].append(
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
                sourceBatch=row["sourceBatch"],
                pilotDocument=row["name"],
                split=row["split"],
                pages=row["pages"],
                status="manually_adjudicated",
                sourceOcrSha256=row["ocrSha256"],
                pdf=row["pdf"],
                pdfSha256=row["pdfSha256"],
                inputTarget=str(path.relative_to(ROOT)),
                inputTargetSha256=sha256_file(path),
                editFormat="explicit_add_remove_v1",
                changes=decision["changes"],
                decisions=[notes[row["reviewKey"]], *decision["notes"]],
                targetSha256=sha256_file(folder / "labels.json"),
                originalAgentStatus=state["status"],
            )
        )
    for split, records in rows.items():
        atomic_publish_bytes(
            stage / f"{split}.jsonl",
            "".join(
                json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records
            ).encode(),
        )
    mapping = {r["documentId"]: r for r in receipts}
    replay_checks = validate_rows(stage, mapping)
    checks = dict(replay_checks)
    assert prior_hashes == {
        str(p.relative_to(prior)): sha256_file(p) for p in prior.rglob("*") if p.is_file()
    }
    checks.update(
        manuallyReviewed=len(receipts),
        remainingAdjudications=0,
        changedDocuments=sum(bool(r["changes"]) for r in receipts),
        unchangedDocuments=sum(not r["changes"] for r in receipts),
        agentStatuses=dict(Counter(r["originalAgentStatus"] for r in receipts)),
        preservedBaseline=str(prior.relative_to(ROOT)),
        preservedBaselineFiles=len(prior_hashes),
        elapsedSeconds=time.monotonic() - start,
        peakRssKiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    checks.update(
        paidSelections=len(receipts) + len(excluded),
        excludedSelections=len(excluded),
        pages=dict(Counter(r["pages"] for r in receipts)),
        noOverlapWithOriginal450=True,
    )
    atomic_publish_json(
        stage / "manifest.json",
        dict(
            samples=len(receipts),
            splits={k: len(v) for k, v in rows.items()},
            selections=selections,
            exclusions=excluded,
            documents=receipts,
            files={f"{s}.jsonl": sha256_file(stage / f"{s}.jsonl") for s in rows},
        ),
    )
    atomic_publish_json(stage / "validation.json", checks)
    atomic_publish_json(stage / "schema.json", BillOfLadingExtractionV7Label.model_json_schema())
    atomic_publish_json(stage / "manual-decisions.json", decisions)
    atomic_publish_json(stage / "source-notes.json", notes)
    atomic_publish_bytes(
        stage / "README.md",
        (
            "# Reviewed real top-up: full annotation contract\n\n"
            + f"{len(receipts)} individually source-reviewed documents: "
            + f"{len(rows['train'])} train / {len(rows['validation'])} validation. "
            + "Original OCR/PDF/splits remain unchanged.\n\n"
            + "These annotations retain vessel flag, marks and forwarding/export references. "
            + "They have not been concatenated with the reduced-field R8 baseline: subsequent "
            + "filtering and an aligned field projection are separate user decisions. "
            + "This full batch and its exact edit receipts remain recoverable.\n"
        ).encode(),
    )
    if commit:
        os.rename(stage, destination)
        assert validate_rows(destination, mapping) == replay_checks
        print("PUBLISHED", destination)
    else:
        print("STAGED", stage)
    print(json.dumps(checks, indent=2))
    return destination if commit else stage


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, nargs="+", required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--commit", action="store_true")
    p.add_argument("--exclusions", type=Path, required=True)
    p.add_argument("--expected-count", type=int, required=True)
    a = p.parse_args()
    publish(
        [ROOT / b for b in a.batch],
        ROOT / a.destination,
        ROOT / a.exclusions,
        a.expected_count,
        a.commit,
    )
