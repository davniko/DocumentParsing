"""Apply the finite, source-reviewed batch-002 decisions and append atomically.

This is a dataset publication tool, not extraction logic. The decision inventory
is mandatory for every document, including unchanged and agent-passed records.
Original OCR, PDFs, model outputs and the previous dataset snapshot are preserved.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import resource
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.hashing import sha256_bytes, sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7Label
from document_ocr.synthesis.container_semantics import review_source_equipment_surface

ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "artifacts/kie-labeling/direct-real-batch002-20261004"
DEST = ROOT / "data/curated/mpci-bl-real-v7-reviewed"
PUBLISHED = ROOT / "data/curated/mpci-bl-real-v7-reviewed-r2"
DECISIONS = BATCH / "manual-review/decisions.json"

# Reviewed complete equipment phrases in this batch. Do not infer types from
# bare lengths, EQ, NOR, or RF/RA without established height. Existing corpus
# grammar supplies the actual pair, not a new parallel conversion table.
COMPLETE_SURFACES = frozenset(
    [
        "20ST",
        "20' Standard Container",
        "40ST",
        "45HC",
        "40RH",
        "40'OT",
        "40'X9'6\" HIGH CUBE CONT.",
        "20' DRY VAN",
    ]
)


def read(path):
    return json.loads(path.read_text())


def apply_decisions(target, changes):
    result = copy.deepcopy(target)
    for change in changes:
        assert change["reason"] and change["path"]
        node = result["documentPatch"]
        for part in change["path"][:-1]:
            node = node[part]
        key = change["path"][-1]
        if "add" in change:
            assert key not in node, change
            assert not any(k in change for k in ("before", "after", "remove"))
            node[key] = copy.deepcopy(change["add"])
        else:
            assert node[key] == change["before"], change
            assert ("after" in change) != bool(change.get("remove")), change
            if change.get("remove"):
                del node[key]
            else:
                node[key] = copy.deepcopy(change["after"])
    return result


def equipment_decisions(target):
    changes = []
    for index, item in enumerate(target["documentPatch"].get("containerInformation", [])):
        surface = item.get("typeDescription")
        if surface not in COMPLETE_SURFACES:
            continue
        reviewed = review_source_equipment_surface(surface, temperature_present=False)
        assert reviewed.resolution == "reviewed_source_grammar"
        assert reviewed.size_category and reviewed.type_category
        reason = "Complete printed equipment phrase: " + surface + "; " + reviewed.review_rule
        changes.extend(
            [
                dict(
                    path=["containerInformation", index, "typeDescription"],
                    before=surface,
                    remove=True,
                    reason=reason,
                ),
                dict(
                    path=["containerInformation", index, "sizeCategory"],
                    add=reviewed.size_category,
                    reason=reason,
                ),
                dict(
                    path=["containerInformation", index, "typeCategory"],
                    add=reviewed.type_category,
                    reason=reason,
                ),
            ]
        )
    return changes


def validate_rows(dataset, receipts):
    seen, hashes, counts = set(), set(), Counter()
    for split in ("train", "validation"):
        for row in map(json.loads, (dataset / f"{split}.jsonl").read_text().splitlines()):
            doc = row["documentId"]
            assert doc not in seen and row["joinedRawTextSha256"] not in hashes
            seen.add(doc)
            hashes.add(row["joinedRawTextSha256"])
            receipt = receipts[doc]
            assert receipt["split"] == split
            assert sha256_bytes(row["joinedRawText"].encode()) == receipt["sourceOcrSha256"]
            assert row["joinedRawTextSha256"] == receipt["sourceOcrSha256"]
            assert sha256_file(ROOT / receipt["pdf"]) == receipt["pdfSha256"]
            assert sha256_file(ROOT / receipt["inputTarget"]) == receipt["inputTargetSha256"]
            baseline = read(ROOT / receipt["inputTarget"])
            # Batch001 receipts use after:null for deletion; no add operation.
            changes = receipt["changes"]
            if receipt.get("editFormat") != "explicit_add_remove_v1":
                changes = [
                    {**{k: v for k, v in c.items() if k != "after"}, "remove": True}
                    if c["after"] is None
                    else c
                    for c in changes
                ]
            expected = apply_decisions(baseline, changes)
            assert row["target"] == expected
            parsed = BillOfLadingExtractionV7Label.model_validate_json(json.dumps(expected))
            assert parsed.canonical_target() == expected
            sample = dataset / "samples" / doc
            assert read(sample / "labels.json") == expected
            assert sha256_file(sample / "labels.json") == receipt["targetSha256"]
            assert (sample / "ocr.txt").read_bytes() == row["joinedRawText"].encode()
            for goods in row["target"]["documentPatch"].get("goodsItemDetails", []):
                if "splitGoodsPlacement" in goods:
                    assert list(goods)[-1] == "splitGoodsPlacement"
            counts[split] += 1
            counts["schemasAndExactReplays"] += 1
            # Falsify integrity gates, not a claimed semantic accuracy estimate.
            assert sha256_bytes((row["joinedRawText"] + "X").encode()) != receipt["sourceOcrSha256"]
            mutant = copy.deepcopy(expected)
            mutant["documentPatch"]["billOfLadingNumber"] = "UNRECORDED_MUTATION"
            assert mutant != row["target"]
            assert receipt["split"] != ("validation" if split == "train" else "train")
            counts["integrityMutantsRejected"] += 3
    assert seen == set(receipts)
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    assert not (DEST / "batches/002").exists(), "Batch already published; no overwrite"
    assert not PUBLISHED.exists(), "Published revision already exists"
    selection, decisions = read(BATCH / "selection.json"), read(DECISIONS)
    assert decisions["batch"] == "002"
    assert len(selection["documents"]) == 100
    assert set(decisions["documents"]) == {r["name"][:3] for r in selection["documents"]}
    old_manifest = read(DEST / "batches/001/manifest.json")
    assert old_manifest["samples"] == 50
    old_receipts = {r["documentId"]: r for r in old_manifest["documents"]}
    before_validation = validate_rows(DEST, old_receipts)
    for name, digest in old_manifest["files"].items():
        assert sha256_file(DEST / name) == digest
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
    stage = Path(tempfile.mkdtemp(prefix=".mpci-v7-batch002-", dir=DEST.parent))
    shutil.copytree(
        DEST,
        stage,
        dirs_exist_ok=True,
        ignore=lambda directory, names: (
            set(names) & {"train.jsonl", "validation.jsonl", "README.md"}
            if Path(directory) == DEST
            else set()
        ),
    )
    records = {
        s: list(map(json.loads, (DEST / f"{s}.jsonl").read_text().splitlines()))
        for s in ("train", "validation")
    }
    receipts, repaired, operations = [], [], Counter()
    for row in selection["documents"]:
        doc, name = row["documentId"], row["name"]
        assert doc not in old_receipts
        decision = decisions["documents"][name[:3]]
        assert decision["notes"].strip()
        path = BATCH / "runs" / name / "refine/target.json"
        baseline = read(path)
        target = apply_decisions(baseline, decision["changes"])
        extra = equipment_decisions(target)
        changes = decision["changes"] + extra
        target = apply_decisions(baseline, changes)
        parsed = BillOfLadingExtractionV7Label.model_validate_json(json.dumps(target))
        assert parsed.canonical_target() == target
        target = parsed.canonical_target()  # Canonical field order, no value changes.
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
        state = read(BATCH / "runs" / name / "refine/status.json")
        receipt = dict(
            documentId=doc,
            pilotDocument=name,
            split=row["split"],
            status="manually_adjudicated",
            sourceOcrSha256=row["ocrSha256"],
            pdf=row["pdf"],
            pdfSha256=row["pdfSha256"],
            inputTarget=str(path.relative_to(ROOT)),
            inputTargetSha256=sha256_file(path),
            editFormat="explicit_add_remove_v1",
            changes=changes,
            decisions=[decision["notes"]],
            targetSha256=sha256_file(folder / "labels.json"),
            originalAgentStatus=state["status"],
        )
        receipts.append(receipt)
        repaired.append(dict(name=name, target=target, decision=decision["notes"]))
        for c in changes:
            operations[c["path"][0]] += 1
    for split, rows in records.items():
        atomic_publish_bytes(
            stage / f"{split}.jsonl",
            "".join(
                json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows
            ).encode(),
        )
    all_receipts = {**old_receipts, **{r["documentId"]: r for r in receipts}}
    result = validate_rows(stage, all_receipts)
    assert result["train"] == 130 and result["validation"] == 20
    # Old sample files and all historical receipts remain byte-for-byte intact.
    preserved = 0
    for old in sorted(DEST.rglob("*")):
        if old.is_file() and old.name not in ("train.jsonl", "validation.jsonl", "README.md"):
            assert sha256_file(old) == sha256_file(stage / old.relative_to(DEST))
            preserved += 1
    manifest = dict(
        batch="002",
        samples=100,
        totalDatasetSamples=150,
        splits=dict(Counter(r["split"] for r in receipts)),
        sourceSelection=str((BATCH / "selection.json").relative_to(ROOT)),
        sourceSelectionSha256=sha256_file(BATCH / "selection.json"),
        decisionsSha256=sha256_file(DECISIONS),
        implementationSha256=sha256_file(Path(__file__)),
        files={f"{s}.jsonl": sha256_file(stage / f"{s}.jsonl") for s in records},
        documents=receipts,
    )
    result.update(
        manuallyReviewed=100,
        remainingAdjudications=0,
        changedDocuments=sum(bool(r["changes"]) for r in receipts),
        editOperations=sum(operations.values()),
        editSections=dict(operations),
        preservedHistoricalFiles=preserved,
        priorDatasetValidation=before_validation,
        paidCalls=0,
        estimatedUsd=0,
        elapsedSeconds=time.monotonic() - started,
        peakRssKiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    atomic_publish_json(stage / "batches/002/manifest.json", manifest)
    atomic_publish_json(stage / "batches/002/validation.json", result)
    atomic_publish_json(stage / "batches/002/manual-decisions.json", decisions)
    atomic_publish_json(stage / "batches/002/repaired-targets.json", repaired)
    atomic_publish_bytes(
        stage / "README.md",
        (
            (DEST / "README.md").read_text()
            + "\n## Batch 002\n\n100 additional manually adjudicated documents: "
            "90 train / 10 validation.\n"
            "Current total: 150 (130 train / 20 validation). "
            "All OCR/PDF and original splits preserved.\n"
            "Exact edits and decisions: batches/002/manifest.json and manual-decisions.json.\n"
            "The unchanged 50-record dataset is preserved at ../mpci-bl-real-v7-reviewed/.\n"
            "Historical batch001 aggregate file hashes describe that archived snapshot;\n"
            "batch002 hashes describe the current combined files. "
            "Per-record receipts remain valid.\n"
        ).encode(),
    )
    if args.publish:
        # Publish a new revision without renaming or mutating the existing tree.
        # DrvFS rejected renaming that directory; a fresh destination also keeps
        # the prior dataset available throughout publication.
        os.rename(stage, PUBLISHED)
        assert validate_rows(PUBLISHED, all_receipts)["schemasAndExactReplays"] == 150
        print("PUBLISHED", PUBLISHED)
    else:
        print("STAGED ONLY", stage)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
