"""Publish reviewed full annotations and the reversible 600/60 reduced projection.

Every selected document needs an authored decision or an explicit exclusion.
Historical/new split changes must match a frozen, authorized allocation manifest.
"""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium
from filter_real_v7_starting_dataset import (
    ROOT,
    digest,
    read,
    tree_hashes,
    validate_records,
    write_json,
)
from jsonschema import Draft202012Validator
from project_real_v7_r9 import project
from publish_real_v7_batch2 import apply_decisions, validate_rows
from review_real_v7_batch import candidate
from run_real_v7_batch import SOURCE, verify

from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7Label

BASE = ROOT / "data/curated/mpci-bl-real-v7-reviewed-r11-reduced-472"
FULL = BASE.with_name("mpci-bl-real-v7-reviewed-batch006-full")
DEST = BASE.with_name("mpci-bl-real-v7-reviewed-r12-reduced-660")


def publish(spec_path, apply=False):
    start = time.monotonic()
    assert not FULL.exists() and not DEST.exists()
    baseline_hashes = tree_hashes(BASE)
    baseline_manifest = read(BASE / "projection-manifest.json")
    assert baseline_hashes == baseline_manifest["files"] | {
        "projection-manifest.json": digest(BASE / "projection-manifest.json")
    }
    old = {r["documentId"]: r for r in map(json.loads, SOURCE.read_text().splitlines())}
    spec = read(spec_path)
    expected, seen_hashes = {}, set()
    for split in ("train", "validation"):
        for line in (BASE / f"{split}.jsonl").read_bytes().splitlines(keepends=True):
            r = json.loads(line)
            assert r["documentId"] not in expected and r["joinedRawTextSha256"] not in seen_hashes
            expected[r["documentId"]] = (split, line)
            seen_hashes.add(r["joinedRawTextSha256"])
    validate_records(BASE, expected)
    stage = Path(tempfile.mkdtemp(prefix=".real-v7-final-", dir=DEST.parent))
    full, reduced = stage / "full", stage / "reduced"
    full.mkdir()
    shutil.copytree(BASE / "samples", reduced / "samples")
    for name in ("schema.json", "prompt-schema.json", "training-prompt.txt"):
        shutil.copy2(BASE / name, reduced / name)
    validators = [
        Draft202012Validator(read(BASE / name)) for name in ("schema.json", "prompt-schema.json")
    ]
    full_rows, receipts, projections, excluded, provenance = (
        [],
        [],
        [],
        [],
        list(baseline_manifest["documents"]),
    )
    input_hashes = {str(spec_path.relative_to(ROOT)): digest(spec_path)}
    for item in spec["batches"]:
        assert item["batchTag"].isdigit() and len(item["batchTag"]) == 3
        batch = ROOT / item["path"]
        manifest = read(batch / "selection.json")
        verify(manifest)
        allocation_path = ROOT / item["assignments"]
        assert manifest["sources"][item["assignments"]] == digest(allocation_path)
        allocation = read(allocation_path)
        assert allocation["authorization"].strip()
        splits = {d["documentId"]: d for d in allocation["documents"]}
        decisions = read(batch / "manual-review/decisions.json")
        source_notes = read(batch / "manual-review/source-notes.json")
        omitted = item["exclusions"]
        names = {r["name"] for r in manifest["documents"]}
        assert not (decisions.keys() & omitted.keys())
        assert decisions.keys() | omitted.keys() == names
        all_indices = {f"{d['index']:03}" for d in manifest["documents"]}
        accepted_indices = {
            f"{d['index']:03}" for d in manifest["documents"] if d["name"] in decisions
        }
        assert accepted_indices <= source_notes.keys() <= all_indices
        for path in (
            batch / "selection.json",
            batch / "manual-review/decisions.json",
            batch / "manual-review/source-notes.json",
            allocation_path,
        ):
            input_hashes[str(path.relative_to(ROOT))] = digest(path)
        for row in manifest["documents"]:
            doc = row["documentId"]
            assert (
                splits[doc]["historicalSplit"] == old[doc]["auditSplit"] == row["historicalSplit"]
            )
            assert row["split"] == splits[doc]["split"]
            if row["name"] in omitted:
                assert omitted[row["name"]].strip()
                excluded.append(row | dict(sourceBatch=item["path"], reason=omitted[row["name"]]))
                continue
            assert source_notes[f"{row['index']:03}"].strip()
            decision = decisions[row["name"]]
            assert decision["reviewed"] and decision["notes"]
            path, initial, state = candidate(batch, row)
            assert digest(path) == decision["inputTargetSha256"]
            target = apply_decisions(initial, decision["changes"])
            canonical = BillOfLadingExtractionV7Label.model_validate_json(
                json.dumps(target)
            ).canonical_target()
            assert canonical == target
            target = canonical
            text = (ROOT / row["input"] / "ocr.txt").read_bytes()
            assert text == old[doc]["joinedRawText"].encode()
            assert digest(ROOT / row["input"] / "ocr.txt") == row["ocrSha256"]
            assert digest(ROOT / row["pdf"]) == row["pdfSha256"]
            with pdfium.PdfDocument(ROOT / row["pdf"]) as pdf:
                pages = len(pdf)
            assert 1 <= pages <= 5
            patch = target["documentPatch"]
            goods = patch.get("goodsItemDetails", [])
            assert len(goods) == 1 and goods[0].get("description", "").strip()
            assert any(patch.get("parties", {}).get(k) for k in ("shipper", "consignee"))
            assert doc not in expected and row["ocrSha256"] not in seen_hashes
            seen_hashes.add(row["ocrSha256"])
            record = dict(
                documentId=doc,
                joinedRawText=text.decode(),
                joinedRawTextSha256=row["ocrSha256"],
                target=target,
            )
            sample = full / "samples" / doc
            sample.mkdir(parents=True)
            (sample / "ocr.txt").write_bytes(text)
            write_json(sample / "labels.json", target)
            full_rows.append((row["split"], record))
            receipt = dict(
                documentId=doc,
                sourceBatch=item["path"],
                pilotDocument=row["name"],
                split=row["split"],
                historicalSplit=row["historicalSplit"],
                pages=pages,
                status="manually_adjudicated",
                sourceOcrSha256=row["ocrSha256"],
                pdf=row["pdf"],
                pdfSha256=row["pdfSha256"],
                inputTarget=str(path.relative_to(ROOT)),
                inputTargetSha256=digest(path),
                editFormat="explicit_add_remove_v1",
                changes=decision["changes"],
                decisions=[source_notes[f"{row['index']:03}"], *decision["notes"]],
                targetSha256=digest(sample / "labels.json"),
                originalAgentStatus=state["status"],
            )
            receipts.append(receipt)
            projected, edits = project(record)
            for validator in validators:
                validator.validate(projected["target"])
            expected[doc] = (
                row["split"],
                (json.dumps(projected, ensure_ascii=False, separators=(",", ":")) + "\n").encode(),
            )
            projections.append(dict(documentId=doc, edits=edits))
            folder = reduced / "samples" / doc
            folder.mkdir()
            (folder / "ocr.txt").write_bytes(text)
            write_json(folder / "labels.json", projected["target"])
            provenance.append(
                dict(
                    documentId=doc,
                    sample=f"{item['batchTag']}/{row['name']}",
                    split=row["split"],
                    historicalSplit=row["historicalSplit"],
                    sourceDataset=str(FULL.relative_to(ROOT)),
                    sourceBatch=item["path"],
                    pdf=row["pdf"],
                    sourceOcrSha256=row["ocrSha256"],
                )
            )
    assert len(receipts) == 188 and len(expected) == 660
    assert Counter(s for s, _ in expected.values()) == Counter(train=600, validation=60)
    for split in ("train", "validation"):
        (full / f"{split}.jsonl").write_text(
            "".join(
                json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                for s, r in full_rows
                if s == split
            )
        )
        (reduced / f"{split}.jsonl").write_bytes(
            b"".join(line for s, line in expected.values() if s == split)
        )
    full_checks = validate_rows(full, {r["documentId"]: r for r in receipts})
    validate_records(reduced, expected)
    assert tree_hashes(BASE) == baseline_hashes
    for p, h in input_hashes.items():
        assert digest(ROOT / p) == h
    summary = dict(
        documents=660,
        splits=dict(Counter(s for s, _ in expected.values())),
        additions=188,
        excludedCandidates=len(excluded),
        baselineUnchanged=472,
        changedNewDocuments=sum(bool(r["changes"]) for r in receipts),
        changedHistoricalSplitAssignments=sum(r["split"] != r["historicalSplit"] for r in receipts),
        historicalTrainAssignedValidation=sum(
            r["historicalSplit"] == "train" and r["split"] == "validation" for r in receipts
        ),
        historicalValidationAssignedTrain=sum(
            r["historicalSplit"] == "validation" and r["split"] == "train" for r in receipts
        ),
        fullReadback=full_checks,
        reducedReadback=660,
        ocrEdits=0,
        elapsedSeconds=round(time.monotonic() - start, 3),
        peakRssKiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    write_json(
        full / "manifest.json",
        dict(
            samples=188,
            documents=receipts,
            exclusions=excluded,
            inputFiles=input_hashes,
            files={f"{s}.jsonl": digest(full / f"{s}.jsonl") for s in ("train", "validation")},
        ),
    )
    write_json(full / "schema.json", BillOfLadingExtractionV7Label.model_json_schema())
    write_json(full / "validation.json", full_checks)
    write_json(reduced / "projection-edits.json", projections)
    write_json(reduced / "validation.json", summary)
    (reduced / "README.md").write_text(
        "# Reviewed real baseline R12\n\n"
        "600 training / 60 validation records, each with one described goods item; "
        "PDFs at most five pages.\n\n"
        "R11 preserves the 472-record pre-addition baseline. "
        "The neighboring batch006-full folder preserves all 188 new full annotations "
        "from batches 006-008 and their exact manual-edit receipts.\n\n"
        "This reduced projection removes carrier party, vessel flag, cargo marks, "
        "and forwarding/export references. "
        "OCR and PDFs are unchanged. Each new validation assignment records historical membership: "
        "do not claim this validation set was unseen by older models.\n\n"
        "Use this folder's frozen schema and prompt together. No training was launched.\n"
    )
    write_json(
        reduced / "projection-manifest.json",
        dict(
            dataset=str(DEST.relative_to(ROOT)),
            sourceDataset=str(BASE.relative_to(ROOT)),
            sourceFiles=baseline_hashes,
            inputFiles=input_hashes,
            documents=provenance,
            summary=summary,
            files=tree_hashes(reduced),
        ),
    )
    if apply:
        full.rename(FULL)
        reduced.rename(DEST)
        validate_records(DEST, expected)
        validate_rows(FULL, {r["documentId"]: r for r in receipts})
    print(json.dumps(dict(stage=str(stage), published=apply, **summary), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    publish(ROOT / a.spec, a.apply)
