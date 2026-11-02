"""Repair the eight reviewed old drum targets without resampling shipment facts."""

import copy
import json
import time
from collections import Counter
from pathlib import Path

from document_ocr.atomic import atomic_publish_json, atomic_write_bytes, atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_reflow_policy import FontMetrics
from document_ocr.synthesis.package_alias_refresh import apply_reviewed_cleanup
from document_ocr.training.tasks import get_training_task

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/curated/mpci-bl-r28-positions-v1"
OUT = ROOT / "docs/analysis/r28-expansion2500-20261101/legacy-drum-wording"
FILES = ("synthetic-train.jsonl", "real-train.jsonl", "previous-uae-synthetic.jsonl", "old-uae-addition.jsonl")
# These exact snippets were read against the current positioned OCR and labels.
# Remove unsampled fills/approval claims, retain each printed packaging material.
EDITS = {
    "631b8b2f73ae779ab6105c83": [("PACKED IN 54 X 220KG POLYETHYLENE DRUMS", "PACKED IN POLYETHYLENE DRUMS")],
    "b7774f28ecd6e8317a0f11a2": [("LB POLYETHYLENE DRUM 2.27 KG", "POLYETHYLENE DRUM")],
    "fa7ca50898d6a9022b539a5a": [("IN UN APPROVED STEEL DRUMS", "IN STEEL DRUMS")],
    "89d7faa2fae1f6d17a4fee86": [("NEW STEEL DRUMS 200 L", "NEW STEEL DRUMS")],
    "6e896941668206d0702a6e40": [("EACH POLYETHYLENE DRUM HOLDS 180 KG NET", "IN POLYETHYLENE DRUMS")],
    "551ff13a14e80a9f08cc757d": [("IN 72 POLYETHYLENE DRUMS, 274 KG NET EACH", "IN 72 POLYETHYLENE DRUMS")],
    "80cbb4c1827d83a1a4b1d211": [("STEEL DRUMS X 200 KGS NET EACH", "STEEL DRUMS")],
}


def main():
    started = time.perf_counter()
    manifest = json.loads((DATA / "manifest.json").read_text())
    config = json.loads((ROOT / "configs/synthesis/mpci_bl_curated_v7_multirow.json").read_text())
    font = config["positions"]["reflow"]
    metrics = FontMetrics(Path(font["font_path"]), font["font_sha256"])
    task = get_training_task("bill_of_lading_extraction_v7_reduced_direct")
    repairs, counts, payloads, before_hashes = {}, Counter(), {}, {}
    for name in FILES:
        path = DATA / name
        if sha256_file(path) != manifest["files"][name]:
            raise ValueError("dataset manifest differs")
        before_hashes[name] = sha256_file(path)
        chunks = []
        for line in path.read_bytes().splitlines(keepends=True):
            row = json.loads(line)
            if not row.get("sourceDocumentId", row["documentId"]).startswith("doc_630cd159"):
                chunks.append(line)
                continue
            sample = row["documentId"]
            changes = EDITS.get(sample.removeprefix("syn_full_v7_"))
            if changes is None and not sample.startswith("doc_630cd159"):
                raise ValueError("unreviewed descendant of affected source: " + sample)
            updated = copy.deepcopy(row)
            proof = None
            if changes:
                for old, new in changes:
                    if "\n" in old or "\n" in new or metrics.length(new) > metrics.length(old):
                        raise ValueError("correction expands the measured line footprint")
                repair = {"documentId": sample, "beforeRecordSha256": digest(row),
                          "rawEdits": [{"before": a, "after": b} for a, b in changes],
                          "targetEdits": [{"path": "goodsItemDetails[0].description", "before": a, "after": b}
                                          for a, b in changes]}
                updated, proof = apply_reviewed_cleanup(row, repair)
            goods = updated["target"]["goodsItemDetails"][0]
            category = "PACKAGE_DRUM_PLASTIC" if "POLYETHYLENE DRUM" in goods["description"] else "PACKAGE_DRUM_STEEL"
            if category.endswith("STEEL") and "STEEL DRUM" not in goods["description"]:
                raise ValueError("specific material lacks literal evidence")
            packages = goods["numberAndTypeOfPackages"]
            if packages != [{"packageQuantity": 72, "typeCategory": "PACKAGE_DRUM"}]:
                raise ValueError("reviewed package row changed")
            packages[0]["typeCategory"] = category
            expected_target = copy.deepcopy(row["target"])
            expected_target["goodsItemDetails"][0]["description"] = goods["description"]
            expected_target["goodsItemDetails"][0]["numberAndTypeOfPackages"][0]["typeCategory"] = category
            if updated["target"] != expected_target:
                raise ValueError("unrelated target changed")
            if task.canonicalize(updated["target"]) != updated["target"]:
                raise ValueError("corrected target is not canonical V7")
            if len(updated["joinedRawText"].splitlines()) != len(row["joinedRawText"].splitlines()):
                raise ValueError("line count changed")
            if [s.rsplit(" ||", 1)[-1] for s in updated["positionedText"].splitlines()] != [s.rsplit(" ||", 1)[-1] for s in row["positionedText"].splitlines()]:
                raise ValueError("coordinate suffix changed")
            record = {"beforeRecordSha256": digest(row), "afterRecordSha256": digest(updated),
                      "descriptionBefore": row["target"]["goodsItemDetails"][0]["description"],
                      "descriptionAfter": goods["description"], "categoryAfter": category, "proof": proof}
            if sample in repairs and repairs[sample] != record:
                raise ValueError("sidecar copy differs")
            repairs[sample] = record
            counts[name] += 1
            chunks.append((json.dumps(updated, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
        payloads[name] = b"".join(chunks)
    if len(repairs) != 8 or counts["synthetic-train.jsonl"] != 7 or counts["real-train.jsonl"] != 1:
        raise ValueError("reviewed repair scope differs")
    # Exact original files are already retained in the campaign's before/ snapshot.
    atomic_publish_json(OUT / "review-and-publication.json", {
        "status": "APPROVED_SCOPED_REPAIR", "before": before_hashes,
        "after": {n: digest(v) for n, v in payloads.items()}, "records": repairs, "counts": dict(counts),
        "scope": "Remove previously generated unsupported package fill/count/approval claims. Material explicitly printed in the goods owner establishes PLASTIC or STEEL; public quantities, allocations, mass, routes, parties and geometry unchanged. Real source changes category only.",
        "seconds": time.perf_counter() - started,
    })
    for name, data in payloads.items():
        atomic_write_bytes(DATA / name, data)
        manifest["files"][name] = digest(data)
    manifest["expansionLegacyDrumWordingRepair20261101"] = {
        "records": 8, "evidence": str((OUT / "review-and-publication.json").relative_to(ROOT)),
    }
    atomic_write_json(DATA / "manifest.json", manifest)
    print(json.dumps({"records": 8, "copies": dict(counts)}), flush=True)


if __name__ == "__main__":
    main()
