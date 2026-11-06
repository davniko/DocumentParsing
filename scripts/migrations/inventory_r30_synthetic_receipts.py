"""Inventory current synthetic receipt lineage without changing any dataset."""

from __future__ import annotations

import json
import resource
import time
from collections import Counter
from pathlib import Path

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/analysis/r30-synthetic-migration-20261103"


def read(path):
    return json.loads(path.read_text())


def main():
    started = time.perf_counter()
    base = ROOT / "data/curated/mpci-bl-r29-positions-v1/synthetic-train.jsonl"
    catalog = read(
        ROOT
        / "artifacts/synthesis-templates/mpci-bl-v7-paddle-reviewed/r30/audit-20261103/catalog.json"
    )
    provenance = {
        r["id"]: r
        for r in read(
            ROOT / "docs/analysis/paddle-migration-scope-20261102/synthetic-provenance.json"
        )
    }
    staged = {}
    for cohort in ("r28-combination-refresh300-20261101", "r29-equal-load-refresh-20261102"):
        for kind in ("cargo", "equipment", "prefix", "legacy", "long"):
            for path in sorted(
                (ROOT / "docs/analysis" / cohort / ("staged-" + kind)).glob("*.json")
            ):
                item = read(path)
                did = item["after"]["documentId"]
                staged.setdefault(did, []).append((path, item))
    counts, result = Counter(), []
    for line in base.open():
        record = json.loads(line)
        did, source = record["documentId"], record["sourceDocumentId"]
        row = {"id": did, "source": source, "recordSha256": digest(record)}
        matches = [(p, s) for p, s in staged.get(did, []) if s["after"] == record]
        if matches:
            path, stage = matches[-1]
            row.update(stage=str(path.relative_to(ROOT)), stageSha256=sha256_file(path))
            if "nativePath" in stage["evidence"]:
                receipt_path = ROOT / stage["evidence"]["nativePath"]
                row["lineage"] = "scoped_cargo"
            else:
                receipt_path = ROOT / provenance[did]["path"]
                row["lineage"] = path.parent.name
        else:
            receipt_path = ROOT / provenance[did]["path"]
            row["lineage"] = "historical"
        candidate = read(receipt_path)
        if "documentId" not in candidate:
            row.update(
                candidate=str(receipt_path.relative_to(ROOT)),
                candidateSha256=sha256_file(receipt_path),
                status="COMPOSITE_OR_NONCANDIDATE_RECEIPT",
                receiptKeys=sorted(candidate),
            )
            counts[row["status"]] += 1
            result.append(row)
            continue
        row.update(
            candidate=str(receipt_path.relative_to(ROOT)),
            candidateSha256=sha256_file(receipt_path),
            candidateId=candidate["documentId"],
            exactCurrentText=candidate.get("joinedRawText") == record["joinedRawText"],
            matchingParent=candidate.get("sourceDocumentId") == source,
        )
        row["admittedParent"] = source in catalog["templates"]
        row["matchingSourcePin"] = (
            row["admittedParent"]
            and candidate.get("proof", {}).get("sourceSha256")
            == catalog["templates"][source]["samplingSource"]["sha256"]
        )
        if row["exactCurrentText"] and row["matchingParent"] and row["matchingSourcePin"]:
            row["status"] = "EXACT_CURRENT_RECEIPT"
        elif not row["admittedParent"]:
            row["status"] = "SOURCE_PRESERVING_NATIVE_REBASE_REQUIRED"
        else:
            row["status"] = "RECEIPT_COMPOSITION_REQUIRED"
        counts[row["status"]] += 1
        counts["lineage:" + row["lineage"]] += 1
        result.append(row)
    report = {
        "status": "READ_ONLY_RECEIPT_INVENTORY",
        "baseline": {"path": str(base.relative_to(ROOT)), "sha256": sha256_file(base)},
        "counts": dict(counts),
        "records": result,
        "elapsedSeconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    atomic_write_json(OUT / "receipt-inventory.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
