"""Admit transport receipts with exact retained-text replay, not synthetic samples."""

import json

from migrate_r30_synthetic import ART, OUT, ROOT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.scoped_cargo_refresh import apply_edits, reconstruct_source


def main():
    report_path = OUT / "composition-proposals.json"
    report = read(report_path)
    inventory = read(OUT / "receipt-inventory.json")
    current = {r["documentId"]: r for r in map(json.loads, verify_pin(inventory["baseline"]).open())}
    receipts = []
    for row in report["records"]:
        if row["status"] != "EXACT_REPLAY_PROPOSAL":
            continue
        candidate_pin = {"path": row["path"], "sha256": row["sha256"]}
        candidate = read(verify_pin(candidate_pin))
        evidence = candidate["migrationComposition"]
        previous = read(verify_pin(evidence["historicalCandidate"]))
        retained = current[row["id"]]
        if evidence["currentRecordSha256"] != digest(retained):
            raise ValueError("composition predates retained record")
        if candidate["scenario"] != previous["scenario"]:
            raise ValueError("composition changed sampled shipment facts")
        if candidate["target"]["documentPatch"] != retained["target"]:
            raise ValueError("composition changed retained target")
        source = reconstruct_source(previous)
        if reconstruct_source(candidate) != source or apply_edits(source, candidate["proof"]["edits"]) != retained["joinedRawText"].encode():
            raise ValueError("composition does not reconstruct every current byte")
        destination = ART / "reconciled" / (row["id"] + ".json")
        atomic_publish_json(destination, {
            "status": "REVIEWED_CURRENT_SHIPMENT_RECEIPT", "candidate": candidate_pin,
            "currentRecordSha256": digest(retained), "mechanismReview": pin(report_path),
            "reviewScope": "Transport receipt only: complete exact source/current replay, retained targets and unchanged sampled facts. Full Paddle rendering and semantic review still required before dataset certification.",
        })
        receipts.append(pin(destination))
    atomic_publish_json(OUT / "exact-composition-receipts.json", {"receipts": receipts, "count": len(receipts)})
    print(json.dumps({"transportReceipts": len(receipts), "syntheticCertifications": 0}))


if __name__ == "__main__":
    main()
