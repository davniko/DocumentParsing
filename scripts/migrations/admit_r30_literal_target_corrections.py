"""Pin manually adjudicated literal target additions without changing R29 or OCR."""

import argparse
import json

from migrate_r30_synthetic import ART, DATA, OUT, ROOT, current_for_render, read
from publish_r30_sources import pin, verify_pin
from r30_literal_target_corrections import ABSENT, apply_review, correction_index

from document_ocr.atomic import atomic_publish_json, atomic_write_json
from document_ocr.synthesis.curated import digest


def main(review_path):
    review = read(review_path)
    if review["status"] != "MAIN_AGENT_REVIEWED_LITERAL_TARGET_ADDITIONS":
        raise ValueError("literal additions require explicit adjudication")
    inventory = read(OUT / "receipt-inventory.json")
    rows = {
        row["documentId"]: row
        for row in map(json.loads, verify_pin(inventory["baseline"]).open())
        if row["documentId"] in review["records"]
    }
    if rows.keys() != review["records"].keys():
        raise ValueError("review contains unknown or missing baseline identities")
    manifest = read(DATA / "manifest.json")
    index = correction_index(manifest)
    prepared = []
    for identity, decision in review["records"].items():
        if identity in index:
            raise ValueError("literal addition already has an admitted review")
        row = rows[identity]
        receipt = ART / "reconciled" / (identity + ".json")
        current = current_for_render(row, read(receipt) if receipt.exists() else None)
        correction = {
            **decision,
            "status": "MAIN_AGENT_REVIEWED_LITERAL_TARGET_CORRECTION",
            "documentId": identity,
            "sourceDocumentId": row["sourceDocumentId"],
            "before": decision.get("before", ABSENT),
            "review": pin(review_path),
            "baselineRecordSha256": digest(row),
        }
        apply_review(current, correction)
        path = ART / "literal-target-corrections" / (identity + "-" + digest(correction) + ".json")
        prepared.append((identity, path, correction))
    for identity, path, correction in prepared:
        atomic_publish_json(path, correction)
        index[identity] = pin(path)
    result = {
        "status": "MAIN_AGENT_REVIEWED_LITERAL_TARGET_CORRECTION_INDEX",
        "records": index,
        "review": pin(review_path),
        "previous": manifest.get("literalTargetCorrections"),
    }
    path = ART / "literal-target-corrections" / ("index-" + digest(result) + ".json")
    atomic_publish_json(path, result)
    manifest["literalTargetCorrections"] = pin(path)
    atomic_write_json(DATA / "manifest.json", manifest)
    print(json.dumps({"admitted": len(prepared), "total": len(index), "index": pin(path)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review")
    args = parser.parse_args()
    main(ROOT / args.review)
