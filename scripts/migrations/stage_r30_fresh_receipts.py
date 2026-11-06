"""Add explicitly reviewed fresh campaign receipts to a native pilot source.

Fresh rows are separate from R29 replays. Neither path can silently substitute
an older target or a different shipment's wording.
"""

from __future__ import annotations

import argparse
import hashlib
import json

from prepare_r30_paddle_pilot import ART as PILOT_ART
from prepare_r30_paddle_pilot import ROOT, read
from r30_region_transfer_inventory import locate

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7DirectTarget


def main(prefix, directory, *, scope):
    if scope == "full":
        from rebase_r30_sources import ART
    elif scope == "pilot":
        ART = PILOT_ART
    else:
        raise ValueError("unknown migration scope")
    (path,) = (ART / "transfers").glob("doc_" + prefix + "*.json")
    transfer = read(path)
    source = read(ART / "sources" / path.name)
    # A never-used admitted template has no previous descendant. Its source
    # authority is the pinned contract, not an arbitrarily selected shipment.
    config = read(ROOT / "configs/synthesis/mpci_bl_curated_v7_multirow.json")
    authority = ROOT / config["source_contracts"] / source["documentId"] / "source.txt"
    authority_bytes = authority.read_bytes()
    authority_sha256 = hashlib.sha256(authority_bytes).hexdigest()
    slots = {s["owner"]: s for s in transfer["slots"]}
    for candidate_path in sorted((ROOT / directory).glob("*.json")):
        candidate = read(candidate_path)
        if candidate["sourceDocumentId"] != source["documentId"]:
            continue
        if candidate["proof"]["sourceSha256"] != authority_sha256:
            raise ValueError("fresh receipt uses a different source authority")
        identity = candidate["documentId"]
        if any(s["id"] == identity for s in transfer["descendants"]):
            raise ValueError("fresh descendant already staged")
        for edit in candidate["proof"]["edits"]:
            if authority_bytes[edit["byteStart"] : edit["byteEnd"]].decode() != edit["before"]:
                raise ValueError("fresh edit differs from the pinned source authority")
            owner = f"{edit['key']}@{edit['byteStart']}:{edit['byteEnd']}"
            if owner in slots:
                continue
            same_source = [s for s in slots.values() if s["before"] == edit["before"]]
            hints = [tuple(ref) for s in same_source for ref in s["hints"]]
            mapping = locate(edit["before"], source["pages"], hints)
            slots[owner] = {
                "owner": owner,
                **{k: edit[k] for k in ("key", "byteStart", "byteEnd", "before", "targetPaths")},
                "hints": hints,
                "kind": "fresh_render_proof",
                **mapping,
            }
        target = BillOfLadingExtractionV7DirectTarget.model_validate_json(
            json.dumps(candidate["target"]["documentPatch"])
        ).canonical_target()
        record_path = ART / "fresh-records" / (identity + ".json")
        atomic_write_json(
            record_path,
            {
                "documentId": identity,
                "sourceDocumentId": source["documentId"],
                "joinedRawText": candidate["joinedRawText"],
                "target": target,
                "receiptSha256": sha256_file(candidate_path),
            },
        )
        transfer["descendants"].append(
            {
                "id": identity,
                "path": str(candidate_path.relative_to(ROOT)),
                "freshRecord": str(record_path.relative_to(ROOT)),
                "receiptSha256": sha256_file(candidate_path),
            }
        )
    transfer["slots"] = list(slots.values())
    transfer["remainingDescendants"] = max(0, 3 - len(transfer["descendants"]))
    atomic_write_json(path, transfer)
    print(json.dumps({"descendants": len(transfer["descendants"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    parser.add_argument("directory")
    parser.add_argument("--scope", choices=("pilot", "full"), required=True)
    args = parser.parse_args()
    main(args.prefix, args.directory, scope=args.scope)
