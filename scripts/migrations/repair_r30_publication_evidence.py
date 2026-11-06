"""Repair stale render pins only after an exact current-template replay.

An early amendment preserved the old rendered-evidence pointer while correctly
amending the record. This repairs provenance, never text, targets or geometry.
"""

import copy
import json
import resource
import time

from migrate_r30_synthetic import ART, DATA, OUT, ROOT, current_for_render, read
from publish_r30_sources import jsonl, pin, verify_pin

from document_ocr.atomic import (
    atomic_publish_bytes,
    atomic_publish_json,
    atomic_write_bytes,
    atomic_write_json,
)
from document_ocr.synthesis.curated import digest
from r30_source_preserving import R30Replay

PAYLOAD = ("pages", "target", "joinedRawText", "positionedText", "bboxText")


def require_same_payload(record, rendered):
    if any(record[k] != rendered[k] for k in PAYLOAD):
        raise ValueError("provenance-only repair cannot change published payload")


def main():
    started = time.perf_counter()
    manifest = read(DATA / "manifest.json")
    catalog = R30Replay(ROOT, manifest["templateCatalog"], manifest.get("sourcePreservingReceipts", ()))
    inventory = read(OUT / "receipt-inventory.json")
    baseline = {
        r["documentId"]: r for r in map(json.loads, verify_pin(inventory["baseline"]).open())
    }
    folder = ART / "publication-evidence-repair" / digest(manifest)
    repaired, certificates, aggregate = [], [], []
    for cert_pin in manifest["syntheticBatches"]:
        cert = read(verify_pin(cert_pin))
        rows = list(map(json.loads, verify_pin(cert["records"]).open()))
        touched = []
        for row in rows:
            evidence = row["evidence"]
            old = read(verify_pin(evidence["rendered"]))
            if all(row[k] == old[k] for k in PAYLOAD):
                continue
            review = read(verify_pin(evidence["review"]))
            staging = read(verify_pin(review["staging"]))
            entry = next(e for e in staging["results"] if e["id"] == row["documentId"])
            current = current_for_render(
                baseline[row["documentId"]],
                read(verify_pin(entry["reconciliation"])) if entry.get("reconciliation") else None,
            )
            rendered = catalog.render(
                verify_pin(evidence["candidate"]),
                current=current,
                wording_receipt=read(verify_pin(evidence["wording"]))
                if evidence["wording"]
                else None,
            )
            require_same_payload(row, rendered)
            path = folder / (row["documentId"] + "-rendered.json")
            atomic_publish_json(path, rendered)
            repair = {
                "id": row["documentId"],
                "beforeRecordSha256": digest(row),
                "previousRendered": evidence["rendered"],
                "rendered": pin(path),
                "catalog": manifest["templateCatalog"],
                "payloadChanged": False,
            }
            receipt = folder / (row["documentId"] + "-receipt.json")
            atomic_publish_json(receipt, repair)
            row["evidence"] = {
                **evidence,
                "rendered": pin(path),
                "catalog": manifest["templateCatalog"],
                "evidenceCorrection": pin(receipt),
            }
            touched.append(row["documentId"])
            repaired.append(repair)
        if touched:
            destination = (
                DATA / "synthetic-amendments" / "render-evidence-20261104" / f"{cert['batch']:03d}"
            )
            atomic_publish_bytes(destination / "synthetic-train.jsonl", jsonl(rows))
            updated = {
                **cert,
                "previous": cert_pin,
                "records": pin(destination / "synthetic-train.jsonl"),
                "provenanceOnlyCorrections": touched,
            }
            atomic_publish_json(destination / "certificate.json", updated)
            certificates.append(pin(destination / "certificate.json"))
        else:
            certificates.append(cert_pin)
        aggregate.extend(rows)
    if not repaired:
        raise ValueError("no stale evidence: publication is unnecessary")
    if read(DATA / "manifest.json") != manifest:
        raise ValueError("dataset changed during evidence repair")
    atomic_publish_json(folder / "previous-manifest.json", manifest)
    atomic_write_bytes(DATA / "synthetic-train.jsonl", jsonl(aggregate))
    updated = copy.deepcopy(manifest)
    updated["syntheticBatches"] = certificates
    updated["aggregateRecords"]["synthetic-train"] = pin(DATA / "synthetic-train.jsonl")
    atomic_write_json(DATA / "manifest.json", updated)
    report = {
        "repairs": repaired,
        "elapsedSeconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    atomic_publish_json(folder / "report.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
