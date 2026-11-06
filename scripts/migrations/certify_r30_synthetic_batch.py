"""Replay and publish a complete, explicitly reviewed 100-record Paddle batch."""

import argparse
import copy
import json
import resource
import time

from audit_r30_independent import independent_view
from audit_r30_paddle_pilot import check_training_target
from migrate_r30_synthetic import ART, DATA, OUT, ROOT, current_for_render, read
from publish_r30_sources import jsonl, pin, verify_pin
from r30_literal_target_corrections import apply_literal_target_correction

from document_ocr.atomic import (
    atomic_publish_bytes,
    atomic_publish_json,
    atomic_write_bytes,
    atomic_write_json,
)
from r30_source_preserving import R30Replay


def review_first():
    """Record the main-agent dispositions already inspected for this specific cohort."""
    staging_path = OUT / "batches/001-staging.json"
    staged = read(staging_path)
    if staged["failures"] or len(staged["results"]) != 100:
        raise ValueError("first batch is not completely rendered")
    notes_path = OUT / "batch001-adjudications.json"
    notes = read(notes_path)["decisions"]
    proposals = {}
    for path in (ART / "semantic-reviews").rglob("*.json"):
        result = read(path)
        proposals.setdefault(result["id"], []).append((path, result))
    reviewed = []
    for entry in staged["results"]:
        identity = entry["id"]
        evidence = proposals.get(identity)
        if not evidence:
            raise ValueError(f"no completed semantic review: {identity}")
        prefix = entry["source"][4:12]
        for _, result in evidence:
            if result["source"] != entry["source"]:
                raise ValueError("review belongs to a different source")
            if result["review"]["findings"] and prefix not in notes:
                raise ValueError(f"no main-agent finding disposition: {identity}")
        reviewed.append(
            {
                "id": identity,
                "rendered": entry["rendered"],
                "reviews": [pin(p) for p, _ in evidence],
                "decision": notes.get(
                    prefix, "No semantic findings; full structural replay passed."
                ),
            }
        )
    atomic_publish_json(
        OUT / "batches/001-final-review.json",
        {
            "status": "MAIN_AGENT_REVIEWED_FOR_PUBLICATION",
            "staging": pin(staging_path),
            "adjudications": pin(notes_path),
            "records": reviewed,
            "scope": "First fixed 100 IDs only. All model findings read and adjudicated by the "
            "main agent; corrections replayed through the pinned catalog. "
            "No provider output directly changed targets.",
        },
    )


def publish(batch, review_path=None):
    started = time.perf_counter()
    review_path = review_path or OUT / "batches" / f"{batch:03d}-final-review.json"
    reviewed = read(review_path)
    if reviewed["status"] != "MAIN_AGENT_REVIEWED_FOR_PUBLICATION":
        raise ValueError("publication requires explicit completed review")
    staged = read(verify_pin(reviewed["staging"]))
    if staged["failures"] or staged["selected"] != 100 or len(staged["results"]) != 100:
        raise ValueError("cannot publish an incomplete batch")
    expected = read(verify_pin(staged["plan"]))["batches"][batch - 1]
    items = {e["id"]: e for e in staged["results"]}
    reviews = {e["id"]: e for e in reviewed["records"]}
    if len(items) != 100 or set(items) != set(expected) or set(reviews) != set(expected):
        raise ValueError("publication membership differs from frozen batch/review")
    manifest = read(DATA / "manifest.json")
    if read(verify_pin(staged["catalog"])) != read(verify_pin(manifest["templateCatalog"])):
        raise ValueError("staged catalog is not the currently published authority")
    catalog = R30Replay(ROOT, manifest["templateCatalog"], manifest.get("sourcePreservingReceipts", ()))
    inventory = read(OUT / "receipt-inventory.json")
    old_rows = {
        r["documentId"]: r
        for r in map(json.loads, verify_pin(inventory["baseline"]).open())
        if r["documentId"] in items
    }
    output = []
    for identity in expected:
        entry = items[identity]
        if reviews[identity]["rendered"] != entry["rendered"]:
            raise ValueError("review does not cover current rendered sample")
        rendered = read(verify_pin(entry["rendered"]))
        wording = read(verify_pin(entry["wording"])) if entry["wording"] else None
        replay = catalog.render(
            verify_pin(entry["candidate"]),
            current=current_for_render(
                old_rows[identity],
                read(verify_pin(entry["reconciliation"])) if entry.get("reconciliation") else None,
            ),
            wording_receipt=wording,
        )
        replay = apply_literal_target_correction(replay, entry.get("literalTargetCorrection"))
        if replay != rendered:
            raise ValueError("publication replay differs from reviewed sample")
        check_training_target(rendered["target"])
        for kind, field in (
            ("raw", "joinedRawText"),
            ("centroid", "positionedText"),
            ("bbox", "bboxText"),
        ):
            if independent_view(rendered["pages"], kind) != rendered[field]:
                raise ValueError("independent native serialization differs")
        row = copy.deepcopy(old_rows[identity])
        for field in ("target", "pages", "joinedRawText", "positionedText", "bboxText"):
            row[field] = rendered[field]
        row["evidence"] = {
            "rendered": entry["rendered"],
            "candidate": entry["candidate"],
            "wording": entry["wording"],
            "review": pin(review_path),
            "catalog": manifest["templateCatalog"],
            "previousRecordSha256": entry["currentRecordSha256"],
            "literalTargetCorrection": entry.get("literalTargetCorrection"),
        }
        output.append(row)
    # A schedule amendment can regroup an unpublished batch whose earlier
    # attempt left immutable artifacts. Keep those attempts intact and bind
    # this publication to its exact reviewed membership/content.
    folder = DATA / "synthetic-batches" / f"{batch:03d}" / pin(review_path)["sha256"]
    records_path = folder / "synthetic-train.jsonl"
    atomic_publish_bytes(records_path, jsonl(output))
    certificate = folder / "certificate.json"
    atomic_publish_json(
        certificate,
        {
            "status": "CERTIFIED_NATIVE_PADDLE_SYNTHETIC_BATCH",
            "batch": batch,
            "count": 100,
            "records": pin(records_path),
            "review": pin(review_path),
            "plan": staged["plan"],
            "catalog": manifest["templateCatalog"],
            "fullReplay": True,
            "elapsedSeconds": time.perf_counter() - started,
            "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
    )
    certificates, records, ids = [], [], set()
    active = [*manifest.get("syntheticBatches", []), pin(certificate)]
    for cert_pin in sorted(active, key=lambda p: read(verify_pin(p))["batch"]):
        cert = read(verify_pin(cert_pin))
        batch_rows = [json.loads(line) for line in verify_pin(cert["records"]).open()]
        if len(batch_rows) != cert["count"]:
            raise ValueError("certified batch count differs")
        for row in batch_rows:
            if row["documentId"] in ids:
                raise ValueError("duplicate published synthetic identity")
            ids.add(row["documentId"])
        records.extend(batch_rows)
        certificates.append(cert_pin)
    atomic_write_bytes(DATA / "synthetic-train.jsonl", jsonl(records))
    manifest.update(
        status="AUDITED_REAL_VALIDATION_TEMPLATES_PARTIAL_SYNTHETIC_NOT_TRAINABLE",
        syntheticBatches=certificates,
        syntheticMigration={"certified": len(records), "required": 9400, "plan": staged["plan"]},
    )
    manifest["aggregateRecords"]["synthetic-train"] = pin(DATA / "synthetic-train.jsonl")
    atomic_write_json(DATA / "manifest.json", manifest)
    print(json.dumps({"batch": batch, "certified": 100, "totalCertified": len(records)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("review-first", "publish"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--review", help="Exact content-pinned main-agent review packet")
    args = parser.parse_args()
    review_first() if args.action == "review-first" else publish(
        args.batch, ROOT / args.review if args.review else None
    )
