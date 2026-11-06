"""Publish exact-reviewed batches with complete finding decisions and no API calls."""

import argparse
import json
import resource
import time

from migrate_r30_synthetic import DATA, OUT, ROOT, read, stage
from publish_r30_sources import pin, verify_pin
from review_r30_synthetic_batch import main as review_batch
from certify_r30_synthetic_batch import publish
from r30_finding_dispositions import complete_decision, load_dispositions
from document_ocr.atomic import atomic_publish_json, atomic_write_json


def require_ready(staged, inventory, dispositions=None):
    if staged["selected"] != 100 or staged["failures"] or len(staged["results"]) != 100:
        raise ValueError("batch is not completely rendered")
    for entry in staged["results"]:
        record = inventory[entry["id"]]
        if record["rendered"] != entry["rendered"] or not record["exact"]:
            raise ValueError("readiness evidence is not exact-current")
        if complete_decision(entry, record["exact"], dispositions or {}) is None:
            raise ValueError("findings require individual main-agent adjudication")


def main(inventory_path, batches, destination, hold_unready=False, dispositions_path=None):
    inventory = {r["id"]: r for r in read(inventory_path)["records"]}
    dispositions = load_dispositions(dispositions_path)
    reports = []
    for batch in batches:
        start = time.perf_counter()
        manifest = read(DATA / "manifest.json")
        if any(read(verify_pin(p))["batch"] == batch for p in manifest["syntheticBatches"]):
            raise ValueError("refusing to republish an already certified batch")
        stage(batch)
        staged = read(OUT / "batches" / f"{batch:03d}-staging.json")
        if hold_unready:
            blocked = [entry["id"] for entry in staged["results"]
                       if entry["id"] not in inventory
                       or inventory[entry["id"]]["rendered"] != entry["rendered"]
                       or not inventory[entry["id"]]["exact"]
                       or complete_decision(entry, inventory[entry["id"]]["exact"], dispositions) is None]
            if staged["failures"] or blocked:
                report = {"batch": batch, "status": "HELD_NOT_PUBLISHED",
                          "mechanicalFailures": staged["failures"],
                          "needsExactReviewOrAdjudication": blocked,
                          "providerCostUsd": 0,
                          "elapsedSeconds": time.perf_counter() - start}
                reports.append(report)
                atomic_write_json(destination / "progress.json", {"batches": reports})
                print("READY_HELD " + json.dumps(report), flush=True)
                continue
        require_ready(staged, inventory, dispositions)
        decisions = {e["id"]: complete_decision(e, inventory[e["id"]]["exact"], dispositions)
                     for e in staged["results"]}
        decisions_path = destination / f"{batch:03d}-decisions.json"
        atomic_publish_json(decisions_path, {
            "decisions": decisions,
            "scope": "Only exact-current independently reviewed records; each finding requires "
            "its individually pinned main-agent disposition, including invalid evidence. Fresh "
            "complete rendering and publication replay remain mandatory. Unresolved or changed "
            "records stop this path; no provider request is made.",
            "inventory": pin(inventory_path),
            "findingDispositions": pin(dispositions_path) if dispositions_path else None,
        })
        review = review_batch(batch, decisions_path)
        publish(batch, review)
        report = {"batch": batch, "certified": 100, "providerCostUsd": 0,
                  "elapsedSeconds": time.perf_counter() - start,
                  "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
        reports.append(report)
        atomic_write_json(destination / "progress.json", {"batches": reports})
        print("READY_CERTIFIED " + json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory")
    parser.add_argument("destination")
    parser.add_argument("--batches", nargs="+", type=int, required=True)
    parser.add_argument("--hold-unready", action="store_true",
                        help="Record changed/flagged batches as unpublished and continue other batches")
    parser.add_argument("--dispositions")
    args = parser.parse_args()
    main(ROOT / args.inventory, args.batches, ROOT / args.destination, args.hold_unready,
         ROOT / args.dispositions if args.dispositions else None)
