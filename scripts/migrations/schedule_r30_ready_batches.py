"""Prioritize already reviewed work; retain every hold and immutable old plan."""

import argparse
import copy
import json
from collections import Counter

from migrate_r30_synthetic import DATA, OUT, ROOT, read
from publish_r30_sources import pin, verify_pin
from r30_batch_schedule import reschedule
from r30_finding_dispositions import complete_decision, load_dispositions
from document_ocr.atomic import atomic_publish_json, atomic_write_json


def main(inventory_path, destination, dispositions_path=None):
    manifest_path = DATA / "manifest.json"
    manifest = read(manifest_path)
    previous_manifest = copy.deepcopy(manifest)
    previous_plan = manifest["syntheticMigration"]["plan"]
    plan = read(verify_pin(previous_plan))
    inventory = read(inventory_path)
    reviews = {r["id"]: r for r in inventory["records"]}
    dispositions = load_dispositions(dispositions_path)
    if len(reviews) != len(inventory["records"]):
        raise ValueError("duplicate review inventory identity")
    published = set()
    for cert_pin in manifest["syntheticBatches"]:
        cert = read(verify_pin(cert_pin))
        batch = cert["batch"]
        if batch in published:
            raise ValueError("duplicate published batch")
        actual = [json.loads(line)["documentId"] for line in verify_pin(cert["records"]).open()]
        # Published amendment writers may sort their rows. Membership, rather
        # than that storage order, must match; neither publication nor its plan
        # order is rewritten by this scheduling-only operation.
        if len(actual) != plan["batchSize"] or len(set(actual)) != len(actual) or set(actual) != set(plan["batches"][batch - 1]):
            raise ValueError("published membership differs from prior schedule")
        published.add(batch)
    priorities, staging_pins = {}, []
    for batch in range(1, len(plan["batches"]) + 1):
        if batch in published:
            continue
        staged = read(OUT / "batches" / f"{batch:03d}-staging.json")
        if staged["catalog"] != manifest["templateCatalog"] or staged["plan"] != previous_plan:
            raise ValueError("readiness staging is stale")
        for entry in staged["results"]:
            review = reviews[entry["id"]]
            if review["rendered"] != entry["rendered"]:
                raise ValueError("readiness review belongs to another render")
            priority = 2
            if review["exact"]:
                priority = 0
                for item in review["exact"]:
                    evidence = read(verify_pin(item["pin"]))
                    if (evidence["rendered"] != entry["rendered"]
                            or evidence["status"] != "PROPOSED_REQUIRES_MAIN_AGENT_ADJUDICATION"):
                        raise ValueError("exact review evidence changed")
                    if evidence["review"]["findings"] or evidence["invalidEvidence"]:
                        priority = 1
                if priority == 1 and complete_decision(entry, review["exact"], dispositions) is not None:
                    priority = 0
            priorities[entry["id"]] = priority
        for entry in staged["failures"]:
            priorities[entry["id"]] = 3
        snapshot = destination / "prior-staging" / f"{batch:03d}.json"
        atomic_publish_json(snapshot, staged)
        staging_pins.append(pin(snapshot))
    amended = reschedule(plan, published, priorities)
    amended["schedulingAmendment"] = {
        "previousPlan": previous_plan,
        "reviewInventory": pin(inventory_path),
        "findingDispositions": pin(dispositions_path) if dispositions_path else None,
        "priorStaging": staging_pins,
        "publishedBatchesUnchanged": sorted(published),
        "priorityCounts": dict(Counter(priorities.values())),
        "reason": "Schedule-only regrouping: exact reviewed/no findings, exact reviewed/findings, "
        "rendered needing review, then explicitly held. No identity removed or replaced. "
        "Every batch still requires complete fresh replay and exact-current adjudicated reviews.",
    }
    plan_path = destination / "plan.json"
    atomic_publish_json(plan_path, amended)
    atomic_publish_json(destination / "previous-manifest.json", previous_manifest)
    if read(manifest_path) != previous_manifest:
        raise ValueError("publication changed during schedule preparation")
    manifest["syntheticMigration"]["plan"] = pin(plan_path)
    atomic_write_json(manifest_path, manifest)
    print(json.dumps({"plan": pin(plan_path), "priorities": dict(Counter(priorities.values()))}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_inventory")
    parser.add_argument("destination")
    parser.add_argument("--dispositions")
    args = parser.parse_args()
    main(ROOT / args.review_inventory, ROOT / args.destination,
         ROOT / args.dispositions if args.dispositions else None)
