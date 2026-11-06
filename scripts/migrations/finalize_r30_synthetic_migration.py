"""Finalize only a completely published and independently audited Paddle rebase."""

import argparse
import json
import resource
import time

from migrate_r30_synthetic import DATA, OUT, ROOT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json, atomic_write_json


def verify_completion(manifest, audit, catalog, controls, plan, replay):
    required = manifest["syntheticMigration"]["required"]
    expected = {"real-train": 600, "validation": 60, "synthetic-train": required}
    if (
        manifest["syntheticMigration"]["certified"] != required
        or audit["counts"] != expected
        or audit["syntheticUncertified"] != 0
        or audit["allPublishedViewsAndSchemaPassed"] is not True
        or audit["catalog"] != manifest["templateCatalog"]
        or audit["controls"] != manifest["templateControls"]
    ):
        raise ValueError("complete publication and exact active audit are required")
    if (
        replay["catalog"] != manifest["templateCatalog"]
        or replay["manifest"] != audit["manifest"]
        or replay["records"] != required
        or replay["failures"]
    ):
        raise ValueError("complete exact-current published render replay is required")
    identities = [identity for batch in plan["batches"] for identity in batch]
    if len(identities) != required or len(set(identities)) != required:
        raise ValueError("frozen migration membership is incomplete or duplicated")
    covered = set()
    for report, tested_catalog in controls:
        sources = report["sources"]
        if report["failures"] or report["passed"] != 3 * len(sources):
            raise ValueError("all three controls must pass for every tested source")
        for source in sources:
            if tested_catalog["templates"][source] == catalog["templates"].get(source):
                covered.add(source)
    if covered != set(catalog["templates"]):
        raise ValueError("current template entries lack complete fresh control coverage")
    return set(identities)


def main(audit_path, control_paths, replay_path, name):
    started = time.perf_counter()
    manifest_pin = pin(DATA / "manifest.json")
    manifest = read(verify_pin(manifest_pin))
    audit = read(audit_path)
    if audit["manifest"] != manifest_pin:
        raise ValueError("audit was not run against this exact active manifest")
    catalog = read(verify_pin(manifest["templateCatalog"]))
    controls = [(read(path), read(verify_pin(read(path)["catalog"]))) for path in control_paths]
    expected = verify_completion(
        manifest,
        audit,
        catalog,
        controls,
        read(verify_pin(manifest["syntheticMigration"]["plan"])),
        read(replay_path),
    )
    actual = {
        json.loads(line)["documentId"]
        for line in verify_pin(manifest["aggregateRecords"]["synthetic-train"]).open()
    }
    if actual != expected:
        raise ValueError("published identities differ from frozen migration membership")
    report = {
        "status": "CERTIFIED_PADDLE_DATASET_REBASE_COMPLETE",
        "previousManifest": manifest_pin,
        "independentAudit": pin(audit_path),
        "controlReplays": [pin(path) for path in control_paths],
        "publishedReplay": pin(replay_path),
        "counts": audit["counts"],
        "templates": len(catalog["templates"]),
        "membershipExact": True,
        "scope": "Complete native real/validation/synthetic records and template registry. "
        "No training run or model-quality improvement is implied.",
        "elapsedSeconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    destination = OUT / (name + ".json")
    if pin(DATA / "manifest.json") != manifest_pin:
        raise ValueError("active publication changed during finalization")
    atomic_publish_json(destination, report)
    manifest.update(status=report["status"], completionEvidence=pin(destination))
    atomic_write_json(DATA / "manifest.json", manifest)
    atomic_write_json(OUT / "resume-state.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit")
    parser.add_argument("--controls", nargs="+", required=True)
    parser.add_argument("--published-replay", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    main(
        ROOT / args.audit,
        [ROOT / path for path in args.controls],
        ROOT / args.published_replay,
        args.name,
    )
