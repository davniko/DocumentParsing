"""Verify active publication; never treat staged rows as certified."""

import argparse
import hashlib
import json
import resource
import time

from audit_r30_independent import independent_view
from audit_r30_paddle_pilot import check_training_target
from migrate_r30_synthetic import DATA, OUT, read
from publish_r30_sources import active_source_certificates, pin, verify_pin

from document_ocr.atomic import atomic_publish_json, atomic_write_json


def main(name):
    started = time.perf_counter()
    manifest = read(DATA / "manifest.json")
    manifest_pin = pin(DATA / "manifest.json")
    certified = {"real-train": {}, "validation": {}, "synthetic-train": {}}

    def add(split, record, line):
        identity = record["documentId"]
        if identity in certified[split]:
            raise ValueError("duplicate identity in active certificates")
        # Certificates and aggregates use the same canonical JSONL writer.
        # Compare exact bytes without retaining two full copies of the dataset.
        certified[split][identity] = hashlib.sha256(line.encode()).hexdigest()

    source_certificates = active_source_certificates()
    if {pin(p)["sha256"] for p in source_certificates.values()} != {
        p["sha256"] for p in manifest["batches"]
    }:
        raise ValueError("active source chains differ from published manifest")
    for path in source_certificates.values():
        for split, record_pin in read(path)["records"].items():
            for line in verify_pin(record_pin).open():
                add(split, json.loads(line), line)
    for certificate_pin in manifest.get("syntheticBatches", []):
        certificate = read(verify_pin(certificate_pin))
        count = 0
        for line in verify_pin(certificate["records"]).open():
            count += 1
            row = json.loads(line)
            rendered = read(verify_pin(row["evidence"]["rendered"]))
            for key in ("target", "pages", "joinedRawText", "positionedText", "bboxText"):
                if row[key] != rendered[key]:
                    raise ValueError("published payload differs from pinned render")
            add("synthetic-train", row, line)
        if count != certificate["count"]:
            raise ValueError("synthetic certificate count differs")

    counts, all_ids = {}, set()
    for split, expected in certified.items():
        rows = {}
        for line in verify_pin(manifest["aggregateRecords"][split]).open():
            row = json.loads(line)
            identity = row["documentId"]
            if identity in all_ids:
                raise ValueError("duplicate aggregate identity")
            all_ids.add(identity)
            if check_training_target(row["target"]):
                raise ValueError("published target fails schema")
            for kind, key in (
                ("raw", "joinedRawText"),
                ("centroid", "positionedText"),
                ("bbox", "bboxText"),
            ):
                if independent_view(row["pages"], kind) != row[key]:
                    raise ValueError("independent native view differs")
            rows[identity] = hashlib.sha256(line.encode()).hexdigest()
        if rows != expected:
            raise ValueError("aggregate does not equal active certified records")
        counts[split] = len(rows)
    required = manifest["syntheticMigration"]["required"]
    if counts["synthetic-train"] != manifest["syntheticMigration"]["certified"]:
        raise ValueError("manifest publication count differs")
    if read(DATA / "manifest.json") != manifest:
        raise ValueError("publication changed during audit")
    report = {
        "status": (
            "PUBLISHED_DATASET_VERIFIED"
            if counts["synthetic-train"] == required
            else "PUBLISHED_CHECKPOINT_VERIFIED_MIGRATION_IN_PROGRESS"
        ),
        "scope": "Active certificates, exact payloads, schema, native order/boxes and three "
        "independent input views. This does not replace semantic review or certify pending rows.",
        "manifest": manifest_pin,
        "catalog": manifest["templateCatalog"],
        "controls": manifest["templateControls"],
        "counts": counts,
        "syntheticRequired": required,
        "syntheticUncertified": required - counts["synthetic-train"],
        "allPublishedViewsAndSchemaPassed": True,
        "elapsedSeconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "pendingWork": []
        if counts["synthetic-train"] == required
        else [
            "Complete exact-current semantic review and adjudication of pending batches",
            "Generate only new native owners; preserve retained shipment facts",
            "Compose historical repairs without reverting current R29 data",
            "Resolve explicit owner/packing holds and six long-inventory native rebases",
            "Publish all fixed identities and audit the final 9400-record synthetic membership",
        ],
    }
    atomic_publish_json(OUT / (name + ".json"), report)
    atomic_write_json(OUT / "resume-state.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="Unique immutable checkpoint report filename stem")
    main(parser.parse_args().name)
