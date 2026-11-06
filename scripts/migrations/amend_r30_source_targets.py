"""Publish reviewed source amendments, preserving certificates and privacy proofs."""

import argparse
import copy
import json
import resource
import time

from audit_r30_independent import independent_view
from audit_r30_paddle_pilot import check_training_target
from migrate_r30_synthetic import DATA, ROOT, read
from publish_r30_sources import active_source_certificates, jsonl, pin, verify_pin
from stage_r30_independent_repairs import amended_target, repair_record

from document_ocr.atomic import (
    atomic_publish_bytes,
    atomic_publish_json,
    atomic_write_bytes,
    atomic_write_json,
)
from document_ocr.spatial_inputs.paddle import pages_from_record
from document_ocr.synthesis.curated_reflow_policy import ReflowPolicy
from document_ocr.synthesis.paddle_reflow import PaddleReflow


def main(review_path, name):
    started = time.perf_counter()
    review = read(review_path)
    if review["status"] not in {
        "MAIN_AGENT_REVIEWED_TARGET_ONLY_CORRECTIONS",
        "MAIN_AGENT_REVIEWED_SOURCE_CORRECTIONS",
    }:
        raise ValueError("explicit main-agent review required")
    privacy_allowed = review["status"] == "MAIN_AGENT_REVIEWED_SOURCE_CORRECTIONS"
    cfg = read(ROOT / "configs/synthesis/mpci_bl_curated_v7_paddle.json")
    engine = PaddleReflow(ROOT, ReflowPolicy.model_validate(cfg["positions"]["reflow"]))
    manifest = read(DATA / "manifest.json")
    groups = {
        s: [json.loads(line) for line in verify_pin(manifest["aggregateRecords"][s]).open()]
        for s in ("real-train", "validation")
    }
    updated = {r["documentId"]: copy.deepcopy(r) for rows in groups.values() for r in rows}
    changed = []
    folder = DATA / "audit-history" / name
    for sid, row in updated.items():
        decision = review["sources"].get(sid[4:12])
        if decision is None:
            continue
        before = copy.deepcopy(row)
        privacy_evidence = None
        if decision["changes"].get("privacy"):
            if not privacy_allowed:
                raise ValueError("target-only authority cannot change source text")
            repaired, privacy_evidence = repair_record(row, decision["changes"], engine)
            row.update(repaired)
        else:
            row["target"] = amended_target(row["target"], decision["changes"], pages_from_record(row))
        if check_training_target(row["target"]):
            raise ValueError("source amendment fails schema")
        for kind, key in (
            ("raw", "joinedRawText"),
            ("centroid", "positionedText"),
            ("bbox", "bboxText"),
        ):
            if (privacy_evidence is None and row[key] != before[key]) or independent_view(row["pages"], kind) != row[key]:
                raise ValueError("target-only amendment changed OCR or geometry")
        evidence = folder / "evidence" / (sid + ".json")
        atomic_publish_json(
            evidence,
            {
                "previousEvidence": before["evidence"],
                "review": pin(review_path),
                "decision": decision,
                "beforeTarget": before["target"],
                "afterTarget": row["target"],
                "privacyReplay": privacy_evidence,
            },
        )
        row["evidence"] = pin(evidence)
        changed.append(sid)
    if len(changed) != len(review["sources"]):
        raise ValueError("review membership does not match unique current sources")
    for split in groups:
        atomic_publish_bytes(
            folder / (split + ".jsonl"),
            verify_pin(manifest["aggregateRecords"][split]).read_bytes(),
        )
    atomic_publish_json(folder / "manifest.json", manifest)
    certificates = []
    for batch, previous in active_source_certificates().items():
        cert = read(previous)
        rows_by_split = {
            s: [json.loads(line) for line in verify_pin(p).open()] for s, p in cert["records"].items()
        }
        affected = [
            r["documentId"]
            for rows in rows_by_split.values()
            for r in rows
            if r["documentId"] in changed
        ]
        if not affected:
            certificates.append(pin(previous))
            continue
        destination = DATA / "amendments" / (name + "-" + batch)
        records = {}
        for split, rows in rows_by_split.items():
            path = destination / (split + ".jsonl")
            atomic_publish_bytes(path, jsonl([updated[r["documentId"]] for r in rows]))
            records[split] = pin(path)
        certificate = destination / "certificate.json"
        atomic_publish_json(
            certificate,
            {
                "status": "CERTIFIED_SOURCE_AMENDMENT" if privacy_allowed else "CERTIFIED_TARGET_ONLY_AMENDMENT",
                "batch": batch,
                "previous": pin(previous),
                "records": records,
                "changedDocuments": affected,
                "counts": {s: len(r) for s, r in rows_by_split.items()},
                "review": pin(review_path),
            },
        )
        certificates.append(pin(certificate))
    if read(DATA / "manifest.json") != manifest:
        raise ValueError("dataset changed during source amendment; publication not attempted")
    for split, rows in groups.items():
        atomic_write_bytes(
            DATA / (split + ".jsonl"), jsonl([updated[r["documentId"]] for r in rows])
        )
        manifest["aggregateRecords"][split] = pin(DATA / (split + ".jsonl"))
    manifest["batches"] = certificates
    atomic_write_json(DATA / "manifest.json", manifest)
    active_source_certificates()  # Verify the complete predecessor chain immediately.
    report = {
        "changedSources": changed,
        "review": pin(review_path),
        "inputsAndCoordinatesChanged": any(d["changes"].get("privacy") for d in review["sources"].values()),
        "elapsedSeconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    atomic_publish_json(folder / "publication.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review")
    parser.add_argument("name")
    args = parser.parse_args()
    main(ROOT / args.review, args.name)
