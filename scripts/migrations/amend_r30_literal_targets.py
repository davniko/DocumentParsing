"""Replay and publish admitted literal-only target repairs in certified records."""

import argparse
import copy
import json
import resource
import time

from audit_r30_paddle_pilot import check_training_target
from migrate_r30_synthetic import DATA, OUT, ROOT, current_for_render, read
from publish_r30_sources import jsonl, pin, verify_pin
from r30_literal_target_corrections import apply_literal_target_correction, correction_index

from document_ocr.atomic import (
    atomic_publish_bytes,
    atomic_publish_json,
    atomic_write_bytes,
    atomic_write_json,
)
from document_ocr.synthesis.curated import digest
from r30_source_preserving import R30Replay


def checked_repair(row, replay, correction):
    for field in ("target", "pages", "joinedRawText", "positionedText", "bboxText"):
        if replay[field] != row[field]:
            raise ValueError("published record differs from its exact pinned replay")
    repaired = apply_literal_target_correction(replay, correction)
    if check_training_target(repaired["target"]):
        raise ValueError("literal target repair fails schema")
    for field in ("pages", "joinedRawText", "positionedText", "bboxText"):
        if repaired[field] != replay[field]:
            raise ValueError("literal target repair changed input or coordinates")
    return repaired


def main():
    started = time.perf_counter()
    manifest = read(DATA / "manifest.json")
    corrections = correction_index(manifest)
    inventory = read(OUT / "receipt-inventory.json")
    baseline = {
        r["documentId"]: r
        for r in map(json.loads, verify_pin(inventory["baseline"]).open())
        if r["documentId"] in corrections
    }
    directory = (
        DATA
        / "synthetic-amendments"
        / ("literal-" + manifest["literalTargetCorrections"]["sha256"])
    )
    catalogs, prepared, changed = {}, [], []
    # Validate every affected record before publishing any certificate.
    for certificate_pin in manifest["syntheticBatches"]:
        certificate = read(verify_pin(certificate_pin))
        rows = [json.loads(line) for line in verify_pin(certificate["records"]).open()]
        touched = []
        for row in rows:
            identity, evidence = row["documentId"], row["evidence"]
            correction = corrections.get(identity)
            if correction is None or correction == evidence.get("literalTargetCorrection"):
                continue
            if evidence.get("literalTargetCorrection") is not None:
                raise ValueError("cannot replace an already applied literal correction")
            catalog_pin = evidence["catalog"]
            key = catalog_pin["sha256"]
            if key not in catalogs:
                catalogs[key] = R30Replay(ROOT, catalog_pin, manifest.get("sourcePreservingReceipts", ()))
            reviewed = read(verify_pin(evidence["review"]))
            staged = read(verify_pin(reviewed["staging"]))
            entry = next(e for e in staged["results"] if e["id"] == identity)
            replay = catalogs[key].render(
                verify_pin(evidence["candidate"]),
                current=current_for_render(
                    baseline[identity],
                    read(verify_pin(entry["reconciliation"]))
                    if entry.get("reconciliation")
                    else None,
                ),
                wording_receipt=read(verify_pin(evidence["wording"]))
                if evidence["wording"]
                else None,
            )
            repaired = checked_repair(row, replay, correction)
            previous = digest(row)
            rendered_path = directory / (identity + "-rendered.json")
            atomic_publish_json(rendered_path, repaired)
            row["target"] = copy.deepcopy(repaired["target"])
            row["evidence"] = {
                **evidence,
                "previousRendered": evidence["rendered"],
                "rendered": pin(rendered_path),
                "literalTargetCorrection": correction,
                "previousRecordSha256": previous,
            }
            touched.append(identity)
        prepared.append((certificate_pin, certificate, rows, touched))
        changed.extend(touched)
    if not changed:
        print(json.dumps({"changed": 0, "status": "ALREADY_APPLIED_OR_NOT_YET_PUBLISHED"}))
        return
    if read(DATA / "manifest.json") != manifest:
        raise ValueError("dataset changed during literal amendment validation")
    certificates, aggregate = [], []
    for old_pin, certificate, rows, touched in prepared:
        if touched:
            folder = directory / f"{certificate['batch']:03d}"
            atomic_publish_bytes(folder / "synthetic-train.jsonl", jsonl(rows))
            atomic_publish_json(
                folder / "certificate.json",
                {
                    **certificate,
                    "previous": old_pin,
                    "records": pin(folder / "synthetic-train.jsonl"),
                    "literalTargetCorrections": manifest["literalTargetCorrections"],
                    "changedDocuments": touched,
                    "literalAmendmentFullReplay": True,
                },
            )
            certificates.append(pin(folder / "certificate.json"))
        else:
            certificates.append(old_pin)
        aggregate.extend(rows)
    report = {
        "status": "PUBLISHED_REPLAYED_LITERAL_TARGET_CORRECTIONS",
        "changedDocuments": changed,
        "inputAndCoordinatesUnchanged": True,
        "previousManifest": manifest,
        "elapsedSeconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    atomic_publish_json(directory / "report.json", report)
    atomic_write_bytes(DATA / "synthetic-train.jsonl", jsonl(aggregate))
    manifest["syntheticBatches"] = certificates
    manifest["aggregateRecords"]["synthetic-train"] = pin(DATA / "synthetic-train.jsonl")
    manifest["literalAmendment"] = pin(directory / "report.json")
    atomic_write_json(DATA / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in report.items() if k != "previousManifest"}))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
