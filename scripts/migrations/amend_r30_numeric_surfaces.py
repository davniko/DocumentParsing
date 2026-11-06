"""Publish explicitly reviewed additive-wording repairs with exact receipt replay."""

import argparse
import copy
import json
import resource
import time

from audit_r30_independent import independent_view
from audit_r30_paddle_pilot import check_training_target
from migrate_r30_synthetic import DATA, ROOT, read
from publish_r30_sources import jsonl, pin, verify_pin
from repair_r30_disjoint_numeric_surfaces import repair
from validate_r30_migration_catalog import semantic_changes

from document_ocr.atomic import (
    atomic_publish_bytes,
    atomic_publish_json,
    atomic_write_bytes,
    atomic_write_json,
)
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog


def checked_target_changes(before, after, expected):
    changes = list(semantic_changes(before, after))
    if [list(change) for change in changes] != expected:
        raise ValueError("numeric wording target changes differ from explicit review")
    if any(path != "/goodsItemDetails/0/description" for path, _, _ in changes):
        raise ValueError("numeric wording repair may only change literal description target")
    if check_training_target(after):
        raise ValueError("repaired target fails schema")


def main(review_path):
    started = time.perf_counter()
    review = read(review_path)
    if review["status"] != "MAIN_AGENT_REVIEWED_NUMERIC_SURFACE_AMENDMENT":
        raise ValueError("explicit completed review required")
    proposal = read(verify_pin(review["proposal"]))
    authority = read(verify_pin(proposal["review"]))
    original = read(verify_pin(authority["candidate"]))
    certificates = read(verify_pin(authority["numericCertificates"]))
    expected, changes = repair(
        original, certificates["sources"][original["sourceDocumentId"]], authority["groupLabels"]
    )
    expected["documentId"] = proposal["id"]
    expected["migrationNumericScopeRepair"] = {"review": proposal["review"], "changes": changes}
    candidate = read(verify_pin(proposal["candidate"]))
    if candidate != expected or changes != proposal["changes"]:
        raise ValueError("amendment differs from exact reviewed numeric repair")
    manifest = read(DATA / "manifest.json")
    catalog = PaddleTemplateCatalog(ROOT, manifest["templateCatalog"])
    rendered = catalog.render(
        verify_pin(proposal["candidate"]),
        current=read(verify_pin(proposal["replacementRecord"])),
        wording_receipt=read(verify_pin(proposal["wording"])) if proposal["wording"] else None,
    )
    if rendered != read(verify_pin(proposal["rendered"])):
        raise ValueError("current native replay differs from reviewed output")
    for kind, field in (
        ("raw", "joinedRawText"),
        ("centroid", "positionedText"),
        ("bbox", "bboxText"),
    ):
        if independent_view(rendered["pages"], kind) != rendered[field]:
            raise ValueError("native coordinate serialization mismatch")
    prepared, matched = [], 0
    directory = DATA / "synthetic-amendments" / ("numeric-surfaces-" + digest(review))
    for certificate_pin in manifest["syntheticBatches"]:
        certificate = read(verify_pin(certificate_pin))
        rows = list(map(json.loads, verify_pin(certificate["records"]).open()))
        for row in rows:
            if row["documentId"] != proposal["id"]:
                continue
            matched += 1
            if digest(row) != review["beforeRecordSha256"]:
                raise ValueError("published record changed after final readback")
            checked_target_changes(row["target"], rendered["target"], review["targetChanges"])
            old_review = read(verify_pin(row["evidence"]["review"]))
            old_staging = read(verify_pin(old_review["staging"]))
            old_entry = next(e for e in old_staging["results"] if e["id"] == proposal["id"])
            reconciliation = {
                "status": "REVIEWED_CURRENT_SHIPMENT_RECEIPT",
                **{
                    key: proposal[key]
                    for key in ("candidate", "replacementRecord", "wording", "currentRecordSha256")
                },
                "decision": review["decision"],
            }
            atomic_publish_json(directory / "reconciliation.json", reconciliation)
            entry = {
                **old_entry,
                "candidate": proposal["candidate"],
                "wording": proposal["wording"],
                "rendered": proposal["rendered"],
                "reconciliation": pin(directory / "reconciliation.json"),
            }
            atomic_publish_json(
                directory / "staging.json",
                {
                    "catalog": manifest["templateCatalog"],
                    "results": [entry],
                    "scope": "One reviewed amendment, not a new batch membership",
                    "proposal": review["proposal"],
                },
            )
            atomic_publish_json(
                directory / "review.json",
                {
                    "status": "MAIN_AGENT_REVIEWED_FOR_PUBLICATION",
                    "staging": pin(directory / "staging.json"),
                    "adjudications": pin(review_path),
                    "records": [
                        {
                            "id": proposal["id"],
                            "rendered": proposal["rendered"],
                            "decision": review["decision"],
                        }
                    ],
                },
            )
            for key in ("pages", "target", "joinedRawText", "positionedText", "bboxText"):
                row[key] = copy.deepcopy(rendered[key])
            row["evidence"] = {
                **row["evidence"],
                "candidate": proposal["candidate"],
                "wording": proposal["wording"],
                "rendered": proposal["rendered"],
                "catalog": manifest["templateCatalog"],
                "review": pin(directory / "review.json"),
                "previousRecordSha256": review["beforeRecordSha256"],
            }
        prepared.append((certificate_pin, certificate, rows))
    if matched != 1 or read(DATA / "manifest.json") != manifest:
        raise ValueError("amendment membership or active authority changed")
    pins, aggregate = [], []
    for old_pin, certificate, rows in prepared:
        if any(r["documentId"] == proposal["id"] for r in rows):
            folder = directory / f"{certificate['batch']:03d}"
            atomic_publish_bytes(folder / "synthetic-train.jsonl", jsonl(rows))
            atomic_publish_json(
                folder / "certificate.json",
                {
                    **certificate,
                    "previous": old_pin,
                    "records": pin(folder / "synthetic-train.jsonl"),
                    "changedDocuments": [proposal["id"]],
                    "amendmentReview": pin(review_path),
                },
            )
            pins.append(pin(folder / "certificate.json"))
        else:
            pins.append(old_pin)
        aggregate.extend(rows)
    report = {
        "changedDocuments": [proposal["id"]],
        "review": pin(review_path),
        "previousManifest": manifest,
        "elapsedSeconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    atomic_publish_json(directory / "publication.json", report)
    atomic_write_bytes(DATA / "synthetic-train.jsonl", jsonl(aggregate))
    manifest["syntheticBatches"] = pins
    manifest["aggregateRecords"]["synthetic-train"] = pin(DATA / "synthetic-train.jsonl")
    atomic_write_json(DATA / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in report.items() if k != "previousManifest"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review")
    main(ROOT / parser.parse_args().review)
