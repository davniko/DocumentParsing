"""Publish the fully replayed native repetition contracts, preserving prior certificates."""

import argparse
import copy
import json

from migrate_r30_synthetic import DATA, OUT, ROOT, read
from publish_r30_sources import jsonl, pin, verify_pin

from document_ocr.atomic import (
    atomic_publish_bytes,
    atomic_publish_json,
    atomic_write_bytes,
    atomic_write_json,
)
from document_ocr.synthesis.curated import digest
from document_ocr.spatial_inputs.paddle import text_view_digests


def main(report_path, destination, amendments_path=None):
    report = read(report_path)
    catalog_path = verify_pin(report["catalog"])
    catalog = read(catalog_path)
    manifest = read(DATA / "manifest.json")
    review = read(verify_pin(report["review"]))
    if manifest["templateCatalog"] != review["baseCatalog"]:
        raise ValueError("active authority changed after the reviewed replay")
    if report["priorControls"] != manifest["templateControls"]:
        raise ValueError("baseline controls changed after replay")
    previous = read(verify_pin(manifest["templateCatalog"]))
    if set(previous["templates"]) != set(catalog["templates"]):
        raise ValueError("correction changes template membership")
    if {k: v for k, v in previous.items() if k != "templates"} != {
        k: v for k, v in catalog.items() if k != "templates"
    }:
        raise ValueError("source-scoped correction changes shared catalog settings")
    changed = {
        sid
        for sid in previous["templates"]
        if previous["templates"][sid] != catalog["templates"][sid]
    }
    if {sid[4:12] for sid in changed} != set(review["sources"]):
        raise ValueError("catalog difference exceeds reviewed source membership")
    if (
        report["failures"]
        or report["passed"] != 3 * len(report["sources"])
        or not changed <= set(report["sources"])
    ):
        raise ValueError("every changed source needs all three fresh control replays")
    replayed = read(catalog_path.parent / "controls.json")
    if set(replayed) != set(report["sources"]):
        raise ValueError("replay evidence does not match the reported source set")
    controls = read(verify_pin(manifest["templateControls"]))
    controls.update(replayed)
    if set(controls) != set(catalog["templates"]) or any(len(v) != 3 for v in controls.values()):
        raise ValueError("catalog control membership differs")
    for sid, group in controls.items():
        for control in group:
            if read(verify_pin(control))["sourceDocumentId"] != sid:
                raise ValueError("replay control belongs to another source")
    affected = []
    for cert_pin in manifest.get("syntheticBatches", []):
        cert = read(verify_pin(cert_pin))
        for line in verify_pin(cert["records"]).open():
            row = json.loads(line)
            if row["sourceDocumentId"] in changed:
                affected.append(row["documentId"])
    amendments = {}
    if amendments_path:
        approval = read(amendments_path)
        staged = read(verify_pin(approval["staged"]))
        if (
            approval["status"] != "MAIN_AGENT_REVIEWED_SYNTHETIC_AMENDMENTS"
            or staged["replay"] != pin(report_path)
            or staged["previousCatalog"] != manifest["templateCatalog"]
        ):
            raise ValueError("published amendment approval has the wrong authority")
        amendments = {v["id"]: v for v in staged["changes"]}
        if set(amendments) != set(affected) or set(approval["decisions"]) != set(affected):
            raise ValueError("every affected published record needs exact main-agent review")
    if affected and not amendments:
        raise ValueError(
            f"changed templates affect published records requiring amendment: {affected}"
        )
    atomic_publish_json(destination / "catalog.json", catalog)
    atomic_publish_json(destination / "controls.json", controls)
    publication = {
        "status": "CERTIFIED_TEMPLATE_CORRECTIONS_PUBLISHED",
        "previousCatalog": manifest["templateCatalog"],
        "catalog": pin(destination / "catalog.json"),
        "controls": pin(destination / "controls.json"),
        "replay": pin(report_path),
        "reviewedSources": sorted(changed),
        "freshControlReplays": report["passed"],
        "unchangedPinnedControlsReused": 1368 - report["passed"],
        "trainingRecordsChanged": len(amendments),
        "affectedPublishedSyntheticRecords": affected,
        "retainedBatchCompatibility": "Affected records are explicitly amended; unchanged rows "
        "and original certificates retain their pinned authorities.",
        "publishedAmendmentReview": pin(amendments_path) if amendments_path else None,
    }
    publication_path = OUT / (destination.name + "-publication.json")
    atomic_publish_json(publication_path, publication)
    atomic_publish_json(destination / "previous-dataset-manifest.json", manifest)
    updated = copy.deepcopy(manifest)
    updated.update(
        templateCatalog=publication["catalog"],
        templateControls=publication["controls"],
        migrationRepetitionAudit=pin(publication_path),
    )
    if amendments:
        certificates, aggregate = [], []
        for cert_pin in manifest["syntheticBatches"]:
            cert = read(verify_pin(cert_pin))
            rows = [json.loads(line) for line in verify_pin(cert["records"]).open()]
            touched = []
            for i, row in enumerate(rows):
                change = amendments.get(row["documentId"])
                if not change:
                    continue
                if change["certificate"] != cert_pin or change["beforeRecordSha256"] != digest(row):
                    raise ValueError("published row changed after amendment readback")
                after = read(verify_pin(change["after"]))
                rendered = read(verify_pin(change["rendered"]))
                for key in ("pages", "target", "joinedRawText", "positionedText", "bboxText"):
                    if after[key] != rendered[key]:
                        raise ValueError("amended record differs from its fresh rendered evidence")
                if any(after[key] != value for key, value in text_view_digests(after).items()):
                    raise ValueError("amended record retains stale input-view checksums")
                after["evidence"] = {
                    **after["evidence"],
                    "catalog": publication["catalog"],
                    "rendered": change["rendered"],
                    "amendment": pin(amendments_path),
                    "previousRecordSha256": digest(row),
                }
                rows[i] = after
                touched.append(row["documentId"])
            if touched:
                folder = DATA / "synthetic-amendments" / destination.name / f"{cert['batch']:03d}"
                atomic_publish_bytes(folder / "synthetic-train.jsonl", jsonl(rows))
                new_cert = {
                    **cert,
                    "previous": cert_pin,
                    "records": pin(folder / "synthetic-train.jsonl"),
                    "amendmentReview": pin(amendments_path),
                    "changedDocuments": touched,
                    "catalog": publication["catalog"],
                }
                atomic_publish_json(folder / "certificate.json", new_cert)
                certificates.append(pin(folder / "certificate.json"))
            else:
                certificates.append(cert_pin)
            aggregate.extend(rows)
        atomic_write_bytes(DATA / "synthetic-train.jsonl", jsonl(aggregate))
        updated["syntheticBatches"] = certificates
        updated["aggregateRecords"]["synthetic-train"] = pin(DATA / "synthetic-train.jsonl")
    atomic_write_json(DATA / "manifest.json", updated)
    print(json.dumps(publication, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("destination")
    parser.add_argument("--amendments")
    args = parser.parse_args()
    main(
        ROOT / args.report,
        ROOT / args.destination,
        ROOT / args.amendments if args.amendments else None,
    )
