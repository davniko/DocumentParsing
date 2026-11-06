"""Publish fully replayed migration corrections without replacing prior history."""

import copy
import json
import resource
import time

from audit_r30_independent import independent_view
from audit_r30_paddle_pilot import check_training_target
from migrate_r30_synthetic import DATA, OUT, ROOT, read
from publish_r30_sources import active_source_certificates, jsonl, pin, verify_pin
from stage_r30_independent_repairs import amended_target

from document_ocr.atomic import (
    atomic_publish_bytes,
    atomic_publish_json,
    atomic_write_bytes,
    atomic_write_json,
)
from document_ocr.spatial_inputs.paddle import pages_from_record


def main():
    started = time.perf_counter()
    full_path = OUT / "semantic-revision-08-replay.json"
    full = read(full_path)
    follow_path = OUT / "semantic-revision-09-replay.json"
    follow = read(follow_path)
    if full["failures"] or full["passed"] != 1368 or len(full["sources"]) != 456:
        raise ValueError("complete catalog replay is required")
    before_catalog = read(verify_pin(full["catalog"]))
    catalog_path = verify_pin(follow["catalog"])
    after_catalog = read(catalog_path)
    changed_templates = [
        s for s, e in before_catalog["templates"].items() if e != after_catalog["templates"][s]
    ]
    if changed_templates != follow["sources"] or follow["failures"] or follow["passed"] != 3:
        raise ValueError("incremental replay must cover precisely the changed template")
    controls = read(verify_pin(full["catalog"]).parent / "controls.json")
    controls.update(read(catalog_path.parent / "controls.json"))
    if len(controls) != 456 or sum(map(len, controls.values())) != 1368:
        raise ValueError("composed control coverage changed")
    replay_path = OUT / "semantic-combined-replay.json"
    atomic_publish_json(
        replay_path,
        {
            "fullReplay": pin(full_path),
            "changedTemplateReplay": pin(follow_path),
            "catalog": follow["catalog"],
            "templates": 456,
            "controls": 1368,
            "reviewedChanges": full["reviewedChanges"] + follow["reviewedChanges"],
            "failures": [],
        },
    )
    old_manifest = read(DATA / "manifest.json")
    destination = (
        ROOT
        / "artifacts/synthesis-templates/mpci-bl-v7-paddle-reviewed/r30/migration-audit-20261103"
    )
    atomic_publish_json(destination / "catalog.json", read(catalog_path))
    atomic_publish_json(destination / "controls.json", controls)
    current = {
        split: [json.loads(line) for line in (DATA / (split + ".jsonl")).open()]
        for split in ("real-train", "validation")
    }
    updated = {r["documentId"]: copy.deepcopy(r) for group in current.values() for r in group}
    changes = []
    for sid, row in updated.items():
        prefix = sid[4:12]
        if prefix == "751f068c":
            decision = {"remove": ["/parties/consignee/contactDetails/emailAddresses"]}
        elif prefix == "d684b168":
            regions = row["pages"][0]["regions"]
            texts = [r["text"] for r in regions]
            cleaning = next(t for t in texts if t.startswith("-Receivers are compelled"))
            continuation = next(t for t in texts if t == "estimated by depot.")
            release = next(t for t in texts if "Any transportation" in t)
            release = release[release.index("Any transportation") :]
            release_end = next(
                t for t in texts if t.startswith("containers at Alexandria terminal")
            )
            decision = {
                "set": {
                    "/goodsItemDetails/0/handlingInstructions": [
                        (cleaning + " " + continuation).upper(),
                        (release + " " + release_end).upper(),
                    ]
                }
            }
        else:
            continue
        before = copy.deepcopy(row)
        row["target"] = amended_target(row["target"], decision, pages_from_record(row))
        problems = check_training_target(row["target"])
        if problems:
            raise ValueError(f"source amendment fails schema: {problems}")
        for kind, key in (
            ("raw", "joinedRawText"),
            ("centroid", "positionedText"),
            ("bbox", "bboxText"),
        ):
            if row[key] != before[key] or independent_view(row["pages"], kind) != row[key]:
                raise ValueError("target-only amendment changed source text or geometry")
        evidence_path = destination / "source-evidence" / (sid + ".json")
        atomic_publish_json(
            evidence_path,
            {
                "previousEvidence": before["evidence"],
                "decision": decision,
                "beforeTarget": before["target"],
                "afterTarget": row["target"],
                "review": "Main-agent native-region readback under direct-consignee-contact and operational handling scope. No OCR or coordinate changes.",
            },
        )
        row["evidence"] = pin(evidence_path)
        changes.append(sid)
    if len(changes) != 2:
        raise ValueError("reviewed real source membership changed")
    archive = DATA / "audit-history/pre-migration-semantic-20261103"
    for split in current:
        atomic_publish_bytes(archive / (split + ".jsonl"), (DATA / (split + ".jsonl")).read_bytes())
    atomic_publish_json(archive / "manifest.json", old_manifest)
    certificates = []
    for batch, previous in active_source_certificates().items():
        cert = read(previous)
        groups = {
            s: [json.loads(line) for line in verify_pin(p).open()]
            for s, p in cert["records"].items()
        }
        changed = [
            r["documentId"]
            for group in groups.values()
            for r in group
            if r["documentId"] in changes
        ]
        if not changed:
            certificates.append(pin(previous))
            continue
        folder = DATA / "amendments" / ("zzzzz-migration-semantic-20261103-" + batch)
        records = {}
        for split, group in groups.items():
            path = folder / (split + ".jsonl")
            atomic_publish_bytes(path, jsonl([updated[r["documentId"]] for r in group]))
            records[split] = pin(path)
        path = folder / "certificate.json"
        atomic_publish_json(
            path,
            {
                "status": "CERTIFIED_MIGRATION_SEMANTIC_AMENDMENT",
                "batch": batch,
                "previous": pin(previous),
                "records": records,
                "changedDocuments": changed,
                "counts": {s: len(group) for s, group in groups.items()},
                "review": pin(replay_path),
            },
        )
        certificates.append(pin(path))
    for split, group in current.items():
        atomic_write_bytes(
            DATA / (split + ".jsonl"), jsonl([updated[r["documentId"]] for r in group])
        )
    manifest = copy.deepcopy(old_manifest)
    manifest.update(
        batches=certificates,
        templateCatalog=pin(destination / "catalog.json"),
        templateControls=pin(destination / "controls.json"),
        aggregateRecords={s: pin(DATA / (s + ".jsonl")) for s in current},
        migrationSemanticAudit=pin(replay_path),
    )
    atomic_write_json(DATA / "manifest.json", manifest)
    atomic_publish_json(
        OUT / "authorities-publication.json",
        {
            "changedSources": changes,
            "templates": 456,
            "controls": 1368,
            "catalog": manifest["templateCatalog"],
            "controlsPin": manifest["templateControls"],
            "elapsedSeconds": time.perf_counter() - started,
            "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
    )


if __name__ == "__main__":
    main()
