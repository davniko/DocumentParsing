"""Publish the reviewed audit revision after a complete independent replay.

Keep the initial handoff and all earlier batches immutable. A new pinned catalog
revision selects audited overlays; real records extend their certificate chains.
"""

from __future__ import annotations

import argparse
import copy
import json
import resource
import time

from audit_r30_independent import independent_view
from audit_r30_paddle_pilot import check_training_target
from publish_r30_sources import DATA, TEMPLATES, active_source_certificates, jsonl, pin, verify_pin
from rebase_r30_sources import ART as REBASE_ART
from rebase_r30_sources import BASE, PILOT_ART, ROOT, read, rows
from rebase_r30_sources import OUT as REBASE_OUT
from render_r30_paddle_pilot import load_current
from stage_r30_independent_repairs import REVIEW_FILES, merged_reviews, repair_record

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, atomic_write_json
from document_ocr.synthesis.curated_reflow_policy import ReflowPolicy
from document_ocr.synthesis.paddle_reflow import PaddleReflow
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog

OUT = ROOT / "docs/analysis/r30-independent-audit-20261103"
ART = ROOT / "artifacts/paddle-migration/r30-independent-audit-20261103"
REVISION = TEMPLATES / "audit-20261103"


def prepare():
    repairs = read(OUT / "template-repair-staging.json")
    if repairs["failures"] or len(repairs["results"]) != 3 * repairs["templates"]:
        raise ValueError("all reviewed templates must pass before preparing publication")
    manifest = read(verify_pin(repairs["candidateCatalog"]))
    controls = read(REBASE_OUT / "handoff-controls.json")
    # Copy changed authorities out of mutable staging; unchanged authorities stay pinned.
    changed = {r["source"] for r in repairs["results"]}
    for did in changed:
        path = REVISION / "cases" / (did + ".json")
        atomic_publish_json(path, read(verify_pin(manifest["templates"][did]["template"])))
        manifest["templates"][did]["template"] = pin(path)
    for did in changed:
        controls[did] = [r["after"] for r in repairs["results"] if r["source"] == did]
    product = read(ART / "product-repair/receipt.json")
    candidate = read(verify_pin(product["after"]))
    did = candidate["sourceDocumentId"]
    template = read(verify_pin(manifest["templates"][did]["template"]))
    transfer = read(verify_pin(template["transfer"]))
    selected = [s for s in transfer["descendants"] if s["id"] == candidate["documentId"]]
    if len(selected) != 1 or pin(ROOT / selected[0]["path"]) != product["before"]:
        raise ValueError("product correction does not match the retained control")
    selected[0].update(path=product["after"]["path"], receiptSha256=product["after"]["sha256"])
    transfer_path = REVISION / "transfers" / (did + ".json")
    atomic_publish_json(transfer_path, transfer)
    template["transfer"] = pin(transfer_path)
    path = REVISION / "cases" / (did + ".json")
    atomic_publish_json(path, template)
    manifest["templates"][did]["template"] = pin(path)
    controls[did] = [
        product["after"] if read(verify_pin(p))["documentId"] == candidate["documentId"] else p
        for p in controls[did]
    ]
    atomic_publish_json(REVISION / "catalog.json", manifest)
    atomic_publish_json(REVISION / "expected-controls.json", controls)
    print(
        json.dumps({"status": "PREPARED_AUDIT_REVISION", "catalog": pin(REVISION / "catalog.json")})
    )


def validate():
    started = time.perf_counter()
    catalog_pin = pin(REVISION / "catalog.json")
    catalog = PaddleTemplateCatalog(ROOT, catalog_pin)
    expected = read(REVISION / "expected-controls.json")
    prior = {r["documentId"]: r for r in rows(BASE / "synthetic-train.jsonl")}
    product = read(ART / "product-repair/receipt.json")
    product_id = read(verify_pin(product["after"]))["documentId"]
    controls, failures, render_seconds = {}, [], 0.0
    for did in sorted(catalog.entries):
        template, _, transfer, _ = catalog._template(did)
        artifacts = (
            PILOT_ART
            if catalog.entries[did]["privacy"]["kind"] == "reviewed_replacement_spans"
            else REBASE_ART
        )
        old_by_id = {read(verify_pin(p))["documentId"]: p for p in expected[did]}
        controls[did] = []
        for selection in transfer["descendants"]:
            identity, path = selection["id"], ROOT / selection["path"]
            try:
                spec = catalog.wording_specification(path)
                wording = read(artifacts / "new-wording" / (identity + ".json")) if spec else None
                current = load_current(selection, prior)
                if identity == product_id:
                    current = copy.deepcopy(current)
                    corrected = read(verify_pin(product["after"]))
                    current["target"]["goodsItemDetails"][0]["description"] = corrected["target"][
                        "documentPatch"
                    ]["goodsItemDetails"][0]["description"]
                    current["joinedRawText"] = corrected["joinedRawText"]
                tick = time.perf_counter()
                result = catalog.render(path, current=current, wording_receipt=wording)
                render_seconds += time.perf_counter() - tick
                if identity != product_id:
                    old = read(verify_pin(old_by_id[identity]))
                    old.pop("migrationPins", None)
                    if result != old:
                        raise ValueError("full-catalog replay differs from reviewed control")
                else:
                    reviewed = read(ART / "product-repair/paddle-control.json")
                    if (
                        result["target"] != reviewed["target"]
                        or result["joinedRawText"] != reviewed["joinedRawText"]
                    ):
                        raise ValueError("regenerated product does not match reviewed readback")
                for kind, key in (
                    ("raw", "joinedRawText"),
                    ("centroid", "positionedText"),
                    ("bbox", "bboxText"),
                ):
                    if independent_view(result["pages"], kind) != result[key]:
                        raise ValueError("independent geometry serialization differs")
                pins = {
                    "template": catalog.entries[did]["template"]["sha256"],
                    "transfer": template["transfer"]["sha256"],
                    "shipment": pin(path)["sha256"],
                }
                if wording is not None:
                    pins["wording"] = pin(artifacts / "new-wording" / (identity + ".json"))[
                        "sha256"
                    ]
                result["migrationPins"] = pins
                destination = REVISION / "controls" / (identity + ".json")
                atomic_publish_json(destination, result)
                controls[did].append(pin(destination))
            except (ValueError, KeyError, IndexError) as exc:
                failures.append({"source": did, "sample": identity, "error": str(exc)})
        if len(controls) % 20 == 0:
            print(json.dumps({"templates": len(controls), "failures": len(failures)}), flush=True)
    report = {
        "catalog": catalog_pin,
        "templates": len(controls),
        "controls": sum(map(len, controls.values())),
        "failures": failures,
        "renderAndValidationSeconds": render_seconds,
        "elapsedSeconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    atomic_write_json(OUT / "complete-audit-replay.json", report)
    if failures or any(len(v) != 3 for v in controls.values()):
        raise ValueError("complete replay failed; nothing published")
    atomic_publish_json(REVISION / "controls.json", controls)
    print(json.dumps(report, indent=2))


def publish():
    started = time.perf_counter()
    replay = read(OUT / "complete-audit-replay.json")
    if (
        replay["catalog"] != pin(REVISION / "catalog.json")
        or replay["failures"]
        or replay["templates"] != 456
        or replay["controls"] != 1368
    ):
        raise ValueError("audit publication requires the complete current replay")
    stage = read(OUT / "source-repair-staging.json")
    if any(
        pin(DATA / (split + ".jsonl")) != evidence for split, evidence in stage["before"].items()
    ):
        raise ValueError("source baseline changed since audit")
    reviews = merged_reviews([OUT / name for name in REVIEW_FILES])
    cfg = read(ROOT / "configs/synthesis/mpci_bl_curated_v7_paddle.json")
    engine = PaddleReflow(ROOT, ReflowPolicy.model_validate(cfg["positions"]["reflow"]))
    current = {r["documentId"]: r for p in stage["before"].values() for r in rows(verify_pin(p))}
    after = {r["documentId"]: r for p in stage["after"].values() for r in rows(verify_pin(p))}
    if current.keys() != after.keys() or len(after) != 660:
        raise ValueError("audit changes source membership")
    for did, record in after.items():
        expected = current[did]
        if did[4:12] in reviews:
            expected, _ = repair_record(expected, reviews[did[4:12]], engine)
            expected["evidence"] = record["evidence"]
        if expected != record:
            raise ValueError("published source differs from explicit reviewed repair")
        verify_pin(record["evidence"])
        check_training_target(record["target"])
        for kind, key in (
            ("raw", "joinedRawText"),
            ("centroid", "positionedText"),
            ("bbox", "bboxText"),
        ):
            if independent_view(record["pages"], kind) != record[key]:
                raise ValueError("source input serialization differs")
    review_path = OUT / "publication-review.json"
    review = {
        "status": "MAIN_AGENT_REVIEWED_COMPLETE_AUDIT",
        "sourceStage": pin(OUT / "source-repair-staging.json"),
        "templateStage": pin(OUT / "template-repair-staging.json"),
        "replay": pin(OUT / "complete-audit-replay.json"),
        "screenReadback": pin(OUT / "screen-adjudications.json"),
        "sourceDecisions": stage["decisions"],
        "templateDecisions": read(OUT / "template-repair-staging.json")["reviewPins"],
        "adjudications": [pin(p) for p in sorted(OUT.glob("*adjudications*.json"))],
        "sourceMembership": len(after),
        "mainAgentSourceReadback": stage["changedDocuments"],
    }
    atomic_publish_json(review_path, review)
    # Freeze the superseded aggregates before publishing certificates. Repair
    # evidence must reference that immutable archive, not a mutable active alias.
    old_manifest = read(DATA / "manifest.json")
    archive = DATA / "audit-history/pre-independent-20261103"
    atomic_publish_json(archive / "manifest.json", old_manifest)
    archived = {}
    for split, evidence in stage["before"].items():
        path = archive / (split + ".jsonl")
        atomic_publish_bytes(path, verify_pin(evidence).read_bytes())
        archived[evidence["path"]] = pin(path)
    for did in stage["changedDocuments"]:
        record = after[did]
        evidence = read(verify_pin(record["evidence"]))
        previous = evidence["previousRecords"]
        replacement = archived[previous["path"]]
        if previous["sha256"] != replacement["sha256"]:
            raise ValueError("archived baseline differs from repair evidence")
        evidence["previousRecords"] = replacement
        path = REVISION / "source-repair-evidence" / (did + ".json")
        atomic_publish_json(path, evidence)
        record["evidence"] = pin(path)
    certificates = active_source_certificates()
    updated = []
    for batch, previous in certificates.items():
        cert = read(previous)
        groups = {split: list(rows(verify_pin(p))) for split, p in cert["records"].items()}
        changed = [
            r["documentId"]
            for group in groups.values()
            for r in group
            if r != after[r["documentId"]]
        ]
        if not changed:
            updated.append(pin(previous))
            continue
        destination = DATA / "amendments" / ("zzzz-independent-audit-20261103-" + batch)
        output = {}
        for split, group in groups.items():
            path = destination / (split + ".jsonl")
            atomic_publish_bytes(path, jsonl([after[r["documentId"]] for r in group]))
            output[split] = pin(path)
        certificate = destination / "certificate.json"
        atomic_publish_json(
            certificate,
            {
                "status": "CERTIFIED_INDEPENDENT_AUDIT_AMENDMENT",
                "batch": batch,
                "previous": pin(previous),
                "review": pin(review_path),
                "records": output,
                "counts": {s: len(r) for s, r in groups.items()},
                "changedDocuments": sorted(changed),
            },
        )
        updated.append(pin(certificate))
    for split, evidence in stage["after"].items():
        from document_ocr.atomic import atomic_write_bytes

        ids = [r["documentId"] for r in rows(verify_pin(evidence))]
        atomic_write_bytes(DATA / (split + ".jsonl"), jsonl([after[did] for did in ids]))
    manifest = copy.deepcopy(old_manifest)
    manifest.update(
        status="AUDITED_REAL_VALIDATION_AND_TEMPLATES_COMPLETE_SYNTHETIC_MIGRATION_PENDING",
        batches=updated,
        aggregateRecords={s: pin(DATA / (s + ".jsonl")) for s in stage["after"]},
        independentAudit=pin(review_path),
        templateCatalog=pin(REVISION / "catalog.json"),
        templateControls=pin(REVISION / "controls.json"),
    )
    atomic_write_json(DATA / "manifest.json", manifest)
    report = {
        "status": manifest["status"],
        "changedSources": len(stage["changedDocuments"]),
        "templates": replay["templates"],
        "controls": replay["controls"],
        "sourceCounts": stage["counts"],
        "catalog": pin(REVISION / "catalog.json"),
        "elapsedSeconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    atomic_publish_json(OUT / "publication.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate", "publish"))
    globals()[parser.parse_args().action]()
