"""Assemble the complete reviewed source baseline and native template handoff.

Preparation exports pinned authorities, not training acceptance. Publication
requires an exact full-catalog replay and retains all batch/amendment evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import resource
import time

from audit_r30_paddle_pilot import check_training_target
from publish_r30_sources import (
    DATA,
    TEMPLATES,
    active_source_certificates,
    active_template_entries,
    jsonl,
    pin,
    update_manifest,
    verify_pin,
)
from rebase_r30_sources import OUT, PILOT_ART, ROOT, read, rows

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.spatial_inputs.paddle import pages_from_record
from document_ocr.synthesis.paddle_validation import check_views


def prepare():
    config_path = ROOT / "configs/synthesis/mpci_bl_curated_v7_multirow.json"
    cfg = read(config_path)
    decisions = {}
    for path in sorted((TEMPLATES / "batches").glob("*.json")):
        cert = read(path)
        source_cert = read(verify_pin(cert["sourceCertificate"]))
        review = read(verify_pin(source_cert["decisions"]))
        for entry in cert["templates"]:
            did = read(verify_pin(entry["template"]))["sourceDocumentId"]
            decisions[did] = review[did[4:12]]
    entries, controls = {}, {}
    numeric = copy.deepcopy(read(verify_pin(cfg["independent"]["numeric_certificates"])))
    numeric_changes = {}
    for did, accepted in active_template_entries().items():
        template = read(verify_pin(accepted["template"]))
        entries[did] = {
            "template": accepted["template"],
            "samplingSource": pin(ROOT / cfg["source_contracts"] / did / "source.txt"),
            "privacy": {"kind": "reviewed_native_owners", "decision": decisions[did]},
        }
        controls[did] = accepted["controls"]
        if "numericCertificate" in template["overlay"]:
            evidence = template["overlay"]["numericCertificate"]
            replacement = read(verify_pin(evidence))["sources"][did]
            if did not in numeric["sources"]:
                raise ValueError("native numeric extension must amend an admitted source")
            numeric["sources"][did] = replacement
            numeric_changes[did] = evidence
    for path in sorted((PILOT_ART / "certified/templates").glob("*.json")):
        template = read(path)
        did = template["sourceDocumentId"]
        if did in entries:
            raise ValueError("duplicate pilot/native template")
        entries[did] = {
            "template": pin(TEMPLATES / "cases" / path.name),
            "samplingSource": pin(ROOT / cfg["source_contracts"] / did / "source.txt"),
            "privacy": {
                "kind": "reviewed_replacement_spans",
                "reference": pin(PILOT_ART / "privacy-proposals" / path.name),
            },
        }
        controls[did] = [
            pin(PILOT_ART / "rendered" / (identity + ".json"))
            for identity in template["testDescendants"]
        ]
    expected = {r["id"] for r in read(OUT / "plan.json")["sources"] if r["activeTemplate"]}
    if entries.keys() != expected or any(len(v) != 3 for v in controls.values()):
        raise ValueError("native catalog must contain exact membership and three controls each")
    numeric_path = TEMPLATES / "numeric-certificates.json"
    atomic_publish_json(numeric_path, numeric)
    cfg["independent"]["numeric_certificates"] = pin(numeric_path)
    cfg["source_ids"] = sorted(entries)
    cfg["output"] = "artifacts/synthesis-candidates/r30-native"
    config_path = ROOT / "configs/synthesis/mpci_bl_curated_v7_paddle.json"
    atomic_publish_json(config_path, cfg)
    manifest = {
        "format": "native-paddle-templates-v1",
        "templateCount": len(entries),
        "schema": pin(DATA / "schema.json"),
        "equipment": pin(ROOT / "artifacts/sampling-registries/equipment-aliases.json"),
        "packages": cfg["aliases"]["presentations"]["vocabulary"],
        "reflow": cfg["positions"]["reflow"],
        "templates": entries,
        "samplingConfig": pin(config_path),
        "nativeNumericExtensions": numeric_changes,
    }
    atomic_publish_json(TEMPLATES / "catalog.json", manifest)
    atomic_publish_json(OUT / "handoff-controls.json", controls)
    print(json.dumps({"catalog": pin(TEMPLATES / "catalog.json"), "templates": len(entries)}))


def publish():
    started = time.perf_counter()
    plan = read(OUT / "plan.json")
    # One explicitly reviewed policy amendment; immutable historic pins remain.
    policy = "docs/kie-real-baseline-label-policy-2026-10-05.md"
    observed = {p: sha256_file(ROOT / p) for p in plan["protectedPins"]}
    changed = {p: sha for p, sha in observed.items() if sha != plan["protectedPins"][p]}
    if set(changed) != {policy}:
        raise ValueError(f"unexpected protected-input changes: {list(changed)}")
    for source in plan["sources"]:
        verify_pin(source)
    replay_path = OUT / "handoff-catalog-validation.json"
    replay = read(replay_path)
    if replay["catalog"] != pin(TEMPLATES / "catalog.json") or replay["failures"]:
        raise ValueError("catalog must pass its current complete replay before publication")
    expected_templates = sum(r["activeTemplate"] for r in plan["sources"])
    if replay["templates"] != expected_templates or replay["controls"] != 3 * expected_templates:
        raise ValueError("catalog replay is incomplete")
    records = {"real-train": [], "validation": []}
    certificates = active_source_certificates()
    for path in certificates.values():
        cert = read(path)
        for split, evidence in cert["records"].items():
            for record in rows(verify_pin(evidence)):
                verify_pin(record["evidence"])
                check_views(record, pages_from_record(record))
                check_training_target(record["target"])
                records[split].append(record)
    for split, members in records.items():
        expected = {r["id"] for r in plan["sources"] if r["split"] == split}
        ids = [r["documentId"] for r in members]
        if set(ids) != expected or len(ids) != len(set(ids)):
            raise ValueError("aggregate source membership differs")
    for split, members in records.items():
        atomic_publish_bytes(DATA / (split + ".jsonl"), jsonl(members))
    update_manifest()
    receipt = {
        "status": "COMPLETE_REAL_VALIDATION_AND_TEMPLATE_BASELINE",
        "sourcePlan": pin(OUT / "plan.json"),
        "records": {s: pin(DATA / (s + ".jsonl")) for s in records},
        "counts": {s: len(v) for s, v in records.items()},
        "sourceCertificates": [pin(p) for p in certificates.values()],
        "catalog": pin(TEMPLATES / "catalog.json"),
        "catalogValidation": pin(replay_path),
        "protectedInputsChecked": len(plan["protectedPins"]),
        "authorizedPolicyChange": {
            "before": plan["protectedPins"][policy],
            "after": changed[policy],
            "review": pin(OUT / "KOS-REVIEW.md"),
        },
        "syntheticMigration": "9400 records remain; no R29 synthetic row is a Paddle row",
        "elapsedSeconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    atomic_write_json(OUT / "completion.json", receipt)
    manifest = read(DATA / "manifest.json")
    manifest.update(
        status="COMPLETE_SOURCE_TEMPLATE_BASELINE_SYNTHETIC_MIGRATION_PENDING",
        aggregateRecords=receipt["records"],
        completion=pin(OUT / "completion.json"),
    )
    atomic_write_json(DATA / "manifest.json", manifest)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "publish"))
    action = parser.parse_args().action
    prepare() if action == "prepare" else publish()
