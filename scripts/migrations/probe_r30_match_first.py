"""Exercise match-first drafts; never publish or certify a template.

Native-span proposals remain subject to role and completeness review. Collect
every selected shipment's failure, so a batch review need not stop at its first
unresolved owner. Explicit probe-polish decisions are separate from publication.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections import Counter
from functools import partial

from audit_r30_paddle_pilot import check_training_target
from jsonschema import ValidationError as SchemaValidationError
from prepare_r30_paddle_pilot import load_pages
from rebase_r30_sources import ART, BASE, OUT, ROOT, batch_records, read, rows
from render_r30_paddle_pilot import load_current

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated_equipment_aliases import EquipmentAliasRegistry
from document_ocr.synthesis.curated_reflow_policy import ReflowPolicy
from document_ocr.synthesis.package_presentations import PackageVocabulary
from document_ocr.synthesis.paddle_reflow import PaddleReflow
from document_ocr.synthesis.paddle_render import render_one, requires_wording
from document_ocr.synthesis.paddle_validation import (
    check_conservation,
    check_new_private_ownership,
    check_private_lexemes,
    check_synthetic_relations,
    check_views,
)


def draft_overlay(packet, manual):
    overlay = {
        "overrides": [],
        "extras": [],
        "targetSets": {},
        "targetCopies": [],
        "targetRemovals": [],
        "physicalGroups": [],
        "watermarkRegions": [],
    }
    for proposal in packet["bindingProposals"]:
        if proposal["status"] == "unique_span_role_review":
            overlay["overrides"].append(
                {
                    "key": proposal["key"],
                    "byteStart": proposal["byteStart"],
                    "spans": [
                        [s["page"], s["region"], s["start"], s["end"]]
                        for s in proposal["choices"][0]
                    ],
                }
            )
    for key, value in manual.items():
        if key == "overrides":
            keys = {(r["key"], r["byteStart"]) for r in value}
            overlay[key] = [
                r for r in overlay[key] if (r["key"], r["byteStart"]) not in keys
            ] + value
        else:
            overlay[key] = value
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    args = parser.parse_args()
    directory = OUT / "batches" / args.batch
    manual_path = directory / "probe-polish.json"
    manual = read(manual_path)
    decisions = read(directory / "main-agent-decisions.json")
    current = {r["documentId"]: r for r in rows(BASE / "synthetic-train.jsonl")}
    config = read(ROOT / "configs/synthesis/mpci_bl_curated_v7_multirow.json")
    engine = PaddleReflow(ROOT, ReflowPolicy.model_validate(config["positions"]["reflow"]))
    equipment = EquipmentAliasRegistry.from_data(
        read(ROOT / "artifacts/sampling-registries/equipment-aliases.json")
    )
    packages = PackageVocabulary(
        read(ROOT / config["aliases"]["presentations"]["vocabulary"]["path"])
    )
    started = time.perf_counter()
    results, overlays, counts = [], {}, Counter()
    for source in batch_records(args.batch):
        if not source["activeTemplate"]:
            continue
        prefix = source["documentId"][4:12]
        packet = read(directory / "review-packets" / (source["documentId"] + ".json"))
        if packet["nativeFingerprint"] != source["nativeFingerprint"]:
            raise ValueError("review packet predates the native source")
        counts.update(p["status"] for p in packet["bindingProposals"])
        overlay = draft_overlay(packet, manual.get(prefix, {}))
        overlays[prefix] = overlay
        transfer = read(ART / "transfers" / (source["documentId"] + ".json"))
        for selected in transfer["descendants"]:
            try:
                candidate = read(ROOT / selected["path"])
                existing = load_current(selected, current)
                wording = (
                    read(ART / "new-wording" / (selected["id"] + ".json"))
                    if requires_wording(overlay)
                    else None
                )

                render = partial(
                    render_one,
                    source,
                    transfer,
                    candidate,
                    existing,
                    overlay,
                    engine,
                    selected["id"],
                    equipment,
                    packages,
                    wording_receipt=wording,
                    project_root=ROOT,
                )

                result = render()
                if result != render():
                    raise ValueError("paired draft replay differs")
                # Check the JSON representation used by the publication auditor;
                # in-memory geometry receipts still contain tuples.
                result = json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))
                check_training_target(result["target"])
                check_views(result, load_pages(result))
                check_conservation(source, result, overlay)
                check_synthetic_relations(result["target"])
                check_private_lexemes(source, result, overlay.get("retainedLexemes", []))
                check_new_private_ownership(source, result, decisions[prefix])
                path = ART / "match-first-probe" / (selected["id"] + ".json")
                atomic_write_json(path, result)
                status, error = "GATES_PASSED_REQUIRES_SEMANTIC_AND_COMPLETENESS_REVIEW", None
                evidence = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
            except (
                ValueError,
                KeyError,
                IndexError,
                FileNotFoundError,
                SchemaValidationError,
            ) as exc:
                status, error, evidence = "HELD", str(exc), None
            results.append(
                {
                    "source": prefix,
                    "id": selected["id"],
                    "status": status,
                    "error": error,
                    "evidence": evidence,
                }
            )
    report = {
        "status": "EFFICIENCY_PROBE_NOT_TEMPLATE_ACCEPTANCE",
        "bindingCounts": dict(counts),
        "results": results,
        "seconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "polishSha256": sha256_file(manual_path),
    }
    atomic_write_json(
        directory / "draft-overlays.json", {"status": report["status"], "sources": overlays}
    )
    atomic_write_json(directory / "match-first-probe.json", report)
    print(
        json.dumps(
            {
                "results": dict(Counter(r["status"] for r in results)),
                **{k: v for k, v in report.items() if k != "results"},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
