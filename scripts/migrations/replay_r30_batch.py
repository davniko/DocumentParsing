"""Replay explicitly reviewed full-source overlays; do not publish candidates."""

from __future__ import annotations

import argparse
import json
import time

from audit_r30_paddle_pilot import check_training_target
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


def replay(batch, prefixes):
    started = time.perf_counter()
    cfg = read(ROOT / "configs/synthesis/mpci_bl_curated_v7_multirow.json")
    vocabulary = cfg["aliases"]["presentations"]["vocabulary"]
    if sha256_file(ROOT / vocabulary["path"]) != vocabulary["sha256"]:
        raise ValueError("package vocabulary differs from the configured pin")
    packages = PackageVocabulary(read(ROOT / vocabulary["path"]))
    equipment_pin = read(OUT / "plan.json")["protectedPins"]
    equipment_path = ROOT / "artifacts/sampling-registries/equipment-aliases.json"
    # The completed pilot pins the same equipment registry used by the current replay.
    pin = read(ROOT / "docs/analysis/r30-paddle-pilot-20261102/paddle-overlays.json")[
        "equipmentRegistry"
    ]
    if (
        str(equipment_path.relative_to(ROOT)) in equipment_pin
        and sha256_file(equipment_path) != equipment_pin[str(equipment_path.relative_to(ROOT))]
    ):
        raise ValueError("pinned equipment registry changed")
    if sha256_file(equipment_path) != pin["sha256"]:
        raise ValueError("pilot equipment registry changed")
    equipment = EquipmentAliasRegistry.from_data(read(equipment_path))
    engine = PaddleReflow(ROOT, ReflowPolicy.model_validate(cfg["positions"]["reflow"]))
    overlay_path = OUT / "batches" / batch / "paddle-overlays.json"
    overlays = read(overlay_path)["sources"]
    decisions = read(OUT / "batches" / batch / "main-agent-decisions.json")
    sources = {r["documentId"][4:12]: r for r in batch_records(batch) if r["activeTemplate"]}
    if set(prefixes) - sources.keys() or set(prefixes) - overlays.keys():
        raise ValueError("every requested member needs an explicit reviewed overlay")
    current = {
        r["documentId"]: r
        for r in rows(BASE / "synthetic-train.jsonl")
        if r["sourceDocumentId"][4:12] in prefixes
    }
    results = []
    for prefix in prefixes:
        source, overlay = sources[prefix], overlays[prefix]
        transfer_path = ART / "transfers" / (source["documentId"] + ".json")
        transfer = read(transfer_path)
        if transfer["sourceFingerprint"] != source["nativeFingerprint"]:
            raise ValueError("binding inventory belongs to a different native input")
        if len(transfer["descendants"]) != 3:
            raise ValueError("template needs three explicit test shipments")
        for selection in transfer["descendants"]:
            identity = selection["id"]
            print(json.dumps({"source": prefix, "replaying": identity}), flush=True)
            candidate = read(ROOT / selection["path"])
            args = (
                source,
                transfer,
                candidate,
                load_current(selection, current),
                overlay,
                engine,
                identity,
                equipment,
                packages,
            )
            wording_path = ART / "new-wording" / (identity + ".json")
            wording_receipt = read(wording_path) if requires_wording(overlay) else None
            rendered = render_one(*args, wording_receipt=wording_receipt, project_root=ROOT)
            if rendered != render_one(*args, wording_receipt=wording_receipt, project_root=ROOT):
                raise ValueError("paired full replay differs")
            # Audit the actual JSON artifact representation, as the pilot does
            # after disk readback; geometry receipts contain tuples in memory.
            rendered = json.loads(json.dumps(rendered, ensure_ascii=False, allow_nan=False))
            check_training_target(rendered["target"])
            check_views(rendered, load_pages(rendered))
            check_conservation(source, rendered, overlay)
            check_synthetic_relations(rendered["target"])
            check_private_lexemes(source, rendered, overlay.get("retainedLexemes", []))
            check_new_private_ownership(source, rendered, decisions[prefix])
            rendered["migrationPins"] = {
                "overlay": sha256_file(overlay_path),
                "transfer": sha256_file(transfer_path),
                "shipment": sha256_file(ROOT / selection["path"]),
            }
            if wording_receipt is not None:
                rendered["migrationPins"]["wording"] = sha256_file(wording_path)
            atomic_write_json(ART / "rendered" / (identity + ".json"), rendered)
            results.append(
                {
                    "id": identity,
                    "source": source["documentId"],
                    "status": "FULL_REPLAY_PASSED_REQUIRES_MAIN_AGENT_READBACK",
                }
            )
    receipt = {
        "status": "REPLAY_ONLY_NOT_TEMPLATE_CERTIFICATION",
        "records": results,
        "elapsedSeconds": time.perf_counter() - started,
    }
    atomic_write_json(OUT / "batches" / batch / "replay-results.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--prefixes", nargs="+", required=True)
    args = parser.parse_args()
    replay(args.batch, args.prefixes)
