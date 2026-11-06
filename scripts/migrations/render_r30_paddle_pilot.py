"""Replay reviewed native templates from pinned migration receipts."""

from __future__ import annotations

import argparse
import json
import resource
import time

from prepare_r30_paddle_pilot import ART, BASE, OUT, ROOT, read, rows

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated_equipment_aliases import EquipmentAliasRegistry
from document_ocr.synthesis.curated_reflow_policy import ReflowPolicy
from document_ocr.synthesis.package_presentations import PackageVocabulary
from document_ocr.synthesis.paddle_reflow import PaddleReflow
from document_ocr.synthesis.paddle_render import render_one, requires_wording


def load_current(selection, current):
    if "freshRecord" not in selection:
        return current[selection["id"]]
    record = read(ROOT / selection["freshRecord"])
    digest = sha256_file(ROOT / selection["path"])
    if record["documentId"] != selection["id"] or not (
        digest == selection["receiptSha256"] == record["receiptSha256"]
    ):
        raise ValueError("fresh shipment record differs from its reviewed receipt")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefixes", nargs="+")
    args = parser.parse_args()
    specification = read(OUT / "paddle-overlays.json")
    overlays = specification["sources"]
    cfg = read(ROOT / "configs/synthesis/mpci_bl_curated_v7_multirow.json")
    pin = specification["equipmentRegistry"]
    if sha256_file(ROOT / pin["path"]) != pin["sha256"]:
        raise ValueError("equipment registry differs from the pilot pin")
    equipment_registry = EquipmentAliasRegistry.from_data(read(ROOT / pin["path"]))
    package_pin = cfg["aliases"]["presentations"]["vocabulary"]
    if sha256_file(ROOT / package_pin["path"]) != package_pin["sha256"]:
        raise ValueError("package vocabulary differs from its pin")
    package_vocabulary = PackageVocabulary(read(ROOT / package_pin["path"]))
    engine = PaddleReflow(ROOT, ReflowPolicy.model_validate(cfg["positions"]["reflow"]))
    current = {
        r["documentId"]: r
        for r in rows(BASE / "synthetic-train.jsonl")
        if r["sourceDocumentId"][4:12] in args.prefixes
    }
    results = []
    for prefix in args.prefixes:
        paths = list((ART / "transfers").glob("doc_" + prefix + "*.json"))
        if len(paths) != 1:
            raise ValueError("nonunique transfer source")
        transfer = read(paths[0])
        record = read(ART / "sources" / paths[0].name)
        for selection in transfer["descendants"]:
            start = time.perf_counter()
            try:
                candidate = read(selection["path"])
                baseline = load_current(selection, current)
                wording_receipt = (
                    read(ART / "new-wording" / (selection["id"] + ".json"))
                    if requires_wording(overlays[prefix])
                    else None
                )
                result = render_one(
                    record,
                    transfer,
                    candidate,
                    baseline,
                    overlays[prefix],
                    engine,
                    selection["id"],
                    equipment_registry,
                    package_vocabulary,
                    wording_receipt=wording_receipt,
                    project_root=ROOT,
                )
                replay = render_one(
                    record,
                    transfer,
                    candidate,
                    baseline,
                    overlays[prefix],
                    engine,
                    selection["id"],
                    equipment_registry,
                    package_vocabulary,
                    wording_receipt=wording_receipt,
                    project_root=ROOT,
                )
                if result != replay:
                    raise ValueError("full deterministic replay differs")
                atomic_write_json(ART / "rendered" / f"{selection['id']}.json", result)
                status = {
                    "status": "RENDERED_NOT_CERTIFIED",
                    "regions": sum(len(p["regions"]) for p in result["pages"]),
                }
            except (ValueError, KeyError) as exc:
                status = {"status": "HELD", "error": str(exc)}
            results.append(
                {
                    "id": selection["id"],
                    "source": prefix,
                    **status,
                    "seconds": time.perf_counter() - start,
                }
            )
    atomic_write_json(
        OUT / "full-render-results.json",
        {
            "results": results,
            "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
