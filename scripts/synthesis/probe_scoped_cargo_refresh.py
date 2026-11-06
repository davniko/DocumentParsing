"""Probe exact cargo-only migration against current R28; never publishes records."""

import json
import time
from collections import Counter
from pathlib import Path

from document_ocr.atomic import atomic_write_json
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7DirectTarget
from document_ocr.synthesis.curated_campaign import create_campaign
from document_ocr.synthesis.curated_layout import PositionPolicy
from document_ocr.synthesis.curated_positions import load_page_geometry, transfer_positions
from document_ocr.synthesis.curated_reflow import ReflowEngine
from document_ocr.synthesis.scoped_cargo_refresh import (
    cargo_auxiliary_keys,
    cargo_surface_keys,
    migrate,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/analysis/r28-combination-refresh300-20261101"
NATIVE = (
    ROOT
    / "artifacts/kie-synthesis-production/r28-targeted-cargo-refresh-probe-20261101/native/campaign"
)


def main():
    start = time.perf_counter()
    config = json.loads(
        next(NATIVE.glob("declared-package-preparation/*/prepared-config.json")).read_text()
    )
    campaign = create_campaign(ROOT, config)
    auxiliary = json.loads((ROOT / config["auxiliary"]).read_text())["sources"]
    scoped_owners = json.loads((OUT / "scoped-owner-reviews.json").read_text())
    dataset = ROOT / config["dataset"]
    alignments = {
        sid: json.loads((dataset / "alignments" / f"{sid}.json").read_text())
        for sid in config["source_ids"]
    }
    boxes, geometry = load_page_geometry(ROOT, dataset, alignments)
    engine = ReflowEngine(ROOT, PositionPolicy.model_validate(config["positions"]).reflow)
    current = {
        r["documentId"]: r
        for line in (ROOT / "data/curated/mpci-bl-r28-positions-v1/synthetic-train.jsonl").open()
        if (r := json.loads(line))["sourceDocumentId"] in config["source_ids"]
        and len(r["target"]["goodsItemDetails"][0]["numberAndTypeOfPackages"]) == 1
    }
    index = json.loads(
        (
            ROOT
            / "docs/analysis/r28-quality-and-variability-20261019/donor/per-record-inventory.json"
        ).read_text()
    )
    results = []
    for sid in config["source_ids"]:
        bp, scenario, sample, planned_target = campaign.plan(sid, 1)
        native = json.loads((NATIVE / "candidates" / f"{sample}.json").read_text())
        physical = campaign.physical_render(bp, scenario, planned_target)
        private = (
            set(physical.surface_values)
            | set(physical.variable_values)
            | {r.key for r in bp.regions if r.kind == "product"}
            | cargo_auxiliary_keys(auxiliary.get(sid, {}).get("bindings", {}))
            | cargo_surface_keys(bp)
            | set(scoped_owners.get(sid, {}).get("keys", []))
        )
        for item in [
            r for r in index if r["sourceDocumentId"] == sid and r["documentId"] in current
        ][:4]:
            before = current[item["documentId"]]
            try:
                old = json.loads((ROOT / item["candidatePath"]).read_text())
                after, evidence = migrate(
                    before, old, native, private_keys=private, regions=bp.regions
                )
                BillOfLadingExtractionV7DirectTarget.model_validate_json(
                    json.dumps(after["target"])
                )
                _, receipt = transfer_positions(
                    bp.source,
                    after["joinedRawText"],
                    evidence["geometryProof"],
                    alignments[sid],
                    page_boxes=boxes[sid],
                    page_dimensions=geometry["pageDimensions"][sid],
                )
                positioned, receipt, _ = engine.apply(
                    source=bp.source,
                    rendered=after["joinedRawText"],
                    proof=evidence["geometryProof"],
                    alignment=alignments[sid],
                    receipt=receipt,
                    boxes=boxes[sid],
                    dimensions=geometry["pageDimensions"][sid],
                    sample_id=before["documentId"],
                )
                rejected = [p for p in receipt["reflowPages"] if p["status"] == "rejected"]
                if rejected:
                    raise ValueError("reflow rejected: " + str(rejected)[:700])
                after["positionedText"] = positioned
                from document_ocr.synthesis.curated import digest

                after["positionedTextSha256"] = digest(positioned.encode())
                atomic_write_json(
                    OUT / "pilot" / f"{before['documentId']}.json",
                    dict(
                        before=before,
                        after=after,
                        evidence=evidence,
                        geometry=receipt,
                        native=str(NATIVE / "candidates" / f"{sample}.json"),
                        historical=item["candidatePath"],
                    ),
                )
                results.append(
                    dict(
                        documentId=before["documentId"], source=sid, status="STAGED_REQUIRES_REVIEW"
                    )
                )
            except (ValueError, KeyError, StopIteration) as e:
                results.append(
                    dict(documentId=before["documentId"], source=sid, status="HELD", reason=str(e))
                )
    report = dict(
        seconds=time.perf_counter() - start,
        results=results,
        counts=dict(Counter(r["status"] for r in results)),
    )
    atomic_write_json(OUT / "pilot-report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
