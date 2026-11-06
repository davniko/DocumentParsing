"""Stage reviewed native cargo into current R28 records, never publish it."""

import argparse
import fcntl
import json
import re
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from stage_combination_refresh import OUT, ROOT

from document_ocr.atomic import atomic_write_json
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7DirectTarget
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_campaign import create_campaign
from document_ocr.synthesis.curated_layout import PositionPolicy
from document_ocr.synthesis.curated_positions import load_page_geometry, transfer_positions
from document_ocr.synthesis.curated_reflow import ReflowEngine
from document_ocr.synthesis.scoped_cargo_refresh import (
    cargo_auxiliary_keys,
    cargo_surface_keys,
    migrate,
)


def position_preservation(before, after):
    old, new = before["joinedRawText"].splitlines(), after["joinedRawText"].splitlines()
    a, b = before["positionedText"].splitlines(), after["positionedText"].splitlines()
    if len(a) != len(old) or len(b) != len(new):
        raise ValueError("positioned/raw line count differs")
    missing = []
    checked = 0
    for block in SequenceMatcher(None, old, new, autojunk=False).get_matching_blocks():
        for i in range(block.size):
            x, y = a[block.a + i], b[block.b + i]
            if re.search(r" \|\| \d+,\d+$", x):
                checked += 1
                if not re.search(r" \|\| \d+,\d+$", y):
                    missing.append(old[block.a + i])
    if missing:
        raise ValueError("unchanged lines lost coordinates: " + repr(missing[:3]))
    return checked


def validate_literals(after):
    text = after["joinedRawText"].upper()
    for g in after["target"]["goodsItemDetails"]:
        if " ".join(g["description"].upper().split()) not in " ".join(text.split()):
            raise ValueError("description target not literal in current-scoped text")
        for code in g.get("hsCodes", []):
            if not re.search(r"(?<![\d.])" + re.escape(code.upper()) + r"(?![\d.])", text):
                raise ValueError("complete HS target absent or stale suffix remains")


def stage(config_path, refresh_native=()):
    start = time.perf_counter()
    directory = config_path.parent
    summary_paths = list((directory / "stage-runs").glob("*/summary.json"))
    if not summary_paths:
        raise ValueError("native pipeline has not finished")
    summary_path = directory / "corrected-stage-summary.json"
    summary = json.loads(
        (summary_path if summary_path.exists() else max(summary_paths)).read_text()
    )
    cfg = json.loads((ROOT / summary["preparedConfig"]).read_text())
    campaign = create_campaign(ROOT, cfg)
    pool = json.loads((OUT / "eligible-old-records.json").read_text())
    current = {
        r["documentId"]: r
        for line in (ROOT / "data/curated/mpci-bl-r28-positions-v1/synthetic-train.jsonl").open()
        if (r := json.loads(line))["sourceDocumentId"] in cfg["source_ids"]
    }
    used = {
        json.loads(p.read_text())["after"]["documentId"]
        for p in (OUT / "staged-cargo").glob("*.json")
    }
    staged_by_native = {
        json.loads(p.read_text())["evidence"]["nativeCandidateId"]: p
        for p in (OUT / "staged-cargo").glob("*.json")
    }
    staged_native = set(staged_by_native)
    auxiliary = json.loads((ROOT / cfg["auxiliary"]).read_text())["sources"]
    decisions_path = OUT / "reviewed-native-decisions.json"
    approvals = json.loads(decisions_path.read_text())["approve"] if decisions_path.exists() else {}
    scoped_owners = json.loads((OUT / "scoped-owner-reviews.json").read_text())
    dataset = ROOT / cfg["dataset"]
    alignments = {
        sid: json.loads((dataset / "alignments" / f"{sid}.json").read_text())
        for sid in cfg["source_ids"]
    }
    boxes, geo = load_page_geometry(ROOT, dataset, alignments)
    engine = ReflowEngine(ROOT, PositionPolicy.model_validate(cfg["positions"]).reflow)
    rows = []
    for geom in summary["stages"]["geometry"]["rows"]:
        sid = geom["sourceId"]
        if geom["status"] != "PASS":
            rows.append(geom)
            continue
        nid = geom["sampleId"]
        if nid in staged_native and nid not in refresh_native:
            continue
        review_path = campaign.output / "reviews" / f"{sid}.json"
        if not review_path.exists():
            rows.append(dict(native=nid, status="HELD", reason="native review missing"))
            continue
        findings = json.loads(review_path.read_text())["review"]["findings"]
        relevant = [f for f in findings if f["sample_id"] == nid]
        if relevant and nid not in approvals:
            rows.append(
                dict(native=nid, status="HELD", reason="native review findings", findings=relevant)
            )
            continue
        native = json.loads((ROOT / geom["candidate"]).read_text())
        campaign.replay_candidate(native)
        bp, scenario, _, planned_target = campaign.plan(sid, geom["variant"])
        physical = campaign.physical_render(bp, scenario, planned_target)
        # Consume the complete production physical plan, including equipment
        # tares and loaded-mass sums, not only its cargo-measure subset.
        private = (
            set(physical.surface_values)
            | set(physical.variable_values)
            | {r.key for r in bp.regions if r.kind == "product"}
            | cargo_auxiliary_keys(auxiliary.get(sid, {}).get("bindings", {}))
            | cargo_surface_keys(bp)
        )
        private.update(scoped_owners.get(sid, {}).get("keys", []))
        options = pool.get(sid, [])
        if nid in staged_native:
            previous = json.loads(staged_by_native[nid].read_text())
            original_id = previous["after"]["documentId"]
            used.remove(original_id)
            options = [i for i in options if i["documentId"] == original_id]
            atomic_write_json(
                OUT / "staged-cargo-history" / f"{original_id}-{digest(previous)}.json", previous
            )
        attempts = []
        for item in options:
            did = item["documentId"]
            if did in used:
                continue
            before = current[did]
            try:
                old = json.loads((ROOT / item["candidatePath"]).read_text())
                after, evidence = migrate(
                    before, old, native, private_keys=private, regions=bp.regions
                )
                BillOfLadingExtractionV7DirectTarget.model_validate_json(
                    json.dumps(after["target"])
                )
                validate_literals(after)
                _, receipt = transfer_positions(
                    bp.source,
                    after["joinedRawText"],
                    evidence["geometryProof"],
                    alignments[sid],
                    page_boxes=boxes[sid],
                    page_dimensions=geo["pageDimensions"][sid],
                )
                positioned, receipt, _ = engine.apply(
                    source=bp.source,
                    rendered=after["joinedRawText"],
                    proof=evidence["geometryProof"],
                    alignment=alignments[sid],
                    receipt=receipt,
                    boxes=boxes[sid],
                    dimensions=geo["pageDimensions"][sid],
                    sample_id=did,
                )
                if any(p["status"] == "rejected" for p in receipt["reflowPages"]):
                    raise ValueError("hybrid geometry rejected")
                after.update(
                    positionedText=positioned, positionedTextSha256=digest(positioned.encode())
                )
                checked = position_preservation(before, after)
                evidence.update(
                    unchangedPositionedLinesChecked=checked,
                    nativePath=geom["candidate"],
                    historicalPath=item["candidatePath"],
                    preparedConfig=summary["preparedConfig"],
                    stratum=directory.name,
                )
                evidence.update(
                    nativeSha256=digest(native),
                    privateOwnerKeys=sorted(private),
                    physicalPlanOwnerKeys=sorted(
                        set(physical.surface_values) | set(physical.variable_values)
                    ),
                )
                if relevant:
                    evidence["nativeReviewAdjudication"] = dict(
                        reason=approvals[nid], candidateSha256=digest(native), findings=relevant
                    )
                atomic_write_json(
                    OUT / "staged-cargo" / f"{did}.json",
                    dict(
                        before=before,
                        after=after,
                        evidence=evidence,
                        geometry=receipt,
                        reviewStatus="REQUIRES_HYBRID_TEXT_REVIEW",
                    ),
                )
                used.add(did)
                rows.append(dict(native=nid, documentId=did, status="STAGED", attempts=attempts))
                break
            except (ValueError, KeyError, AssertionError) as e:
                attempts.append(dict(documentId=did, reason=str(e)))
        else:
            rows.append(dict(native=nid, status="HELD", attempts=attempts))
    result = dict(
        config=str(config_path),
        seconds=time.perf_counter() - start,
        counts=dict(Counter(r["status"] for r in rows)),
        rows=rows,
    )
    atomic_write_json(directory / "scoped-staging.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--refresh-native", action="append", default=[])
    a = p.parse_args()
    with (OUT / ".staging.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stage(a.config.resolve(), a.refresh_native)
