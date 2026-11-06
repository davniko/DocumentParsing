"""Replay reviewed private identities into native Paddle, for manual readback.

This stages proposals only. Missing legacy spans and new native-only identities
are explicit review items; a staging result is never a privacy certificate.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict

from apply_r30_adjudications import assign, get
from prepare_r30_paddle_pilot import ART, OUT, ROOT, load_pages, read

from document_ocr.atomic import atomic_write_json
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7DirectTarget
from document_ocr.spatial_inputs.paddle import RegionSpan, fingerprint, serialize
from document_ocr.synthesis.curated_reflow_policy import ReflowPolicy
from document_ocr.synthesis.paddle_bindings import PaddleReplacement, compile_blocks
from document_ocr.synthesis.paddle_reflow import PaddleReflow
from document_ocr.synthesis.paddle_render import spans_for


def target_replay(target, record, explicit):
    """Only exact reviewed leaf identities may be restored automatically."""
    target = copy.deepcopy(target)
    for update in record["privacyPlan"]["targetUpdates"]:
        parts = re.findall(r"[^.\[\]]+", update["path"])
        if parts[0] == "documentPatch":
            parts = parts[1:]
        path = "/" + "/".join(parts)
        try:
            value = get(target, path)
            current = get(record["current"]["target"], path)
        except (KeyError, IndexError):
            continue  # New/removed roles require an explicit reviewed target action.
        if value == update["before"] and current == update["after"]:
            assign(target, path, current)
    for path, value in explicit.items():
        assign(target, path, value)
    return target


def stage(record, transfer, overlay, engine, physical):
    pages = load_pages(record)
    changes, missing = [], []
    overrides = overlay.get("planOverrides", {})
    for edit in transfer["privacyEdits"]:
        ix = str(edit["index"])
        if ix in overlay.get("absentPlanEdits", {}):
            if edit.get("spans") or edit.get("matches"):
                raise ValueError("an absent-plan disposition has a native match")
            continue
        if ix in overrides:
            rule = overrides[ix]
            groups = [spans_for(g, pages) for g in rule["occurrences"]]
            value = rule["after"]
        else:
            groups = [edit["spans"]] if edit.get("spans") else edit.get("matches", [])
            groups = [tuple(RegionSpan(**s) for s in group) for group in groups]
            value = edit["after"]
        if not groups:
            missing.append(edit["index"])
        for n, group in enumerate(groups):
            changes.append(PaddleReplacement(f"privacy:plan:{ix}:{n}", group, value))
    for n, extra in enumerate(overlay.get("extras", [])):
        changes.append(
            PaddleReplacement(f"privacy:new:{n}", spans_for(extra, pages), extra["after"])
        )
    blocks = compile_blocks(
        pages,
        tuple(changes),
        physical_groups=tuple((p, tuple(ix)) for p, ix in physical.get("physicalGroups", [])),
    )
    kwargs = dict(
        identity=record["documentId"],
        reading_order="source",
        watermark_regions=tuple(tuple(ref) for ref in physical.get("watermarkRegions", [])),
    )
    rendered, receipt = engine.render(pages, blocks, **kwargs)
    if engine.render(pages, blocks, **kwargs) != (rendered, receipt):
        raise ValueError("privacy replay differs")
    review = read(ART / "real-target-review" / (record["documentId"] + ".json"))
    target = target_replay(review["target"], record, overlay.get("targetSets", {}))
    BillOfLadingExtractionV7DirectTarget.model_validate_json(json.dumps(target))
    return {
        "status": "PRIVACY_PROPOSAL_REQUIRES_FULL_NATIVE_AND_TARGET_READBACK",
        "documentId": record["documentId"],
        "target": target,
        "missingLegacyPlanEdits": missing,
        "sourceFingerprint": fingerprint(pages),
        "fingerprint": fingerprint(rendered),
        "pages": [asdict(p) for p in rendered],
        "proof": [asdict(c) for c in changes],
        "geometry": receipt,
        "joinedRawText": serialize(rendered, geometry="none"),
        "positionedText": serialize(rendered),
        "bboxText": serialize(rendered, geometry="bbox"),
    }


def main():
    overlays = read(OUT / "real-privacy-overlays.json")["sources"]
    physical = read(OUT / "paddle-overlays.json")["sources"]
    cfg = read(ROOT / "configs/synthesis/mpci_bl_curated_v7_multirow.json")
    engine = PaddleReflow(ROOT, ReflowPolicy.model_validate(cfg["positions"]["reflow"]))
    results = []
    for path in sorted((ART / "transfers").glob("*.json")):
        record, transfer = read(ART / "sources" / path.name), read(path)
        pre = record["documentId"][4:12]
        try:
            result = stage(record, transfer, overlays.get(pre, {}), engine, physical.get(pre, {}))
            atomic_write_json(ART / "privacy-proposals" / path.name, result)
            results.append(
                {
                    "source": pre,
                    "status": result["status"],
                    "changes": len(result["proof"]),
                    "missing": result["missingLegacyPlanEdits"],
                }
            )
        except (ValueError, KeyError) as exc:
            results.append({"source": pre, "status": "HELD", "error": str(exc)})
    atomic_write_json(OUT / "privacy-staging-check.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
