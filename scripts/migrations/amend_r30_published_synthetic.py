"""Replay catalog corrections against already published rows before explicit readback."""

import argparse
import copy
import json

from audit_r30_independent import independent_view
from audit_r30_paddle_pilot import check_training_target
from migrate_r30_synthetic import DATA, OUT, ROOT, current_for_render, read
from publish_r30_sources import pin, verify_pin
from r30_literal_target_corrections import apply_literal_target_correction
from validate_r30_migration_catalog import semantic_changes

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog


def amendment_wording(evidence, identity, additions):
    """Add reviewed owners without replacing previously accepted linguistic values."""
    addition = additions.get(identity)
    if addition is None:
        return evidence["wording"]
    previous_pin = evidence["wording"]
    if previous_pin is not None and "wordingExtension" not in addition:
        raise ValueError("new-owner amendment cannot overwrite existing wording")
    wording_pin = addition.get("receipt", addition)
    wording = read(verify_pin(wording_pin))
    if wording["status"] != "REVIEWED" or wording["documentId"] != identity:
        raise ValueError("new-owner wording is not reviewed for this row")
    if wording["candidateSha256"] != evidence["candidate"]["sha256"]:
        raise ValueError("new-owner wording belongs to another shipment")
    if previous_pin is not None:
        if addition["wordingExtension"] != previous_pin:
            raise ValueError("wording extension has a different prior authority")
        previous = read(verify_pin(previous_pin))
        if any(previous[k] != wording[k] for k in (
            "status", "documentId", "candidateSha256", "sourceFingerprint"
        )):
            raise ValueError("wording extension changes shipment or source authority")
        old_prompt, new_prompt = json.loads(previous["prompt"]), json.loads(wording["prompt"])
        old_requested = old_prompt.pop("requested")
        new_requested = new_prompt.pop("requested")
        if (
            not old_prompt.keys() <= new_prompt.keys()
            or any(new_prompt[k] != v for k, v in old_prompt.items())
            or not old_requested.keys() < new_requested.keys()
            or any(new_requested[k] != v for k, v in old_requested.items())
            or previous["values"].keys() != old_requested.keys()
            or wording["values"].keys() != new_requested.keys()
            or any(wording["values"][k] != v for k, v in previous["values"].items())
        ):
            raise ValueError("wording extension must preserve prior requests, values and context")
    elif "wordingExtension" in addition:
        raise ValueError("wording extension has no prior linguistic owners")
    return wording_pin


def stage(replay_path, wording_review_path=None):
    replay = read(replay_path)
    if replay["failures"]:
        raise ValueError("catalog controls failed")
    scopes = read(verify_pin(replay["review"]))["sources"]
    catalog = PaddleTemplateCatalog(ROOT, replay["catalog"])
    additions = {}
    if wording_review_path:
        wording_review = read(wording_review_path)
        if (
            wording_review["status"] != "MAIN_AGENT_REVIEWED_NEW_LINGUISTIC_OWNERS"
            or wording_review["catalog"] != replay["catalog"]
        ):
            raise ValueError("new linguistic owners have no matching reviewed authority")
        additions = wording_review["wording"]
    manifest = read(DATA / "manifest.json")
    inventory = read(OUT / "receipt-inventory.json")
    baseline = {
        r["documentId"]: r for r in map(json.loads, verify_pin(inventory["baseline"]).open())
    }
    output = verify_pin(replay["catalog"]).parent / "published-amendments"
    changes = []
    for cert_pin in manifest["syntheticBatches"]:
        cert = read(verify_pin(cert_pin))
        for row in map(json.loads, verify_pin(cert["records"]).open()):
            prefix = row["sourceDocumentId"][4:12]
            if prefix not in scopes:
                continue
            evidence = row["evidence"]
            review = read(verify_pin(evidence["review"]))
            staged = read(verify_pin(review["staging"]))
            entry = next(e for e in staged["results"] if e["id"] == row["documentId"])
            current = current_for_render(
                baseline[row["documentId"]],
                read(verify_pin(entry["reconciliation"])) if entry.get("reconciliation") else None,
            )
            wording_pin = amendment_wording(evidence, row["documentId"], additions)
            rendered = catalog.render(
                verify_pin(evidence["candidate"]),
                current=current,
                wording_receipt=read(verify_pin(wording_pin)) if wording_pin else None,
            )
            rendered = apply_literal_target_correction(
                rendered, evidence.get("literalTargetCorrection")
            )
            delta = list(semantic_changes(row["target"], rendered["target"]))
            scope = scopes[prefix]
            if any(not any(p.startswith(a) for a in scope["targetPaths"]) for p, _, _ in delta):
                raise ValueError(
                    f"published amendment exceeds reviewed target scope for {row['documentId']}: {delta}"
                )
            if not scope["inputChanges"] and row["pages"] != rendered["pages"]:
                raise ValueError("target-only correction changed published OCR")
            if check_training_target(rendered["target"]):
                raise ValueError("published amendment fails schema")
            for kind, key in (
                ("raw", "joinedRawText"),
                ("centroid", "positionedText"),
                ("bbox", "bboxText"),
            ):
                if independent_view(rendered["pages"], kind) != rendered[key]:
                    raise ValueError("published amendment serialization mismatch")
            after = copy.deepcopy(row)
            after["evidence"]["wording"] = wording_pin
            for key in ("pages", "target", "joinedRawText", "positionedText", "bboxText"):
                after[key] = rendered[key]
            path = output / (row["documentId"] + ".json")
            atomic_publish_json(path, after)
            rendered_path = output / (row["documentId"] + "-rendered.json")
            atomic_publish_json(rendered_path, rendered)
            text_delta = []
            for p, (a, b) in enumerate(zip(row["pages"], rendered["pages"], strict=True)):
                at, bt = [r["text"] for r in a["regions"]], [r["text"] for r in b["regions"]]
                if at != bt:
                    import difflib

                    text_delta.append({"page": p, "diff": list(difflib.unified_diff(at, bt))})
            changes.append(
                {
                    "id": row["documentId"],
                    "beforeRecordSha256": digest(row),
                    "certificate": cert_pin,
                    "after": pin(path),
                    "rendered": pin(rendered_path),
                    "targetChanges": delta,
                    "textChanges": text_delta,
                }
            )
    result = {
        "status": "REQUIRES_MAIN_AGENT_AMENDMENT_READBACK",
        "replay": pin(replay_path),
        "catalog": replay["catalog"],
        "previousCatalog": manifest["templateCatalog"],
        "newWordingReview": pin(wording_review_path) if wording_review_path else None,
        "changes": changes,
    }
    path = output / ("report-" + digest(result) + ".json")
    atomic_publish_json(path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay")
    parser.add_argument("--wording-review")
    args = parser.parse_args()
    stage(ROOT / args.replay, ROOT / args.wording_review if args.wording_review else None)
