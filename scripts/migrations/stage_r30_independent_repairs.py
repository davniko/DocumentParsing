"""Stage explicit independent-audit repairs without publishing or accepting model proposals.

All unchanged source data remains byte-equivalent at the record-value level.
Privacy changes use the same measured native renderer and conservation checks
as the rebase; geometry repairs replay the existing privacy proof, never labels
or names reconstructed from the private original.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import resource
import time
from collections import Counter
from dataclasses import asdict

from apply_r30_adjudications import assign, get, parts
from audit_r30_paddle_pilot import check_training_target, leaf_changes
from publish_r30_sources import DATA, jsonl, pin, verify_pin
from rebase_r30_sources import ART as REBASE_ART
from rebase_r30_sources import ROOT, read, rows

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.spatial_inputs.paddle import RegionSpan, fingerprint, pages_from_record, serialize
from document_ocr.synthesis.curated_reflow_policy import ReflowPolicy
from document_ocr.synthesis.paddle_bindings import PaddleReplacement, compile_blocks
from document_ocr.synthesis.paddle_reflow import PaddleReflow
from document_ocr.synthesis.paddle_render import spans_for
from document_ocr.synthesis.paddle_validation import check_conservation, check_views

OUT = ROOT / "docs/analysis/r30-independent-audit-20261103"
ART = ROOT / "artifacts/paddle-migration/r30-independent-audit-20261103"
REVIEW_FILES = (
    "source-corrections.json",
    "source-corrections-supplement.json",
    "source-corrections-final.json",
    "source-corrections-template-crosscheck.json",
)
VIEW_KEYS = ("pages", "joinedRawText", "positionedText", "bboxText")


def merged_reviews(paths):
    result = {}
    for path in paths:
        review = read(path)
        if review["status"] != "REVIEWED_CORRECTIONS_AWAITING_STAGING_AND_TEMPLATE_REPLAY":
            raise ValueError("requires explicit reviewed corrections, not model proposals")
        for prefix, decision in review["sources"].items():
            if prefix in result:
                raise ValueError(f"duplicate review identity: {prefix}")
            allowed = {
                "set",
                "remove",
                "privacy",
                "geometryReview",
                "orderContainersByFirstOccurrence",
            }
            if not decision or decision.keys() - allowed:
                raise ValueError(f"unknown or empty review operation: {prefix}")
            result[prefix] = decision
    return result


def amended_target(target, review, pages):
    result = copy.deepcopy(target)
    pointers = list(review.get("set", {})) + review.get("remove", [])
    if len(pointers) != len(set(pointers)) or any(
        b.startswith(a + "/") for a in pointers for b in pointers if a != b
    ):
        raise ValueError("review target paths overlap")
    for pointer, value in review.get("set", {}).items():
        try:
            old = get(result, pointer)
        except KeyError:
            pass  # Adding a reviewed field is explicit; its parents must exist.
        else:
            if old == value:
                raise ValueError(f"reviewed set changes nothing: {pointer}")
        assign(result, pointer, value)
    for pointer in review.get("remove", []):
        keys = parts(pointer)
        parent = result
        for key in keys[:-1]:
            parent = parent[int(key)] if isinstance(parent, list) else parent[key]
        if isinstance(parent, list):
            raise ValueError("replace a whole reviewed array; removal must not shift its indices")
        del parent[keys[-1]]
    # Optional contacts emptied by explicit removals have no remaining fact.
    for pointer in review.get("remove", []):
        marker = "/contactDetails/"
        if marker in pointer:
            contact_path = pointer.split(marker)[0] + "/contactDetails"
            try:
                contact = get(result, contact_path)
            except KeyError:
                continue  # A prior reviewed removal already emptied this object.
            if contact == {}:
                del get(result, contact_path.rsplit("/", 1)[0])["contactDetails"]
    if review.get("orderContainersByFirstOccurrence"):
        containers = result["containerInformation"]
        text = re.sub(r"[^A-Z0-9]", "", serialize(pages, geometry="none").upper())
        identifiers = [c["equipmentIdentifier"] for c in containers]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("cannot order duplicate container identities")
        positions = {}
        for identity in identifiers:
            token = re.sub(r"[^A-Z0-9]", "", identity.upper())
            if not token or token not in text:
                raise ValueError("cannot order an unprinted container identity")
            positions[identity] = text.index(token)
        ordered = sorted(containers, key=lambda c: positions[c["equipmentIdentifier"]])
        if ordered == containers:
            raise ValueError("container order is already native")
        result["containerInformation"] = ordered
    return result


def render_proof(source, proof, physical, engine, identity):
    pages = pages_from_record(source)
    blocks = compile_blocks(
        pages,
        proof,
        physical_groups=tuple((p, tuple(indices)) for p, indices in physical["physicalGroups"]),
    )
    kwargs = {
        "identity": identity,
        "reading_order": "source",
        "watermark_regions": tuple(tuple(r) for r in physical["watermarkRegions"]),
    }
    rendered, geometry = engine.render(pages, blocks, **kwargs)
    if (rendered, geometry) != engine.render(pages, blocks, **kwargs):
        raise ValueError("repair replay is not deterministic")
    result = {
        "sourceFingerprint": fingerprint(pages),
        "pages": [asdict(p) for p in rendered],
        "proof": [asdict(p) for p in proof],
        "geometry": geometry,
        "joinedRawText": serialize(rendered, geometry="none"),
        "positionedText": serialize(rendered),
        "bboxText": serialize(rendered, geometry="bbox"),
    }
    result = json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))
    check_conservation(source, result, physical)
    check_views(result, rendered)
    return result


def repair_record(record, review, engine):
    result = copy.deepcopy(record)
    evidence = {"previousEvidence": record["evidence"], "review": review}
    if review.get("privacy") and review.get("geometryReview"):
        raise ValueError("privacy and geometry corrections require separate explicit replay stages")
    if review.get("privacy"):
        pages = pages_from_record(record)
        changes = tuple(
            PaddleReplacement(f"independent-privacy:{n}", spans_for(rule, pages), rule["after"])
            for n, rule in enumerate(review["privacy"])
        )
        if any("".join(s.expected for s in c.spans) == c.text for c in changes):
            raise ValueError("privacy replacement changes nothing")
        replay = render_proof(
            record,
            changes,
            {"physicalGroups": [], "watermarkRegions": []},
            engine,
            record["documentId"],
        )
        result.update({k: replay[k] for k in VIEW_KEYS})
        evidence["privacyReplay"] = replay
    if "geometryReview" in review:
        correction = review["geometryReview"]
        source_path = REBASE_ART / "sources" / (record["documentId"] + ".json")
        source = read(source_path)
        old_stage = read(verify_pin(record["evidence"]))
        decision_path = ROOT / correction["path"]
        decision = read(decision_path)[record["documentId"][4:12]]
        physical = copy.deepcopy(decision["physical"])
        if physical["physicalGroups"] != correction["beforePhysicalGroups"]:
            raise ValueError("geometry correction does not match reviewed physical groups")
        proof = tuple(
            PaddleReplacement(p["owner"], tuple(RegionSpan(**s) for s in p["spans"]), p["text"])
            for p in old_stage["proof"]
        )
        old_replay = render_proof(source, proof, physical, engine, record["documentId"])
        if any(old_replay[k] != record[k] for k in VIEW_KEYS):
            raise ValueError("existing privacy proof no longer reconstructs current input")
        physical["physicalGroups"] = correction["afterPhysicalGroups"]
        replay = render_proof(source, proof, physical, engine, record["documentId"])

        def tokens(r):
            return Counter(r["joinedRawText"].split())

        if tokens(replay) != tokens(record):
            raise ValueError("geometry-only repair changed printed tokens")
        witness = correction["preservedColumnWitness"]
        matches = [
            r for r in replay["pages"][witness["page"]]["regions"] if r["text"] == witness["text"]
        ]
        if len(matches) != 1 or matches[0]["box"][0] < witness["minimumLeft"]:
            raise ValueError("reviewed column ownership was not restored")
        result.update({k: replay[k] for k in VIEW_KEYS})
        evidence.update(
            geometryReplay=replay, source=pin(source_path), physicalReview=pin(decision_path)
        )
    result["target"] = amended_target(record["target"], review, pages_from_record(result))
    check_training_target(result["target"])
    check_views(result, pages_from_record(result))
    unchanged = set(record) - {*VIEW_KEYS, "target"}
    if any(result[k] != record[k] for k in unchanged):
        raise ValueError("repair changed unrelated record metadata")
    if (
        not review.get("privacy")
        and "geometryReview" not in review
        and any(result[k] != record[k] for k in VIEW_KEYS)
    ):
        raise ValueError("target-only repair changed input or geometry")
    if result == record:
        raise ValueError("reviewed record contains no actual change")
    evidence["targetChanges"] = leaf_changes(record["target"], result["target"])
    return result, evidence


def stage(revision):
    if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", revision) is None:
        raise ValueError("stage revision must be an explicit simple identifier")
    started = time.perf_counter()
    paths = [OUT / name for name in REVIEW_FILES]
    reviews = merged_reviews(paths)
    cfg = read(ROOT / "configs/synthesis/mpci_bl_curated_v7_paddle.json")
    engine = PaddleReflow(ROOT, ReflowPolicy.model_validate(cfg["positions"]["reflow"]))
    source_pins = {s: pin(DATA / (s + ".jsonl")) for s in ("real-train", "validation")}
    groups = {s: list(rows(verify_pin(p))) for s, p in source_pins.items()}
    lookup = {r["documentId"][4:12]: (s, r) for s, rs in groups.items() for r in rs}
    if len(lookup) != sum(map(len, groups.values())) or reviews.keys() - lookup.keys():
        raise ValueError("review prefix is not unique or is outside current source membership")
    prepared, failures = {}, []
    for prefix, review in reviews.items():
        split, record = lookup[prefix]
        try:
            prepared[record["documentId"]] = repair_record(record, review, engine)
        except (ValueError, KeyError, IndexError) as exc:
            failures.append({"id": record["documentId"], "split": split, "error": str(exc)})
    if failures:
        print(json.dumps({"status": "REJECTED_STAGE", "failures": failures}, indent=2))
        raise ValueError("no repairs staged: resolve all explicit failures first")
    destination = ART / revision
    for did, (record, evidence) in prepared.items():
        evidence.update(
            status="STAGED_AUDIT_REPAIR_NOT_PUBLISHED",
            previousRecords=source_pins[lookup[did[4:12]][0]],
            decisions=[pin(p) for p in paths],
        )
        path = destination / "evidence" / (did + ".json")
        atomic_publish_json(path, evidence)
        record["evidence"] = pin(path)
    outputs = {}
    for split, records in groups.items():
        repaired = [
            prepared[r["documentId"]][0] if r["documentId"] in prepared else r for r in records
        ]
        path = destination / (split + ".jsonl")
        atomic_publish_bytes(path, jsonl(repaired))
        outputs[split] = pin(path)
    report = {
        "status": "STAGED_SOURCE_REPAIRS_REQUIRE_READBACK_AND_TEMPLATE_REPLAY",
        "before": source_pins,
        "after": outputs,
        "decisions": [pin(p) for p in paths],
        "changedDocuments": sorted(prepared),
        "counts": {s: len(r) for s, r in groups.items()},
        "elapsedSeconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    # Timing is observational, not an immutable content-authority pin.
    from document_ocr.atomic import atomic_write_json

    atomic_write_json(OUT / "source-repair-staging.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "changedDocuments"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    stage(parser.parse_args().revision)
