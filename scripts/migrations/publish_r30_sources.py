"""Publish reviewed Paddle batches without presenting a partial set as trainable."""

from __future__ import annotations

import argparse
import copy
import json
import re
import resource
import time
from collections import Counter
from pathlib import Path

from apply_r30_adjudications import assign, get
from audit_r30_paddle_pilot import check_training_target
from jsonschema import Draft202012Validator
from prepare_r30_paddle_pilot import load_pages
from rebase_r30_sources import (
    ART,
    BASE,
    OUT,
    PILOT,
    PILOT_ART,
    ROOT,
    audit_staged_record,
    batch_records,
    read,
    rows,
)

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7DirectTarget
from document_ocr.spatial_inputs.paddle import serialize
from document_ocr.synthesis.paddle_render import requires_wording
from document_ocr.synthesis.paddle_validation import (
    check_conservation,
    check_new_private_ownership,
    check_private_lexemes,
    check_synthetic_relations,
    check_views,
)

DATA = ROOT / "data/curated/mpci-bl-r30-paddle-centroids-v1"
TEMPLATES = ROOT / "artifacts/synthesis-templates/mpci-bl-v7-paddle-reviewed/r30"


def jsonl(records):
    return "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for r in sorted(records, key=lambda r: r["documentId"])
    ).encode()


def verify_pin(pin):
    path = ROOT / pin["path"]
    if sha256_file(path) != pin["sha256"]:
        raise ValueError(f"publication evidence changed: {pin['path']}")
    return path


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def active_source_certificates():
    """Resolve explicit source corrections without rewriting pinned history."""
    active = {p.parent.name: p for p in sorted((DATA / "batches").glob("*/certificate.json"))}
    successors = {}
    for path in (DATA / "amendments").glob("*/certificate.json"):
        cert = read(path)
        batch = cert["batch"]
        previous = cert["previous"]
        if batch not in active or previous["path"] in successors:
            raise ValueError("amendment must extend the active certificate without branching")
        verify_pin(cert["review"])
        for evidence in cert["records"].values():
            verify_pin(evidence)
        successors[previous["path"]] = (previous["sha256"], batch, path)
    for batch, path in active.items():
        while True:
            following = successors.pop(str(path.relative_to(ROOT)), None)
            if following is None:
                break
            expected_hash, owner, following_path = following
            if sha256_file(path) != expected_hash:
                raise ValueError("amendment has an unreachable or changed predecessor")
            if owner != batch:
                raise ValueError("amendment predecessor belongs to another source batch")
            path = following_path
        active[batch] = path
    if successors:
        raise ValueError("amendment has an unreachable or changed predecessor")
    return active


def active_template_entries():
    """Resolve reviewed replacements by identity; never overwrite original cases."""
    active = {}
    for path in sorted((TEMPLATES / "batches").glob("*.json")):
        cert = read(path)
        verify_pin(cert["sourceCertificate"])
        for entry in cert["templates"]:
            identity = read(verify_pin(entry["template"]))["sourceDocumentId"]
            if identity in active:
                raise ValueError("duplicate certified template identity")
            active[identity] = entry
    for path in sorted((TEMPLATES / "amendments").glob("*/certificate.json")):
        cert = read(path)
        verify_pin(cert["review"])
        for replacement in cert["templates"]:
            entry = replacement["replacement"]
            identity = read(verify_pin(entry["template"]))["sourceDocumentId"]
            if identity not in active or replacement["previous"] != active[identity]:
                raise ValueError("template amendment must extend active evidence without branching")
            for evidence in entry["controls"]:
                verify_pin(evidence)
            active[identity] = entry
    return active


def amend_templates(review_path):
    """Publish fully replayed, read-back corrections, retaining every old artifact."""
    review = read(review_path)
    active = active_template_entries()
    if not review["rationale"] or not review["templates"]:
        raise ValueError("template amendment requires rationale and reviewed members")
    reviewed = review["mainAgentReadback"]
    identities = [r["sourceDocumentId"] for r in review["templates"]]
    if len(identities) != len(set(identities)) or set(identities) != set(reviewed):
        raise ValueError("template amendment readback must cover exactly its members")
    prepared = []
    for correction in review["templates"]:
        identity = correction["sourceDocumentId"]
        if correction["previous"] != active[identity]:
            raise ValueError("template amendment does not reference current evidence")
        old = read(verify_pin(active[identity]["template"]))
        new = read(verify_pin(correction["replacement"]["template"]))
        if {k: v for k, v in old.items() if k != "overlay"} != {
            k: v for k, v in new.items() if k != "overlay"
        } or old["overlay"] == new["overlay"]:
            raise ValueError("template amendment must change only reviewed overlay ownership")
        source = read(verify_pin(new["source"]))
        transfer = read(verify_pin(new["transfer"]))
        decisions = read(verify_pin(correction["decisions"]))[identity[4:12]]
        controls = correction["replacement"]["controls"]
        if len(controls) != 3 or len({p["path"] for p in controls}) != 3:
            raise ValueError("template amendment needs three distinct controls")
        observed = []
        for evidence, selection in zip(controls, transfer["descendants"], strict=True):
            result = read(verify_pin(evidence))
            pins = result["migrationPins"]
            expected = {
                "template": correction["replacement"]["template"]["sha256"],
                "transfer": new["transfer"]["sha256"],
                "shipment": sha256_file(ROOT / selection["path"]),
            }
            if requires_wording(new["overlay"]):
                expected["wording"] = sha256_file(ART / "new-wording" / (selection["id"] + ".json"))
            if pins != expected:
                raise ValueError("amendment control predates reviewed evidence")
            observed.append(result["documentId"])
            check_training_target(result["target"])
            check_views(result, load_pages(result))
            check_conservation(source, result, new["overlay"])
            check_synthetic_relations(result["target"])
            check_private_lexemes(source, result, new["overlay"].get("retainedLexemes", []))
            check_new_private_ownership(source, result, decisions)
        if observed != new["testDescendants"]:
            raise ValueError("amendment controls change selected shipments")
        prepared.append(
            {"previous": correction["previous"], "replacement": correction["replacement"]}
        )
    atomic_publish_json(
        TEMPLATES / "amendments" / review_path.stem / "certificate.json",
        {
            "status": "CERTIFIED_PADDLE_REPLAY_TEMPLATE_AMENDMENT",
            "review": pin(review_path),
            "templates": prepared,
        },
    )


def amend_targets(batch, review_path):
    """Publish reviewed atomic-field changes without changing pinned inputs."""
    previous = active_source_certificates()[batch]
    review = read(review_path)
    if review["previous"] != pin(previous) or not review["rationale"]:
        raise ValueError("target amendment needs current evidence and review rationale")
    cert = read(previous)
    if not cert["records"] or not set(cert["records"]) <= {"real-train", "validation"}:
        raise ValueError("target amendment requires real-training/validation records")
    records = {split: list(rows(verify_pin(e))) for split, e in cert["records"].items()}
    lookup = {r["documentId"]: r for group in records.values() for r in group}
    if len(lookup) != sum(map(len, records.values())):
        raise ValueError("target amendment has duplicate source identities")
    changes = review["changes"]
    if not changes or not set(changes) <= lookup.keys():
        raise ValueError("amendment must name existing records")
    for did, edits in changes.items():
        if not edits or len({e["path"] for e in edits}) != len(edits):
            raise ValueError("amendment requires distinct target leaves")
        target = copy.deepcopy(lookup[did]["target"])
        for edit in edits:
            if edit.get("operation") == "addMeasurement":
                # A newly resolved unit can establish a previously absent atomic
                # measurement. Do not permit replacing a cargo/party subtree or
                # silently treating a missing before-value as null.
                if (
                    not re.fullmatch(
                        r"/goodsItemDetails/[0-9]+/(grossWeight|netWeight|volume)", edit["path"]
                    )
                    or edit.get("beforeAbsent") is not True
                    or "before" in edit
                    or not isinstance(edit["after"], dict)
                    or set(edit["after"]) != {"unit", "value"}
                    or not edit.get("rationale")
                ):
                    raise ValueError(
                        "measurement addition requires an explicit absent field and rationale"
                    )
                parent_path, key = edit["path"].rsplit("/", 1)
                parent = get(target, parent_path)
                if key in parent:
                    raise ValueError("measurement addition would overwrite an existing field")
                assign(target, edit["path"], edit["after"])
                continue
            if "operation" in edit:
                raise ValueError("unsupported target amendment operation")
            before = get(target, edit["path"])
            # A phone/email list is one atomic extraction field. Replacing that
            # exact reviewed list permits removing a fax-only value without
            # altering input text or rewriting a previously pinned certificate.
            composite = isinstance(before, dict) or (
                isinstance(before, list) and any(isinstance(v, (dict, list)) for v in before)
            )
            if composite or before != edit["before"]:
                raise ValueError("amendment before-value differs or is not a leaf")
            if edit["before"] == edit["after"]:
                raise ValueError("amendment contains an unchanged value")
            assign(target, edit["path"], edit["after"])
        BillOfLadingExtractionV7DirectTarget.model_validate_json(json.dumps(target))
        Draft202012Validator(read(BASE / "schema.json")).validate(target)
        lookup[did]["target"] = target
    # Publication leaves the previous rows, source decisions and all replay pins intact.
    destination = DATA / "amendments" / review_path.stem
    published = {}
    for split, group in records.items():
        record_path = destination / (split + ".jsonl")
        atomic_publish_bytes(record_path, jsonl(group))
        published[split] = pin(record_path)
    atomic_publish_json(
        destination / "certificate.json",
        {
            "status": "CERTIFIED_TARGET_ONLY_AMENDMENT",
            "batch": batch,
            "previous": pin(previous),
            "review": pin(review_path),
            "records": published,
            "counts": {split: len(group) for split, group in records.items()},
            "changedDocuments": sorted(changes),
            "scope": (
                "reviewed target scalars/flat lists and explicit measurement additions only; "
                "input, geometry, membership "
                "and template replays unchanged"
            ),
        },
    )


def order_only_record(previous, candidate):
    """Prove a source-order correction is only a permutation, including geometry."""
    view_keys = {"pages", "joinedRawText", "positionedText", "bboxText"}
    if {k: v for k, v in previous.items() if k not in view_keys} != {
        k: v for k, v in candidate.items() if k not in view_keys
    }:
        raise ValueError("order amendment changes targets, identity or other metadata")
    before, after = load_pages(previous), load_pages(candidate)
    if len(before) != len(after):
        raise ValueError("order amendment changes page membership")
    for a, b in zip(before, after, strict=True):
        if (a.index, a.width, a.height) != (b.index, b.width, b.height) or Counter(
            (r.text, r.box) for r in a.regions
        ) != Counter((r.text, r.box) for r in b.regions):
            raise ValueError("order amendment changes text or geometry")
    for key, mode in (
        ("joinedRawText", "none"),
        ("positionedText", "centroid"),
        ("bboxText", "bbox"),
    ):
        if candidate[key] != serialize(after, geometry=mode):
            raise ValueError("order amendment has an inconsistent input view")


def amend_input_order(batch, review_path):
    """Publish independently audited native-order permutations; retain pinned history."""
    previous = active_source_certificates()[batch]
    review = read(review_path)
    if review["previous"] != pin(previous) or not review["rationale"]:
        raise ValueError("order amendment needs current evidence and review rationale")
    for evidence in review["sourceEvidence"]:
        verify_pin(evidence)
    if not review["sourceEvidence"]:
        raise ValueError("native-order evidence is required")
    cert = read(previous)
    if set(review["records"]) != set(cert["records"]):
        raise ValueError("order amendment changes split membership")
    prepared, changed = {}, []
    for split, evidence in cert["records"].items():
        before = {r["documentId"]: r for r in rows(verify_pin(evidence))}
        after = list(rows(verify_pin(review["records"][split])))
        if len(after) != len(before) or {r["documentId"] for r in after} != before.keys():
            raise ValueError("order amendment changes record membership")
        for record in after:
            old = before[record["documentId"]]
            order_only_record(old, record)
            if old != record:
                changed.append(record["documentId"])
        prepared[split] = after
    if not changed or sorted(changed) != review["changedDocuments"]:
        raise ValueError("order amendment changed-record inventory differs")
    destination = DATA / "amendments" / review_path.stem
    paths = {}
    for split, records in prepared.items():
        path = destination / (split + ".jsonl")
        atomic_publish_bytes(path, jsonl(records))
        paths[split] = pin(path)
    atomic_publish_json(
        destination / "certificate.json",
        {
            "status": "CERTIFIED_NATIVE_ORDER_ONLY_AMENDMENT",
            "batch": batch,
            "previous": pin(previous),
            "review": pin(review_path),
            "records": paths,
            "counts": {s: len(v) for s, v in prepared.items()},
            "changedDocuments": sorted(changed),
            "scope": (
                "native source order only; exact text/rectangle multiset and all targets unchanged"
            ),
        },
    )


def publish_pilot():
    certificate = read(PILOT / "certification.json")
    if certificate["status"] != "CERTIFIED_BOUNDED_PILOT_NOT_FULL_R30_PUBLICATION":
        raise ValueError("pilot does not have a completed review certificate")
    overlays_path = PILOT / "paddle-overlays.json"
    if sha256_file(overlays_path) != certificate["receipts"][str(overlays_path.relative_to(ROOT))]:
        raise ValueError("pilot overlays differ from their reviewed certificate")
    overlays = read(overlays_path)["sources"]
    for old, new in (("real20", "real-train"), ("validation60", "validation")):
        records = list(rows(verify_pin(certificate["published"][old])))
        if len(records) != certificate["published"][old]["count"]:
            raise ValueError("pilot publication count differs")
        path = DATA / "batches/pilot" / (new + ".jsonl")
        atomic_publish_bytes(path, jsonl(records))
    templates = list((PILOT_ART / "certified/templates").glob("*.json"))
    if len(templates) != certificate["templates"]:
        raise ValueError("pilot template membership differs")
    for path in templates:
        entry = read(path)
        if entry["overlay"] != overlays[entry["sourceDocumentId"][4:12]]:
            raise ValueError("published pilot overlay differs from reviewed content")
        verify_pin(entry["source"])
        verify_pin(entry["transfer"])
        atomic_publish_json(TEMPLATES / "cases" / path.name, entry)
    receipt = {
        "status": "CERTIFIED_PILOT_IMPORTED",
        "evidence": pin(PILOT / "certification.json"),
        "records": {
            s: pin(DATA / "batches/pilot" / (s + ".jsonl")) for s in ("real-train", "validation")
        },
        "counts": {"real-train": 20, "validation": 60, "templateOverlays": 20},
        "templateScope": (
            "native overlays and three reviewed shipment replays; "
            "not standalone generation admission"
        ),
    }
    atomic_publish_json(DATA / "batches/pilot/certificate.json", receipt)


def publish_batch(batch, reviewed):
    sources = batch_records(batch)
    expected = {r["documentId"][4:12] for r in sources}
    if len(reviewed) != len(set(reviewed)) or set(reviewed) != expected:
        raise ValueError("explicit main-agent readback must cover exactly this batch")
    splits = {source["split"] for source in sources}
    if splits not in ({"real-train"}, {"template-only"}):
        raise ValueError("source publication requires one explicit real or template-only split")
    split = next(iter(splits))
    decisions_path = OUT / "batches" / batch / "main-agent-decisions.json"
    decisions = read(decisions_path)
    digest = sha256_file(decisions_path)
    records, audits = [], []
    for source in sources:
        did = source["documentId"]
        path = ART / "staged-real" / (did + ".json")
        staged = read(path)
        if staged["reviewSha256"] != digest:
            raise ValueError("staged record predates current main-agent decisions")
        audit = audit_staged_record(source, staged, decisions[did[4:12]])
        audit["evidence"] = pin(path)
        audits.append(audit)
        records.append(
            {
                k: staged[k]
                for k in (
                    "documentId",
                    "pages",
                    "joinedRawText",
                    "positionedText",
                    "bboxText",
                    "target",
                )
            }
            | {"sourceDocumentId": did, "evidence": pin(path)}
        )
    destination = (
        DATA / "batches" / batch if split == "real-train" else TEMPLATES / "source-batches" / batch
    )
    path = destination / (split + ".jsonl")
    atomic_publish_bytes(path, jsonl(records))
    atomic_publish_json(
        destination / "certificate.json",
        {
            "status": "CERTIFIED_SOURCE_BATCH"
            if split == "real-train"
            else "REVIEWED_TEMPLATE_REFERENCE_NOT_TRAINING_DATA",
            "mainAgentReadback": sorted(reviewed),
            "decisions": pin(decisions_path),
            "records": {split: pin(path)},
            "counts": {split: len(records)},
            "audits": audits,
            "templateAcceptance": "tracked separately; source acceptance does not certify bindings",
        },
    )


def publish_templates(batch, reviewed):
    sources = {r["documentId"][4:12]: r for r in batch_records(batch) if r["activeTemplate"]}
    if len(reviewed) != len(set(reviewed)) or set(reviewed) != sources.keys():
        raise ValueError("template readback must cover exactly the admitted batch membership")
    splits = {source["split"] for source in sources.values()}
    if splits not in ({"real-train"}, {"template-only"}):
        raise ValueError("template batch requires one explicit source split")
    source_certificate = (
        DATA / "batches" / batch / "certificate.json"
        if splits == {"real-train"}
        else TEMPLATES / "source-batches" / batch / "certificate.json"
    )
    if not source_certificate.exists():
        raise ValueError("publish the reviewed source batch before its template overlays")
    overlay_path = OUT / "batches" / batch / "paddle-overlays.json"
    overlays = read(overlay_path)["sources"]
    decisions = read(OUT / "batches" / batch / "main-agent-decisions.json")
    replay_path = OUT / "batches" / batch / "replay-results.json"
    replay = read(replay_path)
    passed = {
        r["id"]
        for r in replay["records"]
        if r["status"] == "FULL_REPLAY_PASSED_REQUIRES_MAIN_AGENT_READBACK"
    }
    prepared = []
    for prefix in sorted(sources):
        source, overlay = sources[prefix], overlays[prefix]
        did = source["documentId"]
        transfer_path = ART / "transfers" / (did + ".json")
        transfer = read(transfer_path)
        if len(transfer["descendants"]) != 3:
            raise ValueError("three complete descendant controls are required")
        controls = []
        for selection in transfer["descendants"]:
            if selection["id"] not in passed:
                raise ValueError("batch replay receipt is incomplete")
            path = ART / "rendered" / (selection["id"] + ".json")
            result = read(path)
            expected_pins = {
                "overlay": sha256_file(overlay_path),
                "transfer": sha256_file(transfer_path),
                "shipment": sha256_file(ROOT / selection["path"]),
            }
            if requires_wording(overlay):
                expected_pins["wording"] = sha256_file(
                    ART / "new-wording" / (selection["id"] + ".json")
                )
            if result["migrationPins"] != expected_pins:
                raise ValueError("template control predates its current bindings/wording")
            check_training_target(result["target"])
            check_views(result, load_pages(result))
            check_conservation(source, result, overlay)
            check_synthetic_relations(result["target"])
            check_private_lexemes(source, result, overlay.get("retainedLexemes", []))
            check_new_private_ownership(source, result, decisions[prefix])
            controls.append(pin(path))
        entry = {
            "sourceDocumentId": did,
            "source": pin(ART / "sources" / (did + ".json")),
            "transfer": pin(transfer_path),
            "overlay": overlay,
            "testDescendants": [r["id"] for r in transfer["descendants"]],
            "scope": "reviewed native owners and three complete shipment replays",
        }
        destination = TEMPLATES / "cases" / (did + ".json")
        prepared.append((destination, entry, controls))
    # Validate the complete batch before publishing any member. Only the final
    # certificate admits its entries to the manifest (including after interruption).
    published = []
    for destination, entry, controls in prepared:
        atomic_publish_json(destination, entry)
        published.append({"template": pin(destination), "controls": controls})
    atomic_publish_json(
        TEMPLATES / "batches" / (batch + ".json"),
        {
            "status": "CERTIFIED_PADDLE_REPLAY_TEMPLATES",
            "mainAgentReadback": sorted(reviewed),
            "sourceCertificate": pin(source_certificate),
            "replay": pin(replay_path),
            "overlays": pin(overlay_path),
            "templates": published,
            "scope": (
                "Paddle presentation and targets over pinned accepted shipment receipts; "
                "fresh generation provenance, when used, is pinned in the transfer selections"
            ),
        },
    )


def update_manifest():
    plan = read(OUT / "plan.json")
    expected = {
        split: {r["id"] for r in plan["sources"] if r["split"] == split}
        for split in ("real-train", "validation")
    }
    seen, certificates = {s: set() for s in expected}, []
    for path in active_source_certificates().values():
        cert = read(path)
        certificates.append(pin(path))
        for split, evidence in cert["records"].items():
            records = list(rows(verify_pin(evidence)))
            ids = [r["documentId"] for r in records]
            if len(ids) != len(set(ids)) or seen[split] & set(ids):
                raise ValueError("duplicate published source identity")
            if not set(ids) <= expected[split]:
                raise ValueError("publication changes frozen split membership")
            seen[split].update(ids)
    templates = set()
    pilot_certificate = DATA / "batches/pilot/certificate.json"
    if pilot_certificate.exists():
        historical = read(verify_pin(read(pilot_certificate)["evidence"]))
        pilot_overlays = read(PILOT / "paddle-overlays.json")["sources"]
        for path in (PILOT_ART / "certified/templates").glob("*.json"):
            entry = read(path)
            published = TEMPLATES / "cases" / path.name
            if read(published) != entry or entry["overlay"] != pilot_overlays[path.stem[4:12]]:
                raise ValueError("published pilot template differs from reviewed entry")
            templates.add(path.stem)
        if len(templates) != historical["templates"]:
            raise ValueError("pilot template certificate membership differs")
    active_templates = active_template_entries()
    if templates & active_templates.keys():
        raise ValueError("duplicate certified template identity")
    templates.update(active_templates)
    expected_templates = {r["id"] for r in plan["sources"] if r["activeTemplate"]}
    if not templates <= expected_templates:
        raise ValueError("publication changes template membership")
    for filename in ("schema.json",):
        atomic_publish_bytes(DATA / filename, (BASE / filename).read_bytes())
    manifest = {
        "status": "PARTIAL_CERTIFIED_BATCHES_NOT_TRAINABLE",
        "sourcePlan": pin(OUT / "plan.json"),
        "batches": certificates,
        "publishedCounts": {s: len(v) for s, v in seen.items()},
        "requiredCounts": {s: len(v) for s, v in expected.items()},
        "publishedTemplateOverlays": len(templates),
        "requiredTemplates": len(expected_templates),
        "templateAmendments": [
            pin(p) for p in sorted((TEMPLATES / "amendments").glob("*/certificate.json"))
        ],
        "syntheticMigration": "not started; R29 synthetic rows are not Paddle rows",
        "inputOrder": (
            "native Paddle region order; reviewed reflow retains native ordering semantics"
        ),
    }
    atomic_write_json(DATA / "manifest.json", manifest)
    return {k: v for k, v in manifest.items() if k not in {"batches", "sourcePlan"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--batch")
    parser.add_argument("--reviewed-sources", nargs="+", default=[])
    parser.add_argument("--templates", action="store_true")
    parser.add_argument("--amend-targets", type=Path)
    parser.add_argument("--amend-order", type=Path)
    parser.add_argument("--amend-templates", type=Path)
    args = parser.parse_args()
    if args.amend_templates:
        if args.pilot or args.batch or args.templates or args.amend_order or args.amend_targets:
            parser.error("template amendments use their own reviewed membership")
        amend_templates(args.amend_templates.resolve())
        print(json.dumps(update_manifest(), indent=2))
        return
    if args.pilot == bool(args.batch):
        parser.error("choose pilot import or a reviewed batch")
    started = time.perf_counter()
    if args.amend_targets and (args.templates or args.amend_order):
        parser.error("target amendment cannot also amend templates or input order")
    if args.amend_order and args.templates:
        parser.error("order amendment changes real inputs, not synthetic template controls")
    if args.amend_order:
        amend_input_order("pilot" if args.pilot else args.batch, args.amend_order.resolve())
    elif args.amend_targets:
        amend_targets("pilot" if args.pilot else args.batch, args.amend_targets.resolve())
    elif args.pilot:
        if args.templates:
            parser.error("pilot import already includes its certified overlays")
        publish_pilot()
    elif args.templates:
        publish_templates(args.batch, args.reviewed_sources)
    else:
        publish_batch(args.batch, args.reviewed_sources)
    print(
        json.dumps(
            update_manifest()
            | {
                "seconds": time.perf_counter() - started,
                "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
