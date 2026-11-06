"""Certify the bounded Paddle pilot, never a full R30 training publication.

Publication requires separately pinned main-agent readback receipts. Automatic
checks prove replay, conservation and provenance, not semantic interpretation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from functools import cache

from jsonschema import Draft202012Validator
from prepare_r30_paddle_pilot import ART, BASE, OUT, PREFIXES, ROOT, load_pages, read

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7DirectTarget
from document_ocr.spatial_inputs.paddle import fingerprint
from document_ocr.synthesis.paddle_validation import (
    check_conservation,
    check_private_residue,
    check_synthetic_relations,
    check_views,
    require,
)


@cache
def training_target_validator():
    # Full V7 permits fields deliberately excluded from the frozen training
    # projection. Replays and publication must enforce both contracts.
    return Draft202012Validator(read(BASE / "schema.json"))


def check_training_target(target):
    training_target_validator().validate(target)
    BillOfLadingExtractionV7DirectTarget.model_validate_json(json.dumps(target))


def leaves(value, path=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from leaves(child, path + "/" + key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from leaves(child, path + "/" + str(index))
    else:
        yield path, value


def leaf_changes(old, new):
    old, new = dict(leaves(old)), dict(leaves(new))
    return [
        {"path": p, "action": "add" if p not in old else "remove" if p not in new else "change"}
        for p in sorted(old.keys() | new.keys())
        if p not in old or p not in new or old[p] != new[p]
    ]


def audit():
    start = time.perf_counter()
    plan = read(OUT / "plan.json")
    for path, digest in plan["protectedPins"].items():
        require(sha256_file(ROOT / path) == digest, f"protected baseline changed: {path}")
    overlays = read(OUT / "paddle-overlays.json")["sources"]
    require(set(overlays) == set(PREFIXES), "overlay cohort differs")
    validator = Draft202012Validator(read(BASE / "schema.json"))
    rows, receipts, templates, changes = (
        {"real20": [], "validation60": [], "synthetic60": []},
        {},
        [],
        [],
    )
    geometry_scales = []
    for pin in plan["sources"]:
        path = ROOT / pin["path"]
        require(sha256_file(path) == pin["sha256"], "pinned native source changed")
        source = read(path)
        did, pre = source["documentId"], source["documentId"][4:12]
        if source["split"] == "validation":
            records = [("validation60", ART / "validation-rebased" / path.name)]
            privacy = None
        else:
            privacy_path = ART / "privacy-proposals" / path.name
            privacy = read(privacy_path)
            require(not privacy["missingLegacyPlanEdits"], "unresolved private owner")
            records = [("real20", privacy_path)]
            transfer_path = ART / "transfers" / path.name
            transfer = read(transfer_path)
            receipts[str(transfer_path.relative_to(ROOT))] = sha256_file(transfer_path)
            require(len(transfer["descendants"]) == 3, "template must have three descendants")
            for selected in transfer["descendants"]:
                candidate_path = ROOT / selected["path"]
                receipts[str(candidate_path.relative_to(ROOT))] = sha256_file(candidate_path)
                records.append(("synthetic60", ART / "rendered" / (selected["id"] + ".json")))
            templates.append(
                {
                    "sourceDocumentId": did,
                    "source": {"path": pin["path"], "sha256": pin["sha256"]},
                    "transfer": {
                        "path": str(transfer_path.relative_to(ROOT)),
                        "sha256": sha256_file(transfer_path),
                    },
                    "overlay": overlays[pre],
                    "testDescendants": [s["id"] for s in transfer["descendants"]],
                    "scope": "reviewed native owners and three complete shipment replays",
                }
            )
        for split, record_path in records:
            row = read(record_path)
            validator.validate(row["target"])
            BillOfLadingExtractionV7DirectTarget.model_validate_json(json.dumps(row["target"]))
            pages = load_pages(source if split == "validation60" else row)
            check_views(row, pages)
            if split != "validation60":
                require(row["fingerprint"] == fingerprint(pages), "rendered fingerprint differs")
                check_conservation(source, row, overlays[pre])
                check_private_residue(privacy, row)
                geometry_scales.extend(p.get("verticalScale", 1) for p in row["geometry"]["pages"])
            if split == "synthetic60":
                check_synthetic_relations(row["target"])
            else:
                changes.append(
                    {
                        "id": did,
                        "split": split,
                        "fields": leaf_changes(source["current"]["target"], row["target"]),
                    }
                )
            key = str(record_path.relative_to(ROOT))
            receipts[key] = sha256_file(record_path)
            rows[split].append(
                {
                    "documentId": row["documentId"],
                    "sourceDocumentId": did,
                    "pages": row.get("pages", source["pages"]),
                    "joinedRawText": row["joinedRawText"],
                    "positionedText": row["positionedText"],
                    "bboxText": row["bboxText"],
                    "target": row["target"],
                    "evidence": {"path": key, "sha256": receipts[key]},
                }
            )
    require(
        {k: len(v) for k, v in rows.items()}
        == {"real20": 20, "validation60": 60, "synthetic60": 60},
        "incomplete cohort",
    )
    require(len({r["documentId"] for rr in rows.values() for r in rr}) == 140, "duplicate identity")
    rendered = read(OUT / "full-render-results.json")
    require(
        len(rendered["results"]) == 60
        and all(r["status"] == "RENDERED_NOT_CERTIFIED" for r in rendered["results"]),
        "combined replay contains holds",
    )
    for name in ("adjudications.json", "paddle-overlays.json", "real-privacy-overlays.json"):
        path = OUT / name
        receipts[str(path.relative_to(ROOT))] = sha256_file(path)
    code_paths = [
        *sorted((ROOT / "scripts/migrations").glob("*.py")),
        ROOT / "src/document_ocr/spatial_inputs/paddle.py",
        *sorted((ROOT / "src/document_ocr/synthesis").glob("paddle_*.py")),
    ]
    for path in code_paths:
        receipts[str(path.relative_to(ROOT))] = sha256_file(path)
    report = {
        "status": "AUTOMATIC_GATES_PASS_REQUIRES_PINNED_MAIN_AGENT_READBACK",
        "counts": {k: len(v) for k, v in rows.items()},
        "templates": len(templates),
        "protectedPinsUnchanged": len(plan["protectedPins"]),
        "receipts": receipts,
        "targetChanges": changes,
        "changesBySplit": {
            k: dict(Counter(x["action"] for r in changes if r["split"] == k for x in r["fields"]))
            for k in ("real20", "validation60")
        },
        "changedDocumentsBySplit": {
            k: sum(bool(r["fields"]) for r in changes if r["split"] == k)
            for k in ("real20", "validation60")
        },
        "geometryScaleMinMedian": [min(geometry_scales), statistics.median(geometry_scales)],
        "replayPairSeconds": sum(r["seconds"] for r in rendered["results"]),
        "auditSeconds": time.perf_counter() - start,
        "limits": [
            "not a whole-dataset privacy proof",
            "not model parity or re-OCR certification",
            "not full production source admission",
        ],
    }
    return report, rows, templates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    report, rows, templates = audit()
    if args.publish:
        review = read(OUT / "main-agent-readback.json")
        require(review["status"] == "REVIEWED_ACCEPTED", "readback is not accepted")
        require(
            review["receipts"] == report["receipts"], "readback pins differ from final artifacts"
        )
        require(set(review["sourceDispositions"]) == set(PREFIXES), "source readback incomplete")
        published = {}
        for split, records in rows.items():
            path = ART / "certified" / (split + ".jsonl")
            payload = "".join(
                json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                for r in sorted(records, key=lambda r: r["documentId"])
            )
            atomic_publish_bytes(path, payload.encode())
            published[split] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "count": len(records),
            }
        for template in templates:
            atomic_publish_json(
                ART / "certified/templates" / (template["sourceDocumentId"] + ".json"), template
            )
        report["status"] = "CERTIFIED_BOUNDED_PILOT_NOT_FULL_R30_PUBLICATION"
        report["published"] = published
        report["mainAgentReadbackSha256"] = sha256_file(OUT / "main-agent-readback.json")
    atomic_write_json(OUT / "certification.json", report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"receipts", "targetChanges"}}, indent=2
        )
    )


if __name__ == "__main__":
    main()
