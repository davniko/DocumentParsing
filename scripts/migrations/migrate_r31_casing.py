"""Audit source-casing recovery before publishing an R31 casing-only dataset."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import resource
import tempfile
import time
from collections import Counter
from functools import cache, lru_cache
from pathlib import Path

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, atomic_write_json
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7DirectTarget
from document_ocr.labeling_agents.source_casing import (
    PrintedCaseIndex,
    case_key,
    recover_case,
    selected_fields,
)
from document_ocr.synthesis.paddle_source_casing import POLICY, assign, rendered_case_scopes

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/curated/mpci-bl-r30-paddle-centroids-v1"
OUT = ROOT / "docs/analysis/r31-casing-migration-20261109"
DEST = ROOT / "data/curated/mpci-bl-r31-paddle-centroids-v1"
CATALOG = (
    ROOT / "artifacts/synthesis-templates/mpci-bl-v7-paddle-reviewed/r31/printed-casing-20261109"
)


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


@cache
def verify(path, digest):
    if sha256_file(ROOT / path) != digest:
        raise ValueError(f"casing evidence pin changed: {path}")


def prepare_catalog():
    parent = read(BASE / "manifest.json")["templateCatalog"]
    verify(parent["path"], parent["sha256"])
    result = read(parent["path"])
    result["targetCasing"] = POLICY
    result["casingParentCatalog"] = parent
    result["casingOccurrenceReview"] = pin(OUT / "reviewed-regions.json")
    matched = set()
    expected = set(reviews()["templateRegions"]) | set(reviews()["templateProjections"])
    for identity, entry in result["templates"].items():
        short = identity[4:12]
        if short not in expected:
            continue
        verify(entry["template"]["path"], entry["template"]["sha256"])
        native = read(entry["template"]["path"])
        native["overlay"] = reviewed_overlay(native["overlay"], short)
        path = CATALOG / "templates" / f"{identity}.json"
        atomic_publish_json(path, native)
        entry["template"] = pin(path)
        matched.add(short)
    if matched != expected:
        raise ValueError("casing reviews contain a nonmember template")
    atomic_publish_json(CATALOG / "catalog.json", result)
    return pin(CATALOG / "catalog.json")


def reviewed_overlay(overlay, short):
    result = copy.deepcopy(overlay)
    result["sourceCaseRegions"] = reviews()["templateRegions"].get(short, {})
    for key, actions in reviews()["templateProjections"].get(short, {}).items():
        if any(a["target"] == b["target"] for a in actions for b in result.get(key, [])):
            raise ValueError("new casing ownership conflicts with an existing projection")
        result.setdefault(key, []).extend(actions)
    return result


def validate_case_only(before, after):
    restored = copy.deepcopy(after)
    fields = dict(selected_fields(before))
    if fields.keys() != dict(selected_fields(after)).keys():
        raise ValueError("casing migration changed field membership")
    for path, value in selected_fields(after):
        if case_key(value) != case_key(fields[path]):
            raise ValueError(f"casing migration changed content: {path}")
        assign(restored, path, fields[path])
    if restored != before:
        raise ValueError("casing migration changed an out-of-scope target")
    if (
        BillOfLadingExtractionV7DirectTarget.model_validate_json(
            json.dumps(after)
        ).canonical_target()
        != after
    ):
        raise ValueError("casing migration changed canonical target structure")


def concatenate_batches(paths, destination):
    # Streaming publication keeps the 450 MB synthetic set out of RAM. A
    # manifest is published only after every completed batch passes readback.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
        for path in paths:
            with path.open("rb") as source:
                while block := source.read(1024 * 1024):
                    stream.write(block)
        stream.flush()
        os.fsync(stream.fileno())
    if destination.exists():
        if sha256_file(destination) != sha256_file(temporary):
            temporary.unlink()
            raise ValueError("R31 publication already exists with different bytes")
        temporary.unlink()
    else:
        os.replace(temporary, destination)


@lru_cache(maxsize=1)
def reviews():
    return read(OUT / "reviewed-regions.json")


def read(path):
    return json.loads((ROOT / path).read_text())


def pointer(path):
    return "/" + re.sub(r"\[(\d+)\]", r".\1", path.removeprefix("documentPatch.")).replace(".", "/")


@lru_cache(maxsize=64)
def catalog(path):
    return read(path)


@lru_cache(maxsize=512)
def template(path):
    return read(path)


def native_scopes(row):
    manifest = catalog(str((BASE / "manifest.json").relative_to(ROOT)))
    active = catalog(manifest["templateCatalog"]["path"])
    entry = active["templates"].get(row["sourceDocumentId"])
    if entry is None:
        return {}
    native = template(entry["template"]["path"])
    source = template(native["source"]["path"])
    original = source["pages"]
    if len(original) != len(row["pages"]) or any(
        len(a["regions"]) != len(b["regions"])
        or any(
            not all(
                math.isclose(x["box"][k], y["box"][k], rel_tol=0, abs_tol=1e-8) for k in (0, 1, 3)
            )
            for x, y in zip(a["regions"], b["regions"], strict=True)
        )
        for a, b in zip(original, row["pages"], strict=True)
    ):
        return {}  # No native-coordinate authority for a geometrically changed input.
    transfer = template(native["transfer"]["path"])
    refs = {}
    for slot in transfer["slots"]:
        overrides = [
            o
            for o in native["overlay"].get("overrides", [])
            if o["key"] == slot["key"]
            and ("byteStart" not in o or o["byteStart"] == slot["byteStart"])
            and ("byteEnd" not in o or o["byteEnd"] == slot["byteEnd"])
        ]
        if len(overrides) > 1:
            raise ValueError("ambiguous native casing owner")
        rule = overrides[0] if overrides else slot
        if "fragments" in rule:
            positions = [(p, i) for p, i, _ in rule["fragments"]]
        elif "regions" in rule:
            positions = [tuple(r) for r in rule["regions"]]
        else:
            positions = [
                (s["page"], s["region"]) if isinstance(s, dict) else tuple(s[:2])
                for s in rule.get("spans", [])
            ]
        for path in slot.get("targetPaths", []):
            refs.setdefault(pointer(path), set()).update(positions)
    for action in native["overlay"].get("extras", []) + native["overlay"].get(
        "targetFromSourceSpans", []
    ):
        if "target" not in action:
            continue
        positions = set()
        for part in action.get("groups", [action]):
            positions.update(tuple(r) for r in part.get("regions", []))
            positions.update((p, i) for p, i, _ in part.get("fragments", []))
            positions.update(tuple(s[:2]) for s in part.get("spans", []))
        refs.setdefault(action["target"], set()).update(positions)
    return {
        path: PrintedCaseIndex(
            "\n".join(row["pages"][p]["regions"][i]["text"] for p, i in sorted(positions))
        )
        for path, positions in refs.items()
        if positions
    }


def owned_scopes(row):
    evidence = row.get("evidence", {})
    if "rendered" not in evidence or "candidate" not in evidence:
        return {}
    for key in ("rendered", "candidate", "catalog"):
        verify(evidence[key]["path"], evidence[key]["sha256"])
    rendered = read(evidence["rendered"]["path"])
    if rendered["pages"] != row["pages"]:
        raise ValueError(f"rendered evidence differs from current input: {row['documentId']}")
    candidate = read(evidence["candidate"]["path"])
    if candidate.get("status") == "REVIEWED_SOURCE_PRESERVING_REBASE":
        if candidate["documentId"] != row["documentId"]:
            raise ValueError("source-preserving case evidence belongs to another record")
        # This reviewed cohort has no generated linguistic owners. Its literal
        # source casing is audited by the same path as anonymized real inputs.
        return {}
    entry = catalog(evidence["catalog"]["path"])["templates"][row["sourceDocumentId"]]
    native = template(entry["template"]["path"])
    overlay = reviewed_overlay(native["overlay"], row["sourceDocumentId"][4:12])
    return rendered_case_scopes(rendered, candidate, overlay, template(native["source"]["path"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--splits", nargs="+", default=["validation", "real-train", "synthetic-train"]
    )
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--catalog-only", action="store_true")
    args = parser.parse_args()
    if args.catalog_only:
        print(json.dumps(prepare_catalog()))
        return
    if args.publish and (
        args.limit or args.splits != ["validation", "real-train", "synthetic-train"]
    ):
        raise ValueError("publication requires every record in all three splits")
    started = time.perf_counter()
    issues, counts, source_hashes = [], {}, {}
    all_batches, aggregates = [], {}
    parent_manifest = read(BASE / "manifest.json")
    published_catalog = prepare_catalog() if args.publish else None
    for split in args.splits:
        count = Counter()
        source_hashes[split] = sha256_file(BASE / f"{split}.jsonl")
        if source_hashes[split] != parent_manifest["aggregateRecords"][split]["sha256"]:
            raise ValueError("R30 aggregate differs from its current manifest")
        batch_rows, batch_receipts, batch_paths = [], [], []

        def flush_batch(
            batch_rows=batch_rows,
            batch_receipts=batch_receipts,
            batch_paths=batch_paths,
            split=split,
        ):
            if not batch_rows:
                return
            batch_path = DEST / "batches" / f"{split}-{len(batch_paths) + 1:03}.jsonl"
            certificate_path = batch_path.with_suffix(".certificate.json")
            certificate = {
                "policy": POLICY,
                "parent": parent_manifest["aggregateRecords"][split],
                "review": pin(OUT / "reviewed-regions.json"),
                "records": batch_receipts,
            }
            atomic_publish_json(certificate_path, certificate)
            for updated in batch_rows:
                updated["casingMigration"] = pin(certificate_path)
            payload = b"".join(
                json.dumps(r, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
                + b"\n"
                for r in batch_rows
            )
            atomic_publish_bytes(batch_path, payload)
            if list(map(json.loads, batch_path.open())) != batch_rows:
                raise ValueError("published batch readback differs")
            all_batches.append(
                {
                    "records": pin(batch_path),
                    "certificate": pin(certificate_path),
                    "count": len(batch_rows),
                }
            )
            batch_paths.append(batch_path)
            batch_rows.clear()
            batch_receipts.clear()

        for n, row in enumerate(map(json.loads, (BASE / f"{split}.jsonl").open())):
            if args.limit is not None and n >= args.limit:
                break
            index = PrintedCaseIndex(
                "\n".join(r["text"] for p in row["pages"] for r in p["regions"])
            )
            scopes = owned_scopes(row) if split == "synthetic-train" else native_scopes(row)
            if not row["documentId"].startswith("syn_") or "source-preserving" in row.get(
                "evidence", {}
            ).get("candidate", {}).get("path", ""):
                for path, refs in (
                    reviews()["recordRegions"].get(row["sourceDocumentId"][4:12], {}).items()
                ):
                    scopes[path] = PrintedCaseIndex(
                        "\n".join(row["pages"][p]["regions"][i]["text"] for p, i in refs)
                    )
            count["documents"] += 1
            updated = copy.deepcopy(row) if args.publish else None
            receipts = []
            changed = False
            for path, value in selected_fields(row["target"]):
                count["fields"] += 1
                result = recover_case(value, index, owned=scopes.get(path))
                uppercase_review = (
                    reviews().get("recordUppercase", {}).get(row["documentId"][4:12], {}).get(path)
                )
                if uppercase_review:
                    scope = scopes[path].text
                    if scope.upper() != scope or value.upper() != value:
                        raise ValueError(
                            "reviewed all-uppercase passage no longer has uniform case"
                        )
                    result = {
                        "value": value,
                        "method": "reviewed_uppercase_continuation",
                        "reason": uppercase_review,
                    }
                if "unresolved" in result:
                    count["unresolved_fields"] += 1
                    issues.append(
                        {
                            "split": split,
                            "id": row["documentId"],
                            "parent": row["sourceDocumentId"],
                            "path": path,
                            "before": value,
                            **result,
                        }
                    )
                else:
                    count[result["method"]] += 1
                    count["changed_fields"] += result["value"] != value
                    if result["value"] != value:
                        count["changed_" + path.split("/")[-1]] += 1
                    changed |= result["value"] != value
                    if args.publish:
                        assign(updated["target"], path, result["value"])
                        receipts.append(
                            {
                                "path": path,
                                "before": value,
                                "after": result["value"],
                                "method": result["method"],
                            }
                        )
            count["changed_documents"] += changed
            if args.publish:
                if issues:
                    raise ValueError(f"unresolved casing cannot be published: {issues[-1]}")
                validate_case_only(row["target"], updated["target"])
                if {k: v for k, v in row.items() if k != "target"} != {
                    k: v for k, v in updated.items() if k != "target"
                }:
                    raise ValueError("input, coordinates or provenance changed")
                batch_rows.append(updated)
                batch_receipts.append(
                    {
                        "documentId": row["documentId"],
                        "parentRowSha256": sha256_bytes(canonical_json_bytes(row)),
                        "fields": receipts,
                    }
                )
                if len(batch_rows) == 100:
                    flush_batch()
            if n % 100 == 99:
                print(
                    json.dumps(
                        {
                            "split": split,
                            "processed": n + 1,
                            "unresolved": count["unresolved_fields"],
                        }
                    ),
                    flush=True,
                )
        counts[split] = dict(count)
        if args.publish:
            flush_batch()
            concatenate_batches(batch_paths, DEST / f"{split}.jsonl")
            aggregates[split] = pin(DEST / f"{split}.jsonl")
    for split, digest in source_hashes.items():
        if sha256_file(BASE / f"{split}.jsonl") != digest:
            raise ValueError("R30 changed during migration")
    report = {
        "counts": counts,
        "sourceHashes": source_hashes,
        "seconds": time.perf_counter() - started,
        "maxRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "published": args.publish,
    }
    if args.publish:
        atomic_publish_bytes(DEST / "schema.json", (BASE / "schema.json").read_bytes())
        manifest = {
            "dataset": "mpci-bl-r31-paddle-centroids-v1",
            "stage": "A-casing-only",
            "targetCasing": POLICY,
            "parent": pin(BASE / "manifest.json"),
            "aggregateRecords": aggregates,
            "templateCatalog": published_catalog,
            "schema": pin(DEST / "schema.json"),
            "batches": all_batches,
            "audit": report,
            "unchanged": [
                "input text",
                "Paddle lines and boxes",
                "field boundaries",
                "non-case content",
                "all other targets",
                "IDs and splits",
            ],
            "stageB": "Not generated; boundary coverage is a separate proposed experiment.",
        }
        atomic_publish_json(DEST / "manifest.json", manifest)
    suffix = f"-first{args.limit}" if args.limit else ""
    if args.splits != ["validation", "real-train", "synthetic-train"]:
        suffix += "-" + "-".join(args.splits)
    atomic_write_json(OUT / f"{'publication' if args.publish else 'screen'}{suffix}.json", report)
    atomic_write_json(OUT / f"unresolved{suffix}.json", issues)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
