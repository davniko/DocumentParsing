"""Stage exact retained-receipt proposals; never publish training samples.

Each attempt records its method and failure. Complete compiled private-ID
owners may explain old anonymization repairs, but cannot authorize new facts.
Native rendering and the existing semantic/publication gates remain mandatory.
"""

import argparse
import copy
import json
import resource
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from compose_r30_current_receipts import compose, compose_with_compiled_identifiers
from migrate_r30_synthetic import ART, DATA, OUT, ROOT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.scoped_cargo_refresh import apply_edits, reconstruct_source


def augment_compiled_owners(candidate, compiled):
    source = reconstruct_source(candidate)
    if compiled["source_sha256"] != digest(source) or compiled["document_id"] != candidate["sourceDocumentId"]:
        raise ValueError("complete compiled owners belong to another source")
    augmented = copy.deepcopy(candidate)
    edits, added = augmented["proof"]["edits"], set()
    for binding in compiled["bindings"]:
        for occurrence in binding["occurrences"]:
            a, b, text = occurrence["byte_start"], occurrence["byte_end"], occurrence["source_text"]
            if not 0 <= a < b <= len(source) or source[a:b].decode() != text:
                raise ValueError("complete compiled owner differs from source")
            if any(e["byteStart"] < b and e["byteEnd"] > a for e in edits):
                continue
            e = {"key": binding["logical_key"], "byteStart": a, "byteEnd": b,
                 "before": text, "after": text, "targetPaths": occurrence["target_paths"]}
            edits.append(e)
            added.add((e["key"], a, b))
    edits.sort(key=lambda e: e["byteStart"])
    return augmented, added


def complete_compiled_proposal(candidate, current, compiled, *, method="character_diff", typed_identifiers=False, retained_equipment_ids=None):
    source = reconstruct_source(candidate)
    augmented, added = augment_compiled_owners(candidate, compiled)
    edits = augmented["proof"]["edits"]
    patterns = {}
    if typed_identifiers:
        typed = {(b["logical_key"], o["byte_start"], o["byte_end"])
                 for b in compiled["bindings"] if b["kind"] == "identifier"
                 for o in b["occurrences"] if not o["target_paths"]}
        for e in edits:
            key = (e["key"], e["byteStart"], e["byteEnd"])
            if key in typed:
                # Only complete, explicitly private, non-target identifiers.
                # Phones, countries, HS codes and package counts are excluded.
                patterns[key] = "".join("[0-9]" if c.isascii() and c.isdigit()
                    else "[A-Za-z]" if c.isascii() and c.isalpha() else re.escape(c)
                    for c in e["after"])
    result = compose(augmented, current, method=method, owner_patterns=patterns,
                     retained_equipment_ids=retained_equipment_ids)
    additions = [e for e in result["proof"]["edits"]
                 if (e["key"], e["byteStart"], e["byteEnd"]) in added and e["before"] != e["after"]]
    result["proof"]["edits"] = [e for e in result["proof"]["edits"]
                                if (e["key"], e["byteStart"], e["byteEnd"]) not in added or e["before"] != e["after"]]
    if reconstruct_source(result) != source or apply_edits(source, result["proof"]["edits"]) != current["joinedRawText"].encode():
        raise ValueError("complete compiled owner composition fails exact replay")
    result["migrationComposition"]["addedCompiledOwners"] = additions
    return result


def proposal(job):
    entry, current, contracts, destination, complete = job
    path = verify_pin({"path": entry["candidate"], "sha256": entry["candidateSha256"]})
    if digest(current) != entry["recordSha256"]:
        raise ValueError("retained record changed after inventory")
    candidate = read(path)
    attempts = []
    methods = ["historical_owners", "compiled_private_identifier_owners"]
    if complete:
        methods.extend(["complete_compiled_owners", "complete_compiled_literal_frames"])
    for method in methods:
        try:
            if method == "historical_owners":
                result = compose(candidate, current)
            else:
                authority = ROOT / contracts / entry["source"] / "template.json"
                compiled = read(authority)
                if method == "compiled_private_identifier_owners":
                    result = compose_with_compiled_identifiers(candidate, current, compiled)
                else:
                    # The early hand-authored contract schema has no hash or
                    # per-occurrence targets. Verify its complete source file,
                    # then use its explicit binding-level target contract.
                    sampling_source = authority.parent / "source.txt"
                    if sampling_source.read_bytes() != reconstruct_source(candidate):
                        raise ValueError("compiled source file differs from receipt")
                    if set(compiled) == {"document_id", "bindings"}:
                        compiled = copy.deepcopy(compiled)
                        compiled["source_sha256"] = digest(sampling_source.read_bytes())
                        for binding in compiled["bindings"]:
                            for occurrence in binding["occurrences"]:
                                if "target_paths" not in occurrence:
                                    occurrence["target_paths"] = binding["target_paths"]
                    result = complete_compiled_proposal(candidate, current, compiled,
                        method="literal_frames" if method == "complete_compiled_literal_frames" else "character_diff")
                    result["migrationComposition"]["compiledSourceAuthority"] = pin(sampling_source)
                result["migrationComposition"]["compiledOwnerAuthority"] = pin(authority)
            result["migrationComposition"]["historicalCandidate"] = pin(path)
            output = Path(destination) / (entry["id"] + ".json")
            atomic_publish_json(output, result)
            return {
                "id": entry["id"], "source": entry["source"],
                "status": "EXACT_REPLAY_MANUAL_OWNER_REVIEW" if method.startswith("complete_compiled_") else "EXACT_REPLAY_PROPOSAL",
                "method": method, "attempts": attempts, "proposal": pin(output),
                "addedOwners": result["migrationComposition"].get("addedCompiledOwners", result["migrationComposition"].get("addedCompiledIdentifierOwners", [])),
            }
        except (ValueError, KeyError, IndexError) as exc:
            attempts.append({"method": method, "error": str(exc)})
    return {"id": entry["id"], "source": entry["source"], "status": "HELD", "attempts": attempts}


def main(name, workers, batch, complete, held_report=None):
    started = time.perf_counter()
    inventory = read(OUT / "receipt-inventory.json")
    entries = {e["id"]: e for e in inventory["records"]}
    selected = None
    if batch is not None:
        selected = set(read(verify_pin(read(DATA / "manifest.json")["syntheticMigration"]["plan"]))["batches"][batch - 1])
    if held_report is not None:
        previous = read(held_report)
        if previous["inventory"] != pin(OUT / "receipt-inventory.json"):
            raise ValueError("retry report uses another lineage inventory")
        held = {r["id"] for r in previous["records"] if r["status"] == "HELD"}
        selected = held if selected is None else selected & held
    contracts = read(ROOT / "configs/synthesis/mpci_bl_curated_v7_multirow.json")["source_contracts"]
    jobs = []
    for row in map(json.loads, verify_pin(inventory["baseline"]).open()):
        identity = row["documentId"]
        entry = entries[identity]
        if selected is not None and identity not in selected:
            continue
        if (ART / "reconciled" / (identity + ".json")).exists():
            continue
        if not (entry["status"] == "RECEIPT_COMPOSITION_REQUIRED"
                and entry["lineage"] == "historical" and entry["matchingSourcePin"]):
            continue
        jobs.append((entry, row, contracts, str(ART / name), complete))
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(proposal, jobs, chunksize=4):
            results.append(result)
            if len(results) % 100 == 0:
                print(json.dumps({"processed": len(results), "total": len(jobs),
                                  "counts": dict(Counter(r["status"] for r in results))}), flush=True)
    report = {
        "scope": "Exact transport proposals only. No native or dataset acceptance.",
        "inventory": pin(OUT / "receipt-inventory.json"),
        "records": results, "counts": dict(Counter(r["status"] for r in results)),
        "elapsedSeconds": time.perf_counter() - started,
        "parentPeakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "workerPeakRssMiB": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
    }
    atomic_publish_json(OUT / (name + ".json"), report)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}))


def admit(name):
    """Admit only exact transport with shape-preserving reviewed ID owners."""
    report_path = OUT / (name + ".json")
    report = read(report_path)
    inventory = read(verify_pin(report["inventory"]))
    current = {r["documentId"]: r for r in map(json.loads, verify_pin(inventory["baseline"]).open())}
    receipts = []
    for row in report["records"]:
        if row["status"] != "EXACT_REPLAY_PROPOSAL":
            continue
        result = read(verify_pin(row["proposal"]))
        evidence = result["migrationComposition"]
        previous = read(verify_pin(evidence["historicalCandidate"]))
        retained = current[row["id"]]
        if (evidence["currentRecordSha256"] != digest(retained)
                or result["scenario"] != previous["scenario"]
                or result["target"]["documentPatch"] != retained["target"]
                or result["joinedRawText"] != retained["joinedRawText"]):
            raise ValueError("proposal changes retained shipment facts or targets")
        source = reconstruct_source(previous)
        if reconstruct_source(result) != source or apply_edits(source, result["proof"]["edits"]) != retained["joinedRawText"].encode():
            raise ValueError("proposal does not replay every source and retained byte")
        additions = row["addedOwners"]
        if additions:
            compiled = read(verify_pin(evidence["compiledOwnerAuthority"]))
            if compiled["source_sha256"] != digest(source):
                raise ValueError("compiled private authority differs")
            allowed = {
                (b["logical_key"], o["byte_start"], o["byte_end"], o["source_text"])
                for b in compiled["bindings"] if b["value_kind"] == "identifier" and not b["target_paths"]
                for o in b["occurrences"] if not o["target_paths"]
            }
            if additions != evidence["addedCompiledIdentifierOwners"]:
                raise ValueError("private additions differ from proposal report")
            for edit in additions:
                if (edit["key"], edit["byteStart"], edit["byteEnd"], edit["before"]) not in allowed or edit["targetPaths"]:
                    raise ValueError("addition is not a complete reviewed private identifier")
                def shape(value):
                    return "".join("D" if c.isdigit() else "A" if c.isupper() else "a" if c.islower() else c for c in value)
                if shape(edit["before"]) != shape(edit["after"]):
                    raise ValueError("private identifier shape changed; manual owner review required")
        destination = ART / "reconciled" / (row["id"] + ".json")
        atomic_publish_json(destination, {
            "status": "REVIEWED_CURRENT_SHIPMENT_RECEIPT", "candidate": row["proposal"],
            "currentRecordSha256": digest(retained), "mechanismReview": pin(report_path),
            "reviewScope": "Transport only: exact source/current replay, unchanged targets/scenario, complete shape-preserving compiled private IDs. Native rendering and semantic review remain required.",
        })
        receipts.append(pin(destination))
    atomic_publish_json(OUT / (name + "-admission.json"), {
        "report": pin(report_path), "count": len(receipts), "receipts": receipts,
        "syntheticCertifications": 0,
    })
    print(json.dumps({"transportReceipts": len(receipts), "syntheticCertifications": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--admit", action="store_true")
    parser.add_argument("--complete-owners", action="store_true")
    parser.add_argument("--held-report", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.admit:
        admit(args.name)
    else:
        main(args.name, args.workers, args.batch, args.complete_owners, args.held_report)
