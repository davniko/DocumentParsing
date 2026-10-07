"""Materialize reviewed lexical receipts and falsify the current pilot's guards.

This is an offline pilot utility. Publication approval requires the operator's
explicit reviewed snapshot hash, never an automatic model-pass decision.
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from collections import Counter
from copy import deepcopy
from pathlib import Path

import yaml

from document_ocr.synthesis.curated import (
    LexicalBatch,
    SourceContract,
    digest,
    render,
    save,
    scenario_values,
    validate_candidate,
)

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approve-snapshot")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/synthesis/mpci_bl_curated_v7_pilot24.yaml").read_text())
    output = ROOT / cfg["output"]
    rows = {
        r["documentId"]: r
        for r in map(json.loads, (ROOT / cfg["dataset"] / "train.jsonl").read_text().splitlines())
    }
    amendments = yaml.safe_load(
        (ROOT / "configs/synthesis/contracts/curated_v7_pilot24_adjudication.yaml").read_text()
    )
    calls = sorted((output / "calls").glob("generate-batch*.json"), key=lambda p: p.stat().st_mtime)
    valid = {}
    for path in calls:
        receipt = json.loads(path.read_text())
        if receipt["output"] is None:
            continue
        try:
            bundle = LexicalBatch.model_validate(receipt["output"])
        except ValueError:
            continue
        if len(bundle.variants) == cfg["variants_per_source"]:
            valid[receipt["identity"]] = (path, bundle)
    records, receipts = [], []
    for sid in cfg["source_ids"]:
        row = rows[sid]
        envelope = json.loads((output / "sources" / sid / "contract.json").read_text())
        contract = SourceContract.model_validate(envelope["contract"])
        lexical = {v.key for v in contract.variables if v.kind in {"postal", "name", "product"}}
        call_path, bundle = valid[sid]
        amendment = amendments.get(sid, {})
        for variant, generated in enumerate(bundle.variants):
            sample_id = "syn_v7_" + digest([sid, cfg["seed"], variant])[:24]
            values = scenario_values(
                contract, sample_id, cfg["seed"], variant, cfg["variants_per_source"]
            )
            new = {v.key: v.value for v in generated.values}
            if set(new) != lexical or len(new) != len(generated.values):
                raise ValueError(f"bad lexical key set: {sid}")
            edits = amendment.get("variants", {}).get(variant, {})
            if not set(edits) <= lexical:
                raise ValueError("amendment edits a non-lexical or absent key")
            corrections = {k: {"before": new[k], "after": v} for k, v in edits.items()}
            new.update(edits)
            values.update(new)
            text, target, proof = render(row, contract, values)
            record = dict(
                documentId=sample_id,
                sourceDocumentId=sid,
                joinedRawText=text,
                joinedRawTextSha256=digest(text.encode()),
                target=target,
                values=values,
                proof=proof,
                contractSha256=digest(contract.model_dump(mode="json")),
                seed=cfg["seed"],
                variant=variant,
                variantCount=cfg["variants_per_source"],
                generationReceipt=str(call_path.relative_to(ROOT)),
                generationReceiptSha256=digest(call_path.read_bytes()),
                lexicalCorrections=corrections,
                correctionReason=amendment.get("reason"),
                status="awaiting_snapshot_approval",
            )
            validate_candidate(row, contract, record)
            save(output / "samples" / f"{sample_id}.json", record)
            preview = output / "previews" / sid / f"variant-{variant + 1}.txt"
            preview.parent.mkdir(parents=True, exist_ok=True)
            preview.write_text(
                text
                + "\n\n--- TRAINING TARGET ---\n"
                + json.dumps(target, ensure_ascii=False, indent=2)
                + "\n"
            )
            records.append(record)
            receipts.append(
                dict(
                    sample=sample_id,
                    source=sid,
                    corrections=corrections,
                    reason=amendment.get("reason"),
                )
            )

    started = time.perf_counter()
    tracemalloc.start()
    rejected = Counter()
    positive = Counter()
    for record in records:
        row = rows[record["sourceDocumentId"]]
        contract = SourceContract.model_validate(
            json.loads((output / "sources" / row["documentId"] / "contract.json").read_text())[
                "contract"
            ]
        )
        validate_candidate(row, contract, record)
        positive["exact_replay"] += 1
        if len(record["joinedRawText"].splitlines()) != len(row["joinedRawText"].splitlines()):
            raise ValueError("source line structure changed")
        positive["line_structure"] += 1
        mutations = {}
        bad = deepcopy(record)
        bad["joinedRawText"] += "\nUNRELATED STALE FIELD"
        mutations["out_of_scope_input_edit"] = bad
        bad = deepcopy(record)
        bad["target"]["documentPatch"]["billOfLadingNumber"] = "UNPRINTED"
        mutations["unprinted_label"] = bad
        bad = deepcopy(record)
        bad["joinedRawTextSha256"] = "0" * 64
        mutations["stale_hash"] = bad
        numeric = next((v for v in contract.variables if v.kind == "count"), None)
        if numeric:
            bad = deepcopy(record)
            bad["values"][numeric.key] = "99173"
            mutations["identifier_as_quantity"] = bad
        goods = record["target"]["documentPatch"]["goodsItemDetails"][0]
        if goods.get("hsCodes"):
            bad = deepcopy(record)
            bad["target"]["documentPatch"]["goodsItemDetails"][0]["hsCodes"][0] = "999999"
            mutations["swapped_hs"] = bad
        if record["target"]["documentPatch"].get("containerInformation"):
            bad = deepcopy(record)
            bad["target"]["documentPatch"]["containerInformation"].pop()
            mutations["missing_container"] = bad
        for family, bad in mutations.items():
            try:
                validate_candidate(row, contract, bad)
            except (ValueError, KeyError):
                rejected[family] += 1
            else:
                raise AssertionError(f"false pass: {family}")
        postal = next(
            (v for v in contract.variables if v.kind == "postal" and v.required_literals), None
        )
        if postal:
            bad_values = {
                **record["values"],
                postal.key: record["values"][postal.key] + " " + postal.required_literals[0],
            }
            try:
                render(row, contract, bad_values)
            except ValueError:
                rejected["coherent_duplicate_country_proposal"] += 1
            else:
                raise AssertionError("country duplication passed")
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    snapshot = digest(records)
    report = dict(
        samples=len(records),
        sources=len(cfg["source_ids"]),
        positive=dict(positive),
        negativeRejected=dict(rejected),
        snapshotSha256=snapshot,
        validationSeconds=time.perf_counter() - started,
        tracedPeakBytes=peak,
        correctedSamples=sum(bool(r["lexicalCorrections"]) for r in records),
        sourceTrainSha256=digest((ROOT / cfg["dataset"] / "train.jsonl").read_bytes()),
        sourceValidationSha256=digest((ROOT / cfg["dataset"] / "validation.jsonl").read_bytes()),
        approvalsWritten=False,
    )
    if args.approve_snapshot:
        if args.approve_snapshot != snapshot:
            raise ValueError("reviewed snapshot differs from final materialized samples")
        for sid in cfg["source_ids"]:
            subset = [r for r in records if r["sourceDocumentId"] == sid]
            save(
                output / "sources" / sid / "adjudication.json",
                dict(
                    decision="approved",
                    snapshotSha256=snapshot,
                    rationale="Source ownership and all three lexical variants manually reviewed. "
                    "Current-label replay, numeric repetition inventory, frozen-fact checks, "
                    "source-specific corrections and negative probes reviewed together. "
                    "Approval covers this snapshot only, not future generated content.",
                    contractSha256=subset[0]["contractSha256"],
                    sampleSha256={r["documentId"]: digest(r) for r in subset},
                ),
            )
        report["approvalsWritten"] = True
    save(output / "pilot-validation.json", report)
    save(output / "lexical-corrections.json", receipts)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
