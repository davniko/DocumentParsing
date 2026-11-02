"""Replay the 100 approved footprint-safe equipment counterfactuals into staging."""

import copy
import json
import shutil
from collections import Counter, defaultdict

from stage_combination_refresh import OUT, ROOT

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7DirectTarget
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_positions import transfer_existing_positions
from document_ocr.synthesis.declared_packages import DeclaredPackages
from document_ocr.synthesis.scoped_cargo_refresh import apply_edits


def main():
    data = ROOT / "data/curated/mpci-bl-r28-positions-v1"
    report = json.loads(
        (
            ROOT / "docs/analysis/r28-targeted-cargo-refresh-probe-20261101/equipment-probe.json"
        ).read_text()
    )
    if sha256_file(data / "synthetic-train.jsonl") != report["datasetSha256"]:
        raise ValueError("current dataset changed since equipment probe")
    backup = OUT / "before"
    backup.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in [
        "synthetic-train.jsonl",
        "previous-uae-synthetic.jsonl",
        "old-uae-addition.jsonl",
        "real-train.jsonl",
        "validation.jsonl",
        "manifest.json",
        "README.md",
    ]:
        source = data / name
        target = backup / name
        hashes[name] = sha256_file(source)
        if not target.exists():
            shutil.copy2(source, target)
        if sha256_file(target) != hashes[name]:
            raise ValueError("backup differs from current data")
    atomic_write_json(backup / "hashes.json", hashes)
    current = {
        r["documentId"]: r
        for line in (data / "synthetic-train.jsonl").open()
        if (r := json.loads(line))
    }
    by_parent = defaultdict(list)
    for r in report["results"]:
        if r["status"] == "PASS_CURRENT_FOOTPRINT":
            by_parent[r["sourceDocumentId"]].append(r)
    chosen = []
    for rows in by_parent.values():
        rows.sort(key=lambda r: digest(["equipment-refresh", r["documentId"]]))
    while len(chosen) < 100:
        for sid in sorted(by_parent, key=lambda s: digest(s)):
            if by_parent[sid]:
                chosen.append(by_parent[sid].pop())
            if len(chosen) == 100:
                break
    for receipt in chosen:
        before = current[receipt["documentId"]]
        after = copy.deepcopy(before)
        if digest(before["joinedRawText"].encode()) != receipt["proof"]["sourceSha256"]:
            raise ValueError("stale equipment proof")
        raw = apply_edits(before["joinedRawText"].encode(), receipt["proof"]["edits"])
        if digest(raw) != receipt["proof"]["renderedSha256"]:
            raise ValueError("equipment replay differs")
        for container, size in zip(
            after["target"]["containerInformation"], receipt["afterSizes"], strict=True
        ):
            container["sizeCategory"] = size
        after.update(joinedRawText=raw.decode(), joinedRawTextSha256=digest(raw))
        positioned, geometry = transfer_existing_positions(
            before["joinedRawText"], before["positionedText"], raw.decode(), receipt["proof"]
        )
        after.update(positionedText=positioned, positionedTextSha256=digest(positioned.encode()))
        BillOfLadingExtractionV7DirectTarget.model_validate_json(json.dumps(after["target"]))
        DeclaredPackages.from_receipt(receipt["ledger"])
        if before["target"]["goodsItemDetails"] != after["target"]["goodsItemDetails"]:
            raise ValueError("equipment-only migration altered cargo")
        if [line.rsplit(" ||", 1)[-1] for line in before["positionedText"].splitlines()] != [
            line.rsplit(" ||", 1)[-1] for line in positioned.splitlines()
        ]:
            raise ValueError("equipment migration changed coordinates")
        atomic_write_json(
            OUT / "staged-equipment" / f"{before['documentId']}.json",
            dict(
                before=before,
                after=after,
                evidence=receipt,
                geometry=geometry,
                reviewStatus="FOOTPRINT_AND_CARGO_LEDGER_VERIFIED",
            ),
        )
    atomic_write_json(
        OUT / "equipment-selection.json",
        dict(
            documents=len(chosen),
            parents=len({r["sourceDocumentId"] for r in chosen}),
            sizes=dict(Counter(s for r in chosen for s in r["afterSizes"])),
            inputHashes=hashes,
        ),
    )
    print("100 equipment-only replacements staged; recoverable dataset snapshot verified.")


if __name__ == "__main__":
    main()
