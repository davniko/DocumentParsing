"""Stage a reviewed description-only correction, preserving all other receipt owners."""

import argparse
import copy
import json

from migrate_r30_synthetic import ART, DATA, OUT, ROOT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog
from document_ocr.synthesis.scoped_cargo_refresh import apply_edits, reconstruct_source


def repair(candidate, decision):
    result = copy.deepcopy(candidate)
    key, before, after = (decision[k] for k in ("key", "before", "after"))
    if not after.strip() or after == before or not decision["reason"].strip():
        raise ValueError("review must provide a nonempty changed description and reason")
    edits = [e for e in result["proof"]["edits"] if e["key"] == key]
    if not edits or any(
        e["after"] != before
        or e["targetPaths"] != ["documentPatch.goodsItemDetails[0].description"]
        for e in edits
    ):
        raise ValueError("review must match every occurrence of one exclusive description owner")
    goods = result["target"]["documentPatch"]["goodsItemDetails"]
    if len(goods) != 1 or goods[0]["description"] != " ".join(before.upper().split()):
        raise ValueError("reviewed text does not match the retained description target")
    for edit in edits:
        edit["after"] = after
    for field in ("renderValues", "surfaceValues"):
        values = result.get(field, {})
        if key in values:
            if values[key] != before:
                raise ValueError("description surface has a different authority")
            values[key] = after
    source = reconstruct_source(candidate)
    raw = apply_edits(source, result["proof"]["edits"]).decode()
    result.update(joinedRawText=raw, joinedRawTextSha256=digest(raw.encode()))
    result["proof"]["renderedSha256"] = digest(raw.encode())
    goods[0]["description"] = " ".join(after.upper().split())
    if reconstruct_source(result) != source:
        raise ValueError("description correction changed the original source")
    return result


def main(review_path):
    review = read(review_path)
    if review["status"] != "MAIN_AGENT_REVIEWED_DESCRIPTION_REPAIR":
        raise ValueError("explicit semantic review is required")
    identity = review["documentId"]
    inventory = read(OUT / "receipt-inventory.json")
    entry = next(e for e in inventory["records"] if e["id"] == identity)
    old_pin = {"path": entry["candidate"], "sha256": entry["candidateSha256"]}
    if review["candidate"] != old_pin:
        raise ValueError("description review belongs to another receipt")
    original = read(verify_pin(old_pin))
    row = next(
        r
        for r in map(json.loads, verify_pin(inventory["baseline"]).open())
        if r["documentId"] == identity
    )
    if original["documentId"] != identity or original["joinedRawText"] != row["joinedRawText"]:
        raise ValueError("description repair requires an exact current receipt")
    candidate = repair(original, review)
    candidate["migrationDescriptionRepair"] = {
        "previousCandidate": old_pin,
        "review": pin(review_path),
        "scope": "One reviewed product-wording owner only; no inferred regex cleanup.",
    }
    replacement = copy.deepcopy(row)
    replacement["joinedRawText"] = candidate["joinedRawText"]
    replacement["target"]["goodsItemDetails"][0]["description"] = candidate["target"][
        "documentPatch"
    ]["goodsItemDetails"][0]["description"]
    folder = ART / "description-repairs" / identity
    path = folder / "candidate.json"
    atomic_publish_json(path, candidate)
    atomic_publish_json(folder / "replacement.json", replacement)
    catalog = PaddleTemplateCatalog(ROOT, read(DATA / "manifest.json")["templateCatalog"])
    if catalog.wording_specification(path) is not None:
        raise ValueError("additional wording needs an explicit context review before transfer")
    rendered = catalog.render(path, current=replacement)
    atomic_publish_json(folder / "rendered.json", rendered)
    atomic_publish_json(
        folder / "proposal.json",
        {
            "status": "DESCRIPTION_REPAIR_REQUIRES_FINAL_READBACK",
            "id": identity,
            "currentRecordSha256": digest(row),
            "candidate": pin(path),
            "replacementRecord": pin(folder / "replacement.json"),
            "rendered": pin(folder / "rendered.json"),
            "review": pin(review_path),
        },
    )
    print(folder / "proposal.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review")
    main(ROOT / parser.parse_args().review)
