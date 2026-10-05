"""Prioritize fresh, low-overlap sources; inventory is selection evidence, not annotation approval."""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter

from run_real_v7_batch import ROOT, SOURCE
from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file

OUT = ROOT / "artifacts/kie-labeling/batch006-screen-20261005"
BASE = ROOT / "data/curated/mpci-bl-real-v7-reviewed-r11-reduced-472"


def normalized(value):
    return re.sub(r"[^A-Z0-9]", "", unicodedata.normalize("NFKD", value).upper())


def signatures(record):
    p = record["target"]["documentPatch"]
    parties = p.get("parties", {})
    names = [parties.get(role, {}).get("name", "") for role in ("shipper", "consignee")]
    names = [re.sub(r"^TO (?:THE )?ORDER(?: OF)?\s*", "", name, flags=re.I) for name in names]
    goods = " ".join(g.get("description", "") for g in p.get("goodsItemDetails", []))
    return tuple(map(normalized, [*names, goods]))


def main():
    inventory = json.loads((OUT / "base-inventory.json").read_text())
    old = {r["documentId"]: r for r in map(json.loads, SOURCE.read_text().splitlines())}
    current = [r for split in ("train", "validation")
               for r in map(json.loads, (BASE / f"{split}.jsonl").read_text().splitlines())]
    seen = [set(), set(), set()]
    for r in current + [old[r["documentId"]] for r in current]:
        for i, key in enumerate(signatures(r)):
            if key:
                seen[i].add(key)
    candidates = []
    for d in inventory["documents"]:
        r = old[d["documentId"]]
        patch = r["target"]["documentPatch"]
        keys = signatures(r)
        reasons = list(d["exclusions"])
        if not any(keys[:2]):
            reasons.append("old_labels_missing_both_primary_parties")
        if not patch.get("billOfLadingNumber"):
            reasons.append("old_bill_number_missing_select_more_complete_source")
        if "amoy.polygonscan.com" in r["joinedRawText"].lower():
            reasons.append("electronic_bl_test_network_wrapper")
        reviewed_multiple = {
            "245db42f": "Four veal products have separate carton counts 551/280/349/73.",
            "c6f1c9d1": "Four chemicals have separate product quantities 5/5/5.01/13 MT.",
        }
        if d["documentId"][4:12] in reviewed_multiple:
            reasons.append("source_review_separate_product_accounting")
        counters = re.findall(r"(?im)^.*\b(?:PAGE|SHEET)\s*[:#]?\s*(\d+)\s*(?:/|OF)\s*(\d+).*$", r["joinedRawText"])
        if counters and max(int(b) for a, b in counters) > d["pages"]:
            reasons.append("printed_page_counter_exceeds_supplied_pdf_review")
        candidates.append(d | dict(exclusions=reasons, keys=keys,
            baselineOverlap=[name for i, name in enumerate(("shipper", "consignee", "goods"))
                             if keys[i] and keys[i] in seen[i]], pageCounters=counters,
            sourceReview=reviewed_multiple.get(d["documentId"][4:12])))
    ranked = []
    for split in ("train", "validation"):
        remaining = [d for d in candidates if d["split"] == split and not d["exclusions"]]
        batch_seen = [set(), set(), set()]
        while remaining:
            def score(d):
                overlaps = sum(bool(k and k in batch_seen[i]) for i, k in enumerate(d["keys"]))
                return len(d["baselineOverlap"]), overlaps, d["pages"], d["documentId"]
            selected = min(remaining, key=score)
            selected["rankWithinSplit"] = len([d for d in ranked if d["split"] == split]) + 1
            selected["earlierRankOverlap"] = [name for i, name in enumerate(("shipper", "consignee", "goods"))
                if selected["keys"][i] and selected["keys"][i] in batch_seen[i]]
            ranked.append(selected)
            remaining.remove(selected)
            for i, key in enumerate(selected["keys"]):
                if key:
                    batch_seen[i].add(key)
    result = dict(sourceSha256=sha256_file(SOURCE), baseline=str(BASE.relative_to(ROOT)),
        baselineFiles={f"{s}.jsonl": sha256_file(BASE / f"{s}.jsonl") for s in ("train", "validation")},
        normalization="Unicode decomposition, case and punctuation; leading TO ORDER instructions removed from entity keys. No company/product semantic equivalence inferred.",
        eligible=dict(Counter(d["split"] for d in ranked)), documents=candidates, ranked=ranked)
    atomic_publish_json(OUT / "novelty-inventory-v2.json", result)
    for split in ("train", "validation"):
        rows = [d for d in ranked if d["split"] == split]
        print(split, len(rows), "no baseline overlap", sum(not d["baselineOverlap"] for d in rows))
        for d in rows[:175 if split == "train" else 13]:
            print(d["rankWithinSplit"], d["documentId"][4:12], d["pages"], d["baselineOverlap"],
                  d["earlierRankOverlap"], d["oldDescription"][:110])


if __name__ == "__main__":
    main()
