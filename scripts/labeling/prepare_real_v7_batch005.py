"""Freeze the reviewed top-up selection; no model calls and no source edits."""
import json
import random
from pathlib import Path

from run_real_v7_batch import ROOT, prepare
from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file

SCREEN = ROOT / "artifacts/kie-labeling/topup-screen-20261005/inventory.json"
DEST = ROOT / "artifacts/kie-labeling/direct-real-batch005-20261005"
# Source-review decisions for this finite selection, not extraction heuristics.
EXCLUDED = {
    "c6f1c9d1": "Four separately mass-accounted chemicals (5MT, 5MT, 5.01MT, 13MT).",
    "c09ab748": "Chocolate and glaze products have separate carton/bag quantities.",
    "c2180108": "Lubricant product rows have separate 50X, 150X and other quantities.",
    "dead9237": "Rider has six independently counted product rows (4200 EA etc.).",
    "f334c5b1": "16 DRUMS NEOSORB and 24 DRUMS LYCASIN are separately accounted.",
    "fdff56bf": "DG jerrican has separate mass accounting from the emulsion assortment.",
    "08c9266d": "Four individually counted/VIN-identified cars plus personal effects.",
    "e9524f55": "No consignee identity in OCR; only orphan CN email/tax continuation.",
    "21883640": "Long individually identified vehicle assortment: conservatively defer goods-scope review.",
}
REPLACEMENTS = ["00388da5", "00ea3f81", "188cb27f", "048dc59d", "049ad12d",
                "05b97047", "05fd9e37", "069cc830", "0750aae7"]


def main():
    inventory = json.loads(SCREEN.read_text())
    rng = random.Random(2026100505)
    selected = []
    for split, count in (("train", 154), ("validation", 13)):
        pool = sorted((r for r in inventory["documents"] if not r["exclusions"] and r["split"] == split), key=lambda r:r["documentId"])
        selected.extend(rng.sample(pool, count))
    assert set(EXCLUDED) <= {r["documentId"][4:12] for r in selected}
    selected = [r for r in selected if r["documentId"][4:12] not in EXCLUDED]
    for prefix in REPLACEMENTS:
        matches = [r for r in inventory["documents"] if r["documentId"][4:12] == prefix]
        assert len(matches) == 1 and not matches[0]["exclusions"] and matches[0]["split"] == "train"
        selected.append(matches[0])
    assert len(selected) == len({r["documentId"] for r in selected}) == 167
    narrowed = {**inventory, "parentInventorySha256": sha256_file(SCREEN),
                "selectionReview": EXCLUDED, "documents": selected,
                "selectionPolicy": "Seeded candidates with recorded source-review replacements; this is not an unbiased prevalence sample."}
    narrowed["replacementReview"] = {"01c3694f": "Not selected: OCR page 4 is a placeholder instead of extracted text; replaced before any model request."}
    path = ROOT / "artifacts/kie-labeling/topup-screen-20261005/selected-inventory-v2.json"
    atomic_publish_json(path, narrowed)
    prepare(DEST, train=154, validation=13, seed=2026100505, inventory=path)


if __name__ == "__main__":
    main()
