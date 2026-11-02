"""Snapshot completed replacement jobs for manual review, without admission."""

import argparse
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/kie-synthesis-production/r28-expansion2500-20261101"
OUT = ROOT / "docs/analysis/r28-expansion2500-20261101"


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--round", type=int, default=2)
    p.add_argument("--batches", nargs="+", type=int)
    a = p.parse_args()
    results, paths = [], []
    for receipt in sorted(RUN.glob(f"batch-*/**/redraw-{a.round}/execution-receipt.json")):
        batch = int(receipt.relative_to(RUN).parts[0].split("-")[1])
        if a.batches and batch not in a.batches:
            continue
        directory = receipt.parent
        prepared = sorted((directory / "campaign/declared-package-preparation").glob("*/prepared-config.json"))
        if not prepared:
            results.append({"component": str(directory.relative_to(RUN)), "preparation": "NO_PREPARED_DRAW"})
            continue
        cfg = read(prepared[-1])
        candidates = []
        for cp in sorted((directory / "campaign/candidates").glob("*.json")):
            row = read(cp)
            paths.append(str(cp.relative_to(ROOT)))
            goods = row["target"]["documentPatch"]["goodsItemDetails"][0]
            scope = RUN / "scoped-cargo-review/reviews" / (cp.stem + "-" + sha256_file(cp) + ".json")
            candidates.append({"id": cp.stem, "source": row["sourceDocumentId"][4:12],
                "description": goods["description"], "packages": goods["numberAndTypeOfPackages"],
                "scopedFindings": read(scope)["review"]["findings"] if scope.exists() else None})
        reviews = {rp.stem[4:12]: read(rp)["review"]["findings"]
                   for rp in sorted((directory / "campaign/reviews").glob("*.json"))}
        results.append({"component": str(directory.relative_to(RUN)),
            "preparedConfig": str(prepared[-1].relative_to(ROOT)),
            "preparedSources": [s[4:12] for s in cfg["source_ids"]],
            "candidates": candidates, "nativeFindings": {s: f for s, f in reviews.items() if f}})
    atomic_publish_json(OUT / "redraw-reviews" / (a.name + "-inventory.json"), results)
    atomic_publish_json(OUT / "redraw-reviews" / (a.name + "-selection.json"), paths)
    print(json.dumps({"jobs": len(results), "candidates": len(paths), "inventory": a.name}))


if __name__ == "__main__":
    main()
