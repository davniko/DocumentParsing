"""Screen unchanged lines moved horizontally by reviewed multi-region groups.

One source line-height is a review threshold, not an acceptance rule. A match
requires unique equal text inside its own recorded physical block. This isolates
column-membership changes from legitimate vertical reflow or anonymized widths.
"""

from __future__ import annotations

import json
import resource
import time
from collections import Counter
from pathlib import Path

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/analysis/r30-independent-audit-20261103"


def read(path):
    return json.loads(path.read_text())


def screen(source, rendered):
    findings = []
    for page in rendered.get("geometry", {}).get("pages", []):
        pi = page["page"]
        for block in page.get("blocks", []):
            originals = [source["pages"][pi]["regions"][i] for i in block["sourceRegions"]]
            emitted = block["physicalFragments"]
            before_counts = Counter(r["text"] for r in originals)
            after_counts = Counter(r["text"] for r in emitted)
            by_text = {r["text"]: r for r in emitted}
            moved = []
            for r in originals:
                if before_counts[r["text"]] != 1 or after_counts[r["text"]] != 1:
                    continue
                after = by_text[r["text"]]["box"]
                height = r["box"][3] - r["box"][1]
                displacement = (after[0] - r["box"][0]) / height
                if abs(displacement) > 1:
                    moved.append({"region": r["index"], "text": r["text"], "before": r["box"], "after": after, "horizontalLineHeights": displacement})
            if moved:
                findings.append({"page": pi, "owner": block["owner"], "sourceRegions": block["sourceRegions"], "moved": moved})
    return findings


def main():
    started = time.perf_counter()
    catalog_path = ROOT / "artifacts/synthesis-templates/mpci-bl-v7-paddle-reviewed/r30/catalog.json"
    catalog = read(catalog_path)
    controls = read(ROOT / "docs/analysis/r30-full-rebase-20261102/handoff-controls.json")
    findings, counts = [], Counter()
    for split in ("real-train", "validation"):
        path = ROOT / "data/curated/mpci-bl-r30-paddle-centroids-v1" / (split + ".jsonl")
        for row in map(json.loads, path.read_text().splitlines()):
            source = read(ROOT / "artifacts/paddle-migration/r30-full-rebase-20261102/sources" / (row["documentId"] + ".json"))
            evidence = ROOT / row["evidence"]["path"]
            if sha256_file(evidence) != row["evidence"]["sha256"]:
                raise ValueError("source evidence pin differs")
            result = screen(source, read(evidence))
            counts[split] += 1
            if result:
                findings.append({"split": split, "source": row["documentId"], "blocks": result})
    for sid, entry in catalog["templates"].items():
        template = read(ROOT / entry["template"]["path"])
        source = read(ROOT / template["source"]["path"])
        for pin in controls[sid]:
            path = ROOT / pin["path"]
            if sha256_file(path) != pin["sha256"]:
                raise ValueError("control evidence pin differs")
            row = read(path)
            result = screen(source, row)
            counts["template-controls"] += 1
            if result:
                findings.append({"split": "template-control", "source": sid, "control": row["documentId"], "blocks": result})
    report = {"status": "SCREEN_REQUIRES_COLUMN_ADJUDICATION", "counts": dict(counts), "findings": findings, "elapsedSeconds": time.perf_counter() - started, "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    atomic_write_json(OUT / "geometry-ownership-screen.json", report)
    print(json.dumps({**{k:v for k,v in report.items() if k != "findings"}, "flaggedRecords":len(findings), "sources":len({x['source'] for x in findings}), "bySplit":dict(Counter(x['split'] for x in findings))}, indent=2))


if __name__ == "__main__":
    main()
