"""Read back fresh outputs and replay; do not convert review status to acceptance."""

import json
import math
import resource
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/synthesis-probes/r30-posttraining-20261108"
OUT = ROOT / "docs/analysis/r30-party-boundary-audit-20261108"


def main():
    started = time.perf_counter()
    report = json.loads((ART / "native/campaign.json").read_text())
    catalog = PaddleTemplateCatalog(ROOT, report["catalog"])
    rows, regions, replays = [], 0, 0
    for entry in report["results"]:
        path = ROOT / entry["path"]
        assert sha256_file(path) == entry["sha256"]
        r = json.loads(path.read_text())
        candidate = ART / "sampling/candidates" / path.name
        wording_path = ART / "wording-reviewed" / path.name
        wording = json.loads(wording_path.read_text()) if wording_path.exists() else None
        replay = catalog.render(candidate, wording_receipt=wording)
        assert replay == r
        replays += 1
        for page in r["pages"]:
            for region in page["regions"]:
                x0, y0, x1, y1 = region["box"]
                assert all(math.isfinite(v) for v in region["box"])
                assert 0 <= x0 < x1 <= page["width"] and 0 <= y0 < y1 <= page["height"]
                regions += 1
        rows.append(
            {
                "id": r["documentId"],
                "parent": r["sourceDocumentId"],
                "regions": sum(len(p["regions"]) for p in r["pages"]),
                "cargo": r["target"]["goodsItemDetails"],
                "containers": r["target"].get("containerInformation", []),
            }
        )
    calls = list((ART / "sampling/calls").glob("*.json")) + list(
        (ART / "wording/provider/calls").glob("*.json")
    )
    costs = Decimal(0)
    usage, stages = Counter(), Counter()
    for path in calls:
        row = json.loads(path.read_text())
        costs += Decimal(row["costUsd"])
        stages[row["stage"]] += 1
        for key in (
            "requests",
            "inputTokens",
            "outputTokens",
            "cacheReadTokens",
            "reasoningTokens",
        ):
            usage[key] += row["usage"].get(key, 0)
    result = {
        "freshAttempted": 8,
        "rendered": len(rows),
        "held": report["failures"],
        "exactFreshReplays": replays,
        "validBoxRegions": regions,
        "samples": rows,
        "allAttemptCostUsd": str(costs),
        "allAttemptUsage": usage,
        "allAttemptStages": stages,
        "seconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "publication": "none; audit artifacts only; frozen training data unchanged",
    }
    (OUT / "fresh-synthesis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
