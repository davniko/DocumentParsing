"""Deduplicate actual provider response IDs; account for successful and failed attempts."""

import argparse
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/kie-synthesis-production/r28-expansion2500-20261101"
OUT = ROOT / "docs/analysis/r28-expansion2500-20261101/final"


def main(output=OUT):
    seen, inventory, duplicates = {}, [], 0
    for path in sorted(RUN.glob("**/calls/*.json")):
        row = json.loads(path.read_text())
        if "stage" not in row:
            continue
        usage = row.get("usage") or {}
        ids = tuple(usage.get("providerResponseIds", []))
        if ids:
            overlaps = [key for key in ids if key in seen]
            if overlaps:
                previous = seen[overlaps[0]]
                if (
                    set(overlaps) != set(ids)
                    or previous["usage"] != usage
                    or previous["costUsd"] != row.get("costUsd")
                ):
                    raise ValueError("provider response overlaps inconsistent cost receipts")
                duplicates += 1
                continue
        entry = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "stage": row["stage"],
            "success": row.get("output") is not None,
            "usage": usage,
            "costUsd": row.get("costUsd"),
            "costStatus": row.get("costStatus"),
        }
        inventory.append(entry)
        for key in ids:
            seen[key] = entry

    def total(rows):
        return {
            "calls": len(rows),
            "providerRequests": sum(r["usage"].get("requests", 0) for r in rows),
            "inputTokens": sum(r["usage"].get("inputTokens", 0) for r in rows),
            "outputTokens": sum(r["usage"].get("outputTokens", 0) for r in rows),
            "recordedUsd": str(sum((Decimal(r["costUsd"] or "0") for r in rows), Decimal(0))),
            "missingCost": sum(r["costUsd"] is None for r in rows),
        }

    grouped = defaultdict(list)
    for row in inventory:
        grouped[row["stage"]].append(row)
    all_usage = total(inventory)
    result = {
        "status": "RECORDED_PROVIDER_USAGE",
        "allAttempts": all_usage,
        "successfulCallsIncludingRejectedDrafts": total([r for r in inventory if r["success"]]),
        "failedCalls": total([r for r in inventory if not r["success"]]),
        "byStage": {k: total(v) for k, v in sorted(grouped.items())},
        "duplicateReceiptsExcluded": duplicates,
        "usdPer1000AcceptedIncludingCampaignOverhead": str(
            Decimal(all_usage["recordedUsd"]) / Decimal("2.5")
        ),
        "scope": (
            "All retained campaign calls including retries, rejected proposals, preflights, "
            "wording corrections and both reviews. Excludes Codex account usage and local "
            "compute. Missing failed-call billing is not assumed zero; not a provider invoice."
        ),
    }
    atomic_publish_json(output / "native-cost-inventory.json", inventory)
    atomic_publish_json(output / "cost.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=OUT, help="New immutable accounting snapshot directory"
    )
    main(parser.parse_args().output)
