"""Inventory fresh campaign drafts and run the separate, non-admitting cargo review."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from audit_r28_multirow_wording import run

from document_ocr.atomic import atomic_publish_json

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/kie-synthesis-production/r28-expansion2500-20261101"
OUT = ROOT / "docs/analysis/r28-expansion2500-20261101"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--retry-transient-refusals", action="store_true")
    parser.add_argument("--concurrency", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--batch-size", type=int, choices=range(1, 7), default=6)
    args = parser.parse_args()
    paths, campaigns = [], []
    for number in range(args.start, args.end + 1):
        batch = RUN / f"batch-{number:02d}"
        directories = [batch] if number == 1 else [batch / "ordinary", batch / "thermal"]
        for directory in directories:
            campaign = directory / "campaign"
            if not (campaign / "candidates").exists():
                continue
            candidates = sorted((campaign / "candidates").glob("*.json"))
            paths.extend(str(p.relative_to(ROOT)) for p in candidates)
            campaigns.append({"path": str(campaign.relative_to(ROOT)), "drafts": len(candidates)})
    if not paths:
        raise ValueError("no draft candidates in selected batches")
    selection = OUT / "scoped-reviews" / (args.name + "-selection.json")
    atomic_publish_json(selection, paths)
    atomic_publish_json(selection.with_name(args.name + "-scope.json"), campaigns)
    asyncio.run(
        run(
            ROOT / "configs/synthesis/mpci_bl_curated_v7_multirow.json",
            selection,
            RUN / "scoped-cargo-review",
            retry_transient_refusals=args.retry_transient_refusals,
            concurrency=args.concurrency,
            batch_size=args.batch_size,
        )
    )


if __name__ == "__main__":
    main()
