"""Execute one bounded native certification batch without automatic admission.

Each stage keeps its own receipts. Failed proposals and review findings remain
held for explicit review/redraw; completing this script is not certification.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from check_package_alias_candidates import check

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.curated_campaign import create_campaign
from document_ocr.synthesis.independent_preparation import prepare_campaign


async def run(path, *, prepared=False):
    root = Path.cwd()
    config = json.loads(path.read_text())
    directory = path.parent / "stage-runs" / str(time.time_ns())
    if prepared:
        prepared_path = path.resolve()
        planned = ready = len(config["source_ids"])
    else:
        preparation = await prepare_campaign(root, config, **config["independent_preparation"])
        atomic_publish_json(directory / "preparation.json", preparation)
        if preparation["preparedConfig"] is None:
            print(json.dumps({"batch": str(path), "status": "ALL_PROPOSALS_HELD"}), flush=True)
            return
        prepared_path = root / preparation["preparedConfig"]["path"]
        planned, ready = preparation["plannedSlots"], preparation["readySlots"]
    campaign = create_campaign(root, json.loads(prepared_path.read_text()))
    stages = {}
    for name, operation in (
        ("generate", campaign.generate_all),
        ("postal", campaign.correct_postal_all),
    ):
        result = await operation()
        atomic_publish_json(directory / f"{name}.json", result)
        stages[name] = {"errors": result.get("errors", [])}
    rendered = campaign.render_all()
    atomic_publish_json(directory / "render.json", rendered)
    stages["render"] = {"accepted": len(rendered["accepted"]), "errors": rendered["errors"]}
    reviewed = await campaign.review_all()
    atomic_publish_json(directory / "review.json", reviewed)
    stages["review"] = reviewed
    stages["geometry"] = await asyncio.to_thread(check, prepared_path, directory / "geometry")
    result = {
        "config": str(path),
        "preparedConfig": str(prepared_path.relative_to(root)),
        "planned": planned,
        "prepared": ready,
        "stages": stages,
        "status": "AWAITING_MAIN_AGENT_REVIEW",
    }
    atomic_publish_json(directory / "summary.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", action="store_true", help="Resume an existing prepared draw")
    args = parser.parse_args()
    asyncio.run(run(args.config, prepared=args.prepared))


if __name__ == "__main__":
    main()
