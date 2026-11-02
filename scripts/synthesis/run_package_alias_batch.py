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
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated_campaign import create_campaign
from document_ocr.synthesis.independent_preparation import prepare_campaign


def credit_refusal_pins(output: Path) -> list[dict]:
    """Authorize only recorded account refusals after an explicit credit top-up."""
    pins = []
    for path in sorted((output / "calls").glob("*.json")):
        row = json.loads(path.read_text())
        error = row.get("error") or ""
        if row.get("output") is None and "status_code: 402" in error:
            pins.append({"path": str(path), "sha256": sha256_file(path)})
    return pins


def transient_refusal_pins(output: Path) -> list[dict]:
    """Retry transport/rate refusals, never invalid model content or arbitrary errors."""
    pins = []
    for path in sorted((output / "calls").glob("*.json")):
        row = json.loads(path.read_text())
        error = row.get("error") or ""
        if row.get("output") is None and any(
            f"status_code: {code}" in error for code in (429, 502, 503, 504)
        ):
            pins.append({"path": str(path), "sha256": sha256_file(path)})
    return pins


async def run(
    path,
    *,
    prepared=False,
    retry_credit_refusals=False,
    retry_transient_refusals=False,
    concurrency=None,
):
    root = Path.cwd()
    config = json.loads(path.read_text())
    if concurrency is not None:
        config["concurrency"] = concurrency
    directory = path.parent / "stage-runs" / str(time.time_ns())
    pins = credit_refusal_pins(root / config["output"]) if retry_credit_refusals else []
    if retry_credit_refusals:
        atomic_publish_json(directory / "authorized-credit-retries.json", pins)
    if retry_transient_refusals:
        transient = transient_refusal_pins(root / config["output"])
        atomic_publish_json(directory / "authorized-transient-retries.json", transient)
        pins.extend(transient)
    if prepared:
        prepared_path = path.resolve()
        planned = ready = len(config["source_ids"])
    else:
        policy = dict(config["independent_preparation"])
        if pins:
            policy["retry_failed_receipt_sha256s"] = [p["sha256"] for p in pins]
        preparation = await prepare_campaign(root, config, **policy)
        atomic_publish_json(directory / "preparation.json", preparation)
        if preparation["preparedConfig"] is None:
            print(json.dumps({"batch": str(path), "status": "ALL_PROPOSALS_HELD"}), flush=True)
            return
        prepared_path = root / preparation["preparedConfig"]["path"]
        planned, ready = preparation["plannedSlots"], preparation["readySlots"]
    campaign = create_campaign(root, json.loads(prepared_path.read_text()))
    if concurrency is not None:
        # Runner owns the semaphore; use a fresh runner with the same provider,
        # cache, budget and request contracts, changing throughput only.
        from document_ocr.synthesis.curated import Runner

        campaign.calls = Runner(root, {**campaign.calls.config, "concurrency": concurrency})
    campaign.calls.retry_failed_receipt_sha256s = frozenset(p["sha256"] for p in pins)
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
    parser.add_argument(
        "--retry-credit-refusals",
        action="store_true",
        help="After a top-up, retry exact cached HTTP 402 receipts once, preserving history.",
    )
    parser.add_argument("--retry-transient-refusals", action="store_true")
    parser.add_argument("--concurrency", type=int, choices=range(1, 9))
    args = parser.parse_args()
    asyncio.run(
        run(
            args.config,
            prepared=args.prepared,
            retry_credit_refusals=args.retry_credit_refusals,
            retry_transient_refusals=args.retry_transient_refusals,
            concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
