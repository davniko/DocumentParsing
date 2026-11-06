"""Independent semantic proposals over rendered migration batches; no publication."""

import argparse
import asyncio
import json
from pathlib import Path

from audit_r30_independent import SYSTEM, Review
from migrate_r30_synthetic import ART, DATA, OUT, ROOT, read
from publish_r30_sources import pin, verify_pin
from r30_provider_resume import authorize_rejected_calls
from r30_semantic_review_cache import rebind_review, rebound_review_name

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.country_presentations import DEFAULT_REGISTRY_MANIFEST
from document_ocr.synthesis.curated import CuratedSynthesisConfig, Runner, digest


def account_blocked(error):
    """Runner serializes provider errors; only an explicit account rejection stops the queue."""
    return any(f"ModelHTTPError: status_code: {code}," in str(error) for code in (401, 402, 403))


def require_review_ready(staged, ids):
    """Do not buy whole-batch reviews while known mechanical holds block it."""
    if ids:
        selected = set(ids)
        if len(selected) != len(ids) or not selected <= {e["id"] for e in staged["results"]}:
            raise ValueError("scoped review identities must be distinct rendered records")
    elif staged["failures"] or staged["selected"] != 100 or len(staged["results"]) != 100:
        raise ValueError("whole-batch semantic review requires complete mechanical readiness; "
                         "resolve holds first or explicitly select a scoped diagnostic")


async def review_pending_entries(entries, one, width):
    """Refill bounded review workers without a slowest-request chunk barrier.

    An account rejection stops dispatch immediately. Already submitted requests
    finish and retain their receipts; unstarted identities remain explicit.
    """
    if type(width) is not int or width <= 0:
        raise ValueError("review concurrency must be a positive integer")
    remaining = iter(enumerate(entries))
    started, failures = set(), {}
    blocked = False

    async def worker():
        nonlocal blocked
        while not blocked:
            item = next(remaining, None)
            if item is None:
                return
            index, entry = item
            started.add(index)
            try:
                await one(entry)
            except Exception as exc:
                failures[index] = {"id": entry["id"], "error": str(exc)}
                if account_blocked(exc):
                    blocked = True

    await asyncio.gather(*(worker() for _ in range(min(width, len(entries)))))
    return (
        [failures[i] for i in sorted(failures)],
        blocked,
        [e["id"] for i, e in enumerate(entries) if i not in started],
    )


def collect_review_cohorts(cohorts):
    """Pin and validate every selection before sharing one bounded provider queue."""
    entries, staging, seen = [], {}, set()
    if not cohorts:
        raise ValueError("at least one review cohort is required")
    for batch, ids in sorted(cohorts.items()):
        path = OUT / "batches" / f"{batch:03d}-staging.json"
        staged = read(path)
        require_review_ready(staged, ids)
        selected = [e for e in staged["results"] if not ids or e["id"] in ids]
        identities = {e["id"] for e in selected}
        if len(identities) != len(selected) or seen & identities:
            raise ValueError("review identities must be distinct across cohorts")
        entries.extend(selected)
        seen.update(identities)
        staging[str(batch)] = pin(path)
    return entries, staging


async def main(batch, max_output_tokens=8192, ids=(), resume_rejected=False, budget_usd=2,
               *, cohorts=None):
    if cohorts is not None and (batch is not None or ids):
        raise ValueError("choose either one batch or explicit review cohorts")
    selections = cohorts if cohorts is not None else {batch: ids}
    entries, staging = collect_review_cohorts(selections)
    cfg = read(ROOT / "configs/synthesis/mpci_bl_curated_v7_paddle.json")
    cfg = {k: cfg[k] for k in CuratedSynthesisConfig.model_fields}
    cfg.update(
        output=str((ART / "semantic-provider").relative_to(ROOT)),
        concurrency=8,
        budget_usd=budget_usd,
    )
    cfg["provider"] = {
        **cfg["provider"],
        "reasoning_effort": "medium",
        "max_output_tokens": max_output_tokens,
    }
    runner = Runner(ROOT, cfg)
    spent_before = runner.spent
    authorized = authorize_rejected_calls(runner) if resume_rejected else []
    policy = (ROOT / "docs/kie-real-baseline-label-policy-2026-10-05.md").read_text()
    policy = (
        policy[: policy.index("## Current published baseline: R12")]
        + policy[policy.index("## Source-owner repair clarifications") :]
    )
    schema = read(DATA / "schema.json")
    names = read(ROOT / Path(DEFAULT_REGISTRY_MANIFEST).parent / "preferred-names.json")["names"]
    receipts = []

    async def one(entry):
        row = read(verify_pin(entry["rendered"]))
        identity = row["documentId"]
        prompt = {
            "policy": policy,
            "schema": schema,
            "preferredCountryNames": names,
            "split": "synthetic-paddle-migration",
            "target": row["target"],
            "pages": row["pages"],
            "scope": (
                "Independently read the actual rendered Paddle regions and target. "
                "Existing shipment facts were retained; new owners are allowed only where "
                "native source templates expose them. Assess final evidence, not how familiar "
                "the goods/package pairing is. Report concrete field-boundary, digit, ownership, "
                "omission, source-remnant or semantic contradictions; "
                "not optional stylistic improvements."
            ),
        }
        request = digest(
            {
                "prompt": prompt,
                "system": SYSTEM,
                "schema": Review.model_json_schema(),
                "provider": cfg["provider"],
            }
        )
        path = ART / "semantic-reviews" / identity / (request + ".json")
        previous = ART / "semantic-reviews" / (identity + ".json")
        cached = previous if previous.exists() and read(previous)["requestSha256"] == request else path
        if cached.exists():
            old = read(cached)
            if old["requestSha256"] != request:
                raise ValueError("semantic receipt belongs to different rendered content")
            if old["rendered"] != entry["rendered"]:
                # Provenance-only replays have new artifact hashes, but do not
                # change the model's input. Preserve the paid review and rebind
                # its evidence after exact comparison; never bypass adjudication.
                reused = rebind_review(old, read(verify_pin(old["rendered"])), row,
                                       entry["rendered"], request, pin(cached))
                cached = path.with_name(rebound_review_name(reused))
                atomic_publish_json(cached, reused)
            receipts.append(pin(cached))
            return
        result = await runner.call(
            "r30-synthetic-migration-review",
            identity,
            Review,
            SYSTEM,
            json.dumps(prompt, ensure_ascii=False, separators=(",", ":")),
            output_retries=1,
        )
        invalid = []
        for i, finding in enumerate(result.findings):
            for quote in finding.evidence:
                try:
                    literal = row["pages"][quote.page]["regions"][quote.region]["text"]
                except IndexError:
                    literal = ""
                if quote.text not in literal:
                    invalid.append({"finding": i, "quote": quote.model_dump()})
        atomic_publish_json(
            path,
            {
                "status": "PROPOSED_REQUIRES_MAIN_AGENT_ADJUDICATION",
                "id": identity,
                "source": row["sourceDocumentId"],
                "rendered": entry["rendered"],
                "requestSha256": request,
                "review": result.model_dump(),
                "invalidEvidence": invalid,
            },
        )
        receipts.append(pin(path))
        print(
            json.dumps(
                {"id": identity, "findings": len(result.findings), "invalidEvidence": len(invalid)}
            ),
            flush=True,
        )

    failures, blocked, unattempted = await review_pending_entries(
        entries, one, cfg["concurrency"]
    )
    report = {
        **({"staging": staging[str(batch)], "ids": sorted(ids)} if cohorts is None else {
            "stagingByBatch": staging,
            "idsByBatch": {str(n): sorted(selected) for n, selected in selections.items()},
        }),
        "maxOutputTokens": max_output_tokens,
        "reviews": sorted(receipts, key=lambda p: p["path"]),
        "failures": failures,
        "allAttemptCostUsd": str(runner.spent - spent_before),
        "cumulativeProviderCostUsd": str(runner.spent),
        "providerAccountBlocked": blocked,
        "unattempted": unattempted,
        "authorizedRejectedReceipts": authorized,
        "cumulativeBudgetUsd": budget_usd,
    }
    # A resumed request set can complete more reviews. Preserve the interrupted
    # report rather than colliding with its immutable path or overwriting it.
    report_identity = digest(report)
    atomic_publish_json(
        OUT / "batches" / f"{f'{batch:03d}' if cohorts is None else 'queue'}-semantic-proposals-{report_identity}.json",
        report,
    )
    if blocked:
        raise RuntimeError("Provider account blocked; report saved, no subsequent batch submitted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=int)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--resume-rejected", action="store_true")
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=2,
        help="Explicit cumulative budget for the retained provider ledger",
    )
    args = parser.parse_args()
    asyncio.run(
        main(args.batch, args.max_output_tokens, args.ids, args.resume_rejected, args.budget_usd)
    )
