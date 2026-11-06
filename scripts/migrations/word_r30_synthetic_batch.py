"""Bounded additional-owner wording proposals; never automatic acceptance."""

import argparse
import asyncio
import json

from migrate_r30_synthetic import ART, OUT, ROOT, read
from r30_provider_resume import authorize_rejected_calls

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.curated import CuratedSynthesisConfig, Runner, digest
from document_ocr.synthesis.paddle_wording import propose


async def main(batch, resume_rejected=False):
    staged = read(OUT / "batches" / f"{batch:03d}-staging.json")
    config = read(ROOT / "configs/synthesis/mpci_bl_curated_v7_paddle.json")
    config = {k: config[k] for k in CuratedSynthesisConfig.model_fields}
    config.update(
        output=str((ART / "wording-provider").relative_to(ROOT)), concurrency=8, budget_usd=2
    )
    runner = Runner(ROOT, config)
    spent_before = runner.spent
    authorized = authorize_rejected_calls(runner) if resume_rejected else []
    errors, outputs = [], []

    async def one(spec):
        destination = ART / "wording-proposals" / (spec["documentId"] + ".json")
        if destination.exists():
            previous = read(destination)
            if previous["candidateSha256"] != spec["candidateSha256"]:
                raise ValueError("existing proposal belongs to a different receipt")
            outputs.append(str(destination.relative_to(ROOT)))
            return
        try:
            result = await propose(runner, spec)
            atomic_publish_json(destination, result)
            outputs.append(str(destination.relative_to(ROOT)))
            print(
                json.dumps(
                    {"id": spec["documentId"], "values": result["values"]}, ensure_ascii=False
                ),
                flush=True,
            )
        except Exception as exc:
            errors.append({"id": spec["documentId"], "error": str(exc)})

    specs = staged["newWording"]
    for start in range(0, len(specs), config["concurrency"]):
        await asyncio.gather(*(one(s) for s in specs[start : start + config["concurrency"]]))
    report = {
        "batch": batch,
        "proposals": outputs,
        "errors": errors,
        "allAttemptCostUsd": str(runner.spent - spent_before),
        "cumulativeProviderCostUsd": str(runner.spent),
        "authorizedRejectedReceipts": authorized,
        "status": "REQUIRES_MAIN_AGENT_READBACK",
    }
    atomic_publish_json(
        OUT / "batches" / f"{batch:03d}-wording-proposals-{digest(report)}.json",
        report,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=int)
    parser.add_argument("--resume-rejected", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.batch, args.resume_rejected))
