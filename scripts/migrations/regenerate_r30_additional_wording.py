"""Regenerate explicitly reviewed one-off wording failures without template special cases."""

import argparse
import asyncio
import json

from migrate_r30_synthetic import ART, OUT, ROOT, read
from publish_r30_sources import pin

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.curated import CuratedSynthesisConfig, Runner, digest
from document_ocr.synthesis.paddle_wording import propose


def validate_decisions(decisions, specs):
    if not isinstance(decisions, dict) or not decisions or not set(decisions) <= set(specs):
        raise ValueError("retry decisions must name a nonempty subset of current missing owners")
    if any(not isinstance(reason, str) or not reason.strip() for reason in decisions.values()):
        raise ValueError("every retry needs explicit textual review feedback")


async def main(batch, decisions_path):
    staged = read(OUT / "batches" / f"{batch:03d}-staging.json")
    specs = {s["documentId"]: s for s in staged["newWording"]}
    decisions = read(decisions_path)
    validate_decisions(decisions, specs)
    config = read(ROOT / "configs/synthesis/mpci_bl_curated_v7_paddle.json")
    config = {k: config[k] for k in CuratedSynthesisConfig.model_fields}
    config.update(
        output=str((ART / "wording-retry-provider").relative_to(ROOT)), concurrency=8, budget_usd=2
    )
    runner = Runner(ROOT, config)
    outputs = []
    for identity, feedback in decisions.items():
        spec = specs[identity]
        before = ART / "wording-proposals" / (identity + ".json")
        spec["context"]["reviewFeedback"] = feedback
        proposal = await propose(runner, spec)
        proposal["regeneration"] = {"previous": pin(before), "feedback": feedback}
        path = ART / "wording-retries" / identity / (digest(proposal) + ".json")
        atomic_publish_json(path, proposal)
        outputs.append(pin(path))
        print(
            json.dumps({"id": identity, "values": proposal["values"]}, ensure_ascii=False),
            flush=True,
        )
    atomic_publish_json(
        OUT / "batches" / f"{batch:03d}-wording-retries.json",
        {"decisions": pin(decisions_path), "proposals": outputs},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=int)
    parser.add_argument("decisions")
    args = parser.parse_args()
    asyncio.run(main(args.batch, ROOT / args.decisions))
