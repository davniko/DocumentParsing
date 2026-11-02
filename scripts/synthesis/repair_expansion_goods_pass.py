"""Execute explicitly authored product-only corrections, keeping all sampled facts."""

import argparse
import asyncio
import json
import subprocess
from pathlib import Path

from manage_r28_expansion import goods_plan

from document_ocr.atomic import atomic_publish_json

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/kie-synthesis-production/r28-expansion2500-20261101"
OUT = ROOT / "docs/analysis/r28-expansion2500-20261101"


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--name", required=True)
    a = p.parse_args()
    selections = json.loads(a.plan.read_text())
    for component, corrections in selections.items():
        config = sorted((RUN / component / "campaign/declared-package-preparation").glob("*/prepared-config.json"))[-1]
        out = OUT / a.name / component
        review = out / "goods-review.json"
        atomic_publish_json(review, {
            "status": "MAIN_REVIEWED_PRODUCT_CORRECTIONS", "goodsCorrections": corrections,
            "basis": str(a.plan), "scope": "Only the explicitly reviewed product owners; all other wording, sampled facts and extraction presence remain unchanged.",
        })
        goods_plan(config, review)
        plan = review.with_name("goods-review-goods-plan.json")
        with (out / "regeneration.log").open("xb") as stream:
            result = await asyncio.to_thread(subprocess.run, [
                str(ROOT / ".venv/bin/python"), "scripts/synthesis/regenerate_reviewed_goods.py",
                "--config", str(config), "--plan", str(plan),
            ], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("goods correction failed; inspect " + str(out))
        print(component, "CORRECTED_REQUIRES_REPLAY_REVIEW", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
