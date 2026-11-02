"""Execute explicitly reviewed commodity-wording corrections, with native replay."""

import json
import subprocess
from collections import defaultdict

from manage_r28_expansion import goods_plan
from stage_combination_refresh import ART, OUT, ROOT

from document_ocr.atomic import atomic_publish_json


def main():
    decisions = json.loads((OUT / "reviewed-native-decisions.json").read_text())["regenerate"]
    grouped = defaultdict(dict)
    for nid, reason in decisions.items():
        matches = list(ART.glob(f"round-*/*/campaign/candidates/{nid}.json"))
        if len(matches) != 1:
            raise ValueError("expected one current candidate for " + nid)
        grouped[matches[0].parents[2]][nid] = reason
    for directory, corrections in grouped.items():
        config = max(
            (directory / "campaign/declared-package-preparation").glob("*/prepared-config.json")
        )
        report = directory / "reviewed-wording-corrections.json"
        atomic_publish_json(
            report, dict(status="MAIN_REVIEWED_PRODUCT_CORRECTIONS", goodsCorrections=corrections)
        )
        goods_plan(config, report)
        plan = report.with_name(report.stem + "-goods-plan.json")
        with (directory / "correction.log").open("w") as stream:
            subprocess.run(
                [
                    str(ROOT / ".venv/bin/python"),
                    "scripts/synthesis/regenerate_reviewed_goods.py",
                    "--config",
                    str(config),
                    "--plan",
                    str(plan),
                ],
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
            subprocess.run(
                [
                    str(ROOT / ".venv/bin/python"),
                    "scripts/synthesis/run_package_alias_batch.py",
                    "--config",
                    str(config),
                    "--prepared",
                    "--concurrency",
                    "4",
                ],
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        latest = max((config.parent / "stage-runs").glob("*/summary.json"))
        atomic_publish_json(
            directory / "corrected-stage-summary.json", json.loads(latest.read_text())
        )
        print(directory, "corrected and rerendered", flush=True)


if __name__ == "__main__":
    main()
