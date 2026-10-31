"""Compare the retained single-row path and the integrated renderer, without I/O mutation."""

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_campaign import create_campaign


def measure(batch, mode, repetitions):
    if mode == "baseline":
        config = json.loads((batch / "base-preparation.json").read_text())["config"]
    else:
        config = json.loads((batch / "prepared-config.json").read_text())
    started = time.perf_counter()
    campaign = create_campaign(Path.cwd(), config)
    initialization = time.perf_counter() - started
    sid = config["source_ids"][0]
    variants = [1] if mode == "baseline" else [1, 2, 3]
    samples = []
    for variant in variants:
        before = campaign.calls.spent
        candidate = campaign.render(sid, variant, persist=False)
        elapsed = []
        for _ in range(repetitions):
            started = time.perf_counter()
            repeated = campaign.render(sid, variant, persist=False)
            elapsed.append(time.perf_counter() - started)
            if digest(repeated) != digest(candidate):
                raise ValueError("benchmark replay is not deterministic")
        if campaign.calls.spent != before:
            raise ValueError("render benchmark made a provider call")
        samples.append(
            {
                "variant": variant,
                "medianMs": statistics.median(elapsed) * 1000,
                "minimumMs": min(elapsed) * 1000,
                "textSha256": candidate["joinedRawTextSha256"],
                "targetSha256": digest(candidate["target"]),
                "publicRows": len(
                    candidate["target"]["documentPatch"]["goodsItemDetails"][0][
                        "numberAndTypeOfPackages"
                    ]
                ),
            }
        )
    return {
        "mode": mode,
        "sourceId": sid,
        "repetitions": repetitions,
        "initializationSeconds": initialization,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "samples": samples,
        "nativeCalls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--mode", choices=["baseline", "integrated"])
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode:
        print(json.dumps(measure(args.batch, args.mode, args.repetitions)))
        return
    if args.output is None:
        parser.error("--output is required for a paired benchmark")
    runs = []
    for mode in ["baseline", "integrated", "integrated", "baseline"]:
        result = subprocess.run(
            [
                sys.executable,
                __file__,
                "--batch",
                str(args.batch),
                "--mode",
                mode,
                "--repetitions",
                str(args.repetitions),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        runs.append(json.loads(result.stdout))
    signatures = {(x["samples"][0]["textSha256"], x["samples"][0]["targetSha256"]) for x in runs}
    if len(signatures) != 1:
        raise ValueError("baseline and integrated single-row control differ")
    atomic_publish_json(args.output, {"singleRowTextAndTargetIdentical": True, "runs": runs})
    print(json.dumps({"singleRowTextAndTargetIdentical": True, "runs": runs}))


if __name__ == "__main__":
    main()
