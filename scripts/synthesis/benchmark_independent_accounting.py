"""Measure the unchanged complete-region ledger on a fixed CPU workload."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import resource
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis import independent_numeric as numeric


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3000)
    args = parser.parse_args()
    blueprint = SimpleNamespace(
        document_id="benchmark-fixed-region",
        target={"documentPatch": {}},
        ownership_data={
            numeric.NUMERIC_LEDGER_OWNER: {
                "certificate": {
                    "regions": [{"role": "packing_clause", "source_text": "PACKING"}],
                    "wordingOwners": [],
                }
            }
        },
    )
    draw = {
        "internalCategory": "PACKAGE_CRATE",
        "package": {"packageQuantity": 37},
        "physical": {"unitNetKg": "80", "unitGrossKg": "95"},
    }
    plan = {"privatePackingRelation": "CONTAINS", "relatedPackageType": "PACKAGE_CARTON"}
    samples = []
    for _ in range(7):
        start = time.perf_counter()
        for index in range(args.iterations):
            result = numeric._build_scope_ledger(blueprint, draw, plan, 73, index)
            assert result["publicQuantity"] == 37
        samples.append((time.perf_counter() - start) * 1000 / args.iterations)
    atomic_publish_json(
        args.output,
        {
            "iterationsPerRepeat": args.iterations,
            "repeats": len(samples),
            "medianMsPerLedger": statistics.median(samples),
            "samplesMs": samples,
            "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "numericModuleSha256": hashlib.sha256(
                Path(inspect.getfile(numeric)).read_bytes()
            ).hexdigest(),
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
