"""Offline, reproducible coverage, accounting and CPU timing for the 24-source pilot."""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import yaml

from document_ocr.synthesis.curated import (
    SourceContract,
    digest,
    flat,
    render,
    save,
    validate_candidate,
)

ROOT = Path(__file__).resolve().parents[2]


def main():
    cfg = yaml.safe_load((ROOT / "configs/synthesis/mpci_bl_curated_v7_pilot24.yaml").read_text())
    output = ROOT / cfg["output"]
    rows = {
        row["documentId"]: row
        for row in map(json.loads, (ROOT / cfg["dataset"] / "train.jsonl").read_text().splitlines())
    }
    records = [json.loads(p.read_text()) for p in sorted((output / "samples").glob("*.json"))]
    contracts = {
        sid: SourceContract.model_validate(
            json.loads((output / "sources" / sid / "contract.json").read_text())["contract"]
        )
        for sid in cfg["source_ids"]
    }
    counters, containers, variation = Counter(), Counter(), Counter()
    for record in records:
        sid = record["sourceDocumentId"]
        target = record["target"]["documentPatch"]
        leaves = flat(target)
        containers[len(target.get("containerInformation", []))] += 1
        counters["dangerousGoodsDocuments"] += any("dangerousGoods" in p for p in leaves)
        counters["temperatureDocuments"] += any("temperature" in p.lower() for p in leaves)
        counters["correctedDocuments"] += bool(record["lexicalCorrections"])
        counters["addressLabels"] += sum(p.endswith("addressLine") for p in leaves)
        counters["singleGoodsDocuments"] += len(target["goodsItemDetails"]) == 1
        changed_kinds = {
            v.kind for v in contracts[sid].variables if record["values"][v.key] != v.value
        }
        variation.update(changed_kinds)

    # Same renderer, two workloads: identity source replay versus generated replay.
    # This is not a throughput comparison against the incompatible historical V5 flow.
    timings = defaultdict(list)
    for _ in range(3):
        for mode in ("identity", "generated", "validate"):
            started = time.perf_counter()
            for record in records:
                sid = record["sourceDocumentId"]
                if mode == "validate":
                    validate_candidate(rows[sid], contracts[sid], record)
                else:
                    values = (
                        {v.key: v.value for v in contracts[sid].variables}
                        if mode == "identity"
                        else record["values"]
                    )
                    render(rows[sid], contracts[sid], values)
            timings[mode].append(time.perf_counter() - started)

    costs, stages = [], defaultdict(lambda: dict(calls=0, confirmed=Decimal(0)))
    for path in sorted((output / "calls").glob("*.json")):
        receipt = json.loads(path.read_text())
        stage = stages[receipt["stage"]]
        stage["calls"] += 1
        if receipt.get("usage"):
            confirmed = Decimal(receipt["costUsd"])
            allowance, basis = confirmed, "provider_reported"
            stage["confirmed"] += confirmed
        elif "status_code: 404" in str(receipt.get("error")):
            confirmed, allowance, basis = Decimal(0), Decimal(0), "request_rejected"
        else:
            confirmed = Decimal(0)
            allowance = Decimal(receipt["costUsd"])
            basis = "recorded_timeout_reservation"
            if allowance == 0:
                # Three early compiler timeouts predate cost reservation. Their
                # 24k output cap costs <= $0.018 at the recorded price ceiling;
                # $0.05/request conservatively also allows >200k prompt tokens.
                if receipt["stage"] != "compile":
                    raise ValueError(f"unaccounted unknown-cost request: {path}")
                allowance, basis = Decimal("0.05"), "early_timeout_conservative_allowance"
        costs.append(
            dict(
                receipt=str(path.relative_to(ROOT)),
                receiptSha256=digest(path.read_bytes()),
                confirmedUsd=str(confirmed),
                allowanceUsd=str(allowance),
                basis=basis,
            )
        )
    confirmed = sum((Decimal(c["confirmedUsd"]) for c in costs), Decimal(0))
    allowance = sum((Decimal(c["allowanceUsd"]) for c in costs), Decimal(0))
    inventory = json.loads(
        (ROOT / "docs/analysis/synthesis-restart-20261006/proposed-pilot.json").read_text()
    )
    report = dict(
        samples=len(records),
        sources=len(contracts),
        carrierFamilies=len(inventory["carrierFamilies"]),
        cohorts=inventory["cohorts"],
        containerCountDistribution=dict(sorted(containers.items())),
        coverage=dict(counters),
        documentsWithChangedVariableKind=dict(variation),
        cpuSecondsFor72={
            mode: dict(trials=values, median=statistics.median(values))
            for mode, values in timings.items()
        },
        costs=dict(
            requests=len(costs),
            providerConfirmedUsd=str(confirmed),
            includingUnknownAllowanceUsd=str(allowance),
            confirmedPerSampleUsd=str(confirmed / len(records)),
            allowancePerSampleUsd=str(allowance / len(records)),
            stages={
                key: dict(calls=value["calls"], confirmedUsd=str(value["confirmed"]))
                for key, value in stages.items()
            },
            receipts=costs,
            note="Unknown timeout charges require provider billing reconciliation; not an invoice.",
        ),
    )
    save(output / "final-audit-statistics.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "costs"}, indent=2))
    print(f"Provider confirmed ${confirmed}; including timeout allowance ${allowance}")


if __name__ == "__main__":
    main()
