"""Offline registry/physical invariants and diversity audit on actual campaign inputs."""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_scenarios import (
    ScenarioCatalog,
    ScenarioSamplingConfig,
    SourceCapabilities,
    _hs6,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--draws-per-source", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    cfg = yaml.safe_load(args.config.read_text())
    dataset = root / cfg["dataset"]
    train = [json.loads(line) for line in (dataset / "train.jsonl").read_text().splitlines()]
    validation = [
        json.loads(line) for line in (dataset / "validation.jsonl").read_text().splitlines()
    ]
    start = time.perf_counter()
    catalog = ScenarioCatalog.load(
        root,
        ScenarioSamplingConfig.model_validate(cfg["sampling"]),
        train,
        frozenset(r["documentId"] for r in validation),
    )
    loading = time.perf_counter() - start
    observed_codes = {
        c
        for r in train
        for g in r["target"]["documentPatch"].get("goodsItemDetails", [])
        for c in _hs6(g)
    }
    before = digest(train)
    groups = defaultdict(list)
    start = time.perf_counter()
    for sid, policy in cfg["capabilities"].items():
        cap = SourceCapabilities.model_validate(policy)
        row = catalog.train[sid]
        old = row["target"]["documentPatch"]["goodsItemDetails"][0]
        for variant in range(1, args.draws_per_source + 1):
            scenario = catalog.sample(row, seed=cfg["seed"], variant=variant, capabilities=cap)
            goods = scenario.cargo["goodsItemDetails"][0]
            codes = [i.hs6 for i in scenario.goods_identities]
            assert len(set(codes)) == cap.identity_count
            assert all(c in catalog.phrases for c in codes)
            assert set(goods) == set(old), "invented/dropped public goods fields"
            if "hsCodes" in goods:
                assert goods["hsCodes"] == codes
            placements = goods.get("splitGoodsPlacement", [])
            if placements and all("packageQuantity" in p for p in placements):
                assert (
                    sum(p["packageQuantity"] for p in placements)
                    == goods["numberAndTypeOfPackages"][0]["packageQuantity"]
                )
            settings = scenario.provenance.get("thermalCommodityContext")
            if settings:
                for c in scenario.cargo["containerInformation"]:
                    assert c["typeCategory"] == "REFRIGERATED"
                    if "temperatureSetpoint" in c:
                        s = c["temperatureSetpoint"]
                        value = s["value"] if s["unit"] == "celsius" else (s["value"] - 32) / 1.8
                        assert abs(value - settings["temperatureCelsius"]) < 1e-8
                assert settings["sampledHS6"] == codes
                if settings["basis"] == "hs_registry_food_profile":
                    assert all(c in catalog.thermal_codes[cap.family] for c in codes)
                    assert settings["ventilationCbmPerHour"] == "0"
            dg = scenario.provenance.get("dgPrintedFacts")
            if dg:
                assert all(g["unNumber"] == dg["unNumber"] for g in goods["dangerousGoods"])
            groups[cap.family].append(
                {
                    "source": sid,
                    "variant": variant,
                    "hs6": codes,
                    "donor": scenario.provenance["donorDocumentId"],
                    "package": goods["numberAndTypeOfPackages"][0]["typeCategory"],
                    "equipment": sorted(
                        {
                            c.get("sizeCategory", "") + "|" + c.get("typeCategory", "")
                            for c in scenario.cargo["containerInformation"]
                        }
                    ),
                    "thermal": settings,
                    "dg": dg,
                    "attempts": scenario.provenance["candidateAttempts"],
                }
            )
    elapsed = time.perf_counter() - start
    assert before == digest(train), "sampling changed training inputs"
    summary = {}
    for family, rows in groups.items():
        codes = {c for r in rows for c in r["hs6"]}
        summary[family] = {
            "draws": len(rows),
            "distinctHS6": len(codes),
            "HS6AbsentFromTraining": sorted(codes - observed_codes),
            "packageCounts": dict(Counter(r["package"] for r in rows)),
            "temperaturesCelsius": sorted(
                {r["thermal"]["temperatureCelsius"] for r in rows if r["thermal"]}
            ),
            "maxAttempts": max(r["attempts"] for r in rows),
        }
    report = {
        "config": str(args.config),
        "summary": summary,
        "registryThermalDomainSizes": {k: len(v) for k, v in catalog.thermal_codes.items()},
        "chemicalCandidates": len(catalog.chemical_candidates),
        "loadSeconds": loading,
        "samplingSeconds": elapsed,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "trainingUnchanged": True,
        "draws": dict(groups),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "draws"}, indent=2))


if __name__ == "__main__":
    main()
