"""Offline route stress, mutation rejection, real-source leg replay and timing.

Uses the production sampler and renderer. Never generates wording or edits the
real dataset. The baseline is an explicit git revision of the scenario module.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
import types
from copy import deepcopy
from pathlib import Path

import yaml

from document_ocr.spatial_inputs.alignment import PAGE_MARKER, verify_preservation
from document_ocr.synthesis.curated import digest, save
from document_ocr.synthesis.curated_auxiliary import _render_recipe
from document_ocr.synthesis.curated_campaign import Campaign, apply_replacements
from document_ocr.synthesis.curated_routes import validate_route_values
from document_ocr.synthesis.curated_scenarios import SourceCapabilities
from document_ocr.synthesis.generators import DeterministicStream

ROOT = Path(__file__).resolve().parents[2]


def geometry_validation(campaign):
    """Recheck published suffixes/targets, bounded coordinates and all hub mentions."""
    position_dir = campaign.output / campaign.config["positions"]["output_subdirectory"]
    plain = {
        row["documentId"]: row
        for row in map(json.loads, (campaign.output / "dataset.jsonl").read_text().splitlines())
    }
    positioned = list(map(json.loads, (position_dir / "dataset.jsonl").read_text().splitlines()))
    assert len(positioned) == len(plain)
    assert {r["documentId"] for r in positioned} == set(plain)
    known = unknown = hub_mentions = 0
    for row in positioned:
        original = plain[row["documentId"]]
        assert all(row[k] == value for k, value in original.items())
        assert digest(row["positionedText"].encode()) == row["positionedTextSha256"]
        receipt = json.loads((position_dir / f"{row['documentId']}.json").read_text())
        expected = {
            n
            for n, line in enumerate(original["joinedRawText"].splitlines(), 1)
            if line.strip() and not PAGE_MARKER.fullmatch(line.strip())
        }
        assert {line["line_number"] for line in receipt["lines"]} == expected
        verify_preservation(
            original["joinedRawText"],
            row["positionedText"],
            expected,
        )
        text_lines = row["positionedText"].splitlines()
        hub = original["target"]["documentPatch"]["route"]["transshipmentPort"]["name"]
        for line in receipt["lines"]:
            suffix = " ||" if line["xy"] is None else f" || {line['xy'][0]},{line['xy'][1]}"
            assert text_lines[line["line_number"] - 1] == line["text"] + suffix
            if line["xy"] is None:
                unknown += 1
            else:
                assert len(line["xy"]) == 2
                assert all(type(v) is int and 0 <= v <= 1000 for v in line["xy"])
                known += 1
            if re.search(r"\b" + re.escape(hub) + r"\b", line["text"], re.I):
                assert line["xy"] is not None
                hub_mentions += 1
    manifest = json.loads((position_dir / "manifest.json").read_text())
    assert digest((position_dir / "dataset.jsonl").read_bytes()) == manifest["datasetSha256"]
    return dict(
        records=len(positioned),
        knownLines=known,
        unknownLines=unknown,
        positionedHubMentions=hub_mentions,
        pageModes=manifest["pageModes"],
        changedPoints=manifest["changedPoints"],
        datasetSha256=manifest["datasetSha256"],
        textAndTargetPreservation=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Pre-change git revision")
    args = parser.parse_args()
    config = yaml.safe_load(
        (ROOT / "configs/synthesis/mpci_bl_curated_v7_transshipment_pilot6.yaml").read_text()
    )
    started = time.perf_counter()
    campaign = Campaign(ROOT, config)
    cold = time.perf_counter() - started
    source_hashes = {
        split: digest((ROOT / config["dataset"] / f"{split}.jsonl").read_bytes())
        for split in ("train", "validation")
    }
    results = []
    for sid in config["source_ids"]:
        rows, elapsed = [], []
        origins, destinations, hubs, codes, quantities = set(), set(), set(), set(), set()
        rejected = 0
        source = campaign.rows[sid]
        for variant in range(1, 501):
            start = time.perf_counter()
            scenario = campaign.catalog.sample(
                source,
                seed=config["seed"],
                variant=variant,
                capabilities=campaign.capabilities[sid],
            )
            elapsed.append(time.perf_counter() - start)
            target = apply_replacements(source["target"], scenario.replacements)["documentPatch"]
            validate_route_values(source["target"]["documentPatch"], target, scenario.replacements)
            repeated = campaign.catalog.sample(
                source,
                seed=config["seed"],
                variant=variant,
                capabilities=campaign.capabilities[sid],
            )
            assert repeated == scenario
            nodes = scenario.provenance["geographyEligibility"]["routeNodes"]
            assert len({node["registry_id"] for node in nodes.values()}) == len(nodes)
            origins.add(scenario.origin.country_code)
            destinations.add(scenario.destination.country_code)
            hubs.add(scenario.route_locations["transshipmentPort"].registry_id)
            codes.update(identity.hs6 for identity in scenario.goods_identities)
            quantities.add(
                target["goodsItemDetails"][0]["numberAndTypeOfPackages"][0]["packageQuantity"]
            )
            for mutation in ("stale", "missing", "invented_country"):
                changed = deepcopy(target)
                if mutation == "stale":
                    changed["route"]["transshipmentPort"]["name"] = source["target"][
                        "documentPatch"
                    ]["route"]["transshipmentPort"]["name"]
                    # A sampled port can coincidentally equal the old one;
                    # that is not corruption. Force a non-sampled sentinel.
                    changed["route"]["transshipmentPort"]["name"] += " WRONG"
                elif mutation == "missing":
                    del changed["route"]["transshipmentPort"]
                else:
                    changed["route"]["transshipmentPort"]["country"] = "UNPRINTED"
                try:
                    validate_route_values(
                        source["target"]["documentPatch"], changed, scenario.replacements
                    )
                except ValueError:
                    rejected += 1
                else:
                    raise AssertionError(mutation)
            if variant <= config["variants_per_source"]:
                rows.append({"variant": variant, "route": target["route"], "nodes": nodes})
        results.append(
            dict(
                sourceDocumentId=sid,
                draws=500,
                mutationRejections=rejected,
                origins=sorted(origins),
                destinations=sorted(destinations),
                hubs=sorted(hubs),
                hs6Count=len(codes),
                quantityCount=len(quantities),
                examples=rows,
                medianSamplingMs=statistics.median(elapsed) * 1000,
            )
        )
    # Exact real-source repeated feeder declaration; no new public vessel field.
    feeder = next(r for sid, r in campaign.rows.items() if sid.startswith("doc_978a3990"))
    source_text = "BIANCA RAMBOW 941 S"
    assert feeder["joinedRawText"].count(source_text) == 2
    recipe = {"transport_leg": "precarriage", "source_voyage": "941 S"}
    _, scenario, _, target = campaign.plan(config["source_ids"][0], 1)
    generated = [
        _render_recipe(
            recipe,
            scenario,
            target,
            DeterministicStream(config["seed"], "leg-probe", feeder["documentId"]),
            source_text,
            {30},
            campaign.vessels,
        )
        for _ in range(2)
    ]
    assert generated[0] == generated[1] and generated[0] != source_text
    # Compare exact old/new direct-route draws, then interleave timing batches.
    code = subprocess.check_output(
        ["git", "show", f"{args.baseline}:src/document_ocr/synthesis/curated_scenarios.py"],
        cwd=ROOT,
        text=True,
    )
    module = types.ModuleType("_baseline_curated_scenarios")
    sys.modules[module.__name__] = module
    exec(compile(code, "<baseline>", "exec"), module.__dict__)
    old = object.__new__(module.ScenarioCatalog)
    old.__dict__.update(campaign.catalog.__dict__)
    for attribute in ("ports", "localities"):
        setattr(
            old,
            attribute,
            {
                country: tuple(
                    module.ScenarioLocation.model_validate(v.model_dump()) for v in values
                )
                for country, values in getattr(old, attribute).items()
            },
        )
    old._classification_contexts = {}
    old._donor_cache = {}
    old._identity_pool_cache = {}
    direct_config = yaml.safe_load(
        (ROOT / "configs/synthesis/mpci_bl_curated_v7_registry_pilot72.yaml").read_text()
    )
    sid = direct_config["source_ids"][0]
    source = campaign.rows[sid]
    policy = direct_config["capabilities"][sid]
    old_cap, new_cap = (
        module.SourceCapabilities.model_validate(policy),
        SourceCapabilities.model_validate(policy),
    )
    for variant in range(1, 101):
        a = old.sample(source, seed=direct_config["seed"], variant=variant, capabilities=old_cap)
        b = campaign.catalog.sample(
            source, seed=direct_config["seed"], variant=variant, capabilities=new_cap
        )
        assert a.model_dump(mode="json") == b.model_dump(mode="json")
    timings = {"before": [], "after": []}
    for cycle in range(8):
        for label in ("before", "after") if cycle % 2 == 0 else ("after", "before"):
            instance, cap = (old, old_cap) if label == "before" else (campaign.catalog, new_cap)
            start = time.perf_counter()
            for variant in range(1, 101):
                instance.sample(
                    source, seed=direct_config["seed"], variant=variant, capabilities=cap
                )
            timings[label].append((time.perf_counter() - start) * 10)
    memory = {}
    for label, instance, cap in (
        ("before", old, old_cap),
        ("after", campaign.catalog, new_cap),
    ):
        gc.collect()
        tracemalloc.start()
        for variant in range(1, 101):
            instance.sample(source, seed=direct_config["seed"], variant=variant, capabilities=cap)
        retained, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory[label] = dict(retainedBytes=retained, peakIncrementalBytes=peak)
    report = dict(
        sourceHashes=source_hashes,
        scenarios=results,
        feederProbe={
            "sourceDocumentId": feeder["documentId"],
            "source": source_text,
            "occurrences": 2,
            "generated": generated,
            "scope": "source-only leg recipe, not whole-document certification",
        },
        directRouteUnchangedDraws=100,
        directComparison="complete serialized scenario including provenance",
        directMedianMs={k: statistics.median(v) for k, v in timings.items()},
        directWarmPythonAllocations=memory,
        geometry=geometry_validation(campaign),
        coldCatalogSeconds=cold,
        processPeakRssMiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        baselineRevision=args.baseline,
    )
    assert source_hashes == {
        split: digest((ROOT / config["dataset"] / f"{split}.jsonl").read_bytes())
        for split in source_hashes
    }
    save(campaign.output / "offline-validation.json", report)
    print(
        json.dumps(
            {
                "report": str(campaign.output / "offline-validation.json"),
                "draws": sum(r["draws"] for r in results),
                "rejectedMutations": sum(r["mutationRejections"] for r in results),
                **{k: v for k, v in report.items() if k not in {"scenarios", "sourceHashes"}},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
