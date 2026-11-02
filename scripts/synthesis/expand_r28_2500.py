"""Stage and inspect the approved R28 expansion; admission remains explicit."""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from document_ocr.atomic import atomic_publish_json, atomic_write_json
from document_ocr.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/analysis/r28-expansion2500-20261101"
CAMPAIGNS = ROOT / "artifacts/kie-synthesis-production/r28-expansion2500-20261101"
BASE = ROOT / "configs/synthesis/mpci_bl_curated_v7_multirow.json"
DATA = ROOT / "data/curated/mpci-bl-r28-positions-v1"


def selected_config(base, counts, output, seed, *, group_counts=(1,), thermal_modes=None):
    config = copy.deepcopy(base)
    config.update(
        source_ids=list(counts),
        capabilities={
            sid: {**base["capabilities"][sid], "destination_country": "AE"} for sid in counts
        },
        output=str(output.relative_to(ROOT)),
        seed=seed,
        variants_per_source=max(counts.values()),
        concurrency=8,
        budget_usd=15,
        template_sampling={"samples": sum(counts.values()), "source_counts": counts},
    )
    config["declared_packages"]["group_counts"] = list(group_counts)
    config["declared_packages"]["source_group_counts"] = {
        sid: values
        for sid, values in base["declared_packages"]["source_group_counts"].items()
        if sid in counts
    }
    config["independent_preparation"]["thermal_goods_modes"] = thermal_modes or {}
    inventory = config["sampling"].get("equipment_inventory")
    if inventory:
        inventory["source_ids"] = [sid for sid in inventory["source_ids"] if sid in counts]
        if not inventory["source_ids"]:
            config["sampling"].pop("equipment_inventory")
    return config


def thermal_preflight(attempt=1):
    base = json.loads(BASE.read_bytes())
    audit = json.loads(
        (ROOT / "docs/analysis/r28-thermal-freight-probe-20261031/audit.json").read_bytes()
    )
    sources = audit["sourceInventory"]
    vents = [s["documentId"] for s in sources if s["ventilation"]]
    frozen = [s["documentId"] for s in sources if s["family"] == "frozen"][:3]
    chilled = [
        s["documentId"] for s in sources if s["family"] == "chilled" and not s["ventilation"]
    ][:3]
    counts = {sid: 3 for sid in [*vents, *frozen, *chilled]}
    assert len(counts) == 12
    name = "thermal-preflight" if attempt == 1 else f"thermal-preflight-{attempt}"
    path = CAMPAIGNS / name / "config.json"
    config = selected_config(
        base,
        counts,
        path.parent / "campaign",
        2026110100 + attempt,
        group_counts=(1, 2, 1),
        thermal_modes={
            **dict.fromkeys(vents, "produce"),
            **dict.fromkeys(chilled, "other"),
        },
    )
    atomic_publish_json(path, config)
    atomic_publish_json(
        OUT / f"{name}-plan.json",
        {
            "status": "STAGED_NOT_ACCEPTED",
            "config": str(path.relative_to(ROOT)),
            "sourceCounts": counts,
            "records": 36,
            "configurationSha256": sha256_file(path),
            "datasetBefore": {p.name: sha256_file(p) for p in DATA.iterdir() if p.is_file()},
            "scope": "Six ventilation, three frozen, three other chilled sources; "
            "three descendants each; one/two public package rows.",
            "payableElsewhere": "excluded from requested expansion",
        },
    )
    print(path.relative_to(ROOT))


def stage():
    """Solve source counts before calls, then split controlled slots into 100s."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    base = json.loads(BASE.read_bytes())
    audit = json.loads(
        (ROOT / "docs/analysis/r28-new2500-postrefresh-20261031/audit.json").read_bytes()
    )
    details = {
        sid: row
        for sid, row in audit["sourceDetails"].items()
        if not row["features"]["payableElsewhere"]
    }
    route_audit = json.loads(
        (ROOT / "docs/analysis/r28-thermal-freight-probe-20261031/audit.json").read_bytes()
    )
    for source in route_audit["sourceInventory"]:
        if source["documentId"] in details:
            details[source["documentId"]]["features"]["transshipment"] = source["transshipment"]
    ids = sorted(details)
    size = len(ids)
    rows, lows, highs = [], [], []

    def constrain(predicate, low, high):
        rows.append([int(predicate(details[sid])) for sid in ids] + [0] * size)
        lows.append(low)
        highs.append(high)

    constrain(lambda s: True, 2500, 2500)
    for families, count in [
        ({"dg_chemical", "dg_vehicle"}, 560),
        ({"frozen"}, 260),
        ({"chilled"}, 200),
        ({"ambient", "vehicle"}, 1480),
    ]:
        constrain(lambda s, families=families: s["family"] in families, count, count)
    for feature, low, high in [
        ("forwarding", 360, 380),
        ("transshipment", 400, 400),
        ("dgPackingGroup", 200, 560),
        ("dgSubsidiary", 80, 560),
        ("dgFlashPoint", 100, 560),
        ("multiDG", 60, 75),
        ("multiContainer", 800, 900),
        ("delivery", 900, 1100),
        ("allocationLinksWithoutQuantities", 100, 200),
        ("containersWithoutAnySizeOrType", 100, 200),
        ("partialAllocations", 20, 25),
        ("masterBL", 15, 20),
        ("vesselIMO", 30, 50),
        ("tank", 25, 30),
        ("ventilationText", 100, 100),
    ]:
        constrain(lambda s, f=feature: s["features"][f], low, high)
    # Minimize absolute deviation from equal contributions within each primary
    # cohort. Feature deficits can override balance; arbitrary IDs cannot.
    families = Counter(s["family"] for s in details.values())
    averages = {
        "ambient": 1480 / (families["ambient"] + families["vehicle"]),
        "vehicle": 1480 / (families["ambient"] + families["vehicle"]),
        "dg_chemical": 560 / (families["dg_chemical"] + families["dg_vehicle"]),
        "dg_vehicle": 560 / (families["dg_chemical"] + families["dg_vehicle"]),
        "frozen": 260 / families["frozen"],
        "chilled": 200 / families["chilled"],
    }
    for index, sid in enumerate(ids):
        average = averages[details[sid]["family"]]
        for sign in (-1, 1):
            row = [0] * (2 * size)
            row[index], row[size + index] = sign, -1
            rows.append(row)
            lows.append(-np.inf)
            highs.append(sign * average)
    lower = [3 if sid in audit["unusedSources"] else 1 for sid in ids] + [0] * size
    result = milp(
        c=np.array([0] * size + [1] * size, dtype=float),
        integrality=np.array([1] * size + [0] * size),
        bounds=Bounds(lower, [50] * size + [np.inf] * size),
        constraints=LinearConstraint(np.array(rows, dtype=float), lows, highs),
        options={"time_limit": 30},
    )
    if not result.success:
        raise ValueError("Requested joint source schedule is not solved: " + result.message)
    counts = dict(zip(ids, map(round, result.x[:size]), strict=True))
    assert sum(counts.values()) == 2500
    # Round-robin by source reduces sibling group size and keeps each batch
    # diverse. Cohort order lets ambient work run while thermal preflight closes.
    rng = random.Random(2026110125)
    slots = []
    for group in [{"ambient", "vehicle"}, {"dg_chemical", "dg_vehicle"}, {"frozen", "chilled"}]:
        selected = [sid for sid in ids if details[sid]["family"] in group]
        rng.shuffle(selected)
        for turn in range(max(counts[sid] for sid in selected)):
            slots.extend(
                {"sourceDocumentId": sid, "groups": 1} for sid in selected if counts[sid] > turn
            )
    for group, target in [({"dg_chemical", "dg_vehicle"}, 120), ({"frozen", "chilled"}, 80)]:
        eligible = [
            i
            for i, row in enumerate(slots)
            if details[row["sourceDocumentId"]]["family"] in group
            and row["sourceDocumentId"] not in base["declared_packages"]["source_group_counts"]
        ]
        for i in rng.sample(eligible, target):
            slots[i]["groups"] = 2
    mixed_sources = set()
    for line in (DATA / "synthetic-train.jsonl").open():
        record = json.loads(line)
        pairs = {
            (e.get("sizeCategory"), e.get("typeCategory"))
            for e in record["target"].get("containerInformation", [])
        }
        if len(pairs) > 1 and all(None not in p for p in pairs):
            mixed_sources.add(record["sourceDocumentId"])
    pins = {p.name: sha256_file(p) for p in DATA.iterdir() if p.is_file()}
    for name, expected in pins.items():
        destination = OUT / "before" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(DATA / name, destination)
        if sha256_file(destination) != expected:
            raise ValueError("Pinned R28 pre-expansion snapshot differs")
    batches = []
    for offset in range(0, 2500, 100):
        number = offset // 100 + 1
        records = slots[offset : offset + 100]
        source_counts = Counter(row["sourceDocumentId"] for row in records)
        sequences = {sid: [] for sid in source_counts}
        for row in records:
            sid = row["sourceDocumentId"]
            sequences[sid].append(row["groups"])
            row["variant"] = len(sequences[sid])
        path = CAMPAIGNS / f"batch-{number:02d}" / "config.json"
        modes = {
            sid: "produce" if details[sid]["features"]["ventilationText"] else "other"
            for sid in source_counts
            if details[sid]["family"] == "chilled"
        }
        config = selected_config(
            base, source_counts, path.parent / "campaign", 2026110200 + number, thermal_modes=modes
        )
        config["declared_packages"]["source_group_counts"] = sequences
        config["independent_preparation"].update(
            mixed_source_ids=sorted(set(source_counts) & mixed_sources),
            mixed_probability=0.5,
        )
        atomic_publish_json(path, config)
        batches.append(
            {
                "batch": number,
                "config": str(path.relative_to(ROOT)),
                "configSha256": sha256_file(path),
                "records": records,
                "sourceCounts": source_counts,
            }
        )
    summary = {
        feature: sum(counts[sid] for sid in ids if details[sid]["features"][feature])
        for feature in next(iter(details.values()))["features"]
    }
    manifest = {
        "status": "STAGED_NOT_GENERATED_NOT_ACCEPTED",
        "records": 2500,
        "beforeFiles": pins,
        "baseConfigSha256": sha256_file(BASE),
        "parents": len(ids),
        "unusedParents": len(set(ids) & set(audit["unusedSources"])),
        "sourceCounts": counts,
        "featurePlan": summary,
        "batches": batches,
        "newPackageRowModes": Counter(r["groups"] for r in slots),
        "scope": "UAE import; independent sampling; no payable-elsewhere sources; R28 append only",
    }
    atomic_publish_json(OUT / "selection.json", manifest)
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k not in {"batches", "sourceCounts"}}, indent=2
        )
    )


def rebalance():
    """Preserve the running first batch; cap subsequent sibling groups efficiently."""
    manifest = json.loads((OUT / "selection.json").read_bytes())
    if manifest.get("balanced"):
        raise ValueError("The approved schedule has already been balanced")
    atomic_publish_json(OUT / "initial-selection.json", manifest)
    base = json.loads(BASE.read_bytes())
    pending = {}
    for batch in manifest["batches"][1:]:
        for row in batch["records"]:
            pending.setdefault(row["sourceDocumentId"], []).append(row)
    buckets = [[] for _ in range(24)]
    source_counts = [Counter() for _ in buckets]
    for sid, records in sorted(pending.items(), key=lambda item: (-len(item[1]), item[0])):
        for row in records:
            index = min(
                (i for i in range(24) if len(buckets[i]) < 100),
                key=lambda i: (
                    source_counts[i][sid],
                    len(buckets[i]),
                    (i + int(sid[4:12], 16)) % 24,
                ),
            )
            buckets[index].append({"sourceDocumentId": sid, "groups": row["groups"]})
            source_counts[index][sid] += 1
    mixed = set()
    modes = {}
    for batch in manifest["batches"]:
        original = json.loads((ROOT / batch["config"]).read_bytes())
        mixed.update(original["independent_preparation"]["mixed_source_ids"])
        modes.update(original["independent_preparation"]["thermal_goods_modes"])
    batches = [manifest["batches"][0]]
    for index, records in enumerate(buckets, 2):
        assert len(records) == 100
        counts = Counter(row["sourceDocumentId"] for row in records)
        if max(counts.values()) > 3:
            raise ValueError("Balanced source groups exceed three variants")
        sequences = {sid: [] for sid in counts}
        for row in records:
            sid = row["sourceDocumentId"]
            sequences[sid].append(row["groups"])
            row["variant"] = len(sequences[sid])
        path = CAMPAIGNS / f"batch-{index:02d}" / "balanced" / "config.json"
        config = selected_config(
            base,
            counts,
            path.parent / "campaign",
            2026110300 + index,
            thermal_modes={sid: modes[sid] for sid in counts if sid in modes},
        )
        config["declared_packages"]["source_group_counts"] = sequences
        config["independent_preparation"].update(
            mixed_source_ids=sorted(set(counts) & mixed),
            mixed_probability=0.5,
        )
        atomic_publish_json(path, config)
        batches.append(
            {
                "batch": index,
                "config": str(path.relative_to(ROOT)),
                "configSha256": sha256_file(path),
                "records": records,
                "sourceCounts": counts,
            }
        )
    manifest.update(
        batches=batches,
        balanced=True,
        balanceReason="At most three sibling variants per source per batch; "
        "avoid large wording calls and sibling rejection amplification.",
    )
    atomic_write_json(OUT / "selection.json", manifest)
    print(
        json.dumps(
            {"batches": 25, "records": 2500, "maximumSiblingCount": 3, "firstBatchUnchanged": True}
        )
    )


def split_cohorts():
    """Preserve every selected slot while thermal preflight and ambient work run separately."""
    manifest = json.loads((OUT / "selection.json").read_text())
    if manifest.get("cohortConfigs"):
        raise ValueError("cohort schedule already exists")
    atomic_publish_json(OUT / "pre-cohort-selection.json", manifest)
    entries = []
    for batch in manifest["batches"][1:]:
        original = json.loads((ROOT / batch["config"]).read_text())
        for phase in ("ordinary", "thermal"):
            ids = [
                sid
                for sid in original["source_ids"]
                if (original["capabilities"][sid]["family"] in {"frozen", "chilled"})
                == (phase == "thermal")
            ]
            if not ids:
                continue
            config = copy.deepcopy(original)
            config["source_ids"] = ids
            config["capabilities"] = {sid: config["capabilities"][sid] for sid in ids}
            counts = {sid: config["template_sampling"]["source_counts"][sid] for sid in ids}
            config["template_sampling"] = {"samples": sum(counts.values()), "source_counts": counts}
            config["declared_packages"]["source_group_counts"] = {
                sid: config["declared_packages"]["source_group_counts"][sid] for sid in ids
            }
            prep = config["independent_preparation"]
            prep["mixed_source_ids"] = [s for s in prep["mixed_source_ids"] if s in ids]
            prep["thermal_goods_modes"] = {
                s: "other" if mode == "non_respiring" else mode
                for s, mode in prep["thermal_goods_modes"].items()
                if s in ids
            }
            if "equipment_inventory" in config["sampling"]:
                inventory = config["sampling"]["equipment_inventory"]
                inventory["source_ids"] = [s for s in inventory["source_ids"] if s in ids]
                if not inventory["source_ids"]:
                    config["sampling"].pop("equipment_inventory")
            path = CAMPAIGNS / f"batch-{batch['batch']:02d}" / phase / "config.json"
            config["output"] = str((path.parent / "campaign").relative_to(ROOT))
            atomic_publish_json(path, config)
            entries.append(
                {
                    "batch": batch["batch"],
                    "phase": phase,
                    "config": str(path.relative_to(ROOT)),
                    "configSha256": sha256_file(path),
                    "planned": sum(counts.values()),
                }
            )
    manifest["cohortConfigs"] = entries
    assert sum(e["planned"] for e in entries) == 2400
    atomic_write_json(OUT / "selection.json", manifest)
    print(json.dumps(entries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["thermal-preflight", "stage", "rebalance", "split"])
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.command == "thermal-preflight":
        if args.attempt < 1:
            raise ValueError("attempt must be positive")
        thermal_preflight(args.attempt)
    elif args.command == "stage":
        stage()
    elif args.command == "rebalance":
        rebalance()
    elif args.command == "split":
        split_cohorts()


if __name__ == "__main__":
    main()
