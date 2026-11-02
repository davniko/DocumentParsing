"""Stage bounded, goods-first R28 replacement draws; never publish training data."""

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

from document_ocr.atomic import atomic_write_json
from document_ocr.synthesis.curated import digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/analysis/r28-combination-refresh300-20261101"
ART = ROOT / "artifacts/kie-synthesis-production/r28-combination-refresh300-20261101"
# Explicit campaign strata, not per-template donor profiles. Native package
# plausibility still decides eligibility; no requested category is forced.
STRATA = {
    "rack": ("PACKAGE_RACK", 30, ["7003", "7004", "7005", "7006", "7007", "7008"]),
    "case": ("PACKAGE_CASE", 30, ["8203", "8204", "8205", "8206", "8301", "9015", "9031"]),
    "reel": ("PACKAGE_REEL", 20, ["5402", "5403", "5404", "5406", "5509", "5510", "5511", "8544"]),
    "superbulk": ("PACKAGE_BAG_SUPER_BULK", 20, ["2505", "3901", "3902", "3903", "3904", "3906"]),
    "parcel": ("PACKAGE_PARCEL", 20, ["4901", "6109", "6110", "6205", "6206", "9503"]),
    "slab": ("PACKAGE_SLAB", 15, ["2515", "2516", "6802"]),
    "block": ("PACKAGE_BLOCK", 15, ["2515", "2516", "4001", "4703"]),
    "flexibag": (
        "PACKAGE_FLEXIBAG",
        15,
        ["1507", "1508", "1509", "1510", "1511", "1512", "1513", "1514", "1515"],
    ),
    "flexitank": (
        "PACKAGE_FLEXITANK",
        15,
        ["1507", "1508", "1509", "1510", "1511", "1512", "1513", "1514", "1515"],
    ),
    "liquid": (
        "PACKAGE_BULK_LIQUID",
        20,
        ["1507", "1508", "1509", "1510", "1511", "1512", "1513", "1514", "1515"],
    ),
}


def load(path):
    return json.loads((ROOT / path).read_text())


def stage(round_number, per_stratum):
    base = load(
        "artifacts/kie-synthesis-production/r28-targeted-cargo-refresh-probe-20261101/native/config.json"
    )
    authority = load(
        "artifacts/synthesis-templates/mpci-bl-v7-reviewed/expansion-20261101-v21/complete-source-authority-config.json"
    )
    census = load("docs/analysis/r28-mix-and-package-coverage-20261101/census.json")
    readiness = load("docs/analysis/r28-mix-and-package-coverage-20261101/mixed-readiness.json")
    sources = {
        r["source"]
        for r in readiness["sources"]
        if r["status"] == "PREREQUISITES_AND_THREE_EQUIPMENT_DRAWS_PASS"
        and r["regime"] == "ambient"
    }
    prior = {
        r["documentId"]: r
        for r in load(
            "docs/analysis/r28-quality-and-variability-20261019/donor/per-record-inventory.json"
        )
    }
    # Do not remove existing scarce coverage to add the same coverage elsewhere.
    abundant = {
        "PACKAGE_CARTON",
        "PACKAGE_PACKAGE",
        "PACKAGE_PALLET",
        "PACKAGE_BOX_FIBREBOARD",
        "PACKAGE_PALLET_WOODEN",
        "PACKAGE_CRATE_WOODEN",
        "PACKAGE_CASE_WOODEN",
        "PACKAGE_BOX",
        "PACKAGE_BAG",
        "PACKAGE_DRUM",
        "PACKAGE_BAG_PLASTIC",
        "PACKAGE_BUNDLE",
    }
    pool = defaultdict(list)
    for r in census["current"]["synthetic-train"]["inventory"]:
        if (
            r["documentId"] in prior
            and r["source"] in sources
            and not r["multirow"]
            and not r["mixedEquipment"]
            and set(r["categories"]) <= abundant
            and prior[r["documentId"]]["cargoFamily"] == "ambient"
        ):
            pool[r["source"]].append(prior[r["documentId"]])
    if not pool:
        raise ValueError("empty eligible historical pool")
    atomic_write_json(OUT / "eligible-old-records.json", dict(pool))
    phrases = [
        json.loads(s)
        for s in (ROOT / base["independent"]["registry"]["phrases"]["path"])
        .read_text()
        .splitlines()
    ]
    # Balance source use across all staged rounds. At most one draw/source per
    # stratum/round avoids coupling one rare package to one layout.
    used = Counter()
    for path in ART.glob("round-*/**/config.json"):
        for sid in json.loads(path.read_text())["source_ids"]:
            used[sid] += 1
    paths = []
    for label, (category, target, headings) in STRATA.items():
        count = per_stratum if per_stratum is not None else target
        sids = sorted(pool, key=lambda s: (used[s], digest([round_number, label, s])))[:count]
        if len(sids) != count:
            raise ValueError("insufficient distinct eligible sources")
        used.update(sids)
        cfg = copy.deepcopy(base)
        cfg["capabilities"] = {s: copy.deepcopy(authority["capabilities"][s]) for s in sids}
        cfg["source_ids"] = sids
        cfg["template_sampling"] = {"samples": count, "source_counts": {s: 1 for s in sids}}
        cfg["declared_packages"]["source_group_counts"] = {
            s: [2 if i % 2 else 3] for i, s in enumerate(sids)
        }
        cfg["seed"] = 202611014000 + round_number * 100 + list(STRATA).index(label)
        cfg["concurrency"] = 8
        cfg["output"] = str(
            (ART / f"round-{round_number:02d}" / label / "campaign").relative_to(ROOT)
        )
        cfg["independent_preparation"].update(
            package_targets={category: count},
            mixed_source_ids=sids,
            hs6_pool=sorted(p["hs6"] for p in phrases if p["hs6"][:4] in headings),
        )
        path = ART / f"round-{round_number:02d}" / label / "config.json"
        if path.exists():
            raise ValueError("refusing to overwrite staged draw")
        atomic_write_json(path, cfg)
        paths.append(str(path.relative_to(ROOT)))
    atomic_write_json(
        OUT / f"round-{round_number:02d}-plan.json",
        dict(
            configs=paths,
            availableRecords=sum(map(len, pool.values())),
            parents=len(pool),
            strata=STRATA,
        ),
    )
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--per-stratum", type=int)
    a = p.parse_args()
    stage(a.round, a.per_stratum)
