"""Independent publication census and exact ledger arithmetic for this refresh."""

import json
import statistics
from collections import Counter
from decimal import Decimal
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/analysis/r28-multirow-refresh1500-20261031"


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    return {r["documentId"]: r for r in map(json.loads, path.open())}


def census():
    admissions = [read(p) for p in (OUT / "batches").glob("*-component-admission.json")]
    receipts = read(OUT / "replacement-receipts.json")
    candidates = read(OUT / "final-scoped-review.json")["candidates"]
    before = rows(OUT / "before/synthetic-train.jsonl")
    after = rows(OUT / "publication/synthetic-train.jsonl")
    selected = {r["documentId"]: r for r in receipts}
    native_ids = {r["candidateId"] for r in receipts}
    hs, group_types, group_counts, projections, nested, slots = (Counter() for _ in range(6))
    source_sets, examples = {}, []
    for pin in candidates:
        path = ROOT / pin["candidate"]["path"]
        assert sha256_file(path) == pin["candidate"]["sha256"]
        assert sha256_file(ROOT / pin["review"]["path"]) == pin["review"]["sha256"]
        candidate = read(path)
        assert candidate["documentId"] in native_ids
        scenario = candidate["scenario"]
        assert scenario["destination"]["country_code"] == "AE"
        provenance = scenario["provenance"]
        ledger = provenance["declaredPackages"]
        groups, matrix, loads = ledger["groups"], ledger["matrix"], ledger["loads"]
        assert len(matrix) == len(loads) == len(ledger["capacities"])
        assert 2 <= len(groups) <= 3
        group_counts[len(groups)] += 1
        group_types.update(g["category"] for g in groups)
        for index, (counts, load, capacity) in enumerate(
            zip(matrix, loads, ledger["capacities"], strict=True)
        ):
            assert len(counts) == len(groups)
            assert all(isinstance(n, int) and n >= 0 for n in counts)
            assert load["row"] == index and load["packageQuantity"] == sum(counts)
            for field in ("netKg", "grossKg", "volumeM3"):
                expected = sum(n * Decimal(g[field]) for n, g in zip(counts, groups, strict=True))
                assert expected == Decimal(load[field])
            assert Decimal(load["netKg"]) <= Decimal(load["grossKg"]) <= Decimal(capacity[0])
            assert Decimal(load["volumeM3"]) <= Decimal(capacity[1])
        for field in ("netKg", "grossKg", "volumeM3"):
            assert sum(Decimal(load[field]) for load in loads) == Decimal(ledger["totals"][field])
        public = candidate["target"]["documentPatch"]["goodsItemDetails"][0]
        assert sum(p["packageQuantity"] for p in public["numberAndTypeOfPackages"]) == sum(
            sum(counts) for counts in matrix
        )
        codes = [g["hs6"] for g in scenario["goods_identities"]]
        hs.update(codes)
        source_sets.setdefault(candidate["sourceDocumentId"], set()).update(codes)
        packing = provenance.get("freshPrivatePacking")
        nested["withPrivatePacking" if packing else "withoutPrivatePacking"] += 1
        for slot in candidate["packagePresentations"]["slots"]:
            slots[slot["style"]] += 1
        if len(examples) < 8 and (len(groups) == 3 or packing):
            examples.append(
                dict(
                    nativeId=candidate["documentId"],
                    source=candidate["sourceDocumentId"],
                    description=public["description"],
                    packageRows=public["numberAndTypeOfPackages"],
                    allocations=public.get("splitGoodsPlacement"),
                    candidate=pin["candidate"],
                )
            )
    assert len(candidates) == len(selected) == 1500
    assert group_counts == {2: 1200, 3: 300}
    for receipt in receipts:
        projections[receipt["projection"]] += 1
    public_rows = Counter(
        len(after[did]["target"]["goodsItemDetails"][0]["numberAndTypeOfPackages"])
        for did in selected
    )
    positions = Counter()
    holds = Counter()
    for admission in admissions:
        positions.update(admission["positions"]["counts"])
        holds.update(admission["positions"]["heldGroups"])
    result = dict(
        status="PASS",
        candidates=1500,
        sources=len(source_sets),
        privateGroupCounts=group_counts,
        publicRowCounts=public_rows,
        publicProjectionModes=projections,
        groupCategories=group_types,
        uniqueGroupCategories=len(group_types),
        uniqueSampledHs6=len(hs),
        distinctHs6PerParent=Counter(map(len, source_sets.values())),
        lexicalSlotStyles=slots,
        privatePacking=nested,
        previousAliasRefreshed=sum(r["previousAliasRefresh"] for r in receipts),
        previousDonorBound=sum(bool(r["previousDonor"]) for r in receipts),
        coordinateProvenance=positions,
        preservedSourceAnchorLimitations=holds,
        admissionComponentSeconds=sum(a["seconds"] for a in admissions),
        admissionMedianSecondsPerRecord=statistics.median(
            a["seconds"] / len(a["records"]) for a in admissions
        ),
        admissionMaxRssMiB=max(a["maxRssMiB"] for a in admissions),
        datasetRecordsBefore=len(before),
        datasetRecordsAfter=len(after),
        selectedRecordsWithChangedText=sum(
            before[d]["joinedRawText"] != after[d]["joinedRawText"] for d in selected
        ),
        examples=examples,
    )
    atomic_publish_json(OUT / "completion-census.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "examples"}), flush=True)


if __name__ == "__main__":
    census()
