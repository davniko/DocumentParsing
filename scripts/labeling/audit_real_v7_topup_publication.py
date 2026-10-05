"""Read-only content/readback audit for the manually reviewed batch005 top-up.

Checks do not silently repair data. Narrow screening exceptions below are exact,
source-reviewed observations for this finite publication, not production rules.
"""

from __future__ import annotations

import argparse
import json
import re
import resource
import time
from collections import Counter
from pathlib import Path

from publish_real_v7_batch2 import ROOT, read, validate_rows

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.labeling_agents.direct_grounding import source_fidelity_findings

PARTIAL_COUNTS = {
    "020-1b3c9ed4": (
        "OCR retains 36 of the 56 declared packages in listed container portions; "
        "omitted rows are not invented."
    ),
    "049-05fd9e37": (
        "39 valid printed container IDs own 990 bags each; a 40th malformed "
        "12-character ID cannot be shortened to invent a valid ID. Shipment total "
        "stays 39,600; 990 remain unassigned."
    ),
    "071-5a4230e0": (
        "Only one 720-carton container row survives OCR; the declared 2,880-carton "
        "shipment total remains independent."
    ),
    "083-24c091c8": (
        "Shipment total is 241; only one 61-package local quantity survives OCR. "
        "Other printed memberships remain unquantified."
    ),
    "130-e352e3f6": (
        "OCR lists only six of the declared seventeen containers; their 1,225 local "
        "packages do not replace the 3,425 shipment total."
    ),
    "162-13ea92fa": (
        "54 is the shipment package total. Only CMAU2325606 has eight pallets printed "
        "locally; six other memberships have no printed local counts."
    ),
}


def audit(dataset, output):
    started = time.monotonic()
    manifest = read(dataset / "manifest.json")
    receipts = {r["documentId"]: r for r in manifest["documents"]}
    integrity = validate_rows(dataset, receipts)
    for name, digest in manifest["files"].items():
        assert sha256_file(dataset / name) == digest
    counts, changes, statuses, pages = Counter(), Counter(), Counter(), Counter()
    observations, defects, seen_partial, bills = [], [], set(), {}
    prior = ROOT / "data/curated/mpci-bl-real-v7-reviewed-r8-reduced-fields"
    for split in ("train", "validation"):
        for row in map(json.loads, (prior / f"{split}.jsonl").read_text().splitlines()):
            number = row["target"]["documentPatch"].get("billOfLadingNumber")
            if number:
                bills.setdefault(re.sub(r"\W", "", number.upper()), []).append(
                    (split, row["documentId"])
                )
    for split in ("train", "validation"):
        for row in map(json.loads, (dataset / f"{split}.jsonl").read_text().splitlines()):
            rec = receipts[row["documentId"]]
            name = rec["pilotDocument"]
            target = row["target"]
            patch = target["documentPatch"]
            text = row["joinedRawText"]
            findings = [
                f
                for section in ("cargo", "equipment", "route_transport")
                for f in source_fidelity_findings(target, text, section)
            ]
            if findings:
                codes = patch["goodsItemDetails"][0].get("hsCodes", [])
                if name == "093-3c2a9bfe" and len(findings) == 1:
                    assert codes == ["68022390"] and "HS CODE:-68022390" in text
                    note = (
                        "HS code explicitly prints after colon-hyphen. The literal gate "
                        "rejects the preceding hyphen; no digits are unsupported."
                    )
                elif name == "109-1e66b2bc" and len(findings) == 16:
                    hs_block = text.split("HS CODE:", 1)[1].split("According to Shipper", 1)[0]
                    printed = re.findall(r"\d{6,18}", hs_block)
                    assert printed == codes
                    note = (
                        "Sixteen separate full HS codes print as a hyphen-delimited list. "
                        "Current strict boundary gate rejects adjacent hyphens; reviewed "
                        "codes match complete tokens exactly."
                    )
                else:
                    defects.append(dict(document=name, findings=[f.model_dump() for f in findings]))
                    note = "UNRESOLVED"
                observations.append(
                    dict(
                        document=name,
                        targetSha256=rec["targetSha256"],
                        note=note,
                        findings=[f.model_dump() for f in findings],
                    )
                )
            goods = patch.get("goodsItemDetails", [])
            assert len(goods) == 1 and goods[0].get("description", "").strip()
            assert not any(
                k in json.dumps(target) for k in ('"additionalInformation"', '"additionalGoods"')
            )
            containers = patch.get("containerInformation", [])
            identifiers = {c["equipmentIdentifier"] for c in containers}
            assert len(identifiers) == len(containers) and identifiers
            counts["documents"] += 1
            counts[split] += 1
            counts["singleGoodsDocuments"] += 1
            counts["multiContainerDocuments"] += len(containers) > 1
            counts["containers"] += len(containers)
            counts["temperatureDocuments"] += any(c.get("temperatureSetpoint") for c in containers)
            counts["vesselFlagDocuments"] += "vesselFlagCountry" in patch.get("transport", {})
            counts["referenceDocuments"] += bool(patch.get("forwardingAndExportReferences"))
            pages[rec["pages"]] += 1
            for g in goods:
                counts["marksDocuments"] += bool(g.get("marksAndNumbers"))
                counts["dangerousGoodsDocuments"] += bool(g.get("dangerousGoods"))
                counts["hsCodeValues"] += len(g.get("hsCodes", []))
                gross, net = g.get("grossWeight"), g.get("netWeight")
                if gross and net and gross.get("unit") == net.get("unit"):
                    assert net["value"] <= gross["value"], name
                placements = g.get("splitGoodsPlacement", [])
                assert len({p["equipmentIdentifier"] for p in placements}) == len(placements)
                assert {p["equipmentIdentifier"] for p in placements} <= identifiers
                counts["placements"] += len(placements)
                pq = sum(p.get("packageQuantity", 0) for p in placements)
                total = sum(
                    p.get("packageQuantity", 0) for p in g.get("numberAndTypeOfPackages", [])
                )
                if pq and total:
                    assert pq <= total, name
                    if pq != total:
                        assert name in PARTIAL_COUNTS, (name, total, pq)
                        seen_partial.add(name)
                        observations.append(
                            dict(
                                document=name,
                                targetSha256=rec["targetSha256"],
                                total=total,
                                allocated=pq,
                                note=PARTIAL_COUNTS[name],
                            )
                        )
            baseline = read(ROOT / rec["inputTarget"])
            changed = baseline != target
            counts["netChangedDocuments"] += changed
            counts["netUnchangedDocuments"] += not changed
            statuses[f"{rec['originalAgentStatus']}:{'changed' if changed else 'unchanged'}"] += 1
            changes.update({c["path"][0] for c in rec["changes"]})
            state = read(ROOT / rec["sourceBatch"] / "runs" / name / "refine/status.json")
            waves = [d["wave"] for d in state["correctionDecisions"]]
            assert set(waves) <= {1}, (name, waves)
            number = patch.get("billOfLadingNumber")
            if number:
                key = re.sub(r"\W", "", number.upper())
                if key in bills:
                    defects.append(
                        dict(document=name, issue="repeated_bill_identifier", others=bills[key])
                    )
                bills.setdefault(key, []).append((split, row["documentId"]))
    assert seen_partial == set(PARTIAL_COUNTS)
    result = dict(
        integrity=integrity,
        counts=counts,
        pages=pages,
        changedDocumentCountsByField=changes,
        automatedStatusVsManualDecision=statuses,
        reviewedScreeningObservations=observations,
        unresolved=defects,
        elapsedSeconds=time.monotonic() - started,
        peakRssKiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    atomic_publish_json(output, result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "reviewedScreeningObservations"}, indent=2
        )
    )
    assert not defects


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(ROOT / args.dataset, ROOT / args.output)
