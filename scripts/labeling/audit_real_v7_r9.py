"""Read-only page, scope, rarity and source-quality inventory for R9/batch005."""

from __future__ import annotations

import csv
import json
import re
import resource
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import pypdfium2 as pdfium
from filter_real_v7_starting_dataset import ANALYSIS, ROOT, digest, read, tree_hashes, write_json
from profile_real_v7_dataset import CONTAINER, NONMARINE, PAGE, family, flatten, normalized_page
from project_real_v7_r9 import DEST, NEW, OLD, OUT, records

QUALITY = {
    "1b3c9ed4": "One of two declared IDs; 36 of 56 packages placed.",
    "a7ede7d8": "Front and attachment voyage suffix conflict; front voyage retained.",
    "dc2f1401": "Conflicting 20/53.856 CBM; shipment volume omitted.",
    "05fd9e37": (
        "40 containers declared but malformed MSKU93333590 cannot supply a valid ID; "
        "39 retained, 990 bags unassigned."
    ),
    "5a4230e0": "One of four IDs present; 720 of 2880 cartons placed.",
    "24c091c8": "Three container memberships, incomplete local package/mass amounts.",
    "63db6a27": "Primary parties, package counts and HS information missing in OCR.",
    "67d49789": "143-bale total versus three 48-bale portions (144); conflicting total omitted.",
    "e352e3f6": "Six of 17 declared IDs; unclear main vessel omitted.",
    "5e03e6b2": "Malformed gross 3.926.05 omitted; PDF-only route information excluded.",
    "13ea92fa": "Seven memberships, only one quantified portion; no printed volume unit.",
    "75006aac": "PDF agency/delivery/date information absent from OCR.",
    "4b109cdf": (
        "Populated Consigned to order of conflicts with non-negotiable title; "
        "target follows consignee wording."
    ),
}


def main():
    started = time.monotonic()
    before = {str(p): tree_hashes(p) for p in (OLD, NEW)}
    manifest = read(NEW / "manifest.json")
    receipts = {r["documentId"]: r for r in manifest["documents"]}
    old_rows = list(records(OLD))
    old_bills = defaultdict(list)
    shared_containers = defaultdict(list)
    old_inventory = {r["documentId"]: r for r in read(ANALYSIS / "inventory.json")}
    for _, row in old_rows:
        bill = row["target"]["documentPatch"].get("billOfLadingNumber")
        if bill:
            old_bills[re.sub(r"\W", "", bill).upper()].append(row["documentId"])
        for container in row["target"]["documentPatch"].get("containerInformation", []):
            shared_containers[container["equipmentIdentifier"]].append(
                old_inventory[row["documentId"]]["sample"]
            )
    rows = []
    for split, record in records(NEW):
        key = record["documentId"]
        receipt = receipts[key]
        pdf_path = ROOT / receipt["pdf"]
        assert digest(pdf_path) == receipt["pdfSha256"]
        with pdfium.PdfDocument(pdf_path) as pdf:
            dimensions = []
            for page in pdf:
                width, height = page.get_size()
                dimensions.append([round(width, 2), round(height, 2)])
                page.close()
            pages = len(pdf)
        assert pages == receipt["pages"] <= 5
        text = record["joinedRawText"]
        patch = record["target"]["documentPatch"]
        goods, containers = patch.get("goodsItemDetails", []), patch.get("containerInformation", [])
        placements = [p for g in goods for p in g.get("splitGoodsPlacement", [])]
        page_matches = list(PAGE.finditer(text))
        page_texts = [
            text[
                m.end() : page_matches[i + 1].start() if i + 1 < len(page_matches) else len(text)
            ].strip()
            for i, m in enumerate(page_matches)
        ]
        flag_lines = {
            category: [
                {"line": i + 1, "text": line}
                for i, line in enumerate(text.splitlines())
                if pattern.search(line)
            ]
            for category, pattern in {
                "nonmaritime": NONMARINE,
                "test_demo": re.compile(
                    r"\b(?:DCSA|HAPPYSUN|DUMMY|SPECIMEN|EXAMPLE|SAMPLE DOCUMENT|TEST DOCUMENT)\b",
                    re.I,
                ),
                "draft": re.compile(r"\b(?:DRAFT|PROFORMA)\b", re.I),
                "wrapper": re.compile(
                    r"Audit log|Document transfer log|CargoX|amendment|correction letter", re.I
                ),
            }.items()
        }
        flags = [category for category, hits in flag_lines.items() if hits]
        printed_page_counts = [
            m.group()
            for m in re.finditer(r"\b(?:PAGE|SHEET)\s*:?\s*\d+\s*(?:OF|/)\s*(\d+)", text, re.I)
            if int(m[1]) > pages
        ]
        if printed_page_counts:
            flags.append("printed_page_count_exceeds_pdf")
        if not patch.get("billOfLadingNumber"):
            flags.append("no_bill_number")
        if pages != len(page_matches):
            flags.append("page_count_mismatch")
        if not containers:
            flags.append("no_containers")
        if len(containers) >= 8:
            flags.append("8plus_containers")
        if len(goods) != 1:
            flags.append("not_single_goods")
        if any(not g.get("description") for g in goods):
            flags.append("no_description")
        if any(w > h for w, h in dimensions):
            flags.append("landscape_page")
        duplicate_pages = len(page_texts) - len({normalized_page(t) for t in page_texts})
        if duplicate_pages:
            flags.append("duplicate_page_text")
        if not patch.get("parties", {}).get("shipper"):
            flags.append("no_shipper")
        if not patch.get("parties", {}).get("consignee"):
            flags.append("no_named_consignee")
        if not patch.get("transport", {}).get("vesselName"):
            flags.append("no_main_vessel")
        if not patch.get("route", {}).get("portOfLoading"):
            flags.append("no_loading_port")
        if not patch.get("route", {}).get("portOfDischarge"):
            flags.append("no_discharge_port")
        values = list(flatten(patch))
        zeros = [
            {"path": list(p), "value": v}
            for p, v in values
            if isinstance(v, (float, int)) and not isinstance(v, bool) and v == 0
        ]
        placeholders = [
            {"path": list(p), "value": v}
            for p, v in values
            if isinstance(v, str)
            and re.search(r"\b(?:X{4,}|TBA|TBN|TBC|TBD|DUMMY|TEST)\b", v, re.I)
        ]
        zero_postal = [
            {"path": list(p), "value": v}
            for p, v in values
            if p[-1] == "addressLine" and re.search(r"(?<!\d)0{4,}(?!\d)", v)
        ]
        short_seals = [
            s
            for c in containers
            for s in c.get("sealNumbers", [])
            if len(re.sub(r"\W", "", s)) <= 3
        ]
        if zeros:
            flags.append("zero_numeric_value")
        if placeholders:
            flags.append("placeholder_value")
        if zero_postal:
            flags.append("zero_postal")
        if short_seals:
            flags.append("short_seal")
        hs = [v for g in goods for v in g.get("hsCodes", [])]
        if any(len(v) not in (6, 8, 10, 12) for v in hs):
            flags.append("uncommon_hs_length")
        quantity_total = sum(
            p["packageQuantity"]
            for g in goods
            for p in g.get("numberAndTypeOfPackages", [])
            if "packageQuantity" in p
        )
        allocated_total = sum(p.get("packageQuantity", 0) for p in placements)
        if quantity_total and allocated_total and quantity_total != allocated_total:
            flags.append("package_total_differs_from_allocated")
        carrier = patch.get("parties", {}).get("carrier", {}).get("name")
        sample = (
            "005/"
            + ("replacement/" if "replacements" in receipt["sourceBatch"] else "")
            + receipt["pilotDocument"]
        )
        for container in containers:
            shared_containers[container["equipmentIdentifier"]].append(sample)
        if key[4:12] in QUALITY:
            flags.append("reviewed_source_quality")
        bill = patch.get("billOfLadingNumber")
        rows.append(
            {
                "documentId": key,
                "sample": sample,
                "split": split,
                "pdf": receipt["pdf"],
                "ocr": str((NEW / "samples" / key / "ocr.txt").relative_to(ROOT)),
                "labels": str((NEW / "samples" / key / "labels.json").relative_to(ROOT)),
                "pdfPages": pages,
                "ocrPages": len(page_matches),
                "dimensions": dimensions,
                "ocrCharacters": len(text),
                "ocrWords": len(text.split()),
                "targetCharacters": len(json.dumps(record["target"], ensure_ascii=False)),
                "goodsCount": len(goods),
                "containerCount": len(containers),
                "placements": len(placements),
                "quantityPlacements": sum("packageQuantity" in p for p in placements),
                "packageTotal": quantity_total,
                "allocatedPackages": allocated_total,
                "containerIds": [c["equipmentIdentifier"] for c in containers],
                "containerLikeOcrIds": sorted(set(CONTAINER.findall(text.upper()))),
                "descriptions": [g.get("description") for g in goods],
                "hsCodes": hs,
                "carrierFamilyBeforeRemoval": family(carrier),
                "billNumber": bill,
                "oldBillCollisions": old_bills.get(re.sub(r"\W", "", bill).upper(), [])
                if bill
                else [],
                "dg": any(g.get("dangerousGoods") for g in goods),
                "thermal": any("temperatureSetpoint" in c for c in containers),
                "duplicatePages": duplicate_pages,
                "printedPageCountsExceedingPdf": printed_page_counts,
                "zeroValues": zeros,
                "placeholderValues": placeholders,
                "zeroPostal": zero_postal,
                "shortSeals": short_seals,
                "sourceQuality": QUALITY.get(key[4:12]),
                "flags": flags,
                "flagLines": flag_lines,
                "decisions": receipt["decisions"],
            }
        )
    assert len(rows) == 167
    groups = defaultdict(list)
    for row in rows:
        if row["billNumber"]:
            groups[re.sub(r"\W", "", row["billNumber"]).upper()].append(row["sample"])
    field_docs = Counter()
    for _, record in records(NEW):
        field_docs.update(
            {
                ".".join("[]" if isinstance(k, int) else k for k in p)
                for p, _ in flatten(record["target"]["documentPatch"])
            }
        )
    summary = {
        "documents": len(rows),
        "pageCounts": dict(Counter(r["pdfPages"] for r in rows)),
        "containerCounts": dict(sorted(Counter(r["containerCount"] for r in rows).items())),
        "goodsCounts": dict(Counter(r["goodsCount"] for r in rows)),
        "containers": sum(r["containerCount"] for r in rows),
        "multiContainer": sum(r["containerCount"] > 1 for r in rows),
        "dgDocuments": sum(r["dg"] for r in rows),
        "thermalDocuments": sum(r["thermal"] for r in rows),
        "multiHsDocuments": sum(len(r["hsCodes"]) > 1 for r in rows),
        "hsValues": sum(len(r["hsCodes"]) for r in rows),
        "flags": dict(Counter(f for r in rows for f in r["flags"])),
        "carrierFamiliesBeforeRemoval": dict(
            Counter(r["carrierFamilyBeforeRemoval"] for r in rows)
        ),
        "withinBatchBillCollisions": {k: v for k, v in groups.items() if len(v) > 1},
        "withOldBillCollisions": [r["sample"] for r in rows if r["oldBillCollisions"]],
        "sharedContainerCandidates": {
            cid: samples
            for cid, samples in shared_containers.items()
            if len(samples) > 1 and any(s.startswith("005/") for s in samples)
        },
        "combinedMultiContainer": sum(
            len(r["target"]["documentPatch"].get("containerInformation", [])) > 1
            for _, r in old_rows
        )
        + sum(r["containerCount"] > 1 for r in rows),
        "ocrCharacters": {
            "min": min(r["ocrCharacters"] for r in rows),
            "max": max(r["ocrCharacters"] for r in rows),
            "median": statistics.median(r["ocrCharacters"] for r in rows),
        },
        "elapsedSeconds": round(time.monotonic() - started, 3),
        "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scope": (
            "Screening flags, not automatic defects or new exclusions. "
            "Original full labels used for carrier/field inventories."
        ),
    }
    OUT.mkdir(exist_ok=True)
    write_json(OUT / "new-batch-inventory.json", rows)
    write_json(OUT / "new-batch-summary.json", summary)
    write_json(OUT / "new-batch-field-documents.json", dict(sorted(field_docs.items())))
    review_groups = {
        "Missing both primary parties": [
            r for r in rows if {"no_shipper", "no_named_consignee"} <= set(r["flags"])
        ],
        "Previously reviewed source-quality cases": [r for r in rows if r["sourceQuality"]],
        "Printed page totals exceed supplied PDF length": [
            r for r in rows if r["printedPageCountsExceedingPdf"]
        ],
        "Very large container lists": [r for r in rows if r["containerCount"] >= 8],
        "Unusual HS lengths": [r for r in rows if "uncommon_hs_length" in r["flags"]],
        "Missing main vessel": [r for r in rows if "no_main_vessel" in r["flags"]],
        "Missing loading or discharge location": [
            r for r in rows if {"no_loading_port", "no_discharge_port"} & set(r["flags"])
        ],
        "Missing B/L identifier": [r for r in rows if not r["billNumber"]],
        "Wrapper and shipment-duplicate review": [
            r for r in rows if r["documentId"][4:12] in {"fe260ee7", "0750aae7", "b7eadb8c"}
        ],
    }
    review_notes = {
        "fe260ee7": (
            "Actual eBL/audit wrapper with placeholder cjcjgls Company name; inspect for "
            "scope/test-like exclusion. Other amendment-link hits are boilerplate, not this case."
        ),
        "0750aae7": (
            "Same valve shipment as 002/015-9be27592: same parties, 689 cartons, container/seal, "
            "ACID and voyage. New copy includes issue/on-board dates and named carrier; "
            "prefer it if deduplicating."
        ),
        "b7eadb8c": (
            "Shared container with 001/45-2809b1ea, but different consignee, VIN, model, mass "
            "and volume. Distinct part-container consignments, not duplicate records."
        ),
        "1e66b2bc": (
            "Eleven-digit HS 48191000000/48025400000 print literally; do not pad to twelve digits."
        ),
        "b62c4e54": "Seven-digit HS8455300 prints literally; do not pad or truncate it.",
    }
    write_json(
        OUT / "outlier-adjudications.json",
        {
            "sourceQuality": QUALITY,
            "additionalReview": review_notes,
            "falsePositives": {
                "zeroValue": "060 has explicitly printed0C, not a fabricated zero.",
                "testWords": "128 TEST TUBES and159 TEST STRIP/METER are product names.",
            "wrapperScreen": (
                "014 is a true wrapper; 048 is an appended CargoX registration reference; "
                "13 other hits concern routine amendment wording."
            ),
            "draftScreen": (
                "Eight genuine draft/proforma headers; seven proforma-invoice references and "
                "one unselected draft/invoice caption are not draft-status evidence."
            ),
            "missingConsignee": (
                "Six of 12 missing named consignees are legitimate bare/order-to-holder/"
                "order-of-shipper instructions, not missing named-party annotations."
            ),
            "landscape": (
                "023 is a portrait document photograph on a landscape canvas, "
                "not non-maritime cargo."
            ),
            },
            "excludedHere": [],
        },
    )
    lines = [
        "# Batch005 outlier review links",
        "",
        "No rows filtered. Groups overlap. PDF/OCR/labels linked for every candidate. "
        "Labels point to the reduced R9 revision.",
        "",
    ]
    for title, group in review_groups.items():
        lines.extend(
            [
                f"## {title} ({len(group)})",
                "",
                "| Record / OCR | PDF / labels | Observation |",
                "|---|---|---|",
            ]
        )
        for row in group:
            label = str((DEST / "samples" / row["documentId"] / "labels.json").relative_to(ROOT))
            note = (
                review_notes.get(row["documentId"][4:12])
                or row["sourceQuality"]
                or ", ".join(row["flags"])
            )
            if title.startswith("Printed page"):
                note = f"{row['pdfPages']} supplied PDF pages; " + ", ".join(
                    row["printedPageCountsExceedingPdf"]
                )
            elif title.startswith("Very large"):
                note = f"{row['containerCount']} containers, {row['pdfPages']} PDF pages. " + (
                    row["sourceQuality"] or "Size outlier, not by itself a defect."
                )
            lines.append(
                f"| [{row['sample']}](../../../../{row['ocr']}) | "
                f"[PDF](../../../../{row['pdf']}) / [labels](../../../../{label}) | "
                f"{note.replace('|', '/')} |"
            )
        lines.append("")
    (OUT / "OUTLIERS.md").write_text("\n".join(lines))
    with (OUT / "new-batch-inventory.csv").open("w", newline="") as stream:
        fields = [
            "sample",
            "split",
            "pdfPages",
            "goodsCount",
            "containerCount",
            "placements",
            "packageTotal",
            "allocatedPackages",
            "ocrCharacters",
            "dg",
            "thermal",
            "sourceQuality",
            "flags",
            "pdf",
            "ocr",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    assert all(tree_hashes(Path(path)) == hashes for path, hashes in before.items())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
