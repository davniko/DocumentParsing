"""Read-only repetition and source-page audit for the approved R10 selection."""

from __future__ import annotations

import json
import re
import resource
import time
import unicodedata
from collections import Counter, defaultdict

import pypdfium2 as pdfium
from filter_real_v7_starting_dataset import ANALYSIS, ROOT, read, write_json
from project_real_v7_r9 import DEST as SOURCE
from project_real_v7_r9 import records

OUT = ANALYSIS / "r10_cleansing"
EXCLUDE_SAMPLES = {
    "002/015-9be27592": "Older duplicate; retain the more complete 005/001-0750aae7.",
    "005/014-fe260ee7": "Electronic B/L test/dummy wrapper, excluded by user.",
}
INSPECT = {13, 20, 35, 43, 49, 60, 71, 83, 99, 120, 130, 150, 157, 162}


def normalized(text):
    """Only case, punctuation and whitespace: no entity or goods synonym guesses."""
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def selection():
    manifest = read(SOURCE / "projection-manifest.json")
    info = {r["documentId"]: r for r in manifest["documents"]}
    rows, excluded = [], []
    for split, record in records(SOURCE):
        item = info[record["documentId"]]
        parties = record["target"]["documentPatch"].get("parties", {})
        reason = EXCLUDE_SAMPLES.get(item["sample"])
        if not parties.get("shipper") and not parties.get("consignee"):
            reason = "Both shipper and consignee absent; primary-party exclusion approved by user."
        if reason:
            excluded.append(item | {"reason": reason})
        else:
            rows.append((split, record, item))
    assert len(excluded) == 26 and len(rows) == 474
    return rows, excluded


def repetition(rows):
    inventory = []
    for split, record, info in rows:
        patch = record["target"]["documentPatch"]
        parties = patch.get("parties", {})
        inventory.append(
            {
                "documentId": record["documentId"],
                "sample": info["sample"],
                "split": split,
                "shipper": parties.get("shipper", {}).get("name"),
                "consignee": parties.get("consignee", {}).get("name"),
                "goods": "\n".join(g.get("description", "") for g in patch["goodsItemDetails"]),
                "bill": patch.get("billOfLadingNumber"),
                "issueDate": patch.get("issueDate"),
                "containers": [
                    c["equipmentIdentifier"] for c in patch.get("containerInformation", [])
                ],
                "goodsFacts": patch["goodsItemDetails"],
                "pdf": info["pdf"],
                "ocr": str(
                    (SOURCE / "samples" / record["documentId"] / "ocr.txt").relative_to(ROOT)
                ),
            }
        )
    groups, summaries = {}, {}
    for fields in [
        ("shipper",),
        ("consignee",),
        ("goods",),
        ("shipper", "consignee"),
        ("shipper", "goods"),
        ("consignee", "goods"),
        ("shipper", "consignee", "goods"),
    ]:
        name = "+".join(fields)
        mapping = defaultdict(list)
        for item in inventory:
            if all(item[k] for k in fields):
                mapping[tuple(normalized(item[k]) for k in fields)].append(item)
        repeated = [
            {
                "key": key,
                "count": len(items),
                "documents": items,
                "crossSplit": len({i["split"] for i in items}) > 1,
            }
            for key, items in mapping.items()
            if len(items) > 1
        ]
        repeated.sort(key=lambda g: (-g["count"], g["key"]))
        groups[name] = repeated
        summaries[name] = {
            "eligibleDocuments": sum(len(v) for v in mapping.values()),
            "distinctKeys": len(mapping),
            "repeatedGroups": len(repeated),
            "documentsInRepeatedGroups": sum(g["count"] for g in repeated),
            "crossSplitGroups": sum(g["crossSplit"] for g in repeated),
            "validationDocumentsInCrossSplitGroups": sum(
                sum(i["split"] == "validation" for i in g["documents"])
                for g in repeated
                if g["crossSplit"]
            ),
        }
    # Word-order-only name differences are candidates, not inferred entity identities.
    name_variants = {}
    for role in ("shipper", "consignee"):
        buckets = defaultdict(list)
        for item in inventory:
            if item[role]:
                buckets[tuple(sorted(normalized(item[role]).split()))].append(item)
        name_variants[role] = [
            {"names": sorted({i[role] for i in items}), "samples": [i["sample"] for i in items]}
            for items in buckets.values()
            if len({normalized(i[role]) for i in items}) > 1
        ]
    # Shared identifiers supplement, but never imply duplicate shipments.
    collisions = {}
    for kind in ("bill", "containers"):
        buckets = defaultdict(list)
        for item in inventory:
            values = item[kind] if kind == "containers" else [item[kind]]
            for value in values:
                if value:
                    buckets[normalized(value)].append(item["sample"])
        collisions[kind] = {k: v for k, v in buckets.items() if len(v) > 1}
    write_json(OUT / "repetition-inventory.json", inventory)
    write_json(OUT / "repetition-groups.json", groups)
    write_json(OUT / "name-variant-candidates.json", name_variants)
    write_json(OUT / "identifier-collisions.json", collisions)
    lines = [
        "# Repeated parties and goods in R10",
        "",
        "Case/spacing/punctuation normalized only. Missing names are excluded from keys. "
        "Repeated wording is not a duplicate decision. No records filtered by these groups.",
        "",
    ]
    for name, values in groups.items():
        lines.extend([f"## {name}", "", json.dumps(summaries[name]), ""])
        for group in values:
            lines.extend(
                [
                    "### " + " / ".join(group["key"]),
                    "",
                    f"{group['count']} documents; cross-split={group['crossSplit']}",
                    "",
                ]
            )
            for item in group["documents"]:
                lines.append(
                    f"- {item['sample']} ({item['split']}): B/L={item['bill']}; "
                    f"date={item['issueDate']}; "
                    f"containers={', '.join(item['containers']) or 'none'}; "
                    f"[OCR](../../../../{item['ocr']}) / [PDF](../../../../{item['pdf']})"
                )
            lines.append("")
    (OUT / "REPETITIONS.md").write_text("\n".join(lines))
    return summaries


def pages(rows):
    result = []
    for _, record, info in rows:
        if (
            not info["sample"].startswith("005/")
            or int(info["sample"].split("/")[-1][:3]) not in INSPECT
        ):
            continue
        stem = info["sample"].replace("/", "-")
        folder = OUT / "sources" / stem
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "ocr.txt").write_text(record["joinedRawText"])
        write_json(folder / "labels.json", record["target"])
        counts, native = [], []
        with pdfium.PdfDocument(ROOT / info["pdf"]) as pdf:
            for index in range(len(pdf)):
                page = pdf[index]
                text_page = page.get_textpage()
                text = text_page.get_text_range()
                native.append(f"--- PDF PAGE {index + 1} ---\n{text}")
                text_page.close()
                bitmap = page.render(scale=1.4)
                bitmap.to_pil().convert("RGB").save(folder / f"page-{index + 1}.jpg", quality=90)
                bitmap.close()
                counts.append(
                    {"page": index + 1, "nativeCharacters": len(text), "size": page.get_size()}
                )
                page.close()
        (folder / "pdf-native.txt").write_text("\n\n".join(native))
        result.append(
            info | {"pdfPages": counts, "evidenceDirectory": str(folder.relative_to(ROOT))}
        )
    write_json(OUT / "page-evidence.json", result)
    return result


def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, excluded = selection()
    write_json(OUT / "exclusions.json", excluded)
    summary = {
        "documents": len(rows),
        "splits": dict(Counter(s for s, _, _ in rows)),
        "excluded": len(excluded),
        "repetition": repetition(rows),
    }
    pages(rows)
    summary.update(
        elapsedSeconds=round(time.monotonic() - start, 3),
        peakRssKiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    write_json(OUT / "analysis-summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
