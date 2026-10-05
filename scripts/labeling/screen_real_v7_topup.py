"""Read-only source selection screen; old labels never enter annotation requests.

Conservative selection eligibility is not an annotation-quality verdict. Sources
excluded here remain immutable and can be considered in a later, broader cohort.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium

from run_real_v7_batch import ROOT, SOURCE, WORK
from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file


def norm(value):
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def shipment_key(record):
    patch = record["target"]["documentPatch"]
    number = patch.get("billOfLadingNumber")
    shipper = patch.get("parties", {}).get("shipper", {}).get("name")
    return (norm(number), norm(shipper)) if number and shipper else None


def screen(destination):
    started = time.monotonic()
    records = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    prior_ids, prior_hashes = set(), set()
    for path in sorted((ROOT / "artifacts/kie-labeling").glob("direct-*/selection.json")):
        for row in json.loads(path.read_text()).get("documents", []):
            prior_ids.add(row["documentId"])
            prior_hashes.add(row["ocrSha256"])
    for split in ("train", "validation"):
        path = ROOT / f"data/curated/mpci-bl-real-v7-reviewed-r4_source_20261005/{split}.jsonl"
        for row in map(json.loads, path.read_text().splitlines()):
            prior_ids.add(row["documentId"])
            prior_hashes.add(row["joinedRawTextSha256"])
    seen_shipments = {shipment_key(r) for r in records if r["documentId"] in prior_ids}
    seen_shipments.discard(None)
    documents = []
    for r in sorted(records, key=lambda x: (x["auditSplit"] != "validation", x["documentId"])):
        if r["documentId"] in prior_ids or r["joinedRawTextSha256"] in prior_hashes:
            continue
        patch = r["target"]["documentPatch"]
        goods = patch.get("goodsItemDetails", [])
        description = goods[0].get("description", "") if len(goods) == 1 else ""
        reasons = []
        if len(goods) != 1:
            reasons.append("old_labels_not_single_goods")
        if not description.strip():
            reasons.append("old_description_missing")
        if not patch.get("containerInformation"):
            reasons.append("outside_container_topup_selection")
        if re.fullmatch(r"(?:TEST|DUMMY|SAMPLE|AS PER ATTACHED(?: LIST)?|AS PER ATTACHMENT)[ .]*", description, re.I):
            reasons.append("test_or_nondescriptive_cargo")
        text = r["joinedRawText"]
        scope_hits = re.findall(r"(?im)^.*(?:\bLAND BILL OF LADING\b|\bAIR WAYBILL\b|\bSHIPPING INSTRUCTIONS\b|\bDUMMY (?:BILL|DOCUMENT)\b|\bSAMPLE BILL OF LADING\b|\bTEST DOCUMENT\b).*$", text)
        if scope_hits:
            reasons.append("scope_caption_review")
        key = shipment_key(r)
        if key is not None and key in seen_shipments:
            reasons.append("repeated_shipment_number_and_shipper")
        matches = []
        for directory in WORK:
            path = directory / f"{r['documentId']}.json"
            if path.exists():
                source = json.loads(path.read_text())["source"]
                pdf = ROOT / source["localCanonicalPath"].split("/DocumentParsing/", 1)[-1]
                if pdf.is_file() and sha256_file(pdf) == source["sourceSha256"]:
                    matches.append((path, pdf, source["sourceSha256"]))
        assert matches and len({m[2] for m in matches}) == 1, r["documentId"]
        work, pdf, pdf_hash = sorted(matches)[0]
        with pdfium.PdfDocument(pdf) as opened:
            pages = len(opened)
        if not 1 <= pages <= 5:
            reasons.append("pdf_over_five_pages")
        markers = [int(x) for x in re.findall(r"--- PAGE (\d+) ---", text)]
        if markers and max(markers) != pages:
            reasons.append("ocr_pdf_page_coverage_review")
        if not reasons and key is not None:
            seen_shipments.add(key)
        documents.append(dict(documentId=r["documentId"], split=r["auditSplit"],
                              exclusions=reasons, pdf=str(pdf.relative_to(ROOT)),
                              pdfSha256=pdf_hash, pages=pages, oldGoods=len(goods),
                              oldDescription=description, oldContainers=len(patch.get("containerInformation", [])),
                              billNumber=patch.get("billOfLadingNumber"), scopeHits=scope_hits))
    output = dict(source=str(SOURCE.relative_to(ROOT)), sourceSha256=sha256_file(SOURCE),
                  eligible=dict(Counter(d["split"] for d in documents if not d["exclusions"])),
                  exclusions=dict(Counter(reason for d in documents for reason in d["exclusions"])),
                  elapsedSeconds=time.monotonic()-started, documents=documents)
    atomic_publish_json(destination, output)
    print(json.dumps({k: v for k, v in output.items() if k != "documents"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    screen(ROOT / args.output)
