"""Read-only structural/outlier inventory of the reviewed real V7 dataset.

Writes analysis artifacts only. Flags describe review candidates, not label
defects or deletion decisions. PDF metadata is read without re-OCR or API calls.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import resource
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium

from document_ocr.hashing import sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7Label

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/curated/mpci-bl-real-v7-reviewed-r4"
OUTPUT = ROOT / "docs/analysis/real-v7-starting-dataset-2026-10-04"
PAGE = re.compile(r"^--- PAGE (\d+) ---\s*$", re.MULTILINE)
CONTAINER = re.compile(r"(?<![A-Z0-9])[A-Z]{3}[UJZ]\s?\d{7}(?!\d)")
NONMARINE = re.compile(
    r"\b(?:LAND BILL OF LADING|AIR\s*WAYBILL|AIR CONSIGNMENT|CMR|RAIL CONSIGNMENT|ROAD CONSIGNMENT)\b",
    re.IGNORECASE,
)


def read(path):
    return json.loads(path.read_text())


def flatten(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten(child, (*path, index))
    else:
        yield path, value


def family(name):
    if not name:
        return "No carrier label"
    normalized = re.sub(r"[^A-Z0-9]", "", unicodedata.normalize("NFKD", name.upper()))
    for token, group in (
        ("CMACGM", "CMA CGM"), ("MAERSK", "Maersk"),
        ("MEDITERRANEANSHIPPING", "MSC"), ("HAPAGLLOYD", "Hapag-Lloyd"),
        ("OCEANNETWORKEXPRESS", "ONE"), ("YANGMING", "Yang Ming"),
        ("VASCOMARITIME", "Vasco"), ("TURKON", "Turkon"),
        ("COSCO", "COSCO"), ("EVERGREEN", "Evergreen"),
        ("WANHAI", "Wan Hai"), ("EMIRATESSHIPPING", "Emirates Shipping"),
    ):
        if token in normalized:
            return group
    if normalized.startswith("HMMCO"):
        return "HMM"
    return re.sub(r"\s+", " ", name.upper()).strip().rstrip(".")


def normalized_page(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def md(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def link(path, label):
    return f"[{label}](../../../{Path(path).as_posix()})"


def row_links(row):
    return (link(row["ocr"], row["sample"]) + " · " +
            link(row["labels"], "labels") + " · " + link(row["pdf"], "PDF"))


def main():
    started = time.monotonic()
    cohorts = read(OUTPUT / "reviewed_cohorts.json")
    annotations = {}
    for group, cohort in cohorts.items():
        for sample in cohort["samples"]:
            annotations.setdefault(sample, []).append(group)
    # Hash all published files, not only the two input streams.
    before = {str(p.relative_to(DATASET)): sha256_file(p)
              for p in DATASET.rglob("*") if p.is_file()}
    manifests = [read(p) for p in sorted((DATASET / "batches").glob("*/manifest.json"))]
    receipts = {r["documentId"]: (m["batch"], r)
                for m in manifests for r in m["documents"]}
    rows = []
    for split in ("train", "validation"):
        for raw in (DATASET / f"{split}.jsonl").read_text().splitlines():
            source = json.loads(raw)
            document_id = source["documentId"]
            batch, receipt = receipts[document_id]
            text, target = source["joinedRawText"], source["target"]
            patch = target["documentPatch"]
            BillOfLadingExtractionV7Label.model_validate_json(json.dumps(target))
            assert hashlib.sha256(text.encode()).hexdigest() == receipt["sourceOcrSha256"]
            pdf_path = ROOT / receipt["pdf"]
            assert sha256_file(pdf_path) == receipt["pdfSha256"]
            with pdfium.PdfDocument(pdf_path) as pdf:
                dimensions = []
                for page in pdf:
                    width, height = page.get_size()
                    dimensions.append(dict(width=round(width, 2), height=round(height, 2),
                                           rotation=page.get_rotation()))
                    page.close()
                page_count = len(pdf)
            page_markers = list(PAGE.finditer(text))
            page_texts = [text[m.end():page_markers[i+1].start() if i+1 < len(page_markers) else len(text)].strip()
                          for i, m in enumerate(page_markers)]
            goods, containers = patch.get("goodsItemDetails", []), patch.get("containerInformation", [])
            parties = patch.get("parties", {})
            carrier = parties.get("carrier", {}).get("name")
            flags = []
            if page_count > 5: flags.append("pdf_over_5_pages")
            if page_count != len(page_markers): flags.append("pdf_ocr_page_count_mismatch")
            if not containers: flags.append("no_labeled_containers")
            if not goods: flags.append("no_labeled_goods")
            if any(not g.get("description") for g in goods): flags.append("goods_without_description")
            if len(goods) > 1: flags.append("multiple_goods")
            if len(goods) >= 5: flags.append("at_least_5_goods")
            if len(containers) >= 8: flags.append("at_least_8_containers")
            if not patch.get("billOfLadingNumber"): flags.append("no_bill_number_label")
            if not patch.get("transport", {}).get("vesselName"): flags.append("no_vessel_label")
            if not carrier: flags.append("no_carrier_label")
            if not patch.get("route", {}).get("portOfLoading"): flags.append("no_loading_port_label")
            if not patch.get("route", {}).get("portOfDischarge"): flags.append("no_discharge_port_label")
            if any(d["width"] > d["height"] for d in dimensions): flags.append("landscape_pdf_page")
            duplicate_pages = len(page_texts) - len({normalized_page(p) for p in page_texts})
            if duplicate_pages: flags.append("identical_normalized_ocr_pages")
            all_values = list(flatten(patch))
            zero_values = [dict(path=list(p), value=v) for p, v in all_values
                           if isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0]
            if zero_values: flags.append("zero_numeric_label")
            zero_postal = [dict(path=list(p), value=v) for p, v in all_values
                          if p[-1] == "addressLine" and re.search(r"(?<!\d)0{4,}(?!\d)", v)]
            if zero_postal: flags.append("zero_postal_text")
            placeholders = [dict(path=list(p), value=v) for p, v in all_values if isinstance(v, str)
                            and re.search(r"\b(?:X{4,}|TBA|TBN|TBC|TBD|TEST|DUMMY|SAMPLE)\b", v, re.I)]
            if placeholders: flags.append("placeholder_or_test_word_label")
            hs = [v for g in goods for v in g.get("hsCodes", [])]
            if any(len(v) not in (6, 8, 10, 12) for v in hs): flags.append("uncommon_hs_length")
            seals = [s for c in containers for s in c.get("sealNumbers", [])]
            short_seals = [s for s in seals if len(re.sub(r"\W", "", s)) <= 3]
            if short_seals: flags.append("seal_at_most_3_alphanumerics")
            duplicate_descriptions = [k for k, n in Counter(normalized_page(g["description"])
                                      for g in goods if g.get("description")).items() if n > 1]
            if duplicate_descriptions: flags.append("repeated_goods_description")
            dg = sum(bool(g.get("dangerousGoods")) for g in goods)
            thermal = sum("temperatureSetpoint" in c for c in containers)
            header = text[:4500]
            nonmarine = [(i+1, line) for i, line in enumerate(text.splitlines())
                         if NONMARINE.search(line)]
            if nonmarine: flags.append("nonmaritime_document_keyword")
            if re.search(r"Audit log|Document transfer log", text, re.I):
                flags.append("electronic_bill_audit_wrapper")
            if re.search(r"^\s*DRAFT\b", text, re.I | re.M):
                flags.append("draft_heading")
            if re.search(r"^\s*PROFORMA\s*$", text, re.I | re.M):
                flags.append("proforma_heading")
            demo = [(i+1, line) for i, line in enumerate(text.splitlines())
                    if re.search(r"\b(?:DCSA|HAPPYSUN|DUMMY|SPECIMEN|EXAMPLE|SAMPLE DOCUMENT|TEST DOCUMENT)\b", line, re.I)]
            layout_headers = [line for line in header.splitlines()
                              if re.search(r"CONGENBILL|CONLINE|SEA\s*WAYBILL|EXPRESS RELEASE|BILL OF LADING|CARGO RECEIPT|SHIPPING INSTRUCTION|BOOKING CONFIRMATION", line, re.I)]
            package_counts = [p["packageQuantity"] for g in goods for p in g.get("numberAndTypeOfPackages", [])
                              if "packageQuantity" in p]
            gross = [g["grossWeight"] for g in goods if "grossWeight" in g]
            placements = [p for g in goods for p in g.get("splitGoodsPlacement", [])]
            row = dict(
                documentId=document_id, sample=f"{batch}/{receipt['pilotDocument']}", split=split,
                pdf=receipt["pdf"], ocr=str((DATASET / "samples" / document_id / "ocr.txt").relative_to(ROOT)),
                labels=str((DATASET / "samples" / document_id / "labels.json").relative_to(ROOT)),
                pdfPages=page_count, ocrPages=len(page_markers), pdfDimensions=dimensions,
                ocrCharacters=len(text), ocrWords=len(text.split()), ocrLines=len(text.splitlines()),
                targetCharacters=len(json.dumps(target, ensure_ascii=False, separators=(",", ":"))),
                scalarLabels=len(all_values), goodsCount=len(goods), containerCount=len(containers),
                carrier=carrier, carrierFamily=family(carrier), billNumber=patch.get("billOfLadingNumber"),
                goodsDescriptions=[g.get("description") for g in goods],
                descriptionsCharacters=[len(g.get("description", "")) for g in goods],
                containers=[c["equipmentIdentifier"] for c in containers],
                containerLikeOcrIds=sorted(set(CONTAINER.findall(text.upper()))),
                placements=len(placements), quantityPlacements=sum("packageQuantity" in p for p in placements),
                multiHsGoods=sum(len(g.get("hsCodes", [])) > 1 for g in goods), hsCodes=hs,
                packageQuantities=package_counts, grossWeights=gross,
                dgGoods=dg, thermalContainers=thermal,
                zeroValues=zero_values, zeroPostal=zero_postal, placeholderValues=placeholders,
                shortSeals=short_seals, repeatedDescriptions=duplicate_descriptions,
                exactDuplicateOcrPages=duplicate_pages, ocrPageCharacters=[len(p) for p in page_texts],
                ocrPageHeaders=[p[:180].replace("\n", " | ") for p in page_texts],
                layoutHeaders=layout_headers[:12], nonmarineKeywordLines=nonmarine,
                demoKeywordLines=demo, reviewDecisions=receipt["decisions"], flags=flags,
                reviewedCohorts=annotations.get(f"{batch}/{receipt['pilotDocument']}", []),
            )
            rows.append(row)
    assert len(rows) == 450 and len({r["documentId"] for r in rows}) == 450
    assert Counter(r["split"] for r in rows) == {"train": 400, "validation": 50}
    assert set(annotations) <= {r["sample"] for r in rows}
    no_container_groups = [c for name, c in cohorts.items() if name.startswith("no_container_")]
    classified = [s for c in no_container_groups for s in c["samples"]]
    assert len(classified) == len(set(classified)) == 69
    assert set(classified) == {r["sample"] for r in rows if not r["containerCount"]}
    assert NONMARINE.search("LAND BILL OF LADING")
    assert not NONMARINE.search("Sealand Bill of Lading")
    assert before == {str(p.relative_to(DATASET)): sha256_file(p)
                      for p in DATASET.rglob("*") if p.is_file()}
    distributions = {k: dict(sorted(Counter(r[k] for r in rows).items(), key=lambda x: str(x[0])))
                     for k in ("pdfPages", "ocrPages", "goodsCount", "containerCount", "carrierFamily")}
    summary = dict(
        dataset=str(DATASET.relative_to(ROOT)), samples=len(rows),
        splitCounts=dict(Counter(r["split"] for r in rows)), distributions=distributions,
        flags=dict(Counter(f for r in rows for f in r["flags"])),
        totals={k: sum(r[k] for r in rows) for k in ("pdfPages", "goodsCount", "containerCount", "placements", "quantityPlacements")},
        numericProfile={k: dict(min=min(r[k] for r in rows), median=statistics.median(r[k] for r in rows),
                               max=max(r[k] for r in rows))
                        for k in ("ocrCharacters", "ocrWords", "targetCharacters", "scalarLabels")},
        uniqueLiteralCarrierNames=len({r["carrier"] for r in rows if r["carrier"]}),
        reviewedCohortCounts={k: len(v["samples"]) for k, v in cohorts.items()},
        originalDatasetFilesHashed=len(before), originalDatasetHashes=before,
        datasetUnchanged=True, elapsedSeconds=time.monotonic()-started,
        peakRssKiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        apiCalls=0, apiCostUsd=0,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "inventory.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False)+"\n")
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    columns = ("sample", "documentId", "split", "pdfPages", "ocrPages", "goodsCount", "containerCount",
               "placements", "quantityPlacements", "carrier", "carrierFamily", "billNumber",
               "ocrCharacters", "targetCharacters", "scalarLabels", "dgGoods", "thermalContainers",
               "flags", "reviewedCohorts", "ocr", "labels", "pdf")
    with (OUTPUT / "inventory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "; ".join(row[k]) if isinstance(row[k], list) else row[k] for k in columns})
    header = "| Sample / files | Split | PDF pages | Goods | Containers | Details |\n|---|---|---:|---:|---:|---|\n"
    for filename, title, selected, describe in (
        ("MULTI_GOODS.md", "Every document with more than one labeled goods item",
         sorted((r for r in rows if r["goodsCount"] > 1), key=lambda r: (-r["goodsCount"], r["sample"])),
         lambda r: "<br>".join(f"{i+1}. {md(d) if d else '[no description]'}" for i, d in enumerate(r["goodsDescriptions"]))),
        ("LONG_DOCUMENTS.md", "Every source PDF with more than five pages",
         sorted((r for r in rows if r["pdfPages"] > 5), key=lambda r: (-r["pdfPages"], r["sample"])),
         lambda r: md(r["carrier"] or "No carrier label") + "; OCR page characters: " + str(r["ocrPageCharacters"])),
        ("NO_CONTAINERS.md", "Every document with no labeled container",
         [r for r in rows if not r["containerCount"]],
         lambda r: md("; ".join(c for c in r["reviewedCohorts"] if c.startswith("no_container_"))) + "; " +
                   (md("; ".join(d or "[no description]" for d in r["goodsDescriptions"])) or "[no goods label]")),
    ):
        lines = [f"# {title}\n\n{len(selected)} documents. Links point to unchanged OCR, labels and source PDFs.\n\n", header]
        for row in selected:
            lines.append(f"| {row_links(row)} | {row['split']} | {row['pdfPages']} | {row['goodsCount']} | {row['containerCount']} | {describe(row)} |\n")
        (OUTPUT / filename).write_text("".join(lines))
    # Separate human-reviewed classifications from mechanically screened traits.
    queue = ["# Outlier review inventory\n\nAnalysis only; no exclusion or repair has been applied. "
             "Cohorts overlap. Identifiers are batch / original review-directory names.\n\n"]
    for name, cohort in cohorts.items():
        if name.startswith("no_container_"):
            continue
        selected = [r for r in rows if r["sample"] in cohort["samples"]]
        queue.extend([f"## {name}\n\n{cohort['meaning']}\n\n", header])
        for row in selected:
            queue.append(f"| {row_links(row)} | {row['split']} | {row['pdfPages']} | {row['goodsCount']} | "
                         f"{row['containerCount']} | {md('; '.join(d or '[no description]' for d in row['goodsDescriptions']))} |\n")
        queue.append("\n")
    for flag in ("no_labeled_goods", "goods_without_description", "at_least_8_containers",
                 "zero_postal_text", "uncommon_hs_length", "landscape_pdf_page", "draft_heading", "proforma_heading"):
        selected = [r for r in rows if flag in r["flags"]]
        queue.extend([f"## {flag}\n\n{len(selected)} screened documents; the flag alone is not a label error.\n\n", header])
        for row in selected:
            queue.append(f"| {row_links(row)} | {row['split']} | {row['pdfPages']} | {row['goodsCount']} | "
                         f"{row['containerCount']} | {md(row['carrier'] or '[no carrier label]')} |\n")
        queue.append("\n")
    (OUTPUT / "REVIEW_QUEUE.md").write_text("".join(queue))
    print(json.dumps({k: v for k, v in summary.items() if k != "originalDatasetHashes"}, indent=2))


if __name__ == "__main__":
    main()
