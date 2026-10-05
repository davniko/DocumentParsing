"""Publish the source-reviewed single-labeled-goods baseline (2026-10-05).

Selection only: nine description-empty records, all 42 multi-goods targets and
ten redundant shipment copies are excluded. Prior datasets remain unchanged.
The shipment choice and PDF/OCR comparison are recorded alongside the report.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import resource
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium

from filter_real_v7_starting_dataset import (
    ANALYSIS, ROOT, digest, read, tree_hashes, validate_records, write_json,
)

SOURCE = ROOT / "data/curated/mpci-bl-real-v7-reviewed-r5-filtered"
DESTINATION = SOURCE.with_name("mpci-bl-real-v7-reviewed-r6-single-goods")
SELECTED = "001/02-44a6df44"
PDF_CONTAINER = re.compile(r"(?<![A-Z])[A-Z]{3}[UJZ][\s-]*\d(?:[\s-]*\d){6}(?!\d)")
VISUAL_CONTAINER_FINDINGS = {
    "001/01-88e5a4eb": "PDF marks column has two container IDs; both absent from OCR.",
    "004/022-eec05b59": "PDF lower-left container/seal/type line is absent from OCR; CHEMRIDER is a mark, not its container ID.",
    "003/050-753394dc": "PDF prints ONEU5008417; OCR prints ONEEU5008417 (extra E), so the existing target cannot represent the corrupted identifier.",
    "003/068-df5d72cd": "DFDS trailer shipment: HROB2297 / WKESD000000844299 are vehicle/trailer identifiers, not ISO container IDs.",
}


def exclusions(row: dict, patch: dict, repeated: set[str]) -> list[str]:
    result = []
    goods = patch.get("goodsItemDetails", [])
    if any(not g.get("description") for g in goods):
        result.append("goods_without_description")
    if len(goods) > 1:
        result.append("multiple_labeled_goods_out_of_scope")
    if row["sample"] in repeated and row["sample"] != SELECTED:
        result.append("redundant_shipment_copy")
    return result


def pdf_text(path: str) -> str:
    texts = []
    with pdfium.PdfDocument(ROOT / path) as pdf:
        for page in pdf:
            textpage = page.get_textpage()
            texts.append(textpage.get_text_range())
            textpage.close()
            page.close()
    return "\n".join(texts)


def links(row: dict) -> str:
    sample = f"../../../{SOURCE.relative_to(ROOT)}/samples/{row['documentId']}"
    return (f"[{row['sample']}]({sample}/ocr.txt) · [labels]({sample}/labels.json) · "
            f"[PDF](../../../{row['pdf']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    previous = read(SOURCE / "filter-manifest.json")
    before = tree_hashes(SOURCE)
    assert before == previous["files"] | {"filter-manifest.json": digest(SOURCE / "filter-manifest.json")}
    inventory = {r["documentId"]: r for r in read(ANALYSIS / "inventory.json")}
    cohorts = read(ANALYSIS / "reviewed_cohorts.json")
    repeated = set(cohorts["repeated_ferro_molybdenum_shipment"]["samples"])
    originals, patches = {}, {}
    for split in ("train", "validation"):
        for line in (SOURCE / f"{split}.jsonl").read_bytes().splitlines(keepends=True):
            record = json.loads(line)
            key = record["documentId"]
            assert key not in originals
            originals[key] = (split, line)
            patches[key] = record["target"]["documentPatch"]
    validate_records(SOURCE, originals)
    selected_ids = [k for k in originals if inventory[k]["sample"] == SELECTED]
    assert len(selected_ids) == 1
    removed = {k: exclusions(inventory[k], patches[k], repeated) for k in originals}
    removed = {k: reasons for k, reasons in removed.items() if reasons}
    retained = {k: v for k, v in originals.items() if k not in removed}
    assert len(removed) == 61 and len(retained) == 360
    assert Counter(r for values in removed.values() for r in values) == {
        "goods_without_description": 9, "multiple_labeled_goods_out_of_scope": 42,
        "redundant_shipment_copy": 10,
    }
    assert selected_ids[0] in retained
    assert all(len(patches[k]["goodsItemDetails"]) == 1 and patches[k]["goodsItemDetails"][0].get("description") for k in retained)

    comparison = []
    for key in originals:
        row = inventory[key]
        if row["sample"] not in repeated:
            continue
        text = (SOURCE / "samples" / key / "ocr.txt").read_text()
        body, wrapper = text.split("File details", 1)
        patch = patches[key]
        assert all(c in body for c in ("GCXU2131234", "TLLU2582822", "TRHU3400993", "212776446", "212776401", "212776447"))
        comparison.append({
            "sample": row["sample"], "documentId": key, "pdf": row["pdf"],
            "ocrCharacters": len(text), "bodyCharacters": len(body), "wrapperCharacters": len(wrapper),
            "measurementValuePrinted": "60.0000" in body, "measurementUnitPrinted": "CBM" in body,
            "noMarksPrinted": bool(re.search(r"\bN/M\b", body)),
            "blNumberPresentInOcr": "PUSE97516700" in text,
            "volumeLabel": patch["goodsItemDetails"][0].get("volume"),
            "deliveryAgentLabel": patch.get("parties", {}).get("deliveryAgent"),
            "equipmentLabels": patch["containerInformation"],
            "routeCountries": {role: loc.get("country") for role, loc in patch["route"].items()},
            "pdfTextCharacters": len(pdf_text(row["pdf"])),
            "selected": row["sample"] == SELECTED,
        })
    assert len(comparison) == 11
    assert sum(r["measurementValuePrinted"] and r["measurementUnitPrinted"] for r in comparison) == 2

    no_containers = []
    for key in originals:
        if patches[key].get("containerInformation"):
            continue
        row = inventory[key]
        category = next(c for c in row["reviewedCohorts"] if c.startswith("no_container_"))
        details = {
            "sample": row["sample"], "documentId": key, "split": originals[key][0],
            "category": category, "retained": key in retained,
            "exclusionReasons": removed.get(key, []), "pdf": row["pdf"],
            "descriptions": row["goodsDescriptions"],
            "visualFinding": VISUAL_CONTAINER_FINDINGS.get(row["sample"]),
        }
        if category == "no_container_other_marine":
            native = pdf_text(row["pdf"])
            ocr = (SOURCE / "samples" / key / "ocr.txt").read_text()
            details["pdfNativeTextCharacters"] = len(native.strip())
            details["pdfNativeContainerIds"] = sorted({re.sub(r"[\s-]", "", s) for s in PDF_CONTAINER.findall(native.upper())})
            details["ocrContainerIds"] = sorted({re.sub(r"[\s-]", "", s) for s in PDF_CONTAINER.findall(ocr.upper())})
        no_containers.append(details)
    assert len(no_containers) == 64 and sum(r["retained"] for r in no_containers) == 56
    audit = {
        "selected": SELECTED, "selectedDocumentId": selected_ids[0],
        "selectionBasis": "Joint best OCR body coverage (including 60 CBM); canonical equipment and route-country extraction, and fewer explicit QA/test-account cues in wrapper than the tied OCR candidate. No content was synthesized or repaired.",
        "selectionLimits": "All eleven OCR variants omit the printed B/L number; the selected variant also omits PDF-only N/M. Selection is relative quality, not certification of complete PDF transcription. Existing labels and OCR are preserved.",
        "shipmentComparison": comparison, "noContainerInventory": no_containers,
    }
    result = {
        "date": "2026-10-05", "schemaVersion": "7.0.0", "sourceDataset": str(SOURCE.relative_to(ROOT)),
        "dataset": str(DESTINATION.relative_to(ROOT)), "priorManifestSha256": digest(SOURCE / "filter-manifest.json"),
        "sourceFiles": before, "before": dict(Counter(s for s, _ in originals.values())),
        "after": dict(Counter(s for s, _ in retained.values())), "excludedCount": len(removed), "retainedCount": len(retained),
        "excludedReasons": dict(Counter(r for values in removed.values() for r in values)),
        "selectedDuplicateRepresentative": SELECTED,
        "excluded": [{"documentId": k, "sample": inventory[k]["sample"], "split": originals[k][0], "reasons": v} for k, v in removed.items()],
        "retained": [{"documentId": k, "sample": inventory[k]["sample"], "split": v[0]} for k, v in retained.items()],
        "scope": "Exactly one described goods entry per current label; no semantic merging or certification of latent source topology.",
        "remainingNoContainerDocuments": 56, "labelEdits": 0, "ocrEdits": 0, "apiCostUsd": 0,
    }
    if not args.apply:
        print(json.dumps({k: result[k] for k in ("before", "after", "excludedReasons", "selectedDuplicateRepresentative", "remainingNoContainerDocuments")}, indent=2))
        return

    # Preserve R5 and the full source backup; publish only into a new revision.
    stage = Path(tempfile.mkdtemp(prefix=".real-v7-single-goods-", dir=SOURCE.parent))
    shutil.copytree(SOURCE / "batches", stage / "batches")
    shutil.copy2(SOURCE / "schema.json", stage / "schema.json")
    (stage / "samples").mkdir()
    for key in retained:
        shutil.copytree(SOURCE / "samples" / key, stage / "samples" / key)
    for split in ("train", "validation"):
        with (stage / f"{split}.jsonl").open("wb") as stream:
            for s, line in retained.values():
                if s == split:
                    stream.write(line)
    validate_records(stage, retained)
    for name, sha in tree_hashes(stage).items():
        if name not in ("train.jsonl", "validation.jsonl"):
            assert before[name] == sha
    assert tree_hashes(SOURCE) == before
    (stage / "README.md").write_text(
        "# Reviewed real baseline R6 — one labeled goods entry\n\n"
        "360 documents: **319 training / 41 validation**. User-approved selection, 2026-10-05.\n\n"
        "From R5, excludes nine goods-without-description records, 42 multi-goods-labeled records and "
        "ten redundant shipment copies. Keeps `001/02-44a6df44` from the eleven-copy group after OCR/label/PDF comparison. "
        "Retained OCR, labels, record order and split assignments are unchanged.\n\n"
        "This is a label-count-based scope restriction, not a claim that every remaining source is semantically "
        "unambiguous about goods grouping. No goods entries were merged. Fifty-six no-container records remain; "
        "no additional container-based filter was authorized.\n\n"
        "`filter-manifest.json` defines current membership. `batches/` contains unchanged historical receipts "
        "including excluded records, and must not be used as current membership. Current described schema is "
        "`batches/004/label-schema.json`.\n\n"
        "R5 remains intact at `../mpci-bl-real-v7-reviewed-r5-filtered/`. Complete original source backup: "
        "`../mpci-bl-real-v7-reviewed-r4_source_20261005/`. Every excluded record is recoverable.\n\n"
        "Analysis: `docs/analysis/real-v7-starting-dataset-2026-10-04/BASELINE_REFINEMENT_2026-10-05.md`. "
        "Training configs were not redirected or launched.\n"
    )
    result["files"] = tree_hashes(stage)
    result["elapsedSeconds"] = time.monotonic() - started
    result["peakRssKiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(stage / "filter-manifest.json", result)
    stage.rename(DESTINATION)
    write_json(ANALYSIS / "BASELINE_REFINEMENT_AUDIT.json", audit)
    lines = ["# No-container cohort: all 64 R5 records\n\n"
             "56 remain in R6; eight are excluded for the requested goods-scope filters, not for container absence. "
             "Native-PDF identifier matches are source evidence; no match in a scanned PDF is inconclusive. "
             "No PDF-only labels were added.\n\n"
             "| Sample / files | Category | R6 status | PDF/OCR finding |\n|---|---|---|---|\n"]
    for r in no_containers:
        finding = r["visualFinding"] or ""
        if r.get("pdfNativeContainerIds"):
            finding += " PDF text contains " + ", ".join(r["pdfNativeContainerIds"]) + "; absent from OCR identifier screen."
        status = "Retained" if r["retained"] else "; ".join(r["exclusionReasons"])
        lines.append(f"| {links(inventory[r['documentId']])} | {r['category'].removeprefix('no_container_')} | {status} | {finding or 'No new full-PDF adjudication; see cohort discussion.'} |\n")
    (ANALYSIS / "NO_CONTAINER_REVIEW_2026-10-05.md").write_text("".join(lines))
    with (ANALYSIS / "R6_RETAINED_INVENTORY.csv").open("w", newline="") as stream:
        fields = ["sample", "documentId", "split", "pdfPages", "containerCount", "ocr", "labels", "pdf"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in retained:
            row = dict(inventory[key])
            for f, filename in (("ocr", "ocr.txt"), ("labels", "labels.json")):
                row[f] = str((DESTINATION / "samples" / key / filename).relative_to(ROOT))
            writer.writerow({k: row[k] for k in fields})
    print(json.dumps({k: result[k] for k in ("dataset", "after", "excludedReasons", "selectedDuplicateRepresentative", "elapsedSeconds", "peakRssKiB")}, indent=2))


if __name__ == "__main__":
    main()
