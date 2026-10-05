"""Publish the approved R8 field projection, preserving the complete R7 source.

Dry-run by default. Excludes only the two reviewed product-OCR omissions. Saves
exact field-removal receipts and proves every surviving fact and OCR byte unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import resource
import shutil
import tempfile
import time

from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7Label
from document_ocr.training.config import PromptConfig
from document_ocr.training.prompting import load_prompt
from document_ocr.training.tasks import get_training_task

from filter_real_v7_starting_dataset import (
    ANALYSIS, ROOT, digest, read, tree_hashes, validate_records, write_json,
)

SOURCE = ROOT / "data/curated/mpci-bl-real-v7-reviewed-r7-input-filtered"
DESTINATION = SOURCE.with_name("mpci-bl-real-v7-reviewed-r8-reduced-fields")
EXCLUDED = {"004/023-5fe4e926", "004/106-d7cd79d3"}
FIELDS = {"vesselFlagCountry", "marksAndNumbers", "forwardingAndExportReferences"}


def encode(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()


def project(record: dict) -> tuple[dict, list[dict]]:
    result = deepcopy(record)
    patch = result["target"]["documentPatch"]
    receipts = []

    def remove(owner, field, path):
        if field in owner:
            receipts.append({"path": path + [field], "before": owner.pop(field)})

    remove(patch, "forwardingAndExportReferences", ["documentPatch"])
    if "transport" in patch:
        remove(patch["transport"], "vesselFlagCountry", ["documentPatch", "transport"])
        if not patch["transport"]:
            del patch["transport"]
    for index, goods in enumerate(patch.get("goodsItemDetails", [])):
        remove(goods, "marksAndNumbers", ["documentPatch", "goodsItemDetails", index])

    # Exact inverse proves that removal did not alter other values. Keep the
    # archive as the byte/order-preserving inverse, not only these field receipts.
    restored = deepcopy(result)
    for receipt in receipts:
        owner = restored["target"]
        for part in receipt["path"][:-1]:
            if isinstance(part, str) and part not in owner:
                owner[part] = {}
            owner = owner[part]
        owner[receipt["path"][-1]] = receipt["before"]
    assert restored == record, record["documentId"]
    return result, receipts


def property_names(value):
    if isinstance(value, dict):
        yield from value.get("properties", {})
        for child in value.values():
            yield from property_names(child)
    elif isinstance(value, list):
        for child in value:
            yield from property_names(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    prior = read(SOURCE / "filter-manifest.json")
    before = tree_hashes(SOURCE)
    assert before == prior["files"] | {"filter-manifest.json": digest(SOURCE / "filter-manifest.json")}
    review = ANALYSIS / "r7_field_audit"
    assert read(review / "validation.json")["datasetManifestSha256"] == digest(SOURCE / "filter-manifest.json")
    inventory = {r["documentId"]: r for r in read(review / "records.json")}
    approved = {r["documentId"] for r in inventory.values() if r["sample"] in EXCLUDED}
    assert len(approved) == 2
    task = get_training_task("bill_of_lading_extraction_v7")
    schema = BillOfLadingExtractionV7Label.model_json_schema()
    prompt_schema = json.loads(task.prompt_schema_json())
    assert not FIELDS.intersection(property_names(schema))
    assert not FIELDS.intersection(property_names(prompt_schema))
    prompt = load_prompt(ROOT, PromptConfig(
        path="prompts/training/mpci_bl_extraction_v7.txt",
        placeholder="{{document_text}}", schema_placeholder="{{output_schema}}",
    ), task)

    expected, originals, receipts, exclusions = {}, {}, [], []
    field_documents, field_values = Counter(), Counter()
    projected_seconds = 0.0
    for split in ("train", "validation"):
        for line in (SOURCE / f"{split}.jsonl").read_bytes().splitlines(keepends=True):
            record = json.loads(line)
            key = record["documentId"]
            assert key not in originals
            originals[key] = record
            sample = SOURCE / "samples" / key
            assert (sample / "ocr.txt").read_text() == record["joinedRawText"]
            assert read(sample / "labels.json") == record["target"]
            if key in approved:
                exclusions.append({"documentId": key, "sample": inventory[key]["sample"], "split": split,
                                   "reason": "pdf_confirmed_omitted_product_description"})
                continue
            tick = time.perf_counter()
            projected, edits = project(record)
            assert task.canonicalize(projected["target"]) == projected["target"]
            assert len(projected["target"]["documentPatch"]["goodsItemDetails"]) == 1
            assert projected["target"]["documentPatch"]["goodsItemDetails"][0].get("description")
            assert projected["joinedRawText"] in prompt.render(projected["joinedRawText"])
            projected_seconds += time.perf_counter() - tick
            expected[key] = (split, encode(projected))
            if edits:
                receipts.append({"documentId": key, "sample": inventory[key]["sample"], "edits": edits})
            for field in {r["path"][-1] for r in edits}:
                field_documents[field] += 1
            for receipt in edits:
                field_values[receipt["path"][-1]] += len(receipt["before"]) if isinstance(receipt["before"], list) else 1

    assert len(originals) == 335 and len(expected) == 333
    splits = dict(Counter(s for s, _ in expected.values()))
    assert splits == {"train": 296, "validation": 37}
    manifest = {
        "date": "2026-10-05", "schemaVersion": "7.0.0",
        "contractRevision": "real-v7-reduced-fields-20261005",
        "dataset": str(DESTINATION.relative_to(ROOT)), "sourceDataset": str(SOURCE.relative_to(ROOT)),
        "sourceFiles": before, "before": prior["after"], "after": splits,
        "excludedCount": len(exclusions), "excluded": exclusions, "retainedCount": len(expected),
        "retained": [{"documentId": k, "sample": inventory[k]["sample"], "split": s} for k, (s, _) in expected.items()],
        "removedFieldDocuments": dict(field_documents), "removedFieldValues": dict(field_values),
        "documentsWithLabelEdits": len(receipts), "ocrEdits": 0,
        "policy": "Only two product-description OCR omissions excluded. Vessel flag, cargo marks and forwarding/export references removed from both splits and active schema. IMO retained; five partial-container cases remain. No fields moved into descriptions or other targets.",
        "topUpTo450": {"total": 117, "train": 104, "validation": 13},
        "topUpTo500": {"total": 167}, "apiCalls": 0, "apiCostUsd": 0,
        "validation": {"schemaValidDocuments": len(expected), "exactInverseChecks": len(expected),
                       "sameOcrAndSplits": True, "trainingPromptSmokeDocuments": len(expected)},
        "projectionValidationSeconds": round(projected_seconds, 3),
    }
    summary_keys = ("before", "after", "excludedCount", "removedFieldDocuments", "removedFieldValues", "documentsWithLabelEdits", "projectionValidationSeconds", "topUpTo450")
    if not args.apply:
        print(json.dumps({k: manifest[k] for k in summary_keys}, indent=2))
        assert tree_hashes(SOURCE) == before
        return

    stage = Path(tempfile.mkdtemp(prefix=".real-v7-field-projection-", dir=SOURCE.parent))
    (stage / "samples").mkdir()
    for key, (_, line) in expected.items():
        sample = stage / "samples" / key
        sample.mkdir()
        shutil.copy2(SOURCE / "samples" / key / "ocr.txt", sample / "ocr.txt")
        write_json(sample / "labels.json", json.loads(line)["target"])
    for split in ("train", "validation"):
        (stage / f"{split}.jsonl").write_bytes(b"".join(line for s, line in expected.values() if s == split))
    write_json(stage / "schema.json", schema)
    write_json(stage / "prompt-schema.json", prompt_schema)
    (stage / "training-prompt.txt").write_text(prompt.text)
    write_json(stage / "removed-fields.json", receipts)
    (stage / "README.md").write_text(
        "# Reviewed real baseline R8 — reduced fields\n\n"
        "**333 documents: 296 train / 37 validation.** One described goods entry per record.\n\n"
        "Excluded only `004/023-5fe4e926` and `004/106-d7cd79d3` for omitted product OCR. "
        "The five partial-container OCR candidates remain included. Removed vesselFlagCountry, "
        "marksAndNumbers and forwardingAndExportReferences; vesselImoNumber is retained. "
        "OCR, split assignments and every other target value are unchanged.\n\n"
        "`schema.json` is the active validation contract; `prompt-schema.json` and "
        "`training-prompt.txt` snapshot the runtime training prompt. The V7 working contract "
        "was narrowed; historical revisions retain their original schema snapshots. "
        "Use this revision with the reduced active schema, not older unprojected V7 labels.\n\n"
        "`projection-manifest.json` defines current membership and integrity. "
        "`removed-fields.json` contains exact removals, not model targets. "
        "The complete unchanged R7 archive, historical schemas and adjudication receipts "
        "remain at `../mpci-bl-real-v7-reviewed-r7-input-filtered/`. "
        "No training config has been redirected and no new documents were processed.\n\n"
        "Top-up to the original 450 (400 train / 50 validation): 117 accepted documents, "
        "comprising 104 train and 13 validation. To reach 500 total: 167.\n"
    )
    validate_records(stage, expected)
    for key in expected:
        assert digest(stage / "samples" / key / "ocr.txt") == digest(SOURCE / "samples" / key / "ocr.txt")
    assert tree_hashes(SOURCE) == before
    manifest["implementation"] = {str(p.relative_to(ROOT)): digest(p) for p in (
        Path(__file__), ROOT / "src/document_ocr/label_schemas/bill_of_lading_v7.py",
        ROOT / "src/document_ocr/labeling_agents/direct_models.py",
        ROOT / "src/document_ocr/labeling_agents/direct_cargo.py",
        ROOT / "prompts/training/mpci_bl_extraction_v7.txt",
    )}
    manifest["elapsedSeconds"] = round(time.monotonic() - start, 3)
    manifest["peakRssKiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    manifest["files"] = tree_hashes(stage)
    write_json(stage / "projection-manifest.json", manifest)
    stage.rename(DESTINATION)
    print(json.dumps({k: manifest[k] for k in (*summary_keys, "elapsedSeconds", "peakRssKiB")}, indent=2))


if __name__ == "__main__":
    main()
