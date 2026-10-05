"""Publish approved exclusions and reviewed abbreviated-phone removals; no OCR edits."""

from __future__ import annotations

import argparse
import json
import re
import resource
import shutil
import tempfile
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

from audit_real_v7_r10 import OUT, SOURCE, selection
from filter_real_v7_starting_dataset import (
    ROOT,
    digest,
    read,
    tree_hashes,
    validate_records,
    write_json,
)
from jsonschema import Draft202012Validator
from project_real_v7_r9 import ARCHIVE, records

from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7Label, ContactsV7

DEST = SOURCE.with_name("mpci-bl-real-v7-reviewed-r10-reduced-474")
REPAIRS = {
    "doc_5242647c15b8d9847aef3c4321fb5d93df660576d1aaab7e6e4e2f261124de04": {
        "role": "consignee",
        "remove": ["353", "131"],
        "evidence": "AHMED TEL.+20 2 44874385 , 353 , 131 MOB:+201271781112",
    },
    "doc_eec05b59509a883c3f5cc69080b6c675d074690bd3886b4d741a14f16afd3fe2": {
        "role": "deliveryAgent",
        "remove": ["97"],
        "evidence": "TEL:+2034210096 / 97",
    },
}


def short_phones(record):
    """Broad audit candidates, not a worldwide minimum phone-length validation rule."""
    result = []
    for role, value in record["target"]["documentPatch"].get("parties", {}).items():
        for index, party in enumerate(value if isinstance(value, list) else [value]):
            for phone in party.get("contactDetails", {}).get("phoneNumbers", []):
                if len(re.sub(r"\D", "", phone)) < 7:
                    result.append({"role": role, "index": index, "phone": phone})
    return result


def repair(record):
    result = deepcopy(record)
    change = REPAIRS.get(record["documentId"])
    if change is None:
        return result, []
    assert change["evidence"] in record["joinedRawText"].splitlines()
    owner = result["target"]["documentPatch"]["parties"][change["role"]]["contactDetails"]
    before = owner["phoneNumbers"]
    assert set(change["remove"]) <= set(before)
    after = [p for p in before if p not in change["remove"]]
    assert after
    owner["phoneNumbers"] = after
    edit = {
        "path": ["documentPatch", "parties", change["role"], "contactDetails", "phoneNumbers"],
        "before": before,
        "after": after,
        "reason": "Printed suffixes are not independent telephone numbers.",
        "evidence": change["evidence"],
    }
    restored = deepcopy(result)
    restored["target"]["documentPatch"]["parties"][change["role"]]["contactDetails"][
        "phoneNumbers"
    ] = before
    assert restored == record
    return result, [edit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    if args.apply and DEST.exists():
        raise FileExistsError(DEST)
    original = tree_hashes(SOURCE)
    manifest = read(SOURCE / "projection-manifest.json")
    assert original == manifest["files"] | {
        "projection-manifest.json": digest(SOURCE / "projection-manifest.json")
    }
    archive_hashes = tree_hashes(ARCHIVE)
    rows, excluded = selection()
    # Scan all 617 unique reviewed sources, including records filtered previously.
    pool = {r["documentId"]: r for _, r in records(ARCHIVE)}
    pool.update({r["documentId"]: r for _, r in records(SOURCE)})
    assert len(pool) == 617
    candidates = {key: short_phones(r) for key, r in pool.items() if short_phones(r)}
    assert set(candidates) == set(REPAIRS), "New short-number candidate needs source review."
    receipts, expected, provenance = [], {}, []
    schemas = {name: read(SOURCE / name) for name in ("schema.json", "prompt-schema.json")}
    description = ContactsV7.model_fields["phoneNumbers"].description
    # The compact training schema deliberately omits descriptions; preserve that
    # contract and its prompt byte-for-byte. The full validation schema documents
    # the clarified annotation rule without changing any structural constraint.
    schemas["schema.json"]["$defs"]["ContactsV7"]["properties"]["phoneNumbers"]["description"] = (
        description
    )
    for schema in schemas.values():
        Draft202012Validator.check_schema(schema)
    validators = [Draft202012Validator(s) for s in schemas.values()]
    baseline_seconds, repaired_seconds = 0.0, 0.0
    for split, record, info in rows:
        tick = time.perf_counter()
        BillOfLadingExtractionV7Label.model_validate_json(json.dumps(record["target"]))
        baseline_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        result, edits = repair(record)
        BillOfLadingExtractionV7Label.model_validate_json(json.dumps(result["target"]))
        repaired_seconds += time.perf_counter() - tick
        for validator in validators:
            validator.validate(result["target"])
        assert not short_phones(result)
        assert result["joinedRawText"] == record["joinedRawText"]
        key = result["documentId"]
        expected[key] = (
            split,
            (json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n").encode(),
        )
        provenance.append(info | {"sourceDataset": str(SOURCE.relative_to(ROOT))})
        if edits:
            receipts.append({"documentId": key, "sample": info["sample"], "edits": edits})
    assert len(receipts) == 1
    assert len(expected) == 474
    # The user-cited sample was excluded in R7. Repair a reference copy without
    # resurrecting it or rewriting immutable source datasets.
    reference_key = next(k for k in REPAIRS if k not in expected)
    reference, reference_edits = repair(pool[reference_key])
    BillOfLadingExtractionV7Label.model_validate_json(json.dumps(reference["target"]))
    assert not short_phones(reference)
    summary = {
        "documents": len(expected),
        "splits": dict(Counter(s for s, _ in expected.values())),
        "excluded": len(excluded),
        "missingBothPrimaryParties": 24,
        "olderDuplicateRemoved": 1,
        "dummyWrapperRemoved": 1,
        "activeEditedDocuments": 1,
        "activePhoneSuffixesRemoved": 2,
        "excludedReferenceEditedDocuments": 1,
        "excludedReferenceSuffixesRemoved": 1,
        "uniqueSourcesScreenedForShortPhones": len(pool),
        "ocrEdits": 0,
        "schemaValidated": len(expected),
        "apiCalls": 0,
        "apiCostUsd": 0,
        "baselineModelValidationSeconds": round(baseline_seconds, 6),
        "repairAndModelValidationSeconds": round(repaired_seconds, 6),
    }
    OUT.mkdir(exist_ok=True)
    write_json(
        OUT / "short-phone-audit.json",
        {"sources": len(pool), "candidates": candidates, "decisions": REPAIRS},
    )
    if not args.apply:
        assert tree_hashes(SOURCE) == original and tree_hashes(ARCHIVE) == archive_hashes
        print(json.dumps(summary, indent=2))
        return
    stage = Path(tempfile.mkdtemp(prefix=".real-v7-r10-", dir=DEST.parent))
    for key, (_, line) in expected.items():
        folder = stage / "samples" / key
        folder.mkdir(parents=True)
        shutil.copy2(SOURCE / "samples" / key / "ocr.txt", folder / "ocr.txt")
        write_json(folder / "labels.json", json.loads(line)["target"])
    for split in ("train", "validation"):
        (stage / f"{split}.jsonl").write_bytes(
            b"".join(line for s, line in expected.values() if s == split)
        )
    for name, schema in schemas.items():
        write_json(stage / name, schema)
    shutil.copy2(SOURCE / "training-prompt.txt", stage / "training-prompt.txt")
    write_json(stage / "label-edits.json", receipts)
    write_json(stage / "exclusions.json", excluded)
    (stage / "README.md").write_text(
        "# Current real baseline: R10\n\n474 documents: 427 train / 47 validation.\n\n"
        "Removed 24 records without shipper and consignee, one older shipment duplicate "
        "and one user-identified dummy electronic B/L wrapper. Removed two abbreviated "
        "telephone suffix targets from one retained document. OCR, PDFs, other labels "
        "and retained split assignments are unchanged. R9 remains the complete pre-edit backup.\n\n"
        "Use this revision's frozen reduced schema and prompt together. No training was launched "
        "or config redirected. Page-set/container/source-value review candidates remain included "
        "pending discussion; no extra exclusions are implied. Full analysis: "
        "docs/analysis/real-v7-starting-dataset-2026-10-04/R10_CLEANSING_2026-10-05.md.\n"
    )
    validate_records(stage, expected)
    assert tree_hashes(SOURCE) == original and tree_hashes(ARCHIVE) == archive_hashes
    summary.update(
        readbackDocuments=len(expected),
        elapsedSeconds=round(time.monotonic() - start, 3),
        peakRssKiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    write_json(stage / "validation.json", summary)
    write_json(
        stage / "projection-manifest.json",
        {
            "dataset": str(DEST.relative_to(ROOT)),
            "sourceDataset": str(SOURCE.relative_to(ROOT)),
            "sourceFiles": original,
            "documents": provenance,
            "summary": summary,
            "implementationSha256": digest(Path(__file__)),
            "files": tree_hashes(stage),
        },
    )
    stage.rename(DEST)
    folder = OUT / "excluded-reference-004-022-eec05b59"
    folder.mkdir(exist_ok=True)
    write_json(folder / "corrected-record.json", reference)
    write_json(folder / "labels.json", reference["target"])
    (folder / "ocr.txt").write_text(reference["joinedRawText"])
    write_json(
        folder / "receipt.json",
        {
            "sourceDataset": str(ARCHIVE.relative_to(ROOT)),
            "sourceRecordSha256": digest(ARCHIVE / "samples" / reference_key / "labels.json"),
            "documentId": reference_key,
            "includedInDataset": False,
            "edits": reference_edits,
        },
    )
    write_json(OUT / "publication-summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
