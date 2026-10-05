"""Publish the manually adjudicated 50-source pilot, without changing source data.

These finite decisions are dataset receipts, not rules in the extraction pipeline.
All outputs are built in a staging directory and validated before publication.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.hashing import sha256_bytes, sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7Label

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "artifacts/kie-labeling/direct-full-20261004-luna-high50-r5"
DEST = ROOT / "data/curated/mpci-bl-real-v7-reviewed"


def edits(index, patch):
    """Apply only individually inspected decisions; return exact-path edit receipts."""
    changes = []
    decisions = []

    def edit(path, before, after, reason):
        node = patch
        for part in path[:-1]:
            node = node[part]
        assert node[path[-1]] == before, (index, path, node[path[-1]])
        if after is None:
            del node[path[-1]]
        else:
            node[path[-1]] = after
        changes.append(dict(path=list(path), before=before, after=after, reason=reason))

    if index == 15:
        edit(
            ("parties", "consignee", "addressLine"),
            "UMAR IBN AL-KHATTAB BUILD 5A PART 9, PORT SAID(PORT SAID-EGYPT)",
            "UMAR IBN AL-KHATTAB BUILD 5A PART 9, PORT SAID-EGYPT",
            "Remove duplicate PORT SAID and its redundant parentheses, "
            "retaining all numbers and country.",
        )
    if index == 16:
        edit(
            ("transport", "voyageNumber"),
            "0MRFKE1MA",
            None,
            "OCR has competing voyage identifiers 0MRFKE1MA and OMRFKE1WA. "
            "Single undecidable value is absent under the extractor contract; "
            "no majority vote or PDF transcription.",
        )
        decisions.append(
            "Keep both distinct literal VAT references and both distinct printed phone strings: "
            "these fields are lists of extracted declarations, not a selected authoritative "
            "VAT/telephone. Do not claim the underlying document is internally consistent."
        )
    if index == 17:
        edit(
            ("parties", "consignee", "addressLine"),
            "EL GHARBIYA, 23 JULY STREET, 31714 ELTANZIM, BAS/EGYPT Egypt",
            "EL GHARBIYA, 23 JULY STREET, 31714 ELTANZIM, BAS/EGYPT",
            "Remove repeated terminal country; retain BAS qualifier and postcode.",
        )
    if index in (18, 43):
        old = patch["parties"]["notifyParties"][0]["addressLine"]
        assert old.endswith("TAIWAN, R.O.C, TW")
        edit(
            ("parties", "notifyParties", 0, "addressLine"),
            old,
            old.removesuffix(", TW"),
            "TW repeats the preceding complete postal country expression TAIWAN, R.O.C; "
            "keep that expression once.",
        )
        if index == 43:
            edit(
                ("parties", "notifyParties", 0, "country"),
                "TAIWAN, R.O.C, TW",
                "TAIWAN",
                "Return the printed country name once, consistent with the equivalent "
                "notify block in document 18.",
            )
    if index == 36:
        decisions.append(
            "Keep both ACID strings exactly as OCR prints them in the reference list. "
            "The target extracts distinct captioned references; it neither selects a valid "
            "customs identifier nor silently repairs OCR. PDF confirms both blocks concern "
            "this shipment. Do not add PDF-only 500/430 local bag counts."
        )
    if index == 39:
        edit(
            ("parties", "consignee", "addressLine"),
            "NASR CITY, PUBLIC FREE ZONE, BLOCK G, LOT NO. (7,10,11), "
            "NASR CITY, CAIRO, 11816, EGYPT",
            "NASR CITY, PUBLIC FREE ZONE, BLOCK G, LOT NO. (7,10,11), CAIRO, 11816, EGYPT",
            "Remove the second identical NASR CITY component; preserve block, lots and postcode.",
        )
    if index == 49:
        edit(
            ("goodsItemDetails", 0, "origin"),
            {"name": "Finland"},
            None,
            "FINNISH WHITEWOOD is printed product wording, not a separate origin/manufacture "
            "declaration. Retain the complete description; do not infer origin from the "
            "loading country.",
        )
        decisions.append(
            "Retain absence of grossWeight: the PDF also prints KOS and does not establish "
            "a supported mass unit. Keep 148 packages and 465.332 cubic metres. Reg.No is "
            "unrelated company registration, not a shipment reference; signing agent is "
            "not automatically a forwarding agent."
        )
    if index == 50:
        decisions.append(
            "Manual source/PDF review confirms contradictory local versus total gross/net masses. "
            "Apply the existing undecidable-value absence rule to both mass fields; do not swap "
            "headers or prefer totals. Keep 1400 bags and two explicit 700-bag placements. "
            "This closes the target decision, not the factual contradiction "
            "in the original shipment."
        )
    return changes, decisions


def replay(before, changes):
    value = copy.deepcopy(before)
    for change in changes:
        node = value
        for part in change["path"][:-1]:
            node = node[part]
        key = change["path"][-1]
        assert node[key] == change["before"]
        if change["after"] is None:
            del node[key]
        else:
            node[key] = change["after"]
    return value


def main():
    started = time.monotonic()
    assert not DEST.exists(), "Never overwrite an existing dataset"
    selection = json.loads((PILOT / "selection.json").read_text())
    audit = json.loads((PILOT / "adjudicated-pilot/manifest.json").read_text())
    prior = {r["document"]: r for r in audit["documents"]}
    original = {
        r["documentId"]: r
        for r in map(
            json.loads,
            (
                ROOT
                / "artifacts/kie-training/analysis/real-data-baseline-audit-20261001"
                / "trained_real_v6.jsonl"
            ).open(),
        )
    }
    # Re-evaluate the source-fixed controls on the published targets, not old booleans.
    sys.path.insert(0, str(PILOT))
    spec = importlib.util.spec_from_file_location("readiness", PILOT / "readiness_checks.py")
    controls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controls)
    previous_checks = json.loads((PILOT / "final-audit.json").read_text())["checks"]
    assert len(previous_checks) == 140 and all(c["passed"] for c in previous_checks)
    checked = []
    receipts = []
    records = {"train": [], "validation": []}
    DEST.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".mpci-bl-real-v7-batch1-", dir=DEST.parent))
    for row in selection["documents"]:
        name, i = row["name"], row["index"]
        old = prior[name]
        baseline = json.loads((ROOT / old["target"]).read_text())
        target = copy.deepcopy(baseline)
        changes, decisions = edits(i, target["documentPatch"])
        assert replay(baseline["documentPatch"], changes) == target["documentPatch"]
        target = BillOfLadingExtractionV7Label.model_validate_json(
            json.dumps(target)
        ).canonical_target()
        assert target == replay(
            baseline,
            [
                {
                    "path": ["documentPatch", *c["path"]],
                    **{k: v for k, v in c.items() if k != "path"},
                }
                for c in changes
            ],
        )
        text = (ROOT / row["input"] / "ocr.txt").read_text()
        source = original[row["documentId"]]
        assert text == source["joinedRawText"]
        assert source["auditSplit"] == row["split"]
        assert sha256_bytes(text.encode()) == row["ocrSha256"]
        assert sha256_file(ROOT / row["pdf"]) == row["pdfSha256"]
        p = target["documentPatch"]
        # Historical control 50 checked that the conflict was caught; its receipt is
        # retained, and publication independently verifies the adjudicated omission.
        status = {"reviews": {"cargo": {"findings": old["problems"]}}}
        named = {
            **controls.controls.values(i, p),
            **controls.extra(i, p),
            **controls.expanded(i, p),
            **controls.supplemental(i, p),
            **controls.new_checks(i, p, status),
        }
        if i == 10:
            named["consignee_form_instruction_is_not_order_consigning"] = (
                p.get("negotiability") == "non_negotiable"
            )
        expected = (
            "negotiable"
            if i in (3, 6, 36, 41)
            else None
            if i in (8, 19, 23, 42, 46, 50)
            else "non_negotiable"
        )
        named["actual_consignee_not_conditional_boilerplate"] = p.get("negotiability") == expected
        if i == 18:
            named["linked_shipper_identity_continuation"] = (
                "ON BEHALF OF SAMSUNG ELECTRONICS TAIWAN" in p["parties"]["shipper"]["name"]
            )
        if i == 16:
            named["conflicting_single_voyage_absent"] = "voyageNumber" not in p["transport"]
            named["both_literal_VATs_retained"] = all(
                any(v in ref for ref in p["forwardingAndExportReferences"])
                for v in ("298712164", "299712164")
            )
        if i == 36:
            named["both_literal_ACIDs_retained"] = all(
                "ACID: " + v in p["forwardingAndExportReferences"]
                for v in ("565169637202302022", "5651696372023020022")
            )
        if i == 49:
            named["product_adjective_not_origin_declaration"] = (
                not p["goodsItemDetails"][0].get("origin")
                and p["goodsItemDetails"][0]["description"] == "FINNISH WHITEWOOD"
            )
        if i == 50:
            g = p["goodsItemDetails"][0]
            named["conflicting_masses_absent_bags_preserved"] = (
                not g.get("grossWeight")
                and not g.get("netWeight")
                and sum(x["packageQuantity"] for x in g["splitGoodsPlacement"]) == 1400
            )
        assert all(named.values()), (name, named)
        checked.extend(dict(document=name, check=k, passed=v) for k, v in named.items())
        record = dict(
            documentId=row["documentId"],
            joinedRawText=text,
            joinedRawTextSha256=row["ocrSha256"],
            target=target,
        )
        records[row["split"]].append(record)
        folder = stage / "samples" / row["documentId"]
        atomic_publish_bytes(folder / "ocr.txt", text.encode())
        atomic_publish_json(folder / "labels.json", target)
        receipts.append(
            dict(
                documentId=row["documentId"],
                pilotDocument=name,
                split=row["split"],
                status="manually_adjudicated",
                sourceOcrSha256=row["ocrSha256"],
                pdf=row["pdf"],
                pdfSha256=row["pdfSha256"],
                inputTarget=old["target"],
                inputTargetSha256=sha256_file(ROOT / old["target"]),
                priorManualChanges=old["manualChanges"],
                priorManualDecisions=old["manualDecisions"],
                changes=changes,
                decisions=decisions,
                resolvedFindings=old["problems"],
                targetSha256=sha256_file(folder / "labels.json"),
            )
        )
    assert sum(map(len, records.values())) == 50
    assert len({r["documentId"] for rows in records.values() for r in rows}) == 50
    assert len({r["joinedRawTextSha256"] for rows in records.values() for r in rows}) == 50
    for split, rows in records.items():
        atomic_publish_bytes(
            stage / f"{split}.jsonl",
            "".join(
                json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows
            ).encode(),
        )
        assert list(map(json.loads, (stage / f"{split}.jsonl").read_text().splitlines())) == rows
    manifest = dict(
        datasetVersion="real-v7-reviewed-batch001",
        schemaVersion="7.0.0",
        batch="001",
        samples=50,
        splits={k: len(v) for k, v in records.items()},
        adjudication=(
            "Source-grounded extraction under V7 policy; explicit undecidable scalar absences "
            "recorded in receipts. Reference lists preserve conflicting printed strings, "
            "without selecting a true external identifier."
        ),
        sourceSelection=str((PILOT / "selection.json").relative_to(ROOT)),
        sourceSelectionSha256=sha256_file(PILOT / "selection.json"),
        files={f"{s}.jsonl": sha256_file(stage / f"{s}.jsonl") for s in records},
        documents=receipts,
    )
    validation = dict(
        samples=50,
        schemasValid=50,
        sourceTextsUnchanged=50,
        sourcePdfsUnchanged=50,
        exactEditReplays=50,
        splitMembershipPreserved=50,
        checks=checked,
        additionalEditedDocuments=sum(bool(r["changes"]) for r in receipts),
        totalManuallyEditedDocuments=sum(
            bool(r["changes"] or r["priorManualChanges"]) for r in receipts
        ),
        remainingAdjudications=0,
        paidCalls=0,
        estimatedUsd=0,
        elapsedSeconds=time.monotonic() - started,
    )
    atomic_publish_json(stage / "batches/001/manifest.json", manifest)
    atomic_publish_json(stage / "batches/001/validation.json", validation)
    atomic_publish_json(stage / "schema.json", BillOfLadingExtractionV7Label.model_json_schema())
    readme = """# Reviewed real Bill-of-Lading dataset (V7)

Batch 001: 50 manually adjudicated documents (40 train, 10 validation).
Original real OCR is byte-preserved; only labels changed. No synthetic data.

`train.jsonl` and `validation.jsonl` use documentId, joinedRawText, joinedRawTextSha256 and target.
Per-document OCR and labels are in `samples/`. Provenance, exact edits and source decisions
are in `batches/001/manifest.json`; validation results are alongside it.

Targets follow the described V7 extraction policy: full addressLine plus country,
no city or additionalInformation, goods-owned facts and final splitGoodsPlacement.
Missing or genuinely undecidable single values are absent, not guessed. Lists of
references retain distinct explicit strings even when a real source contradicts itself.
These decisions do not certify external shipment/customs truth. PDFs remain at their
original repository paths recorded with hashes; no PDF-only label enrichment.

Only manually adjudicated batches belong here. New automated batch candidates live
in artifacts/kie-labeling until their separate review is complete. Preserve the original
train/validation assignments when adding batches; do not use this pilot to claim an
untouched evaluation set or population-wide accuracy. No training has been launched.
"""
    atomic_publish_bytes(stage / "README.md", readme.encode())
    os.rename(stage, DEST)
    print(json.dumps({k: v for k, v in validation.items() if k != "checks"}, indent=2))
    print("checks", len(checked), "dataset", DEST)


if __name__ == "__main__":
    main()
