"""Publish the approved 500-record reduced projection and negotiability audit.

Dry-run by default. Full annotations and previous projections remain immutable.
The four source-reviewed decisions below are dataset receipts, not generation rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import shutil
import tempfile
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

from filter_real_v7_starting_dataset import (
    ANALYSIS,
    ROOT,
    digest,
    read,
    tree_hashes,
    validate_records,
    write_json,
)
from jsonschema import Draft202012Validator

from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7Label
from document_ocr.labeling_agents.direct_grounding import (
    order_consignment_candidates,
    source_fidelity_findings,
)
from document_ocr.training.tasks import get_training_task

OLD = ROOT / "data/curated/mpci-bl-real-v7-reviewed-r8-reduced-fields"
NEW = OLD.with_name("mpci-bl-real-v7-reviewed-batch005-full")
ARCHIVE = OLD.with_name("mpci-bl-real-v7-reviewed-r4_source_20261005")
DEST = OLD.with_name("mpci-bl-real-v7-reviewed-r9-reduced-500")
OUT = ANALYSIS / "r9_reduced_500"
REMOVED = {
    "TransportV7": "vesselFlagCountry",
    "GoodsItemDetailsV7": "marksAndNumbers",
    "BillOfLadingDocumentPatchV7": "forwardingAndExportReferences",
    "ExtractionPartiesV7": "carrier",
}
CORRECTIONS = {
    "doc_c50046f5ac1ee07d1c94120076535b63b062eb2f34a4f8481bd784f87f8dbb2a",
    "doc_e7267fe14343fac7453b8f995aafeec52d9d1037a780da3543acd190e149a9fc",
    "doc_d97350b6c1883d0c0198bc4a2d410a09429b88be518eacf41a2d3d3ecc3aee84",
    "doc_4b109cdfd9337ccd7315e0c49bbea11862be29ba69770317b470e168ea65ef7b",
}


def records(directory):
    for split in ("train", "validation"):
        for line in (directory / f"{split}.jsonl").read_text().splitlines():
            yield split, json.loads(line)


def reduced_schema(schema):
    result = deepcopy(schema)
    for definition, field in REMOVED.items():
        node = result["$defs"][definition]
        del node["properties"][field]
        if field in node.get("required", []):
            node["required"].remove(field)
    return result


def project(record):
    result = deepcopy(record)
    patch = result["target"]["documentPatch"]
    edits = []

    def remove(owner, field, path):
        if field in owner:
            edits.append({"path": ["documentPatch", *path, field], "before": owner.pop(field)})

    remove(patch, "forwardingAndExportReferences", [])
    for parent, field in (("transport", "vesselFlagCountry"), ("parties", "carrier")):
        if parent in patch:
            remove(patch[parent], field, [parent])
            if not patch[parent]:
                del patch[parent]
    for i, goods in enumerate(patch.get("goodsItemDetails", [])):
        remove(goods, "marksAndNumbers", ["goodsItemDetails", i])
    if record["documentId"] in CORRECTIONS:
        assert patch["negotiability"] == "non_negotiable"
        assert "Consigned to order of" in record["joinedRawText"]
        assert patch["parties"]["consignee"]["name"]
        patch["negotiability"] = "negotiable"
        edits.append(
            {
                "path": ["documentPatch", "negotiability"],
                "before": "non_negotiable",
                "after": "negotiable",
                "reason": (
                    "Populated unconditional Consigned to order of field; consignee-wording policy."
                ),
            }
        )
    restored = deepcopy(result)
    for edit in edits:
        owner = restored["target"]
        for part in edit["path"][:-1]:
            if isinstance(part, str) and part not in owner:
                owner[part] = {}
            owner = owner[part]
        if "after" in edit:
            assert owner[edit["path"][-1]] == edit["after"]
        else:
            assert edit["path"][-1] not in owner
        owner[edit["path"][-1]] = edit["before"]
    assert restored == record, record["documentId"]
    return result, edits


def negotiability_row(record, sample, cohort, active):
    text = record["joinedRawText"]
    patch = record["target"]["documentPatch"]
    # This broad screen is deliberately separate from the narrow flow gate.
    evidence = [
        {"line": i + 1, "text": line}
        for i, line in enumerate(text.splitlines())
        if re.search(r"order|consign|negotiab|waybill", line, re.I)
    ]
    return {
        "documentId": record["documentId"],
        "sample": sample,
        "cohort": cohort,
        "active": active,
        "before": patch.get("negotiability"),
        "after": "negotiable"
        if record["documentId"] in CORRECTIONS
        else patch.get("negotiability"),
        "consignee": patch.get("parties", {}).get("consignee"),
        "affirmativeCandidates": order_consignment_candidates(text),
        "evidence": evidence,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    if args.apply and DEST.exists():
        raise FileExistsError(DEST)
    before = {str(p.relative_to(ROOT)): tree_hashes(p) for p in (OLD, NEW, ARCHIVE)}
    old_manifest = read(OLD / "projection-manifest.json")
    new_manifest = read(NEW / "manifest.json")
    assert before[str(OLD.relative_to(ROOT))] == old_manifest["files"] | {
        "projection-manifest.json": digest(OLD / "projection-manifest.json")
    }
    # The new publication adds readback validation files after its manifest; verify
    # every frozen file, and snapshot all additional receipts rather than ignore them.
    for file, expected_hash in new_manifest["files"].items():
        assert digest(NEW / file) == expected_hash, file
    old_inventory = {r["documentId"]: r for r in read(ANALYSIS / "inventory.json")}
    new_inventory = {r["documentId"]: r for r in new_manifest["documents"]}
    schema = reduced_schema(BillOfLadingExtractionV7Label.model_json_schema())
    prompt_schema = reduced_schema(
        json.loads(get_training_task("bill_of_lading_extraction_v7").prompt_schema_json())
    )
    Draft202012Validator.check_schema(schema)
    schema_validator = Draft202012Validator(schema)
    prompt_validator = Draft202012Validator(prompt_schema)
    prompt = (
        (ROOT / "prompts/training/mpci_bl_extraction_v7.txt")
        .read_text()
        .replace(
            "{{output_schema}}",
            json.dumps(prompt_schema, ensure_ascii=False, separators=(",", ":")),
        )
    )
    expected, source_paths, receipts, audit, provenance = {}, {}, [], [], []
    projected, old_timing, new_timing = [], 0.0, 0.0
    for source in (OLD, NEW):
        for split, record in records(source):
            key = record["documentId"]
            assert key not in expected
            assert read(source / "samples" / key / "labels.json") == record["target"]
            assert (source / "samples" / key / "ocr.txt").read_text() == record["joinedRawText"]
            assert (
                hashlib.sha256(record["joinedRawText"].encode()).hexdigest()
                == record["joinedRawTextSha256"]
            )
            info = old_inventory[key] if source == OLD else new_inventory[key]
            sample = (
                info["sample"]
                if source == OLD
                else "005/"
                + ("replacement/" if "replacements" in info["sourceBatch"] else "")
                + info["pilotDocument"]
            )
            tick = time.perf_counter()
            BillOfLadingExtractionV7Label.model_validate_json(json.dumps(record["target"]))
            old_timing += time.perf_counter() - tick
            tick = time.perf_counter()
            result, edits = project(record)
            BillOfLadingExtractionV7Label.model_validate_json(json.dumps(result["target"]))
            schema_validator.validate(result["target"])
            prompt_validator.validate(result["target"])
            assert not source_fidelity_findings(
                result["target"], result["joinedRawText"], "metadata_freight"
            )
            new_timing += time.perf_counter() - tick
            if order_consignment_candidates(record["joinedRawText"]):
                assert result["target"]["documentPatch"]["negotiability"] == "negotiable"
            elif result["target"]["documentPatch"].get("negotiability") == "negotiable":
                raise AssertionError(f"Negotiable without reviewed affirmative instruction: {key}")
            line = (json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
            expected[key] = (split, line)
            source_paths[key] = source
            projected.append(result)
            if edits:
                receipts.append(
                    {"documentId": key, "sample": sample, "split": split, "edits": edits}
                )
            audit.append(
                negotiability_row(record, sample, "earlier" if source == OLD else "new", True)
            )
            provenance.append(
                {
                    "documentId": key,
                    "sample": sample,
                    "split": split,
                    "sourceDataset": str(source.relative_to(ROOT)),
                    "pdf": info["pdf"],
                    "sourceOcrSha256": record["joinedRawTextSha256"],
                }
            )
    assert len(expected) == 500 and set(expected) >= CORRECTIONS
    assert Counter(s for s, _ in expected.values()) == {"train": 450, "validation": 50}
    assert len({r["joinedRawTextSha256"] for r in projected}) == 500
    for _split, record in records(ARCHIVE):
        if record["documentId"] not in expected:
            row = negotiability_row(
                record, old_inventory[record["documentId"]]["sample"], "previously_filtered", False
            )
            assert not source_fidelity_findings(
                record["target"], record["joinedRawText"], "metadata_freight"
            )
            audit.append(row)
    assert len(audit) == 617

    # Mutation controls: each real affirmative instruction must block a mistaken
    # non-negotiable label. Removed properties must be forbidden, not just optional.
    mutations = 0
    for record in projected:
        if record["target"]["documentPatch"].get("negotiability") == "negotiable":
            mutant = deepcopy(record["target"])
            mutant["documentPatch"]["negotiability"] = "non_negotiable"
            assert source_fidelity_findings(mutant, record["joinedRawText"], "metadata_freight")
            mutations += 1
    rejected_fields = Counter()
    for receipt in receipts:
        target = json.loads(expected[receipt["documentId"]][1])["target"]
        for edit in receipt["edits"]:
            if "after" in edit:
                continue
            mutant = deepcopy(target)
            owner = mutant
            for part in edit["path"][:-1]:
                if isinstance(part, str) and part not in owner:
                    owner[part] = {}
                owner = owner[part]
            owner[edit["path"][-1]] = edit["before"]
            assert not schema_validator.is_valid(mutant)
            assert not prompt_validator.is_valid(mutant)
            rejected_fields[edit["path"][-1]] += 1
    assert set(rejected_fields) == set(REMOVED.values())
    source_summary = {
        cohort: {
            "documents": sum(a["cohort"] == cohort for a in audit),
            "before": dict(
                Counter(a["before"] or "absent" for a in audit if a["cohort"] == cohort)
            ),
            "after": dict(Counter(a["after"] or "absent" for a in audit if a["cohort"] == cohort)),
        }
        for cohort in ("earlier", "new", "previously_filtered")
    }
    summary = {
        "documents": 500,
        "splits": {"train": 450, "validation": 50},
        "negotiability": source_summary,
        "negotiabilityCorrections": len(CORRECTIONS),
        "removalOccurrences": dict(rejected_fields),
        "editedDocuments": len(receipts),
        "ocrEdits": 0,
        "filteredDocuments": 0,
        "apiCalls": 0,
        "apiCostUsd": 0,
        "validation": {
            "schemaAndPromptValid": 500,
            "exactInverseChecks": 500,
            "negotiabilityMutationRejections": mutations,
            "removedPropertyMutationRejections": sum(rejected_fields.values()),
            "beforeModelValidationSeconds": round(old_timing, 4),
            "projectionAndFullValidationSeconds": round(new_timing, 4),
        },
    }
    OUT.mkdir(exist_ok=True)
    write_json(OUT / "negotiability-audit.json", audit)
    write_json(OUT / "summary.json", summary)
    if not args.apply:
        assert all(tree_hashes(ROOT / path) == hashes for path, hashes in before.items())
        print(json.dumps(summary, indent=2))
        return
    stage = Path(tempfile.mkdtemp(prefix=".real-v7-r9-", dir=DEST.parent))
    for key, (_, line) in expected.items():
        sample = stage / "samples" / key
        sample.mkdir(parents=True)
        shutil.copy2(source_paths[key] / "samples" / key / "ocr.txt", sample / "ocr.txt")
        write_json(sample / "labels.json", json.loads(line)["target"])
    for split in ("train", "validation"):
        (stage / f"{split}.jsonl").write_bytes(
            b"".join(line for s, line in expected.values() if s == split)
        )
    for name, value in (
        ("schema.json", schema),
        ("prompt-schema.json", prompt_schema),
        ("label-edits.json", receipts),
        ("negotiability-audit.json", audit),
    ):
        write_json(stage / name, value)
    (stage / "training-prompt.txt").write_text(prompt)
    (stage / "README.md").write_text(
        "# Current reviewed real baseline: R9 (500 documents)\n\n"
        "450 training / 50 validation. Combines the previous R8 333 and batch005 167. "
        "No additional records filtered. Removed carrier party, vessel flag, goods marks "
        "and forwarding/export references. Four negotiability values corrected from "
        "populated order-consignment wording. All OCR and every other label preserved.\n\n"
        "Full annotations and historical revisions are immutable source backups, not "
        "the current training selection. `label-edits.json` permits exact value reversal. "
        "`projection-manifest.json` records sources, membership and hashes.\n\n"
        "Use the frozen reduced `schema.json`, `prompt-schema.json` and "
        "`training-prompt.txt` together. The live annotation flow deliberately retains "
        "the full optional field set; it is not this reduced training contract. No "
        "training config was changed or launched.\n\n"
        "Source-quality outliers remain pending the user's filtering decision. See "
        "`docs/analysis/real-v7-starting-dataset-2026-10-04/R9_REDUCED_500_2026-10-05.md` "
        "for the audit, policy precedence, tests and outlier inventory.\n"
    )
    validate_records(stage, expected)
    assert all(tree_hashes(ROOT / path) == hashes for path, hashes in before.items())
    summary["validation"]["readbackDocuments"] = 500
    summary["elapsedSeconds"] = round(time.monotonic() - start, 3)
    summary["peakRssKiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(stage / "validation.json", summary)
    write_json(
        stage / "projection-manifest.json",
        {
            "contractRevision": "real-v7-reduced-500-20261005",
            "dataset": str(DEST.relative_to(ROOT)),
            "documents": provenance,
            "sourceFiles": before,
            "summary": summary,
            "implementationSha256": digest(Path(__file__)),
            "files": tree_hashes(stage),
        },
    )
    stage.rename(DEST)
    write_json(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
