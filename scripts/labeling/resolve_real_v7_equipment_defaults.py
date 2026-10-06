"""Apply/replay the reviewed final R14 equipment classifications in place.

No OCR, temperature, goods, party, identifier, split or row-order changes.
R13 plus the existing ten-edit receipt reconstructs the exact pre-pass R14.
The independent 36-row OCR inventory is the scope and decision oracle.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
import tracemalloc
from collections import Counter

from jsonschema import Draft202012Validator
from normalize_real_v7_baseline import ROOT, digest, hashes, load_rows, read, write
from reconcile_real_v7_equipment_followup import CONTRACT, DEST, OUT, SOURCE, task_and_contract

from document_ocr.hashing import canonical_json_bytes
from document_ocr.labeling_agents.equipment_normalization import reconcile_equipment_categories

EC_DOCUMENT = "doc_3bb79a8d45f3cd76eda5ed208d63786e7212519188596bcb01e307f6f0d145d3"


def categories(item):
    """Independent decisions from full OCR review and the two source adjudications."""
    group = item["decisionGroup"]
    pair = item["recommendedPair"]
    if pair:
        return {k: "REFRIGERATED" if v == "REEFER" else v for k, v in pair.items()}
    surface = item["typeDescription"]
    if group.startswith("incomplete_"):
        return {
            "typeCategory": "REFRIGERATED" if group == "incomplete_reefer" else "GENERAL_PURPOSE"
        }
    size, kind = {
        "40EC": ("FORTY_FOOT_HIGH_CUBE", "GENERAL_PURPOSE"),
        "40EQ": ("FORTY_FOOT_HIGH_CUBE", "GENERAL_PURPOSE"),
        "40BX": ("FORTY_FOOT_STANDARD_HEIGHT", "GENERAL_PURPOSE"),
        "20DY": ("TWENTY_FOOT_STANDARD_HEIGHT", "GENERAL_PURPOSE"),
        "40RO": ("FORTY_FOOT_STANDARD_HEIGHT", "REFRIGERATED"),
        "40RK": ("FORTY_FOOT_STANDARD_HEIGHT", "REFRIGERATED"),
        "20TANK CONTAINER": ("TWENTY_FOOT_STANDARD_HEIGHT", "PRESSURIZED_TANK"),
    }[surface]
    return {"sizeCategory": size, "typeCategory": kind}


def run(*, apply):
    before_files = hashes(DEST)
    source_files = hashes(SOURCE)
    assert {k: v for k, v in source_files.items() if k != "projection-manifest.json"} == read(
        SOURCE / "projection-manifest.json"
    )["files"]
    manifest = read(DEST / "projection-manifest.json")
    assert {k: v for k, v in before_files.items() if k != "projection-manifest.json"} == manifest[
        "files"
    ]
    task, contract = task_and_contract()
    from document_ocr.training.tasks import RelationExplicitTaskConstraints, get_training_task

    contract["containerCategoryTokens"] = sorted(
        set(contract["containerCategoryTokens"]) | {"PRESSURIZED_TANK"}
    )
    task = get_training_task("bill_of_lading_extraction_v7_reduced").bind_constraints(
        RelationExplicitTaskConstraints.model_validate_json(json.dumps(contract))
    )
    baseline = copy.deepcopy(load_rows(SOURCE))
    by_id = {row["documentId"]: row for _, row in baseline}
    prior_edits = read(DEST / "equipment-followup-edits.json")
    assert len(prior_edits) == 10
    for edit in prior_edits:
        containers = by_id[edit["documentId"]]["target"]["documentPatch"]["containerInformation"]
        assert containers[edit["index"]] == edit["before"]
        containers[edit["index"]] = edit["after"]
    inventory = read(OUT / "ocr-followup-review.json")["decisions"]
    assert len(inventory) == 36 and len({r["documentId"] for r in inventory}) == 32
    owned = {(r["documentId"], r["containerIndex"]): r for r in inventory}
    expected = copy.deepcopy(baseline)
    expected_by_id = {row["documentId"]: row for _, row in expected}
    for (doc, index), item in owned.items():
        row = expected_by_id[doc]
        container = row["target"]["documentPatch"]["containerInformation"][index]
        assert container["equipmentIdentifier"] == item["equipmentIdentifier"]
        assert container.pop("typeDescription") == item["typeDescription"]
        container.update(categories(item))
        for evidence in item["evidence"]:
            assert row["joinedRawText"].splitlines()[evidence["line"] - 1] == evidence["text"]

    planned, edits, decisions = [], [], []
    for split, row in baseline:
        target, detected = reconcile_equipment_categories(
            row["target"], source_text=row["joinedRawText"]
        )
        for decision in detected:
            key = row["documentId"], decision["index"]
            assert key in owned
            item = owned[key]
            current = target["documentPatch"]["containerInformation"][decision["index"]]
            surface = item["typeDescription"]
            if surface in {"40EC", "40EQ"}:
                assert (
                    decision["action"] == "retain_fallback"
                    and current["typeDescription"] == surface
                )
                if surface == "40EC":
                    assert row["documentId"] == EC_DOCUMENT
                    assert current["equipmentIdentifier"] == "CMAU7204660"
                    assert "1 x 40EC 1875 PACKAGE(S)" in row["joinedRawText"]
                    assert (OUT / "40EC-USER-ADJUDICATION.md").is_file()
                    reason = "user_pdf_adjudication_ocr_ec_is_hc_not_global_alias"
                else:
                    assert row["documentId"].startswith("doc_00ea3f81")
                    assert "CCLU7970683 /KLW395165" in row["joinedRawText"]
                    assert (OUT / "references/cosco-40eq-source-40hq-page1.png").is_file()
                    reason = "source_pdf_confirms_ocr_eq_is_hq_not_global_alias"
                del current["typeDescription"]
                current.update(categories(item))
                decision = {
                    **decision,
                    "action": "source_adjudication",
                    "reason": reason,
                    **categories(item),
                }
            assert (
                current
                == expected_by_id[row["documentId"]]["target"]["documentPatch"][
                    "containerInformation"
                ][decision["index"]]
            )
            edits.append(
                {
                    "documentId": row["documentId"],
                    "split": split,
                    "index": decision["index"],
                    "before": row["target"]["documentPatch"]["containerInformation"][
                        decision["index"]
                    ],
                    "after": current,
                    "reason": decision["reason"],
                }
            )
            decisions.append({"documentId": row["documentId"], "split": split, **decision})
        assert target == expected_by_id[row["documentId"]]["target"]
        assert task.canonicalize(target) == target
        assert reconcile_equipment_categories(target, source_text=row["joinedRawText"])[0] == target
        planned.append((split, {**row, "target": target}))
    assert planned == expected and len(edits) == 36
    for schema in (task.target_schema(), json.loads(task.prompt_schema_json())):
        validator = Draft202012Validator(schema)
        for _, row in planned:
            validator.validate(row["target"])
    current_rows = load_rows(DEST)
    # Preserve the exact previous publication as an auditable one-field transition.
    previous_publication = copy.deepcopy(planned)
    previous_row = next(r for _, r in previous_publication if r["documentId"] == EC_DOCUMENT)
    previous_row["target"]["documentPatch"]["containerInformation"][0]["sizeCategory"] = (
        "FORTY_FOOT_STANDARD_HEIGHT"
    )
    replacing_previous = current_rows == previous_publication
    if replacing_previous:
        previous_edits = copy.deepcopy(edits)
        previous_edit = next(e for e in previous_edits if e["documentId"] == EC_DOCUMENT)
        previous_edit["after"]["sizeCategory"] = "FORTY_FOOT_STANDARD_HEIGHT"
        previous_edit["reason"] = "owned_1x40_fcl_standard_policy_not_ec_alias"
        assert read(DEST / "equipment-classification-edits.json") == previous_edits
    assert current_rows in (baseline, planned, previous_publication), (
        "Current data diverged from audited baseline"
    )

    times = []
    for _ in range(11):
        start = time.perf_counter()
        for _, row in baseline:
            target, _ = reconcile_equipment_categories(
                row["target"], source_text=row["joinedRawText"]
            )
            task.canonicalize(target)
        times.append(time.perf_counter() - start)
    tracemalloc.start()
    for _, row in baseline:
        task.canonicalize(
            reconcile_equipment_categories(row["target"], source_text=row["joinedRawText"])[0]
        )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    summary = {
        "documents": 660,
        "train": 600,
        "validation": 60,
        "changedDocuments": 32,
        "changedContainers": 36,
        "automaticNormalizations": 34,
        "sourceAdjudications": 2,
        "remainingTypeDescriptions": 0,
        "typeWithoutSize": 3,
        "fullEquipmentPairs": 1283,
        "withoutPrintedEquipmentSpecification": 85,
        "ocrChanges": 0,
        "unrelatedLabelChanges": 0,
        "temperatureChanges": 0,
        "apiCostUsd": 0,
        "medianSeconds660": statistics.median(times),
        "peakIncrementalTracedBytes": peak,
        "newTypeCategories": dict(Counter(e["after"]["typeCategory"] for e in edits)),
    }
    containers = [
        c for _, r in planned for c in r["target"]["documentPatch"].get("containerInformation", [])
    ]
    assert len(containers) == 1371
    assert sum("sizeCategory" in c and "typeCategory" in c for c in containers) == 1283
    assert sum("typeCategory" in c and "sizeCategory" not in c for c in containers) == 3
    assert not any("typeDescription" in c for c in containers)
    write(
        OUT / "classification-plan.json",
        {"summary": summary, "edits": edits, "decisions": decisions},
    )
    if not apply:
        print(json.dumps({"applied": current_rows == planned, **summary}, indent=2))
        return
    if current_rows == planned:
        assert read(DEST / "equipment-classification-edits.json") == edits
        assert read(DEST / "task-constraints.json") == contract == read(CONTRACT)
        print("Already applied; scope, manifest, receipts, source and schemas revalidated.")
        return
    if replacing_previous:
        write(
            DEST / "equipment-40ec-adjudication.json",
            {
                "documentId": EC_DOCUMENT,
                "equipmentIdentifier": "CMAU7204660",
                "authority": "User source-PDF review, 2026-10-06",
                "path": "documentPatch.containerInformation[0].sizeCategory",
                "before": "FORTY_FOOT_STANDARD_HEIGHT",
                "after": "FORTY_FOOT_HIGH_CUBE",
                "reason": "Source PDF is 40HC; OCR 40EC is a character error.",
                "previousTrainSha256": before_files["train.jsonl"],
                "previousClassificationEdits": read(DEST / "equipment-classification-edits.json"),
                "ocrChanged": False,
                "globalAliasAdded": False,
            },
        )
    for split in ("train", "validation"):
        path = DEST / f"{split}.jsonl"
        output = []
        for line in path.read_text().splitlines(keepends=True):
            old = json.loads(line)
            new = expected_by_id[old["documentId"]]
            output.append(
                line
                if old == new
                else json.dumps(new, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        path.write_text("".join(output))
    changed_ids = {e["documentId"] for e in edits}
    for doc in changed_ids:
        write(DEST / "samples" / doc / "labels.json", expected_by_id[doc]["target"])
    write(DEST / "equipment-classification-edits.json", edits)
    write(DEST / "equipment-classification-decisions.json", decisions)
    write(DEST / "schema.json", task.target_schema())
    write(DEST / "prompt-schema.json", json.loads(task.prompt_schema_json()))
    from document_ocr.training.config import load_training_config
    from document_ocr.training.prompting import load_prompt

    config = load_training_config(
        ROOT / "configs/training/production/"
        "t5gemma2_270m_lora.mpci_bl_real660_r14_reduced_v7_e10_compact_eva_a32_r32_local_schedulefree_v1.yaml"
    )
    (DEST / "training-prompt.txt").write_text(load_prompt(ROOT, config.prompt, task).text)
    for path in (DEST / "task-constraints.json", CONTRACT):
        path.write_bytes(canonical_json_bytes(contract) + b"\n")
    write(DEST / "validation.json", summary)
    (DEST / "README.md").write_text(
        "# R14 reviewed real baseline — completed equipment classification\n\n"
        "600 train / 60 validation records. All printed equipment descriptions are now "
        "classified: 1,283 complete pairs and three type-only records with no printed length; "
        "85 containers have no printed equipment specification. Approved standard-height "
        "defaults apply to known 20/40-foot wording; explicit owned codes take precedence. "
        "NOR is refrigerated equipment without an invented setpoint.\n\n"
        "R13 remains the unchanged backup. equipment-followup-edits.json contains the "
        "earlier ten edits; equipment-classification-edits.json contains this pass's 36. "
        "All OCR, temperatures, identifiers, goods, parties, splits and ordering are unchanged. "
        "Two source-level adjudications are explicit, not universal EC/EQ aliases.\n\n"
        "Audit: docs/analysis/real660-equipment-followup-20261006/CLASSIFICATION-COMPLETION.md.\n"
    )
    after_files = hashes(DEST)
    for path, sha in before_files.items():
        if path == "validation.jsonl" or (
            path.startswith("samples/")
            and (path.endswith("ocr.txt") or path.split("/")[1] not in changed_ids)
        ):
            assert after_files[path] == sha, f"Out-of-scope change: {path}"
    assert load_rows(DEST) == planned and hashes(SOURCE) == source_files
    manifest.update(
        summary=summary,
        files={k: v for k, v in after_files.items() if k != "projection-manifest.json"},
    )
    write(DEST / "projection-manifest.json", manifest)
    assert hashes(DEST) == {
        **manifest["files"],
        "projection-manifest.json": digest(DEST / "projection-manifest.json"),
    }
    write(OUT / "classification-validation.json", summary)
    print(
        json.dumps(
            {
                **summary,
                "trainSha256": digest(DEST / "train.jsonl"),
                "validationSha256": digest(DEST / "validation.jsonl"),
                "contractSha256": digest(CONTRACT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    run(apply=parser.parse_args().apply)
