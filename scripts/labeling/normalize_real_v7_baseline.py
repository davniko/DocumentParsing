"""Audit and publish the approved casing/equipment repair, preserving the R12 source.

No provider calls. Default is an audit plus an exact source backup; --apply also
publishes a new R13 dataset. Existing published data is never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import resource
import shutil
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from document_ocr.labeling_agents.target_normalization import (
    UPPERCASE_FIELDS,
    normalize_target_casing,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/curated/mpci-bl-real-v7-reviewed-r12-reduced-660"
BACKUP = SOURCE.with_name(SOURCE.name + "_source_20261006")
DEST = SOURCE.with_name("mpci-bl-real-v7-reviewed-r13-casing-equipment-660")
OUT = ROOT / "docs/analysis/real660-casing-equipment-20261006"
CONTRACT = ROOT / (
    "configs/training/production/contracts/mpci_bl_real660_reduced_v7_r13/task-constraints.json"
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hashes(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def leaves(value, path="", field=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from leaves(child, f"{path}.{key}".lstrip("."), f"{field}.{key}".lstrip("."))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from leaves(child, f"{path}[{i}]", field + "[]")
    else:
        yield path, field, value


def style(value):
    if not any(c.isalpha() for c in value):
        return "uncased"
    if value.isupper():
        return "uppercase"
    if value.islower():
        return "lowercase"
    if value.istitle():
        return "titlecase"
    return "mixed"


def load_rows(root):
    rows = []
    for split in ("train", "validation"):
        rows.extend(
            (split, json.loads(line)) for line in (root / f"{split}.jsonl").read_text().splitlines()
        )
    assert Counter(s for s, _ in rows) == {"train": 600, "validation": 60}
    assert len({r["documentId"] for _, r in rows}) == 660
    for _, row in rows:
        sample = root / "samples" / row["documentId"]
        assert row["target"] == read(sample / "labels.json")
        assert row["joinedRawText"] == (sample / "ocr.txt").read_text()
        assert (
            hashlib.sha256(row["joinedRawText"].encode()).hexdigest() == row["joinedRawTextSha256"]
        )
    return rows


def backup(source_hashes):
    if BACKUP.exists():
        assert read(BACKUP / "source-snapshot.json")["files"] == source_hashes
        assert {k: digest(BACKUP / k) for k in source_hashes} == source_hashes
        return
    stage = Path(tempfile.mkdtemp(prefix=".real660-backup-", dir=SOURCE.parent))
    shutil.copytree(SOURCE, stage / "source")
    assert hashes(stage / "source") == source_hashes
    write(
        stage / "source/source-snapshot.json",
        {
            "source": str(SOURCE.relative_to(ROOT)),
            "files": source_hashes,
            "purpose": "Exact 600/60 snapshot before uppercase text and equipment reconciliation.",
        },
    )
    (stage / "source/SOURCE_SNAPSHOT.md").write_text(
        "# Pre-repair source snapshot\n\n"
        "Exact R12 labels and OCR before the 2026-10-06 casing/equipment repair. "
        "Retained for reproduction of the real660 V7 e20 run and earlier R12 configs. "
        "Do not substitute these labels for the repaired R13 training dataset.\n"
    )
    (stage / "source").rename(BACKUP)
    stage.rmdir()


def audit(rows):
    counts, candidates, changes = defaultdict(Counter), [], []
    variants = defaultdict(lambda: defaultdict(list))
    for split, row in rows:
        text = re.sub(r"\s+", " ", row["joinedRawText"])
        for path, field, value in leaves(row["target"]["documentPatch"]):
            if not isinstance(value, str):
                continue
            counter = counts[(split, field)]
            counter["values"] += 1
            counter["label_" + style(value)] += 1
            normalized = re.sub(r"\s+", " ", value)
            matches = {
                m.group()
                for m in re.finditer(r"(?<!\w)" + re.escape(normalized) + r"(?!\w)", text, re.I)
            }
            state = (
                "exact_case_present"
                if normalized in matches
                else "case_only_source_difference"
                if matches
                else "no_same_phrase"
            )
            counter[state] += 1
            if state == "case_only_source_difference":
                candidates.append(
                    dict(
                        split=split,
                        documentId=row["documentId"],
                        path=path,
                        field=field,
                        label=value,
                        ocrVariants=sorted(matches),
                        casingPolicy="uppercase_text" if field in UPPERCASE_FIELDS else "preserve",
                    )
                )
            variants[(field, value.casefold())][value].append(row["documentId"])
        _, edits = normalize_target_casing(row["target"])
        changes.extend(dict(split=split, documentId=row["documentId"], **edit) for edit in edits)
    result = {
        "documents": len(rows),
        "uppercaseFields": sorted(UPPERCASE_FIELDS),
        "caseOnlyPhraseCandidates": len(candidates),
        "humanTextCaseOnlyPhraseCandidates": sum(
            r["casingPolicy"] == "uppercase_text" for r in candidates
        ),
        "uppercaseChanges": len(changes),
        "changedDocuments": len({r["documentId"] for r in changes}),
        "note": (
            "Phrase matching measures casing only, not source ownership or annotation correctness. "
            "Normalized enums/units are intentionally unlike OCR. "
            "Missing phrases are not automatically defects."
        ),
        "fields": [
            dict(
                split=s,
                field=f,
                policy="uppercase_text" if f in UPPERCASE_FIELDS else "preserve",
                **c,
            )
            for (s, f), c in sorted(counts.items())
        ],
        "caseVariantsOfSameLabel": [
            dict(field=f, foldedValue=v, variants=dict(forms))
            for (f, v), forms in variants.items()
            if len(forms) > 1
        ],
    }
    write(OUT / "pre-repair-audit.json", result)
    write(OUT / "ocr-case-candidates.json", candidates)
    write(OUT / "casing-edits.json", changes)
    return result


def publish(rows, source_hashes):
    from jsonschema import Draft202012Validator

    from document_ocr.labeling_agents.equipment_normalization import reconcile_equipment_categories
    from document_ocr.training.tasks import RelationExplicitTaskConstraints, get_training_task

    if DEST.exists():
        raise FileExistsError(f"Refusing to overwrite published dataset: {DEST}")
    task = get_training_task("bill_of_lading_extraction_v7_reduced").bind_constraints(
        RelationExplicitTaskConstraints.model_validate_json(CONTRACT.read_text())
    )
    repaired, receipts, decisions = [], [], []
    started = time.perf_counter()
    for split, row in rows:
        original = copy.deepcopy(row)
        target, equipment = reconcile_equipment_categories(
            row["target"], source_text=row["joinedRawText"]
        )
        target, casing = normalize_target_casing(target)
        assert task.canonicalize(target) == target
        again, _ = reconcile_equipment_categories(target, source_text=row["joinedRawText"])
        assert normalize_target_casing(again)[0] == target, "Repair must be idempotent"
        new = {**row, "target": target}
        assert row == original, "Repair mutated source"
        repaired.append((split, new))
        receipts.append(
            dict(
                split=split,
                documentId=row["documentId"],
                casing=casing,
                beforeSha256=hashlib.sha256(
                    json.dumps(row["target"], sort_keys=True).encode()
                ).hexdigest(),
                afterSha256=hashlib.sha256(json.dumps(target, sort_keys=True).encode()).hexdigest(),
            )
        )
        decisions.extend(dict(split=split, documentId=row["documentId"], **d) for d in equipment)
    elapsed = time.perf_counter() - started
    write(OUT / "equipment-decisions.json", decisions)

    # Independent scope proof: reverse approved case-only changes; remaining
    # differences must be fallback removal plus complete canonical equipment pairs.
    changed_documents, changed_containers = set(), 0
    for (_, old), (_, new) in zip(rows, repaired, strict=True):
        assert {k: v for k, v in old.items() if k != "target"} == {
            k: v for k, v in new.items() if k != "target"
        }
        expected = copy.deepcopy(old["target"])
        old_containers = expected["documentPatch"].get("containerInformation", [])
        new_containers = new["target"]["documentPatch"].get("containerInformation", [])
        assert len(old_containers) == len(new_containers)
        for a, b in zip(old_containers, new_containers, strict=True):
            if "typeDescription" in a and "typeDescription" not in b:
                assert "sizeCategory" in b and "typeCategory" in b
                a.pop("typeDescription")
                a.update(sizeCategory=b["sizeCategory"], typeCategory=b["typeCategory"])
                changed_containers += 1
        old_leaves = list(leaves(expected["documentPatch"]))
        new_leaves = list(leaves(new["target"]["documentPatch"]))
        assert len(old_leaves) == len(new_leaves)
        for (path, field, value), (new_path, new_field, new_value) in zip(
            old_leaves, new_leaves, strict=True
        ):
            assert (path, field) == (new_path, new_field), "Structure or order changed"
            wanted = (
                value.upper() if field in UPPERCASE_FIELDS and isinstance(value, str) else value
            )
            assert new_value == wanted, "Unauthorized target change"
        if old != new:
            changed_documents.add(old["documentId"])
        for _, field, value in leaves(new["target"]["documentPatch"]):
            if field in UPPERCASE_FIELDS and isinstance(value, str):
                assert value == value.upper()

    stage_root = Path(tempfile.mkdtemp(prefix=".real660-normalized-", dir=SOURCE.parent))
    stage = stage_root / "dataset"
    shutil.copytree(SOURCE, stage)
    for split in ("train", "validation"):
        (stage / f"{split}.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for current_split, row in repaired
                if current_split == split
            )
        )
    for _, row in repaired:
        write(stage / "samples" / row["documentId"] / "labels.json", row["target"])
    write(stage / "schema.json", task.target_schema())
    write(stage / "prompt-schema.json", json.loads(task.prompt_schema_json()))
    (stage / "training-prompt.txt").write_text(
        (ROOT / "prompts/training/mpci_bl_extraction_v7.txt")
        .read_text()
        .replace("{{output_schema}}", task.prompt_schema_json())
    )
    shutil.copy2(CONTRACT, stage / "task-constraints.json")
    validators = [
        Draft202012Validator(read(stage / name)) for name in ("schema.json", "prompt-schema.json")
    ]
    for _, row in repaired:
        for validator in validators:
            validator.validate(row["target"])
    write(stage / "normalization-edits.json", receipts)
    write(stage / "equipment-decisions.json", decisions)
    summary = dict(
        documents=660,
        train=600,
        validation=60,
        changedDocuments=len(changed_documents),
        casingEdits=sum(len(r["casing"]) for r in receipts),
        equipmentCanonicalized=changed_containers,
        equipmentFallbackRetained=sum(
            "typeDescription" in c
            for _, r in repaired
            for c in r["target"]["documentPatch"].get("containerInformation", [])
        ),
        ocrEdits=0,
        unrelatedLabelEdits=0,
        apiCostUsd=0,
        repairAndValidationSeconds=elapsed,
        peakRssKiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    timings = []
    for _ in range(11):
        start = time.perf_counter()
        for _, row in rows:
            normalize_target_casing(row["target"])
        timings.append(time.perf_counter() - start)
    summary["casingMedianSeconds660"] = statistics.median(timings)
    write(stage / "validation.json", summary)
    (stage / "README.md").write_text(
        "# Reviewed real baseline R13 — casing and equipment\n\n"
        "600 training / 60 validation documents. Membership, ordering, OCR bytes and all "
        "unrelated labels are unchanged from R12. Human-readable text is uppercase; "
        "identifiers, contact endpoints, units and enums retain their existing formats. "
        "Equipment aliases are canonicalized only where source-supported dimensions and "
        "type are established. Ambiguous printed descriptions remain as fallback labels.\n\n"
        "R12 and its `_source_20261006` backup are immutable historical inputs. "
        "`normalization-edits.json` and `equipment-decisions.json` record this repair. "
        "`projection-edits.json` remains the historical reduced-field projection receipt. "
        "The new normalization does not certify unrelated semantic annotation choices.\n"
    )
    assert load_rows(stage) == repaired
    manifest = read(SOURCE / "projection-manifest.json")
    manifest.update(
        dataset=str(DEST.relative_to(ROOT)),
        sourceDataset=str(SOURCE.relative_to(ROOT)),
        sourceBackup=str(BACKUP.relative_to(ROOT)),
        sourceFiles=source_hashes,
        summary=summary,
        files={k: v for k, v in hashes(stage).items() if k != "projection-manifest.json"},
    )
    write(stage / "projection-manifest.json", manifest)
    assert hashes(SOURCE) == source_hashes
    stage.rename(DEST)
    stage_root.rmdir()
    write(OUT / "repair-summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    source_hashes = hashes(SOURCE)
    manifest = read(SOURCE / "projection-manifest.json")
    assert (
        manifest["files"]
        | {"projection-manifest.json": digest(SOURCE / "projection-manifest.json")}
        == source_hashes
    )
    rows = load_rows(SOURCE)
    backup(source_hashes)
    before = audit(rows)
    print(
        json.dumps(
            {
                k: v
                for k, v in before.items()
                if k not in ("fields", "uppercaseFields", "caseVariantsOfSameLabel")
            },
            indent=2,
        )
    )
    if args.apply:
        publish(rows, source_hashes)


if __name__ == "__main__":
    main()
