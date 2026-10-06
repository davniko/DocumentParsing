"""Publish/revalidate the source-preserving R14 equipment-only follow-up.

The immutable R13 directory is the backup. No OCR, temperature, identifiers,
parties, goods, memberships or row ordering may change. No provider calls.
"""

from __future__ import annotations

import argparse
import copy
import json
import resource
import shutil
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path

from normalize_real_v7_baseline import ROOT, digest, hashes, load_rows, read, write

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.labeling_agents.equipment_normalization import reconcile_equipment_categories
from document_ocr.training.tasks import RelationExplicitTaskConstraints, get_training_task

SOURCE = ROOT / "data/curated/mpci-bl-real-v7-reviewed-r13-casing-equipment-660"
DEST = SOURCE.with_name("mpci-bl-real-v7-reviewed-r14-equipment-660")
OUT = ROOT / "docs/analysis/real660-equipment-followup-20261006"
CONTRACT = (
    ROOT
    / "configs/training/production/contracts/mpci_bl_real660_reduced_v7_r14/task-constraints.json"
)
APPROVED = {
    "22GO": ("TWENTY_FOOT_STANDARD_HEIGHT", "GENERAL_PURPOSE"),
    "40RQ": ("FORTY_FOOT_HIGH_CUBE", "REFRIGERATED"),
    "20HO": ("TWENTY_FOOT_HIGH_CUBE", "OPEN_TOP"),
}
ADJUDICATED_DOCUMENT = "doc_a44bb232a1bbaa7dccfe7ca709e0a3d1f03988fb03ed947b218375b4a1c3e972"


def verify_scope(original, repaired):
    """Replay each difference against independently adjudicated alias decisions."""
    edits = []
    for (old_split, old), (split, new) in zip(original, repaired, strict=True):
        assert split == old_split
        assert old.keys() == new.keys()
        assert {k: v for k, v in old.items() if k != "target"} == {
            k: v for k, v in new.items() if k != "target"
        }
        expected = copy.deepcopy(old["target"])
        for i, container in enumerate(expected["documentPatch"].get("containerInformation", [])):
            surface = container.get("typeDescription")
            if surface not in APPROVED:
                continue
            before = copy.deepcopy(container)
            size, kind = APPROVED[surface]
            del container["typeDescription"]
            container.update(sizeCategory=size, typeCategory=kind)
            edits.append(
                {
                    "documentId": old["documentId"],
                    "split": split,
                    "index": i,
                    "before": before,
                    "after": copy.deepcopy(container),
                }
            )
        assert new["target"] == expected, f"Unauthorized difference: {old['documentId']}"
    assert Counter(e["before"]["typeDescription"] for e in edits) == {
        "22GO": 6,
        "40RQ": 3,
        "20HO": 1,
    }
    return edits


def task_and_contract():
    task = get_training_task("bill_of_lading_extraction_v7_reduced")
    contract = read(SOURCE / "task-constraints.json")
    contract.update(
        basePromptSchemaSha256=task.base_prompt_schema_sha256(),
        targetSchemaSha256=sha256_bytes(canonical_json_bytes(task.target_schema())),
    )
    return task.bind_constraints(
        RelationExplicitTaskConstraints.model_validate_json(json.dumps(contract))
    ), contract


def apply_adjudication_in_place(repaired, edits, decisions, task, contract):
    """Apply only the authorized 20HO delta, preserving all other current rows."""
    current = load_rows(DEST)
    before_files = hashes(DEST)
    manifest = read(DEST / "projection-manifest.json")
    assert {k: v for k, v in before_files.items() if k != "projection-manifest.json"} == manifest[
        "files"
    ]
    changed = [
        (s, a, b) for (s, a), (t, b) in zip(current, repaired, strict=True) if (s, a) != (t, b)
    ]
    if not changed:
        assert read(DEST / "equipment-20ho-adjudication.json")["documentId"] == ADJUDICATED_DOCUMENT
        return
    assert len(changed) == 1, "In-place adjudication must change exactly one document"
    split, before, after = changed[0]
    assert split == "validation" and before["documentId"] == ADJUDICATED_DOCUMENT
    expected = copy.deepcopy(before)
    container = expected["target"]["documentPatch"]["containerInformation"][0]
    assert container.pop("typeDescription") == "20HO"
    container.update(sizeCategory="TWENTY_FOOT_HIGH_CUBE", typeCategory="OPEN_TOP")
    assert expected == after
    receipt = {
        "basis": (
            "User adjudication 2026-10-06: TARROS 20HO = 20ft high-cube open top; "
            "not an ISO O/0 correction."
        ),
        "documentId": ADJUDICATED_DOCUMENT,
        "split": split,
        "beforeTarget": before["target"],
        "afterTarget": after["target"],
        "beforeValidationSha256": before_files["validation.jsonl"],
        "beforeManifestSha256": before_files["projection-manifest.json"],
    }
    # Replace the one JSONL row while retaining every other physical line verbatim.
    path = DEST / "validation.jsonl"
    lines = path.read_text().splitlines(keepends=True)
    matches = [
        i for i, line in enumerate(lines) if json.loads(line)["documentId"] == ADJUDICATED_DOCUMENT
    ]
    assert len(matches) == 1
    lines[matches[0]] = json.dumps(after, ensure_ascii=False, separators=(",", ":")) + "\n"
    path.write_text("".join(lines))
    write(DEST / "samples" / ADJUDICATED_DOCUMENT / "labels.json", after["target"])
    write(DEST / "equipment-20ho-adjudication.json", receipt)
    write(DEST / "equipment-followup-edits.json", edits)
    write(DEST / "equipment-followup-decisions.json", decisions)
    write(DEST / "schema.json", task.target_schema())
    write(DEST / "prompt-schema.json", json.loads(task.prompt_schema_json()))
    (DEST / "task-constraints.json").write_bytes(canonical_json_bytes(contract) + b"\n")
    CONTRACT.write_bytes(canonical_json_bytes(contract) + b"\n")
    summary = read(DEST / "validation.json")
    summary.update(
        changedDocuments=len({e["documentId"] for e in edits}),
        changedContainers=len(edits),
        changedContainersBySplit=dict(Counter(e["split"] for e in edits)),
        remainingFallbackContainers=sum(d["action"] == "retain_fallback" for d in decisions),
        remainingFallbackDocuments=len(
            {d["documentId"] for d in decisions if d["action"] == "retain_fallback"}
        ),
    )
    del summary["remainingSurfaces"]["20HO"]
    summary["additionalInPlaceUserAdjudications"] = 1
    write(DEST / "validation.json", summary)
    readme = DEST / "README.md"
    prior = "Six 22GO aliases and three 40RQ aliases now have canonical equipment categories."
    content = readme.read_text()
    assert content.count(prior) == 1
    readme.write_text(
        content.replace(
            prior,
            "Six 22GO aliases, three 40RQ aliases, and the user-adjudicated TARROS "
            "20HO now have canonical equipment categories.",
        )
    )
    manifest.update(
        summary=summary,
        files={k: v for k, v in hashes(DEST).items() if k != "projection-manifest.json"},
    )
    write(DEST / "projection-manifest.json", manifest)
    assert load_rows(DEST) == repaired
    for name, sha in before_files.items():
        if (
            name == "train.jsonl"
            or name.endswith("/ocr.txt")
            or (name.startswith("samples/") and ADJUDICATED_DOCUMENT not in name)
        ):
            assert digest(DEST / name) == sha, f"Unrelated sample file changed: {name}"
    write(OUT / "equipment-decisions.json", decisions)
    write(OUT / "equipment-edits.json", edits)
    write(OUT / "repair-summary.json", summary)
    write(OUT / "20ho-adjudication.json", receipt)
    print(
        json.dumps(
            {
                "inPlaceDocumentChanges": 1,
                "summary": summary,
                "validationSha256": digest(path),
                "contractSha256": digest(CONTRACT),
            },
            indent=2,
        )
    )


def build(*, apply_in_place=False):
    from jsonschema import Draft202012Validator

    source_hashes = hashes(SOURCE)
    assert {k: v for k, v in source_hashes.items() if k != "projection-manifest.json"} == read(
        SOURCE / "projection-manifest.json"
    )["files"]
    task, contract = task_and_contract()
    rows = load_rows(SOURCE)
    repaired, decisions = [], []
    for split, row in rows:
        target, evidence = reconcile_equipment_categories(
            row["target"], source_text=row["joinedRawText"]
        )
        assert task.canonicalize(target) == target
        assert reconcile_equipment_categories(target, source_text=row["joinedRawText"])[0] == target
        repaired.append((split, {**row, "target": target}))
        decisions.extend({"split": split, "documentId": row["documentId"], **d} for d in evidence)
    edits = verify_scope(rows, repaired)
    for schema in (task.target_schema(), json.loads(task.prompt_schema_json())):
        validator = Draft202012Validator(schema)
        for _, row in repaired:
            validator.validate(row["target"])
    if DEST.exists():
        if apply_in_place:
            apply_adjudication_in_place(repaired, edits, decisions, task, contract)
        assert load_rows(DEST) == repaired
        assert read(DEST / "task-constraints.json") == contract
        assert read(DEST / "equipment-followup-edits.json") == edits
        assert {k: v for k, v in hashes(DEST).items() if k != "projection-manifest.json"} == read(
            DEST / "projection-manifest.json"
        )["files"]
        assert read(CONTRACT) == contract
        print("R14 readback, exact scope, schema and manifest validation passed.")
        return

    times = []
    for _ in range(11):
        started = time.perf_counter()
        for _, row in rows:
            target, _ = reconcile_equipment_categories(
                row["target"], source_text=row["joinedRawText"]
            )
            task.canonicalize(target)
        times.append(time.perf_counter() - started)
    remaining = Counter(
        c["typeDescription"]
        for _, row in repaired
        for c in row["target"]["documentPatch"].get("containerInformation", [])
        if "typeDescription" in c
    )
    summary = {
        "documents": 660,
        "train": 600,
        "validation": 60,
        "changedDocuments": len({e["documentId"] for e in edits}),
        "changedContainers": len(edits),
        "changedContainersBySplit": dict(Counter(e["split"] for e in edits)),
        "remainingFallbackContainers": sum(remaining.values()),
        "remainingFallbackDocuments": len(
            {d["documentId"] for d in decisions if d["action"] == "retain_fallback"}
        ),
        "remainingSurfaces": dict(sorted(remaining.items())),
        "ocrChanges": 0,
        "unrelatedLabelChanges": 0,
        "temperatureChanges": 0,
        "apiCostUsd": 0,
        "equipmentFinalizerAndValidationMedianSeconds660": statistics.median(times),
        "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    stage_parent = ROOT / "data/curated"
    with tempfile.TemporaryDirectory(prefix=".real660-equipment-", dir=stage_parent) as temp:
        stage = Path(temp) / "dataset"
        shutil.copytree(SOURCE, stage)
        for split in ("train", "validation"):
            (stage / f"{split}.jsonl").write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for s, row in repaired
                    if s == split
                )
            )
        for _, row in repaired:
            write(stage / "samples" / row["documentId"] / "labels.json", row["target"])
        write(stage / "equipment-followup-edits.json", edits)
        write(stage / "equipment-followup-decisions.json", decisions)
        write(stage / "schema.json", task.target_schema())
        write(stage / "prompt-schema.json", json.loads(task.prompt_schema_json()))
        (stage / "training-prompt.txt").write_text(
            (ROOT / "prompts/training/mpci_bl_extraction_v7.txt")
            .read_text()
            .replace("{{output_schema}}", task.prompt_schema_json())
        )
        (stage / "task-constraints.json").write_bytes(canonical_json_bytes(contract) + b"\n")
        write(stage / "validation.json", summary)
        (stage / "README.md").write_text(
            "# Reviewed real baseline R14 — equipment alias follow-up\n\n"
            "600 training / 60 validation records. "
            "R13 is preserved unchanged as the source backup. "
            "Six 22GO aliases, three 40RQ aliases, and one user-adjudicated TARROS 20HO "
            "now have canonical equipment categories. "
            "All OCR, temperatures, identities, goods, parties, order "
            "and split membership are unchanged.\n\n"
            "See equipment-followup-edits.json and equipment-followup-decisions.json "
            "for this pass. "
            "normalization-edits.json and equipment-decisions.json are historical R13 receipts; "
            "projection-edits.json is the earlier reduced-field projection history. "
            "Unresolved size/type descriptions remain explicit, "
            "never defaulted to dry containers.\n"
        )
        assert load_rows(stage) == repaired
        manifest = read(SOURCE / "projection-manifest.json")
        manifest.update(
            dataset=str(DEST.relative_to(ROOT)),
            sourceDataset=str(SOURCE.relative_to(ROOT)),
            sourceBackup=str(SOURCE.relative_to(ROOT)),
            sourceFiles=source_hashes,
            summary=summary,
            files={k: v for k, v in hashes(stage).items() if k != "projection-manifest.json"},
        )
        write(stage / "projection-manifest.json", manifest)
        assert hashes(SOURCE) == source_hashes
        stage.rename(DEST)
    CONTRACT.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT.write_bytes(canonical_json_bytes(contract) + b"\n")
    write(OUT / "equipment-decisions.json", decisions)
    write(OUT / "equipment-edits.json", edits)
    write(OUT / "repair-summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                "trainSha256": digest(DEST / "train.jsonl"),
                "validationSha256": digest(DEST / "validation.jsonl"),
                "contractSha256": digest(CONTRACT),
            },
            indent=2,
        )
    )


def training_preflight():
    """Exercise the trainer's actual CPU data path, without loading model weights."""
    from document_ocr.training.config import load_training_config
    from document_ocr.training.data import prepare_datasets
    from document_ocr.training.prompting import load_prompt
    from document_ocr.training.runtime import build_training_arguments, load_tokenizer
    from document_ocr.training.tasks import load_training_task

    prepared, results = {}, {}
    source_rows = load_rows(DEST)
    expected = {row["documentId"]: row["target"] for _, row in source_rows}
    for mode in ("compact", "pretty"):
        config = load_training_config(
            ROOT
            / (
                "configs/training/production/t5gemma2_270m_lora.mpci_bl_real660_r14_"
                f"reduced_v7_e10_{mode}_eva_a32_r32_local_schedulefree_v1.yaml"
            )
        )
        task = load_training_task(ROOT, config)
        prompt = load_prompt(ROOT, config.prompt, task)
        assert prompt.text == (DEST / "training-prompt.txt").read_text()
        assert task.target_schema() == read(DEST / "schema.json")
        tokenizer = load_tokenizer(config)
        start = time.perf_counter()
        data = prepare_datasets(
            project_root=ROOT, config=config, prompt=prompt, task=task, tokenizer=tokenizer
        )
        arguments = build_training_arguments(config, OUT / f"{mode}-preflight")
        assert {k: len(v) for k, v in data.datasets.items()} == {"train": 600, "validation": 60}
        for split in ("train", "validation"):
            for row in data.datasets[split]:
                decoded = tokenizer.decode(row["labels"], skip_special_tokens=True)
                assert json.loads(decoded) == expected[row["document_id"]]
                assert ("\n" not in decoded) == (mode == "compact")
                assert row["labels"][-1] == tokenizer.eos_token_id
        prepared[mode] = data
        results[mode] = {
            "tokenLengths": data.token_lengths,
            "seconds": time.perf_counter() - start,
            "cacheIdentity": data.cache_identity,
            "evalSteps": arguments.eval_steps,
            "saveSteps": arguments.save_steps,
        }
    for split in ("train", "validation"):
        for a, b in zip(
            prepared["compact"].datasets[split], prepared["pretty"].datasets[split], strict=True
        ):
            assert a["document_id"] == b["document_id"]
            assert a["input_ids"] == b["input_ids"]
    assert prepared["compact"].cache_identity != prepared["pretty"].cache_identity
    results.update(modelLoaded=False, trainingStarted=False)
    write(OUT / "training-preflight.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-preflight", action="store_true")
    parser.add_argument("--apply-adjudication-in-place", action="store_true")
    args = parser.parse_args()
    training_preflight() if args.training_preflight else build(
        apply_in_place=args.apply_adjudication_in_place
    )
