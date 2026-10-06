"""Independent readback, cost-free metrics replay and CPU preparation of repaired R13.

Run with --training-preflight inside the trainer image; no model is loaded.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
import tracemalloc
from collections import Counter

from normalize_real_v7_baseline import (
    BACKUP,
    DEST,
    OUT,
    ROOT,
    SOURCE,
    hashes,
    leaves,
    load_rows,
    read,
    write,
)

from document_ocr.labeling_agents.equipment_normalization import reconcile_equipment_categories
from document_ocr.labeling_agents.target_normalization import (
    UPPERCASE_FIELDS,
    normalize_target_casing,
)
from document_ocr.training.config import load_training_config
from document_ocr.training.metrics import structured_metrics
from document_ocr.training.prompting import load_prompt
from document_ocr.training.tasks import load_training_task

PREFIX = "configs/training/production/t5gemma2_270m_lora.mpci_bl_real660_reduced_v7_e10_"


def config_for(mode):
    return load_training_config(ROOT / f"{PREFIX}{mode}_eva_a32_r32_local_schedulefree_v1.yaml")


def validate():
    source, repaired = load_rows(SOURCE), load_rows(DEST)
    snapshot = read(BACKUP / "source-snapshot.json")
    assert hashes(SOURCE) == snapshot["files"]
    assert {k: v for k, v in hashes(BACKUP).items() if k in snapshot["files"]} == snapshot["files"]
    manifest = read(DEST / "projection-manifest.json")
    assert {k: v for k, v in hashes(DEST).items() if k != "projection-manifest.json"} == manifest[
        "files"
    ]
    task = load_training_task(ROOT, config_for("compact"))
    prompt = load_prompt(ROOT, config_for("compact").prompt, task)
    assert (DEST / "training-prompt.txt").read_text() == prompt.text
    assert read(DEST / "prompt-schema.json") == json.loads(task.prompt_schema_json())
    assert read(DEST / "schema.json") == task.target_schema()
    case_counts = Counter()
    equipment_changes = Counter()
    for (old_split, old), (split, new) in zip(source, repaired, strict=True):
        assert split == old_split
        assert task.canonicalize(new["target"]) == new["target"]
        assert {k: v for k, v in old.items() if k != "target"} == {
            k: v for k, v in new.items() if k != "target"
        }
        # Replay exact receipts backward independently of both repair functions.
        reversed_target = copy.deepcopy(new["target"])
        for old_container, new_container in zip(
            old["target"]["documentPatch"].get("containerInformation", []),
            reversed_target["documentPatch"].get("containerInformation", []),
            strict=True,
        ):
            if "typeDescription" in old_container and "typeDescription" not in new_container:
                new_container.pop("sizeCategory")
                new_container.pop("typeCategory")
                new_container["typeDescription"] = old_container["typeDescription"]
                equipment_changes[split] += 1
        a = {path: (field, value) for path, field, value in leaves(old["target"]["documentPatch"])}
        b = {
            path: (field, value) for path, field, value in leaves(reversed_target["documentPatch"])
        }
        assert a.keys() == b.keys()
        for path, (field, old_value) in a.items():
            _, new_value = b[path]
            if field in UPPERCASE_FIELDS and isinstance(old_value, str):
                # Restored fallback descriptions above deliberately retain original case.
                assert new_value in (old_value, old_value.upper())
                if new_value != old_value:
                    case_counts[field] += 1
            else:
                assert new_value == old_value, (old["documentId"], path)
        actual, _ = reconcile_equipment_categories(old["target"], source_text=old["joinedRawText"])
        assert normalize_target_casing(actual)[0] == new["target"]
        assert normalize_target_casing(new["target"]) == (new["target"], [])

    predictions = {
        r["document_id"]: r["generated_text"]
        for r in map(
            json.loads,
            (
                ROOT
                / "artifacts/kie-training/analysis/real660-vs-30k-20261006"
                / "predictions/validation.jsonl"
            )
            .read_text()
            .splitlines(),
        )
    }
    metrics = {}
    for name, rows in (("original_r12", source), ("repaired_r13", repaired)):
        validation = [r for s, r in rows if s == "validation"]
        metrics[name] = {}
        for case_sensitive in (True, False):
            values, _ = structured_metrics(
                [predictions[r["documentId"]] for r in validation],
                [json.dumps(r["target"]) for r in validation],
                task,
                case_sensitive=case_sensitive,
            )
            metrics[name]["strict" if case_sensitive else "ignore_case"] = values

    timings = {}
    for mode in ("original_validation", "finalizers_and_validation"):
        elapsed = []
        for _ in range(11):
            start = time.perf_counter()
            for _, row in source:
                target = row["target"]
                if mode == "finalizers_and_validation":
                    target = reconcile_equipment_categories(
                        target, source_text=row["joinedRawText"]
                    )[0]
                    target = normalize_target_casing(target)[0]
                task.canonicalize(target)
            elapsed.append(time.perf_counter() - start)
        timings[mode + "_median_seconds_660"] = statistics.median(elapsed)
    tracemalloc.start()
    for _, row in source:
        target = reconcile_equipment_categories(row["target"], source_text=row["joinedRawText"])[0]
        normalize_target_casing(target)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result = dict(
        validatedDocuments=660,
        train=600,
        validation=60,
        ocrAndMembershipChanges=0,
        unrelatedLeafChanges=0,
        sourceAndBackupHashVerified=True,
        allFinalizerReplaysMatch=True,
        schemaAndPromptSnapshotsMatchTraining=True,
        uppercaseEditsByField=dict(case_counts),
        equipmentConversionsBySplit=dict(equipment_changes),
        metrics=metrics,
        finalizerPeakTracedBytes=peak,
        **timings,
    )
    write(OUT / "independent-validation.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("metrics", "uppercaseEditsByField")},
            indent=2,
        )
    )


def training_preflight():
    from document_ocr.training.data import prepare_datasets
    from document_ocr.training.runtime import build_training_arguments, load_tokenizer

    result, prepared = {}, {}
    for mode in ("compact", "pretty"):
        config = config_for(mode)
        task = load_training_task(ROOT, config)
        prompt = load_prompt(ROOT, config.prompt, task)
        tokenizer = load_tokenizer(config)
        start = time.perf_counter()
        data = prepare_datasets(
            project_root=ROOT, config=config, prompt=prompt, task=task, tokenizer=tokenizer
        )
        args = build_training_arguments(config, OUT / f"{mode}-preflight")
        assert {k: len(v) for k, v in data.datasets.items()} == {"train": 600, "validation": 60}
        assert args.num_train_epochs == 10 and args.eval_steps == args.save_steps == 95
        assert args.per_device_train_batch_size == 1 and args.gradient_accumulation_steps == 32
        result[mode] = dict(
            tokenLengths=data.token_lengths,
            cacheIdentity=data.cache_identity,
            seconds=time.perf_counter() - start,
            targetFormat=config.dataset.preprocessing.target_format,
            evaluationSteps=args.eval_steps,
            savingSteps=args.save_steps,
        )
        prepared[mode] = data
    for split in ("train", "validation"):
        for a, b in zip(
            prepared["compact"].datasets[split], prepared["pretty"].datasets[split], strict=True
        ):
            assert a["document_id"] == b["document_id"] and a["input_ids"] == b["input_ids"]
            left, right = [tokenizer.decode(r["labels"], skip_special_tokens=True) for r in (a, b)]
            assert json.loads(left) == json.loads(right)
            assert "\n" not in left and "\n" in right
            assert a["labels"][-1] == b["labels"][-1] == tokenizer.eos_token_id
    assert prepared["compact"].cache_identity != prepared["pretty"].cache_identity
    result["modelLoaded"] = False
    result["trainingStarted"] = False
    result["runtimeModule"] = __import__(
        "document_ocr.training.runtime", fromlist=["__file__"]
    ).__file__
    write(OUT / "training-preflight.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-preflight", action="store_true")
    args = parser.parse_args()
    training_preflight() if args.training_preflight else validate()
