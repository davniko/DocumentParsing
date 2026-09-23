from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from document_ocr.training.collator import MetadataStrippingCollator
from document_ocr.training.config import load_training_config
from document_ocr.training.data import (
    _apply_target_length_limit,
    _tokenize_batch,
    inspect_dataset,
    prepare_datasets,
)
from document_ocr.training.prompting import load_prompt
from document_ocr.training.tasks import get_training_task

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "t5gemma2_270m_lora.pilot106.yaml"


class _WordTokenizer:
    name_or_path = "test-tokenizer"
    bos_token_id = 98
    eos_token_id = 99
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return 100

    def __call__(self, text: Any = None, **kwargs: Any) -> dict[str, list[list[int]]]:
        self.calls.append(dict(kwargs))
        values = kwargs.get("text_target", text)
        if values is None:
            raise AssertionError("tokenizer input is absent")
        encoded = [[(index % 90) + 1 for index, _ in enumerate(value.split())] for value in values]
        if kwargs.get("add_special_tokens"):
            encoded = [[self.bos_token_id, *item] for item in encoded]
        max_length = kwargs.get("max_length")
        if kwargs.get("truncation") and max_length is not None:
            encoded = [item[:max_length] for item in encoded]
        return {
            "input_ids": encoded,
            "attention_mask": [[1] * len(item) for item in encoded],
        }


def _training_components() -> Any:
    config = load_training_config(CONFIG_PATH)
    task = get_training_task(config.task)
    return (
        config,
        load_prompt(PROJECT_ROOT, config.prompt, task),
        task,
    )


def test_published_training_dataset_passes_full_cpu_inspection() -> None:
    config, prompt, task = _training_components()

    records, report = inspect_dataset(
        project_root=PROJECT_ROOT,
        config=config,
        prompt=prompt,
        task=task,
    )

    assert report.total_records == 106
    assert report.split_records == {"train": 90, "validation": 16, "test": 0}
    assert len(records["train"]) == 90
    assert len(records["validation"]) == 16
    assert records["train"][0].input_text.endswith("\nJSON:\n")
    assert records["train"][0].target_text.startswith('{"documentPatch":')
    assert records["train"][0].target_text.endswith('"schemaVersion":"2.0.0"}')


def test_runtime_partition_inspection_is_reported_and_source_ordered() -> None:
    config, prompt, task = _training_components()
    source = (
        PROJECT_ROOT
        / "artifacts"
        / "kie-training"
        / "datasets"
        / "mpci-bl-pilot106-raw-latin-v1"
        / "records.jsonl"
    )
    payload = source.read_bytes()
    dataset_value = config.dataset.model_dump(mode="python")
    dataset_value.update(
        {
            "splits": None,
            "source": {
                "path": str(source),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "records": 106,
            },
            "partition": {
                "algorithm": "seeded_sha256_rank_v1",
                "seed": 42,
                "validation_size": {"kind": "records", "value": 16},
                "coverage_policy": "retain_each_target_leaf_in_train",
            },
        }
    )
    dataset = type(config.dataset).model_validate(dataset_value, strict=True)
    config = config.model_copy(update={"dataset": dataset})

    records, report = inspect_dataset(
        project_root=PROJECT_ROOT,
        config=config,
        prompt=prompt,
        task=task,
    )

    source_ids = [json.loads(line)["documentId"] for line in payload.decode().splitlines()]
    assert report.dataset_mode == "runtime_partition"
    assert report.split_records == {"train": 90, "validation": 16, "test": 0}
    assert report.partition is not None
    assert report.partition["seed"] == 42
    assert report.partition["resolved_validation_records"] == 16
    assert len(report.partition["outputs"]["train"]["document_ids"]) == 90
    assert len(report.partition["outputs"]["validation"]["document_ids"]) == 16
    assert [record.document_id for record in records["train"]] == [
        document_id
        for document_id in source_ids
        if document_id in {record.document_id for record in records["train"]}
    ]
    assert [record.document_id for record in records["validation"]] == [
        document_id
        for document_id in source_ids
        if document_id in {record.document_id for record in records["validation"]}
    ]


def test_published_training_dataset_preserves_audited_latin_corrections() -> None:
    correction_dir = (
        PROJECT_ROOT / "artifacts" / "kie-training" / "datasets" / "mpci-bl-pilot106-raw-latin-v1"
    )
    correction_manifest = json.loads((correction_dir / "manifest.json").read_text(encoding="utf-8"))
    source_payload = (correction_dir / "records.jsonl").read_bytes()
    assert correction_manifest["corrected_document_count"] == 5
    assert correction_manifest["corrected_target_value_count"] == 13
    assert correction_manifest["output"]["sha256"] == hashlib.sha256(source_payload).hexdigest()

    rows = {
        row["documentId"]: row
        for row in (json.loads(line) for line in source_payload.decode("utf-8").splitlines())
    }

    def pointer_value(root: Any, pointer: str) -> Any:
        value = root
        for part in pointer.removeprefix("/").split("/"):
            value = value[int(part)] if isinstance(value, list) else value[part]
        return value

    for correction in correction_manifest["corrections"]:
        row = rows[correction["document_id"]]
        assert pointer_value(row["target"], correction["target_path"]) == correction["after"]
        assert row["targetCorrectionSet"] == "preserve_printed_latin_v1"
        assert all(evidence in row["joinedRawText"] for evidence in correction["raw_evidence"])

    original = json.loads(
        (
            PROJECT_ROOT
            / "artifacts"
            / "kie-training"
            / "datasets"
            / "mpci-bl-pilot106-seed42"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    corrected = json.loads(
        (
            PROJECT_ROOT
            / "artifacts"
            / "kie-training"
            / "datasets"
            / "mpci-bl-pilot106-seed42-v2"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    for split in ("train", "validation"):
        assert (
            corrected["outputs"][split]["document_ids"]
            == original["outputs"][split]["document_ids"]
        )


def test_dataset_inspection_rejects_input_hash_mismatch(tmp_path: Path) -> None:
    config, prompt, task = _training_components()
    record = {
        "documentId": "doc_test",
        "joinedRawText": "--- PAGE 1 ---\nOCR",
        "joinedRawTextSha256": "0" * 64,
        "target": {
            "schemaVersion": "2.0.0",
            "documentPatch": {"billOfLadingNumber": "ABC"},
        },
    }
    payload = (json.dumps(record) + "\n").encode()
    source = tmp_path / "records.jsonl"
    source.write_bytes(payload)
    source_config = config.dataset.splits.train[0].model_copy(
        update={
            "path": str(source),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "records": 1,
        }
    )
    config = config.model_copy(
        update={
            "dataset": config.dataset.model_copy(
                update={
                    "splits": config.dataset.splits.model_copy(update={"train": [source_config]})
                }
            )
        }
    )

    with pytest.raises(ValueError, match="input text SHA-256 mismatch"):
        inspect_dataset(
            project_root=PROJECT_ROOT,
            config=config,
            prompt=prompt,
            task=task,
        )


def test_tokenization_refuses_source_overflow_and_keeps_complete_targets_for_filtering() -> None:
    config, _, _ = _training_components()
    config = config.model_copy(
        update={
            "dataset": config.dataset.model_copy(
                update={
                    "preprocessing": config.dataset.preprocessing.model_copy(
                        update={"max_source_length": 3, "max_target_length": 3}
                    )
                }
            )
        }
    )
    batch = {
        "document_id": ["one"],
        "input_text": ["one two three"],
        "target_text": ["one"],
    }
    with pytest.raises(ValueError, match="source token length"):
        _tokenize_batch(batch, tokenizer=_WordTokenizer(), config=config)

    batch = {
        "document_id": ["one"],
        "input_text": ["one"],
        "target_text": ["one two three"],
    }
    output = _tokenize_batch(batch, tokenizer=_WordTokenizer(), config=config)
    assert output["target_length"] == [4]
    assert output["labels"] == [[1, 2, 3, _WordTokenizer.eos_token_id]]


def test_target_filter_is_inclusive_order_preserving_and_never_truncates(capsys) -> None:
    from datasets import Dataset

    dataset = Dataset.from_list(
        [
            {"document_id": "short", "target_length": 2, "labels": [1, 99]},
            {"document_id": "long", "target_length": 4, "labels": [1, 2, 3, 99]},
            {"document_id": "boundary", "target_length": 3, "labels": [1, 2, 99]},
        ]
    )
    retained, report = _apply_target_length_limit(dataset, split="train", max_target_length=3)
    assert list(retained["document_id"]) == ["short", "boundary"]
    assert list(retained["labels"]) == [[1, 99], [1, 2, 99]]
    assert report["excluded"] == [{"document_id": "long", "target_length": 4}]
    assert report["original_records"] == 3
    assert report["retained_records"] == 2
    assert report["excluded_records"] == 1
    assert "skipping 1 of 3 samples; retaining 2" in capsys.readouterr().err
    assert len(dataset) == 3
    unchanged, report = _apply_target_length_limit(dataset, split="train", max_target_length=4)
    assert unchanged is dataset
    assert report["excluded_records"] == 0


@pytest.mark.parametrize("split", ["validation", "test"])
def test_target_filter_refuses_to_change_held_out_membership(split) -> None:
    from datasets import Dataset

    dataset = Dataset.from_list([{"document_id": "held-out", "target_length": 4}])
    with pytest.raises(ValueError, match="held-out samples must not be filtered"):
        _apply_target_length_limit(dataset, split=split, max_target_length=3)


def test_target_filter_refuses_an_empty_training_partition() -> None:
    from datasets import Dataset

    dataset = Dataset.from_list([{"document_id": "long", "target_length": 4}])
    with pytest.raises(ValueError, match="removed all train samples"):
        _apply_target_length_limit(dataset, split="train", max_target_length=3)


def test_preparation_filters_on_cache_hits_and_records_exact_exclusions(tmp_path, monkeypatch):
    import document_ocr.training.data as data_module

    config, prompt, task = _training_components()
    records, inspection = inspect_dataset(
        project_root=PROJECT_ROOT,
        config=config,
        prompt=prompt,
        task=task,
    )
    train = [
        replace(record, input_text="source", target_text=" ".join(["token"] * count))
        for record, count in zip(records["train"][:3], (1, 2, 3), strict=True)
    ]
    validation = [replace(records["validation"][0], input_text="source", target_text="token")]
    monkeypatch.setattr(
        data_module,
        "inspect_dataset",
        lambda **kw: (
            {"train": train, "validation": validation, "test": []},
            inspection,
        ),
    )
    preprocessing = config.dataset.preprocessing.model_copy(
        update={
            "cache_dir": str(tmp_path / "cache"),
            "num_proc": 2,
            "max_target_length": 3,
        }
    )
    config = config.model_copy(
        update={
            "dataset": config.dataset.model_copy(update={"preprocessing": preprocessing}),
        }
    )
    kwargs = dict(project_root=PROJECT_ROOT, config=config, prompt=prompt, task=task)
    first = prepare_datasets(**kwargs, tokenizer=_WordTokenizer())
    cached_tokenizer = _WordTokenizer()
    second = prepare_datasets(**kwargs, tokenizer=cached_tokenizer)
    assert not cached_tokenizer.calls
    assert first.cache_identity == second.cache_identity
    assert first.target_length_filter == second.target_length_filter
    assert list(second.datasets["train"]["document_id"]) == [row.document_id for row in train[:2]]
    assert second.token_lengths["train"]["target"]["max"] == 3
    assert second.report()["target_length_filter"]["train"]["excluded"] == [
        {"document_id": train[2].document_id, "target_length": 4},
    ]
    assert list(second.datasets["validation"]["document_id"]) == [validation[0].document_id]
    expanded = config.model_copy(
        update={
            "dataset": config.dataset.model_copy(
                update={
                    "preprocessing": preprocessing.model_copy(update={"max_target_length": 4}),
                }
            )
        }
    )
    third = prepare_datasets(**{**kwargs, "config": expanded}, tokenizer=_WordTokenizer())
    assert third.cache_identity != first.cache_identity
    assert len(third.datasets["train"]) == 3


def test_decoder_targets_exclude_bos_and_end_in_exactly_one_eos() -> None:
    config, _, _ = _training_components()
    tokenizer = _WordTokenizer()

    output = _tokenize_batch(
        {
            "document_id": ["one"],
            "input_text": ["source text"],
            "target_text": ["target text"],
        },
        tokenizer=tokenizer,
        config=config,
    )

    assert tokenizer.calls[0]["add_special_tokens"] is True
    assert tokenizer.calls[1]["add_special_tokens"] is False
    assert output["labels"] == [[1, 2, tokenizer.eos_token_id]]
    assert tokenizer.bos_token_id not in output["labels"][0]
    assert output["labels"][0].count(tokenizer.eos_token_id) == 1
    assert output["target_length"] == [3]


def test_decoder_target_tokenization_requires_eos() -> None:
    config, _, _ = _training_components()
    tokenizer = _WordTokenizer()
    tokenizer.eos_token_id = None

    with pytest.raises(ValueError, match="requires an EOS token"):
        _tokenize_batch(
            {
                "document_id": ["one"],
                "input_text": ["source"],
                "target_text": ["target"],
            },
            tokenizer=tokenizer,
            config=config,
        )


def test_explicit_source_truncation_is_the_only_truncation_path() -> None:
    config, _, _ = _training_components()
    preprocessing = config.dataset.preprocessing.model_copy(
        update={"max_source_length": 3, "max_target_length": 10, "source_overflow": "truncate"}
    )
    config = config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"preprocessing": preprocessing})}
    )

    output = _tokenize_batch(
        {
            "document_id": ["one"],
            "input_text": ["one two three"],
            "target_text": ["one two three"],
        },
        tokenizer=_WordTokenizer(),
        config=config,
    )

    assert output["input_length"] == [3]
    assert output["input_original_length"] == [4]
    assert output["target_length"] == [4]
    assert output["labels"][0][-1] == _WordTokenizer.eos_token_id
    assert output["source_truncated"] == [True]


def test_collator_retains_lengths_for_sampling_but_not_for_the_model() -> None:
    captured: list[Any] = []

    def base(features: Any) -> dict[str, Any]:
        captured.extend(features)
        return {"input_ids": [feature["input_ids"] for feature in features]}

    collator = MetadataStrippingCollator(base)
    result = collator(
        [
            {
                "document_id": "doc_one",
                "input_length": 2,
                "input_original_length": 2,
                "target_length": 1,
                "source_truncated": False,
                "input_ids": [1, 2],
                "attention_mask": [1, 1],
                "labels": [3],
            }
        ]
    )

    assert result == {"input_ids": [[1, 2]]}
    assert captured == [{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [3]}]


def test_arrow_preprocessing_is_cache_keyed_and_complete(tmp_path: Path) -> None:
    config, prompt, task = _training_components()
    preprocessing = config.dataset.preprocessing.model_copy(
        update={"cache_dir": str(tmp_path / "cache"), "num_proc": None}
    )
    config = config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"preprocessing": preprocessing})}
    )

    first = prepare_datasets(
        project_root=PROJECT_ROOT,
        config=config,
        prompt=prompt,
        task=task,
        tokenizer=_WordTokenizer(),
    )
    second = prepare_datasets(
        project_root=PROJECT_ROOT,
        config=config,
        prompt=prompt,
        task=task,
        tokenizer=_WordTokenizer(),
    )

    assert first.cache_identity == second.cache_identity
    assert first.decoder_target_contract == {
        "name": "content_tokens_then_terminal_eos_v1",
        "tokenizer_add_special_tokens": False,
        "append_eos_token_id": _WordTokenizer.eos_token_id,
        "bos_token_id_forbidden_in_labels": _WordTokenizer.bos_token_id,
    }
    assert len(first.datasets["train"]) == 90
    assert len(first.datasets["validation"]) == 16
    assert set(first.datasets["train"].column_names) == {
        "document_id",
        "input_ids",
        "attention_mask",
        "labels",
        "input_length",
        "input_original_length",
        "target_length",
        "source_truncated",
    }
    assert first.token_lengths["train"]["source_truncated_records"] == 0
    assert list((tmp_path / "cache").glob("train-*.arrow"))


def test_preparation_rejects_generation_capacity_below_evaluation_reference(
    tmp_path: Path,
) -> None:
    config, prompt, task = _training_components()
    preprocessing = config.dataset.preprocessing.model_copy(
        update={"cache_dir": str(tmp_path / "cache"), "num_proc": None}
    )
    config = config.model_copy(
        update={
            "dataset": config.dataset.model_copy(update={"preprocessing": preprocessing}),
            "evaluation": config.evaluation.model_copy(update={"generation_max_length": 1}),
        }
    )

    with pytest.raises(
        ValueError,
        match="evaluation generation_max_length cannot reproduce the longest reference",
    ):
        prepare_datasets(
            project_root=PROJECT_ROOT,
            config=config,
            prompt=prompt,
            task=task,
            tokenizer=_WordTokenizer(),
        )
