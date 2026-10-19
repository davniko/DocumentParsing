"""Evaluation order, checkpoint evidence and real Trainer replay regressions."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from document_ocr.training.config import load_training_config
from document_ocr.training.metrics import make_compute_metrics
from document_ocr.training.prediction import (
    PredictionPublicationError,
    SamplerAwarePredictionMixin,
    length_sorted_evaluation_indices,
)
from document_ocr.training.runtime import _EvaluationRecorder
from document_ocr.training.tasks import canonical_json, get_training_task

PROJECT = Path(__file__).parents[1]


class _Dataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        return [row[key] for row in self.rows] if isinstance(key, str) else self.rows[key]


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def batch_decode(self, rows, **kwargs):
        return [
            canonical_json(
                {"schemaVersion": "2.0.0", "documentPatch": {"billOfLadingNumber": str(row[0])}}
            )
            for row in rows
        ]


def _recorder(directory, tokenizer=None):
    return _EvaluationRecorder(
        run_dir=directory,
        config=load_training_config(PROJECT / "configs/training/t5gemma2_270m_lora.pilot106.yaml"),
        tokenizer=tokenizer or _Tokenizer(),
        task=get_training_task("bill_of_lading_semantic_v2"),
        dataset_cache_identity="test-identity",
    )


def _output(tokenizer, task, rows):
    values = np.array(rows)
    output = SimpleNamespace(predictions=values, label_ids=values.copy())
    output.metrics = {
        f"eval_{key}": value for key, value in make_compute_metrics(tokenizer, task)(output).items()
    }
    output.metrics["eval_loss"] = 0.125
    return output


def test_stable_length_order_and_ties():
    assert length_sorted_evaluation_indices([3, 8, 8, 1, 5]) == (1, 2, 4, 0, 3)


@pytest.mark.parametrize("lengths", [[], [True], [1.5], [0], [-1]])
def test_bad_lengths_fail_loudly(lengths):
    with pytest.raises(PredictionPublicationError):
        length_sorted_evaluation_indices(lengths)


def test_eval_grouping_is_repeatable_and_does_not_touch_training_sampler():
    class Base:
        def _get_eval_sampler(self, dataset):
            return SimpleNamespace(lengths=[3, 8, 8, 1, 5])

        def _get_train_sampler(self):
            return "original-training-sampler"

    class Trainer(SamplerAwarePredictionMixin, Base):
        args = SimpleNamespace(world_size=1, train_sampling_strategy="group_by_length")

    trainer = Trainer()
    assert list(trainer._get_eval_sampler(range(5))) == [1, 2, 4, 0, 3]
    assert list(trainer._get_eval_sampler(range(5))) == [1, 2, 4, 0, 3]
    assert trainer._get_train_sampler() == "original-training-sampler"
    trainer.evaluation_indices = (4, 1, 0, 3, 2)
    assert list(trainer._get_eval_sampler(range(5))) == [4, 1, 0, 3, 2]
    trainer.evaluation_indices = (True, 0, 2, 3, 4)
    with pytest.raises(PredictionPublicationError, match="full permutation"):
        trainer._get_eval_sampler(range(5))


def test_recorder_links_only_the_evaluated_step_and_verifies_output_order(tmp_path):
    recorder = _recorder(tmp_path)
    dataset = _Dataset([{"document_id": "a", "labels": [2]}, {"document_id": "b", "labels": [3]}])
    output = _output(recorder.tokenizer, recorder.task, [[3], [2]])
    recorder.record(step=7, dataset=dataset, indices=(1, 0), output=output)
    assert recorder.latest["document_ids"] == ["b", "a"]
    checkpoint = tmp_path / "checkpoint-7"
    checkpoint.mkdir()
    recorder.checkpoint_saved(step=7, output_dir=str(tmp_path))
    receipt = json.loads((checkpoint / "evaluation.json").read_text())
    assert receipt == recorder.latest
    recorder.checkpoint_saved(step=8, output_dir=str(tmp_path))
    assert not (tmp_path / "checkpoint-8/evaluation.json").exists()
    assert len(recorder.artifact_paths) == 2
    with pytest.raises(RuntimeError, match="batch plan"):
        recorder.record(step=8, dataset=dataset, indices=(0, 1), output=output)
    with pytest.raises(RuntimeError, match="exactly once"):
        recorder.record(step=8, dataset=dataset, indices=(1, 1), output=output)
    duplicate = _Dataset([{"document_id": "a", "labels": [2]}, {"document_id": "a", "labels": [3]}])
    with pytest.raises(RuntimeError, match="not unique"):
        recorder.record(step=8, dataset=duplicate, indices=(0, 1), output=output)


@pytest.mark.parametrize("schedule_free", [False, True])
def test_real_trainer_saves_and_replays_evaluated_weights_with_fixed_batches(
    tmp_path, schedule_free
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from test_training_model_contract import _tiny_t5gemma2
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    from document_ocr.training.config import ScheduleFreeAdamWConfig
    from document_ocr.training.schedule_free import ScheduleFreeTrainerMixin

    bases = [SamplerAwarePredictionMixin]
    if schedule_free:
        bases.append(ScheduleFreeTrainerMixin)
    bases.append(Seq2SeqTrainer)
    Trainer = type(
        "ReplayTestTrainer",
        tuple(bases),
        {
            "schedule_free_adamw": (
                ScheduleFreeAdamWConfig(warmup_steps=1, r=0.0, weight_lr_power=2.0, foreach=True)
                if schedule_free
                else None
            )
        },
    )

    class Tokenizer(_Tokenizer):
        def batch_decode(self, rows, **kwargs):
            return [
                canonical_json(
                    {"schemaVersion": "2.0.0", "documentPatch": {"billOfLadingNumber": "TEST"}}
                )
                for _ in rows
            ]

    def collate(rows):
        longest = max(len(row["input_ids"]) for row in rows)
        return {
            "input_ids": torch.tensor(
                [row["input_ids"] + [0] * (longest - len(row["input_ids"])) for row in rows]
            ),
            "attention_mask": torch.tensor(
                [
                    [1] * len(row["input_ids"]) + [0] * (longest - len(row["input_ids"]))
                    for row in rows
                ]
            ),
            "labels": torch.tensor([row["labels"] for row in rows]),
        }

    dataset = Dataset.from_list(
        [
            {
                "document_id": str(index),
                "input_ids": [2] + [3] * length + [1],
                "input_length": length + 2,
                "labels": [7, 8, 1],
            }
            for index, length in enumerate([2, 7, 4, 3])
        ]
    )
    tokenizer = Tokenizer()
    recorder = _recorder(tmp_path, tokenizer)

    def build(directory):
        torch.manual_seed(123)
        model = get_peft_model(
            _tiny_t5gemma2(),
            LoraConfig(
                r=2,
                lora_alpha=2,
                target_modules=["q_proj", "v_proj"],
                task_type=TaskType.SEQ_2_SEQ_LM,
            ),
        )
        return Trainer(
            model=model,
            args=Seq2SeqTrainingArguments(
                output_dir=str(directory),
                max_steps=2,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                train_sampling_strategy="group_by_length",
                length_column_name="input_length",
                optim="schedule_free_adamw" if schedule_free else "adamw_torch",
                lr_scheduler_type="constant",
                warmup_steps=1 if schedule_free else 0,
                eval_strategy="steps",
                eval_steps=1,
                save_strategy="steps",
                save_steps=1,
                predict_with_generate=True,
                generation_max_length=8,
                remove_unused_columns=False,
                use_cpu=True,
                report_to=[],
                disable_tqdm=True,
                label_names=["labels"],
                dataloader_num_workers=1,
                dataloader_persistent_workers=True,
            ),
            train_dataset=dataset,
            eval_dataset=dataset,
            data_collator=collate,
            compute_metrics=make_compute_metrics(tokenizer, recorder.task),
        )

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        trainer = build(tmp_path)
        trainer.evaluation_recorder = recorder
        trainer.train()
        receipt = json.loads((tmp_path / "checkpoint-2/evaluation.json").read_text())
        assert receipt["document_ids"] == ["1", "2", "3", "0"]
        assert (tmp_path / "checkpoint-1/evaluation.json").exists()
        before = torch.get_rng_state().clone()
        list(trainer._get_eval_sampler(dataset))
        assert torch.equal(torch.get_rng_state(), before)
        first = trainer.predict_with_identities(dataset, metric_key_prefix="eval")
        torch.manual_seed(897)
        second = trainer.predict_with_identities(dataset, metric_key_prefix="eval")
        assert np.array_equal(first.output.predictions, second.output.predictions)
        assert first.output.metrics["eval_loss"] == second.output.metrics["eval_loss"]
        # A genuinely new model/Trainer, not merely evaluating the in-memory weights.
        replay = build(tmp_path / "replay")
        set_peft_model_state_dict(
            replay.model, load_file(str(tmp_path / "checkpoint-2/adapter_model.safetensors"))
        )
        third = replay.predict_with_identities(dataset, metric_key_prefix="eval")
        assert first.document_ids == third.document_ids == tuple(receipt["document_ids"])
        assert np.array_equal(first.output.predictions, third.output.predictions)
        assert all(third.output.metrics[key] == value for key, value in receipt["metrics"].items())
        assert json.loads((tmp_path / "checkpoint-2/evaluation.json").read_text()) == receipt
    finally:
        torch.set_num_threads(previous_threads)
