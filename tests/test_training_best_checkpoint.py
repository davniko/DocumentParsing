"""Real CPU Trainer/PEFT regression tests; no pretrained weights or GPU needed."""

# Optional training dependencies must be checked before importing their runtime classes.
# ruff: noqa: E402

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import load_file, save_file
from test_training_model_contract import _tiny_t5gemma2
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator

from document_ocr.training.checkpoints import (
    PeftBestCheckpointMixin,
    adapter_directory,
    verify_best_adapter_export,
)
from document_ocr.training.config import ScheduleFreeAdamWConfig
from document_ocr.training.prediction import SamplerAwarePredictionMixin
from document_ocr.training.schedule_free import ScheduleFreeTrainerMixin


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _trainer(
    directory: Path,
    *,
    adapter: str,
    schedule_free: bool,
    fixed: bool = True,
    compiled: bool = False,
):
    torch.manual_seed(123)
    model = get_peft_model(
        _tiny_t5gemma2(),
        LoraConfig(
            r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"], task_type=TaskType.SEQ_2_SEQ_LM
        ),
        adapter_name=adapter,
    )
    bases = [SamplerAwarePredictionMixin]
    if schedule_free:
        bases.append(ScheduleFreeTrainerMixin)
    if fixed:
        bases.append(PeftBestCheckpointMixin)
    bases.append(Seq2SeqTrainer)
    trainer_type = type(
        "TestTrainer",
        tuple(bases),
        {
            "schedule_free_adamw": ScheduleFreeAdamWConfig(
                warmup_steps=1, r=0.0, weight_lr_power=2.0, foreach=True
            )
            if schedule_free
            else None,
        },
    )
    evaluations = 0

    def metrics(_):
        nonlocal evaluations
        evaluations += 1
        # Deliberately make an earlier checkpoint best, regardless of loss.
        return {"selection": -evaluations}

    data = [
        dict(
            input_ids=torch.tensor([2, 3 + i, 1]),
            attention_mask=torch.ones(3, dtype=torch.long),
            labels=torch.tensor([7, 8 + i, 1]),
        )
        for i in range(2)
    ]
    return trainer_type(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=str(directory),
            max_steps=4,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            learning_rate=0.01,
            optim="schedule_free_adamw" if schedule_free else "adamw_torch",
            lr_scheduler_type="constant",
            warmup_steps=1 if schedule_free else 0,
            eval_strategy="steps",
            eval_steps=1,
            save_strategy="steps",
            save_steps=1,
            save_total_limit=None,
            load_best_model_at_end=True,
            metric_for_best_model="selection",
            greater_is_better=True,
            logging_strategy="no",
            report_to=[],
            use_cpu=True,
            disable_tqdm=True,
            label_names=["labels"],
            torch_compile=compiled,
            torch_compile_backend="eager" if compiled else None,
        ),
        train_dataset=data,
        eval_dataset=data,
        compute_metrics=metrics,
        data_collator=default_data_collator,
    )


def _weights(checkpoint, adapter):
    return load_file(adapter_directory(checkpoint, adapter) / "adapter_model.safetensors")


def test_unpatched_trainer_reproduces_named_adapter_export_failure(tmp_path):
    trainer = _trainer(tmp_path, adapter="named", schedule_free=True, fixed=False)
    trainer.train()
    assert Path(trainer.state.best_model_checkpoint) == tmp_path / "checkpoint-1"
    trainer.save_model(str(tmp_path / "final"))
    final, last, best = [
        _weights(tmp_path / name, "named") for name in ("final", "checkpoint-4", "checkpoint-1")
    ]
    assert all(torch.equal(final[k], last[k]) for k in final)
    assert any(not torch.equal(final[k], best[k]) for k in final)
    with pytest.raises(RuntimeError, match="differs from the best checkpoint"):
        verify_best_adapter_export(
            checkpoint=tmp_path / "checkpoint-1", exported=tmp_path / "final", adapter_name="named"
        )


@pytest.mark.parametrize("adapter", ["default", "experiment_rank48"])
@pytest.mark.parametrize("schedule_free", [False, True])
def test_best_named_adapter_loaded_exported_and_all_checkpoints_retained(
    tmp_path,
    adapter,
    schedule_free,
):
    trainer = _trainer(tmp_path, adapter=adapter, schedule_free=schedule_free)
    parameters_before = {
        name: (id(value), value.requires_grad) for name, value in trainer.model.named_parameters()
    }
    result = trainer.train()
    assert parameters_before == {
        name: (id(value), value.requires_grad) for name, value in trainer.model.named_parameters()
    }
    assert result.global_step == 4
    assert Path(trainer.state.best_model_checkpoint) == tmp_path / "checkpoint-1"
    assert sorted(p.name for p in tmp_path.glob("checkpoint-*")) == [
        "checkpoint-1",
        "checkpoint-2",
        "checkpoint-3",
        "checkpoint-4",
    ]
    for path in tmp_path.glob("checkpoint-*"):
        assert (path / "optimizer.pt").is_file()
        assert (path / "trainer_state.json").is_file()
        if schedule_free:
            state = torch.load(path / "optimizer.pt", weights_only=False)
            assert state["param_groups"][0]["train_mode"] is False
    if schedule_free:
        assert trainer.optimizer.param_groups[0]["train_mode"] is False
    trainer.save_model(str(tmp_path / "final"))
    receipt = verify_best_adapter_export(
        checkpoint=tmp_path / "checkpoint-1", exported=tmp_path / "final", adapter_name=adapter
    )
    assert receipt["status"] == "verified"
    assert receipt["tensor_count"] > 0
    final, best, last = [
        _weights(tmp_path / name, adapter) for name in ("final", "checkpoint-1", "checkpoint-4")
    ]
    assert all(torch.equal(final[k], best[k]) for k in final)
    assert any(not torch.equal(final[k], last[k]) for k in final)


@pytest.mark.parametrize("defect", ["missing_file", "missing_tensor", "extra_tensor", "shape"])
def test_best_loading_fails_loudly_for_incomplete_checkpoint(tmp_path, defect):
    trainer = _trainer(tmp_path, adapter="named", schedule_free=False)
    trainer.save_model(str(tmp_path / "checkpoint"))
    trainer.state.best_model_checkpoint = str(tmp_path / "checkpoint")
    path = adapter_directory(tmp_path / "checkpoint", "named") / "adapter_model.safetensors"
    tensors = load_file(path)
    key = next(iter(tensors))
    if defect == "missing_file":
        path.unlink()
    elif defect == "missing_tensor":
        del tensors[key]
        save_file(tensors, path)
    elif defect == "extra_tensor":
        tensors["not_a_model_parameter"] = torch.ones(1)
        save_file(tensors, path)
    else:
        tensors[key] = torch.ones(1, 1)
        save_file(tensors, path)
    with pytest.raises(RuntimeError):
        trainer._load_best_model()


def test_best_loading_rejects_multiple_active_training_adapters(tmp_path):
    trainer = _trainer(tmp_path, adapter="named", schedule_free=False)
    trainer.model.add_adapter("other", trainer.model.peft_config["named"])
    with pytest.raises(RuntimeError, match="exactly one training adapter"):
        trainer._load_best_model()


def test_compiled_adapter_loads_into_original_model(tmp_path):
    # Exercise Trainer's actual wrapping path, not a manually replaced Trainer.model.
    trainer = _trainer(tmp_path, adapter="named", schedule_free=False, compiled=True)
    trainer.train()
    assert trainer.model_wrapped is not trainer.model
    assert Path(trainer.state.best_model_checkpoint) == tmp_path / "checkpoint-1"
    trainer.save_model(str(tmp_path / "final"))
    verify_best_adapter_export(
        checkpoint=tmp_path / "checkpoint-1", exported=tmp_path / "final", adapter_name="named"
    )


@pytest.mark.parametrize("distributed", ["is_deepspeed_enabled", "is_fsdp_enabled"])
def test_distributed_checkpoint_load_is_explicitly_unsupported(tmp_path, distributed):
    trainer = _trainer(tmp_path, adapter="named", schedule_free=False)
    setattr(trainer, distributed, True)
    with pytest.raises(RuntimeError, match="single-process"):
        trainer._load_best_model()


def test_missing_selection_cannot_silently_export_last_checkpoint(tmp_path):
    trainer = _trainer(tmp_path, adapter="named", schedule_free=False)
    with pytest.raises(RuntimeError, match="selected checkpoint"):
        trainer._load_best_model()


def test_non_peft_model_uses_transformers_loader():
    class Base:
        def _load_best_model(self):
            self.loaded = True

    class Trainer(PeftBestCheckpointMixin, Base):
        model = torch.nn.Linear(2, 2)
        accelerator = SimpleNamespace(unwrap_model=lambda model, **kwargs: model)

    trainer = Trainer()
    trainer._load_best_model()
    assert trainer.loaded


@pytest.mark.parametrize("defect", ["keys", "dtype", "value"])
def test_export_guard_rejects_wrong_weights(tmp_path, defect):
    for name in ("best", "final"):
        directory = tmp_path / name / "named"
        directory.mkdir(parents=True)
        (directory / "adapter_config.json").write_text(json.dumps({}))
        save_file({"a": torch.ones(2)}, directory / "adapter_model.safetensors")
    value = (
        {"b": torch.ones(2)}
        if defect == "keys"
        else {"a": torch.ones(2, dtype=torch.float64) if defect == "dtype" else torch.zeros(2)}
    )
    save_file(value, tmp_path / "final/named/adapter_model.safetensors")
    with pytest.raises(RuntimeError, match="differ"):
        verify_best_adapter_export(
            checkpoint=tmp_path / "best", exported=tmp_path / "final", adapter_name="named"
        )
