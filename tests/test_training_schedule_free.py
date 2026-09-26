from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from document_ocr.training.config import (
    ScheduleFreeAdamWConfig,
    TrainingConfig,
    load_training_config,
)
from document_ocr.training.schedule_free import ScheduleFreeTrainerMixin


def _settings() -> ScheduleFreeAdamWConfig:
    return ScheduleFreeAdamWConfig(
        warmup_steps=161,
        r=0.0,
        weight_lr_power=2.0,
        foreach=True,
    )


def test_schedule_free_configuration_rejects_double_scheduling() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/training/t5gemma2_270m_lora.pilot106.yaml"
    value = load_training_config(path).model_dump(mode="python")
    optimization = value["optimization"]
    optimization["optimizer"] = "schedule_free_adamw"
    optimization["schedule_free_adamw"] = _settings().model_dump(mode="python")

    with pytest.raises(ValidationError, match="constant external scheduler"):
        TrainingConfig.model_validate(value, strict=True)

    optimization["lr_scheduler_type"] = "constant"
    with pytest.raises(ValidationError, match="warmup_steps, not warmup_ratio"):
        TrainingConfig.model_validate(value, strict=True)

    optimization["warmup_ratio"] = 0.0
    optimization["optimizer_args"] = "unsupported_option=0.9"
    with pytest.raises(ValidationError, match="typed settings, not optimizer_args"):
        TrainingConfig.model_validate(value, strict=True)


class _Optimizer:
    def __init__(self, model: SimpleNamespace) -> None:
        self.model = model
        self.train_mode = True
        self.eval_calls = 0

    def eval(self) -> None:
        self.eval_calls += 1
        if self.train_mode:
            self.model.weight += 100
            self.train_mode = False

    def train(self) -> None:
        if not self.train_mode:
            self.model.weight -= 100
            self.train_mode = True


class _BaseTrainer:
    def get_optimizer_cls_and_kwargs(
        self, args: Any, model: Any = None
    ) -> tuple[type[object], dict[str, Any]]:
        return object, {"lr": 0.0001, "warmup_steps": 0}

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        self.saved_weight = self.model.weight  # type: ignore[attr-defined]

    def _save_checkpoint(self) -> None:
        self.save_model("checkpoint")
        self.saved_optimizer_mode = self.optimizer.train_mode  # type: ignore[attr-defined]

    def _load_best_model(self) -> None:
        self.model.weight = 7  # type: ignore[attr-defined]


class _Trainer(ScheduleFreeTrainerMixin, _BaseTrainer):
    schedule_free_adamw = _settings()

    def __init__(self) -> None:
        self.model = SimpleNamespace(weight=1)
        self.optimizer = _Optimizer(self.model)
        self.saved_weight: int | None = None
        self.saved_optimizer_mode: bool | None = None


def test_schedule_free_save_resume_and_best_model_lifecycle() -> None:
    trainer = _Trainer()
    optimizer_cls, kwargs = trainer.get_optimizer_cls_and_kwargs(
        SimpleNamespace(optim="schedule_free_adamw")
    )
    assert optimizer_cls is object
    assert kwargs["warmup_steps"] == 161
    assert kwargs["foreach"] is True

    trainer._save_checkpoint()
    assert trainer.saved_weight == 101
    assert trainer.saved_optimizer_mode is False
    trainer.optimizer.train()
    assert trainer.model.weight == 1

    trainer._load_best_model()
    trainer.save_model("final")
    assert trainer.saved_weight == 7
    assert trainer.optimizer.train_mode is False
