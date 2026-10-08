from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from document_ocr.training.config import load_training_config
from document_ocr.training.runtime import build_training_arguments

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = PROJECT_ROOT / "configs" / "training" / "production"
CONFIG_STEM = (
    "t5gemma2_270m_lora.mpci_bl_real600_synthetic1500_positions_inputonly_v7_e10_compact_eva_a32_"
)


def _config_path(rank: int) -> Path:
    return PRODUCTION / f"{CONFIG_STEM}r{rank}_local_schedulefree_v1.yaml"


def test_rank48_comparison_changes_only_rank_identity_retention_and_warmup() -> None:
    baseline = load_training_config(_config_path(32)).model_dump(mode="python")
    actual = load_training_config(_config_path(48)).model_dump(mode="python")

    baseline["run"]["run_id"] = baseline["run"]["run_id"].replace("a32-r32-", "a32-r48-")
    baseline["peft"]["adapter_name"] = baseline["peft"]["adapter_name"].replace(
        "a32_r32_", "a32_r48_"
    )
    baseline["peft"]["rank"] = 48
    baseline["checkpoint"]["total_limit"] = None
    baseline["logging"]["mlflow"]["tags"]["lora_rank"] = "48"
    baseline["optimization"]["schedule_free_adamw"]["warmup_steps"] = 33

    assert actual == baseline


def test_rank48_training_arguments_keep_all_scheduled_checkpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Capture the real translation without requiring CUDA or loading a model.
    import transformers

    monkeypatch.setattr(transformers, "Seq2SeqTrainingArguments", SimpleNamespace)
    config = load_training_config(_config_path(48))
    arguments = build_training_arguments(config, tmp_path / "run")

    assert config.peft.rank == 48
    assert config.peft.alpha == 32
    assert config.peft.use_rslora is True
    assert config.peft.eva is not None
    assert config.peft.eva.rho == 1.0
    assert arguments.save_total_limit is None
    assert arguments.save_steps == arguments.eval_steps == 165
    assert arguments.num_train_epochs == 10.0
    assert arguments.load_best_model_at_end is True
    assert arguments.metric_for_best_model == "field_value_f1"
    assert arguments.save_only_model is False
    assert arguments.warmup_steps == 33


def test_rank48_warmup_covers_five_percent_of_configured_optimizer_updates() -> None:
    config = load_training_config(_config_path(48))
    optimization = config.optimization
    assert optimization.max_steps == -1
    assert config.runtime.distributed_processes == 1
    records = sum(source.records for source in config.dataset.splits.train)
    batches_per_epoch = math.ceil(records / optimization.per_device_train_batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / optimization.gradient_accumulation_steps)
    total_updates = math.ceil(updates_per_epoch * optimization.num_train_epochs)

    assert total_updates == 660
    assert optimization.schedule_free_adamw is not None
    assert optimization.schedule_free_adamw.warmup_steps == math.ceil(0.05 * total_updates)
    assert optimization.warmup_ratio == 0.0
    assert optimization.lr_scheduler_type == "constant"


def test_rank48_real_trainer_optimizer_applies_warmup_once(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    from document_ocr.training.schedule_free import ScheduleFreeTrainerMixin

    config = load_training_config(_config_path(48))
    optimization = config.optimization

    class WarmupTrainer(ScheduleFreeTrainerMixin, Seq2SeqTrainer):
        schedule_free_adamw = optimization.schedule_free_adamw

    args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to=[],
        num_train_epochs=optimization.num_train_epochs,
        per_device_train_batch_size=optimization.per_device_train_batch_size,
        gradient_accumulation_steps=optimization.gradient_accumulation_steps,
        optim=optimization.optimizer,
        learning_rate=optimization.learning_rate,
        lr_scheduler_type=optimization.lr_scheduler_type,
        warmup_steps=optimization.schedule_free_adamw.warmup_steps,
    )
    trainer = WarmupTrainer(model=torch.nn.Linear(1, 1), args=args)
    records = sum(source.records for source in config.dataset.splits.train)
    dataloader = torch.utils.data.DataLoader(
        torch.arange(records), batch_size=optimization.per_device_train_batch_size
    )
    total_updates = trainer.set_initial_training_values(args, dataloader)[-1]
    assert total_updates == 660
    trainer.create_optimizer_and_scheduler(total_updates)
    optimizer = trainer.optimizer
    optimizer.train()
    assert all(group["warmup_steps"] == 33 for group in optimizer.param_groups)

    for step in range(1, 35):
        for parameter in trainer.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        trainer.lr_scheduler.step()
        expected_lr = optimization.learning_rate * min(step / 33, 1.0)
        for group in optimizer.param_groups:
            assert group["scheduled_lr"] == pytest.approx(expected_lr)
            assert group["lr"] == optimization.learning_rate
