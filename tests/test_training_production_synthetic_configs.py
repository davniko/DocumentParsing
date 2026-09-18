from __future__ import annotations

import math
from pathlib import Path

from document_ocr.training.config import load_training_config

_ROOT = Path(__file__).resolve().parents[1]
_EXACT_CONFIG = (
    _ROOT
    / "configs/training/production/"
    "t5gemma2_270m_lora.mpci_bl_real1057_synth10000_exact_id_v5_e5.yaml"
)
_LAYOUT_CONFIG = (
    _ROOT
    / "configs/training/production/"
    "t5gemma2_270m_lora.mpci_bl_real1057_synth10000_layout_holdout_v5_e5.yaml"
)


def test_production_synthetic_training_arms_share_the_controlled_axes() -> None:
    exact = load_training_config(_EXACT_CONFIG)
    layout = load_training_config(_LAYOUT_CONFIG)

    assert exact.task == layout.task == "bill_of_lading_relation_explicit_v5"
    assert exact.objective == layout.objective == "seq2seq_teacher_forcing"
    assert exact.model == layout.model
    assert exact.prompt == layout.prompt
    assert exact.task_constraints == layout.task_constraints
    assert exact.optimization == layout.optimization
    assert exact.runtime == layout.runtime
    assert exact.dataloader == layout.dataloader
    assert exact.evaluation == layout.evaluation
    assert exact.checkpoint == layout.checkpoint

    assert exact.dataset.splits is not None
    assert layout.dataset.splits is not None
    for config in (exact, layout):
        assert config.dataset.input_mode == "pre_split"
        assert config.dataset.splits is not None
        assert [source.records for source in config.dataset.splits.train] == [1057, 10000]
        assert [source.records for source in config.dataset.splits.validation] == [100]
        assert config.dataset.splits.test == []
        assert config.dataset.preprocessing.max_source_length == 18432
        assert config.dataset.preprocessing.max_target_length == 6144
        assert config.optimization.num_train_epochs == 5.0
        assert config.optimization.max_steps == -1
        assert config.optimization.per_device_train_batch_size == 1
        assert config.optimization.gradient_accumulation_steps == 24
        assert config.optimization.train_sampling_strategy == "group_by_length"
        assert config.runtime.gradient_checkpointing is True
        assert config.runtime.gradient_checkpointing_use_reentrant is True
        assert config.evaluation.on_start is False
        assert config.evaluation.generation_max_length == 2048
        assert config.evaluation.steps == 230
        assert config.checkpoint.steps == 230

        train_records = sum(source.records for source in config.dataset.splits.train)
        updates_per_epoch = math.ceil(
            train_records
            / (
                config.optimization.per_device_train_batch_size
                * config.optimization.gradient_accumulation_steps
            )
        )
        assert updates_per_epoch == 461
        assert updates_per_epoch * int(config.optimization.num_train_epochs) == 2305

    assert exact.dataset.splits.validation == layout.dataset.splits.validation
    assert exact.dataset.splits.train[0] == layout.dataset.splits.train[0]
    assert exact.dataset.splits.train[1] != layout.dataset.splits.train[1]
    assert exact.run.run_id != layout.run.run_id
    assert exact.peft.adapter_name != layout.peft.adapter_name
