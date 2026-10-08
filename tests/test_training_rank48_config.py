from __future__ import annotations

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


def test_rank48_comparison_changes_only_rank_identity_and_retention() -> None:
    baseline = load_training_config(_config_path(32)).model_dump(mode="python")
    actual = load_training_config(_config_path(48)).model_dump(mode="python")

    baseline["run"]["run_id"] = baseline["run"]["run_id"].replace("a32-r32-", "a32-r48-")
    baseline["peft"]["adapter_name"] = baseline["peft"]["adapter_name"].replace(
        "a32_r32_", "a32_r48_"
    )
    baseline["peft"]["rank"] = 48
    baseline["checkpoint"]["total_limit"] = None
    baseline["logging"]["mlflow"]["tags"]["lora_rank"] = "48"

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

