from __future__ import annotations

import math
from pathlib import Path

from document_ocr.training.config import load_training_config
from document_ocr.training.prompting import load_prompt
from document_ocr.training.tasks import load_training_task

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "configs" / "training" / "production"
STEM = (
    "t5gemma2_270m_lora.mpci_bl_real600_synthetic6900_packages_r27_positions_"
    "inputonly_v7_e10_compact_eva_a32_"
)


def config_path(rank: int) -> Path:
    return PRODUCTION / f"{STEM}r{rank}_local_schedulefree_v1.yaml"


def test_r27_rank64_changes_only_rank_identity_and_clears_resume() -> None:
    expected = load_training_config(config_path(48)).model_dump(mode="python")
    actual = load_training_config(config_path(64)).model_dump(mode="python")

    expected["run"]["run_id"] = expected["run"]["run_id"].replace("a32-r48-", "a32-r64-")
    expected["peft"]["adapter_name"] = expected["peft"]["adapter_name"].replace(
        "a32_r48_", "a32_r64_"
    )
    expected["peft"]["rank"] = 64
    expected["logging"]["mlflow"]["tags"]["lora_rank"] = "64"
    expected["checkpoint"]["resume_from_checkpoint"] = None
    expected["logging"]["mlflow"]["resume_run_id"] = None

    assert actual == expected


def test_r27_rank64_retains_data_schedule_and_best_export_contract() -> None:
    config = load_training_config(config_path(64))
    records = sum(source.records for source in config.dataset.splits.train)
    assert records == 7500
    assert sum(source.records for source in config.dataset.splits.validation) == 60
    updates = math.ceil(records / config.optimization.gradient_accumulation_steps) * 10
    assert updates == 2350
    assert config.optimization.schedule_free_adamw.warmup_steps == math.ceil(updates * 0.05)
    assert config.evaluation.steps == config.checkpoint.steps == 587
    assert config.checkpoint.total_limit is None
    assert config.checkpoint.load_best_model_at_end is True
    assert config.checkpoint.metric_for_best_model == "field_value_f1"
    assert config.checkpoint.save_only_model is False
    assert config.peft.alpha == 32
    assert config.peft.use_rslora is True
    assert config.peft.eva is not None
    assert config.peft.eva.rho == 1.0
    assert config.peft.eva.sample_count == 512


def test_r27_rank64_loads_pinned_task_and_unchanged_input_only_prompt() -> None:
    baseline = load_training_config(config_path(48))
    config = load_training_config(config_path(64))
    prompt = load_prompt(ROOT, config.prompt, load_training_task(ROOT, config))
    baseline_prompt = load_prompt(ROOT, baseline.prompt, load_training_task(ROOT, baseline))

    assert prompt.sha256 == baseline_prompt.sha256
    assert prompt.output_schema_sha256 == baseline_prompt.output_schema_sha256
    assert config.dataset == baseline.dataset
    assert config.task_constraints == baseline.task_constraints
