"""Keep the R28 hardware variants equivalent to the rank-64 R27 baseline."""

from copy import deepcopy
from math import ceil
from pathlib import Path

import pytest

from document_ocr.training.config import load_training_config

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "configs/training/production"
BASE = DIRECTORY / (
    "t5gemma2_270m_lora.mpci_bl_real600_synthetic6900_packages_r27_positions_inputonly_"
    "v7_e10_compact_eva_a32_r64_local_schedulefree_v1.yaml"
)
PREFIX = (
    "t5gemma2_270m_lora.mpci_bl_real600_synthetic9400_r28_positions_inputonly_"
    "v7_e10_compact_eva_a32_r64_"
)


def config(hardware):
    return load_training_config(DIRECTORY / f"{PREFIX}{hardware}_schedulefree_v1.yaml")


@pytest.mark.parametrize("hardware,microbatch,accumulation", [("local", 1, 32), ("runpod", 4, 8)])
def test_preserves_training_settings_except_explicit_dataset_scaling(
    hardware, microbatch, accumulation
):
    base = load_training_config(BASE).model_dump(mode="json")
    actual = config(hardware).model_dump(mode="json")
    expected = deepcopy(base)
    for section in ("run", "dataset", "task_constraints", "logging"):
        expected[section] = actual[section]
    expected["peft"]["adapter_name"] = actual["peft"]["adapter_name"]
    expected["optimization"]["per_device_train_batch_size"] = microbatch
    expected["optimization"]["gradient_accumulation_steps"] = accumulation
    expected["optimization"]["schedule_free_adamw"]["warmup_steps"] = 157
    expected["evaluation"]["steps"] = 782
    expected["checkpoint"]["steps"] = 782
    expected["checkpoint"]["resume_from_checkpoint"] = None
    assert actual == expected
    assert actual["logging"]["mlflow"]["resume_run_id"] is None
    assert actual["run"]["run_id"] != base["run"]["run_id"]


@pytest.mark.parametrize("hardware", ["local", "runpod"])
def test_uses_current_r28_membership_and_matching_constraints(hardware):
    value = config(hardware)
    assert [f.records for f in value.dataset.splits.train] == [9400, 600]
    assert [f.records for f in value.dataset.splits.validation] == [60]
    assert not value.dataset.splits.test
    assert value.dataset.fields.input_text == "positionedText"
    assert value.dataset.fields.target == "target"
    assert value.dataset.preprocessing.target_format == "compact"
    root = "data/curated/mpci-bl-r28-positions-v1/"
    for file in (*value.dataset.splits.train, *value.dataset.splits.validation):
        assert str(file.path).startswith(root)
    assert str(value.task_constraints.path) == root + "task-constraints.json"
    steps = (
        ceil(
            ceil(10000 / value.optimization.per_device_train_batch_size)
            / value.optimization.gradient_accumulation_steps
        )
        * 10
    )
    assert steps == 3130
    assert value.optimization.schedule_free_adamw.warmup_steps == ceil(steps * 0.05)
    assert value.evaluation.steps == value.checkpoint.steps == steps // 4


def test_runpod_changes_only_batch_hardware_identity_and_tracking_endpoint():
    local = config("local").model_dump(mode="json")
    runpod = config("runpod").model_dump(mode="json")
    local["run"]["run_id"] = runpod["run"]["run_id"]
    local["peft"]["adapter_name"] = runpod["peft"]["adapter_name"]
    local["optimization"]["per_device_train_batch_size"] = 4
    local["optimization"]["gradient_accumulation_steps"] = 8
    local["logging"]["mlflow"]["tracking_uri"] = "http://127.0.0.1:5003"
    for key in ("hardware", "microbatch_size", "gradient_accumulation_steps"):
        local["logging"]["mlflow"]["tags"][key] = runpod["logging"]["mlflow"]["tags"][key]
    assert local == runpod
    assert runpod["runtime"]["gradient_checkpointing"] is True
    assert runpod["evaluation"]["per_device_batch_size"] == 2
