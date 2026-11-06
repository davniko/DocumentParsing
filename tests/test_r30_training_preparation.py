"""The R30 preparation spec must retain sufficient output capacity on both GPUs."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from document_ocr.training.config import TrainingConfig, load_training_config

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/training/preparation/r30_paddle_centroids.yaml"


@pytest.mark.parametrize("hardware", ["local", "runpod"])
def test_r30_output_override_fits_the_unchanged_training_budget(hardware):
    spec = yaml.safe_load(SPEC.read_text())
    base = load_training_config(ROOT / spec["base_configs"][hardware])
    values = base.model_dump(mode="python")
    values["evaluation"].update(spec["training_overrides"]["evaluation"])
    prepared = TrainingConfig.model_validate(values, strict=True)
    assert prepared.evaluation.generation_max_length == 5500
    assert prepared.evaluation.generation_max_length >= 3129
    assert (
        prepared.evaluation.generation_max_length
        == prepared.dataset.preprocessing.max_target_length
    )
    assert prepared.optimization == base.optimization
    assert prepared.peft == base.peft
    assert prepared.dataset == base.dataset  # Only validating the override, not building R30.


def test_preparation_is_not_an_accidentally_runnable_glm_training_config():
    spec = yaml.safe_load(SPEC.read_text())
    assert spec["status"] == "preparation_only_awaiting_training_config_and_benchmark"
    assert spec["input_contract"]["text"] == "paddle_only"
    assert spec["input_contract"]["coordinate_mode"] == "centroid"
    with pytest.raises(ValidationError):
        load_training_config(SPEC)
