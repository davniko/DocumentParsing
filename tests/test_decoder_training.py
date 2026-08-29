from __future__ import annotations

import json
from pathlib import Path

import pytest
from yaml.constructor import ConstructorError

from document_ocr.decoder_training.config import (
    load_decoder_training_config,
    parse_decoder_training_config,
)
from document_ocr.decoder_training.data import inspect_decoder_dataset, load_task_and_prompt
from document_ocr.decoder_training.rewards import schema_gated_field_f1
from document_ocr.decoder_training.runtime import (
    _grpo_initialization_path,
    _prepare_lora_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs/decoder_training/qwen35_08b_lora_sft.mpci_bl_combined1157.yaml"
)
GRPO_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/decoder_training/qwen35_08b_lora_dr_grpo_probe.mpci_bl_combined1157.yaml"
)


def test_decoder_config_and_full_dataset_inspection_are_pinned() -> None:
    config = load_decoder_training_config(CONFIG_PATH)
    task, prompt = load_task_and_prompt(PROJECT_ROOT, config)
    records, inspection = inspect_decoder_dataset(
        project_root=PROJECT_ROOT,
        config=config,
        task=task,
        prompt=prompt,
    )

    assert config.method == "sft"
    assert inspection.source_sha256 == config.dataset.source.sha256
    assert inspection.split_records == {"train": 1057, "validation": 100}
    assert len(records["train"]) + len(records["validation"]) == 1157
    assert set(records["train"]).isdisjoint(records["validation"])


def test_decoder_yaml_rejects_duplicate_keys() -> None:
    with pytest.raises(ConstructorError, match="duplicate key"):
        parse_decoder_training_config("schema_version: 1\nschema_version: 1\n")


def test_dr_grpo_probe_contract_is_valid_and_schema_gated() -> None:
    config = load_decoder_training_config(GRPO_CONFIG_PATH)

    assert config.method == "grpo"
    assert config.grpo is not None
    assert config.grpo.reward.policy == "schema_gated_field_f1_v1"
    assert config.grpo.policy_optimization.loss_type == "dr_grpo"
    assert config.grpo.policy_optimization.scale_rewards is False
    assert config.optimization.per_device_eval_batch_size == 2
    assert config.grpo.rollout.num_generations == 2


def test_grpo_rejects_eval_batches_that_split_reward_groups() -> None:
    value = load_decoder_training_config(GRPO_CONFIG_PATH).model_dump(mode="json")
    value["optimization"]["per_device_eval_batch_size"] = 1

    with pytest.raises(ValueError, match="eval batch must be divisible"):
        type(load_decoder_training_config(GRPO_CONFIG_PATH)).model_validate(
            value, strict=True
        )


def test_grpo_probe_refuses_to_start_without_sft_adapter(tmp_path: Path) -> None:
    baseline = load_decoder_training_config(GRPO_CONFIG_PATH)
    value = baseline.model_dump(mode="json")
    value["grpo"]["initialize_from"] = str(tmp_path / "missing-adapter")
    config = type(baseline).model_validate(value, strict=True)

    with pytest.raises(ValueError, match="GRPO initialization path does not exist"):
        _grpo_initialization_path(PROJECT_ROOT, config)


def test_grpo_preserves_loaded_sft_adapter_instead_of_stacking_lora(tmp_path: Path) -> None:
    config = load_decoder_training_config(GRPO_CONFIG_PATH)
    already_adapted_model = object()

    class RefuseNewAdapter:
        @staticmethod
        def get_peft_model(*args: object, **kwargs: object) -> object:
            raise AssertionError("GRPO must not initialize a second LoRA adapter")

    result = _prepare_lora_model(
        fast_language_model=RefuseNewAdapter,
        model=already_adapted_model,
        config=config,
        initialization_path=tmp_path,
    )

    assert result is already_adapted_model


def test_schema_gated_reward_uses_exact_shared_metric_contract() -> None:
    config = load_decoder_training_config(CONFIG_PATH)
    task, prompt = load_task_and_prompt(PROJECT_ROOT, config)
    records, _ = inspect_decoder_dataset(
        project_root=PROJECT_ROOT,
        config=config,
        task=task,
        prompt=prompt,
    )
    reference = records["validation"][0].reference_target
    wrong = json.loads(reference)
    wrong["documentPatch"]["billOfLadingNumber"] = "UNSUPPORTED-WRONG-VALUE"
    wrong_text = json.dumps(wrong, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    rewards = schema_gated_field_f1(
        [reference, "not json", wrong_text],
        [reference, reference, reference],
        task=task,
    )

    assert rewards[0] == 1.0
    assert rewards[1] == 0.0
    assert 0.0 < rewards[2] < 1.0
