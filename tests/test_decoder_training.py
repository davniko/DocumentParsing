from __future__ import annotations

import json
from pathlib import Path

import pytest
from yaml.constructor import ConstructorError

from document_ocr.decoder_training.completions import split_completion
from document_ocr.decoder_training.config import (
    load_decoder_training_config,
    parse_decoder_training_config,
)
from document_ocr.decoder_training.data import inspect_decoder_dataset, load_task_and_prompt
from document_ocr.decoder_training.evaluation import generated_evaluation
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
REASONING_GRPO_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/decoder_training/qwen35_08b_lora_dr_grpo_reasoning_probe.mpci_bl_combined1157.yaml"
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
    assert config.grpo.rollout.top_k == 20
    assert config.grpo.rollout.repetition_penalty == 1.0


def test_reasoning_dr_grpo_uses_native_thinking_with_a_larger_rollout_budget() -> None:
    config = load_decoder_training_config(REASONING_GRPO_CONFIG_PATH)

    assert config.method == "grpo"
    assert config.sequence.thinking == "enabled"
    assert config.sequence.max_completion_length == 6144
    assert config.grpo is not None
    assert config.grpo.rollout.max_completion_length == 6144
    assert config.grpo.rollout.top_k == 20


def test_sft_rejects_native_thinking_without_supervised_reasoning_traces() -> None:
    baseline = load_decoder_training_config(CONFIG_PATH)
    value = baseline.model_dump(mode="json")
    value["sequence"]["thinking"] = "enabled"

    with pytest.raises(ValueError, match="supported only for GRPO"):
        type(baseline).model_validate(value, strict=True)


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


def test_grpo_can_initialize_a_fresh_adapter_from_the_configured_base() -> None:
    baseline = load_decoder_training_config(GRPO_CONFIG_PATH)
    value = baseline.model_dump(mode="json")
    value["grpo"]["initialize_from"] = None
    config = type(baseline).model_validate(value, strict=True)

    assert _grpo_initialization_path(PROJECT_ROOT, config) is None

    sentinel_model = object()
    calls: list[tuple[object, dict[str, object]]] = []

    class RecordNewAdapter:
        @staticmethod
        def get_peft_model(model: object, **kwargs: object) -> object:
            calls.append((model, kwargs))
            return "fresh-adapter-model"

    result = _prepare_lora_model(
        fast_language_model=RecordNewAdapter,
        model=sentinel_model,
        config=config,
        initialization_path=None,
    )

    assert result == "fresh-adapter-model"
    assert len(calls) == 1
    assert calls[0][0] is sentinel_model
    assert calls[0][1]["r"] == config.adapter.rank


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
        thinking="disabled",
    )

    assert rewards[0] == 1.0
    assert rewards[1] == 0.0
    assert 0.0 < rewards[2] < 1.0


def test_native_thinking_reward_scores_only_the_final_json() -> None:
    config = load_decoder_training_config(CONFIG_PATH)
    task, prompt = load_task_and_prompt(PROJECT_ROOT, config)
    records, _ = inspect_decoder_dataset(
        project_root=PROJECT_ROOT,
        config=config,
        task=task,
        prompt=prompt,
    )
    reference = records["validation"][0].reference_target

    rewards = schema_gated_field_f1(
        [
            f"I checked OCR {{not JSON}}.\n</think>\n\n{reference}",
            reference,
            f"reasoning</think>\n\n{reference}</think>",
        ],
        [reference, reference, reference],
        task=task,
        thinking="enabled",
    )

    assert rewards == [1.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("text", "valid", "error"),
    [
        ("trace\n</think>\n\n{}", True, None),
        ("{}", False, "missing_close_tag"),
        ("</think>", False, "empty_final_answer"),
        ("<think>x</think>{}", False, "unexpected_open_tag"),
        ("x</think>{}</think>", False, "multiple_close_tags"),
    ],
)
def test_native_thinking_boundary_is_fail_closed(
    text: str, valid: bool, error: str | None
) -> None:
    parts = split_completion(text, thinking="enabled")

    assert parts.boundary_valid is valid
    assert parts.boundary_error == error


def test_generated_evaluation_publishes_trace_but_scores_only_final_answer(
    tmp_path: Path,
) -> None:
    import torch

    config = load_decoder_training_config(CONFIG_PATH)
    task, prompt = load_task_and_prompt(PROJECT_ROOT, config)
    records, _ = inspect_decoder_dataset(
        project_root=PROJECT_ROOT,
        config=config,
        task=task,
        prompt=prompt,
    )
    record = records["validation"][0]
    completion = f"Checked OCR evidence.\n</think>\n\n{record.reference_target}"

    class OneRowDataset:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, key: slice) -> dict[str, list[str]]:
            assert isinstance(key, slice)
            return {
                "prompt": ["rendered prompt"],
                "reference_target": [record.reference_target],
                "document_id": [record.document_id],
            }

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, value: object, **_: object) -> dict[str, object]:
            if isinstance(value, list):
                return {
                    "input_ids": torch.tensor([[10, 11]], dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
                }
            assert isinstance(value, str)
            return {"input_ids": value.split()}

        def batch_decode(self, value: object, **_: object) -> list[str]:
            assert isinstance(value, torch.Tensor)
            return [completion]

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def generate(self, *, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
            suffix = torch.tensor([[7, 8, 1]], dtype=torch.long, device=input_ids.device)
            return torch.cat((input_ids, suffix), dim=1)

    output_path = tmp_path / "reasoning-predictions.jsonl"
    model = FakeModel()
    metrics = generated_evaluation(
        model=model,
        tokenizer=FakeTokenizer(),
        dataset=OneRowDataset(),
        task=task,
        batch_size=1,
        max_new_tokens=16,
        num_beams=1,
        thinking="enabled",
        output_path=output_path,
    )

    published = json.loads(output_path.read_text(encoding="utf-8"))
    assert metrics["field_value_f1"] == 1.0
    assert metrics["reasoning_boundary_valid_fraction"] == 1.0
    assert published["reasoningTrace"] == "Checked OCR evidence."
    assert published["finalAnswer"] == record.reference_target
    assert published["reasoningBoundaryValid"] is True
    assert model.training is True
