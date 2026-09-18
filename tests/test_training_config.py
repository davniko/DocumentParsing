from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from yaml.constructor import ConstructorError

from document_ocr.training.config import (
    TrainingConfig,
    load_dataset_partition_config,
    load_training_config,
)
from document_ocr.training.prompting import load_prompt
from document_ocr.training.runtime import (
    _configure_cuda_allocator,
    _early_stopping_callback,
    build_training_arguments,
)
from document_ocr.training.tasks import get_training_task

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "t5gemma2_270m_lora.pilot106.yaml"
COMBINED_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "training" / "t5gemma2_270m_lora.mpci_bl_combined487.yaml"
)
TASK_FACING_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "training"
    / "t5gemma2_270m_lora.mpci_bl_combined1157_task_facing.yaml"
)
TASK_FACING_R64_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "training"
    / "t5gemma2_270m_lora.mpci_bl_combined1157_task_facing_r64_a96_ga32_e30.yaml"
)
TABLE_INPUT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "training"
    / "t5gemma2_270m_lora.mpci_bl_relation_v3_table_input.yaml"
)
FOLLOWUP_SPLIT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "training" / "mpci_bl_followup381_split.seed42.yaml"
)
PILOT_VAL10_SPLIT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "training" / "mpci_bl_pilot106_split.seed42.val10.yaml"
)
FOLLOWUP_VAL50_SPLIT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "training" / "mpci_bl_followup381_split.seed42.val50.yaml"
)


def test_pilot_training_configuration_is_strict_and_content_pinned() -> None:
    config = load_training_config(CONFIG_PATH)

    assert config.model.name_or_path == "google/t5gemma-2-270m-270m"
    assert config.model.revision == "7c38f16641f455ef0685b18431faf1b17722d5a1"
    assert config.dataset.splits.train[0].records == 90
    assert config.dataset.splits.train[0].sha256 == (
        "a7b2d935d16cb29c1db78a009ee0b90ea1785aae3bf211a9980405e7768469e9"
    )
    assert config.dataset.splits.validation[0].records == 16
    assert config.dataset.splits.validation[0].sha256 == (
        "47029d7b9b663225552c2c031d6088493e8c90d7a5e440aac4ca723cdf757403"
    )
    assert config.evaluation.strategy == "steps"
    assert config.evaluation.steps == 16
    assert config.evaluation.split == "validation"
    assert config.evaluation.predict_with_generate is True
    assert config.evaluation.run_final_evaluation is True
    assert config.evaluation.per_device_batch_size == 8
    assert config.runtime.torch_compile is False
    assert config.runtime.cuda_allocator_conf == "expandable_segments:True"
    assert config.runtime.gradient_checkpointing_use_reentrant is False
    assert config.runtime.skip_memory_metrics is True
    assert config.optimization.optimizer == "adamw_torch_fused"
    assert config.optimization.per_device_train_batch_size == 3
    assert config.optimization.gradient_accumulation_steps == 8
    assert config.optimization.num_train_epochs == 60.0
    assert config.evaluation.early_stopping_patience == 3
    assert config.evaluation.early_stopping_threshold == 0.0
    assert config.evaluation.write_predictions_for == ["validation"]
    assert config.checkpoint.steps == 16
    assert config.checkpoint.load_best_model_at_end is True
    assert config.checkpoint.metric_for_best_model == "field_value_f1"
    assert config.checkpoint.greater_is_better is True
    assert config.logging.report_to == ["mlflow"]
    assert config.logging.mlflow.tracking_uri == "http://mlflow-server:5000"
    assert config.logging.mlflow.system_metrics is True


@pytest.mark.parametrize(
    "task",
    [
        "bill_of_lading_relation_explicit_v3",
        "bill_of_lading_relation_explicit_v4",
        "bill_of_lading_relation_explicit_v5",
    ],
)
def test_relation_explicit_training_requires_frozen_task_constraints(task: str) -> None:
    value = load_training_config(CONFIG_PATH).model_dump(mode="python")
    value["task"] = task

    with pytest.raises(ValidationError, match="requires a frozen task_constraints"):
        TrainingConfig.model_validate(value, strict=True)

    value["task_constraints"] = {"path": "constraints.json", "sha256": "a" * 64}
    constrained = TrainingConfig.model_validate(value, strict=True)
    assert constrained.task_constraints is not None

    value["task"] = "bill_of_lading_semantic_v2"
    with pytest.raises(ValidationError, match="must not configure it"):
        TrainingConfig.model_validate(value, strict=True)


def test_t5gemma2_configuration_rejects_unsupported_flash_attention_2() -> None:
    value = load_training_config(CONFIG_PATH).model_dump(mode="python")
    value["model"]["attention_implementation"] = "flash_attention_2"

    with pytest.raises(ValidationError, match="attention_implementation"):
        TrainingConfig.model_validate(value, strict=True)


def test_evaluation_generation_capacity_can_be_smaller_than_training_capacity() -> None:
    value = load_training_config(CONFIG_PATH).model_dump(mode="python")
    value["evaluation"]["generation_max_length"] = (
        value["dataset"]["preprocessing"]["max_target_length"] - 1
    )

    config = TrainingConfig.model_validate(value, strict=True)

    assert (
        config.evaluation.generation_max_length
        < config.dataset.preprocessing.max_target_length
    )


def test_evaluation_generation_capacity_cannot_exceed_training_capacity() -> None:
    value = load_training_config(CONFIG_PATH).model_dump(mode="python")
    value["evaluation"]["generation_max_length"] = (
        value["dataset"]["preprocessing"]["max_target_length"] + 1
    )

    with pytest.raises(
        ValidationError,
        match=r"generation_max_length must not exceed preprocessing\.max_target_length",
    ):
        TrainingConfig.model_validate(value, strict=True)


def test_combined_training_configuration_pins_both_semantic_v2_cohorts() -> None:
    config = load_training_config(COMBINED_CONFIG_PATH)

    assert [source.records for source in config.dataset.splits.train] == [96, 331]
    assert [source.records for source in config.dataset.splits.validation] == [10, 50]
    assert sum(source.records for source in config.dataset.splits.train) == 427
    assert sum(source.records for source in config.dataset.splits.validation) == 60
    assert [source.sha256 for source in config.dataset.splits.train] == [
        "2e4b01b06a8ab9dbe136259a45c7f6f5231405c75708939972b5bd6aa9f73302",
        "011d9aa2114739831139355ff3e80350070a4c29440a9a72120846c99dd5e08f",
    ]
    assert [source.sha256 for source in config.dataset.splits.validation] == [
        "c4cb5e80032690c7ef1e2b43886043ff595246fab513796968d8537bd9d29b89",
        "87c7f599c4492fe1621243f6bee81a6422e5013bde7647082a985e9682c7d1ed",
    ]
    assert config.dataset.preprocessing.max_source_length == 8192
    assert config.dataset.preprocessing.max_target_length == 3072
    assert config.optimization.per_device_train_batch_size == 2
    assert config.optimization.gradient_accumulation_steps == 12
    assert (
        config.optimization.per_device_train_batch_size
        * config.optimization.gradient_accumulation_steps
        == 24
    )
    assert config.optimization.num_train_epochs == 25.0
    assert config.evaluation.per_device_batch_size == 4
    assert config.evaluation.on_start is True
    assert config.evaluation.generation_max_length == 3072
    assert config.evaluation.steps == config.checkpoint.steps == 90
    assert config.evaluation.early_stopping_patience is None
    assert config.evaluation.early_stopping_threshold is None


def test_task_facing_training_configuration_uses_one_seeded_source() -> None:
    config = load_training_config(TASK_FACING_CONFIG_PATH)

    assert config.task == "bill_of_lading_relation_explicit_v3"
    assert config.task_constraints is not None
    assert config.task_constraints.sha256 == (
        "f55ca917c95f75ddad494067aaaa2ac40d6c58a79688056c27de9dc17241911e"
    )
    assert config.dataset.input_mode == "runtime_partition"
    assert config.dataset.source is not None
    assert config.dataset.source.records == 1157
    assert config.dataset.source.sha256 == (
        "2a3e2ea3231cfff7674e85b54a52f66d0fee98b59b207bc7b04fe0f9916dfc42"
    )
    assert config.dataset.partition is not None
    assert config.dataset.partition.seed == 424
    assert config.dataset.partition.validation_size.kind == "records"
    assert config.dataset.partition.validation_size.value == 100
    assert config.dataset.preprocessing.max_source_length == 13312
    assert config.dataset.preprocessing.max_target_length == 4096
    assert config.optimization.num_train_epochs == 25.0
    assert config.optimization.per_device_train_batch_size == 1
    assert config.optimization.gradient_accumulation_steps == 24
    assert config.evaluation.on_start is True
    assert config.evaluation.generation_max_length == 4096
    assert config.evaluation.steps == config.checkpoint.steps == 225
    assert config.evaluation.early_stopping_patience is None
    assert config.evaluation.early_stopping_threshold is None


def test_task_facing_r64_configuration_changes_only_requested_experiment_axes() -> None:
    baseline = load_training_config(TASK_FACING_CONFIG_PATH)
    config = load_training_config(TASK_FACING_R64_CONFIG_PATH)

    assert config.run.run_id != baseline.run.run_id
    assert config.run.output_dir == baseline.run.output_dir
    assert config.peft.adapter_name != baseline.peft.adapter_name
    assert config.dataset == baseline.dataset
    assert config.prompt == baseline.prompt
    assert config.task_constraints == baseline.task_constraints
    assert config.model == baseline.model
    assert config.runtime == baseline.runtime
    assert config.dataloader == baseline.dataloader
    assert config.task == baseline.task
    assert config.objective == baseline.objective
    assert config.dataset.fields.input_text == "joinedRawText"
    assert config.dataset.fields.input_sha256 == "joinedRawTextSha256"
    assert "table" not in config.prompt.path

    assert config.peft.rank == 64
    assert config.peft.alpha == 96
    baseline_peft = baseline.peft.model_dump(mode="python")
    configured_peft = config.peft.model_dump(mode="python")
    for key in ("adapter_name", "rank", "alpha"):
        baseline_peft.pop(key)
        configured_peft.pop(key)
    assert configured_peft == baseline_peft

    assert config.optimization.num_train_epochs == 30.0
    assert config.optimization.per_device_train_batch_size == 1
    assert config.optimization.gradient_accumulation_steps == 32
    assert (
        config.optimization.per_device_train_batch_size
        * config.optimization.gradient_accumulation_steps
        == 32
    )
    baseline_optimization = baseline.optimization.model_dump(mode="python")
    configured_optimization = config.optimization.model_dump(mode="python")
    for key in ("num_train_epochs", "gradient_accumulation_steps"):
        baseline_optimization.pop(key)
        configured_optimization.pop(key)
    assert configured_optimization == baseline_optimization

    assert config.evaluation.on_start is True
    assert config.evaluation.steps == config.checkpoint.steps == 170
    assert config.evaluation.early_stopping_patience is None
    assert config.evaluation.early_stopping_threshold is None
    baseline_evaluation = baseline.evaluation.model_dump(mode="python")
    configured_evaluation = config.evaluation.model_dump(mode="python")
    baseline_evaluation.pop("steps")
    configured_evaluation.pop("steps")
    assert configured_evaluation == baseline_evaluation
    baseline_checkpoint = baseline.checkpoint.model_dump(mode="python")
    configured_checkpoint = config.checkpoint.model_dump(mode="python")
    baseline_checkpoint.pop("steps")
    configured_checkpoint.pop("steps")
    assert configured_checkpoint == baseline_checkpoint


def test_table_input_training_configuration_is_pinned_and_uses_safe_batch_one() -> None:
    config = load_training_config(TABLE_INPUT_CONFIG_PATH)

    assert config.task == "bill_of_lading_relation_explicit_v3"
    assert config.dataset.input_mode == "pre_split"
    assert config.dataset.splits is not None
    assert config.dataset.splits.train[0].records == 423
    assert config.dataset.splits.validation[0].records == 60
    assert config.dataset.fields.input_text == "modelInputText"
    assert config.dataset.fields.input_sha256 == "modelInputTextSha256"
    assert config.dataset.preprocessing.max_source_length == 13312
    assert config.dataset.preprocessing.max_target_length == 4096
    assert config.optimization.per_device_train_batch_size == 1
    assert config.optimization.gradient_accumulation_steps == 24
    assert (
        config.optimization.per_device_train_batch_size
        * config.optimization.gradient_accumulation_steps
        == 24
    )
    assert config.evaluation.per_device_batch_size == 2
    assert config.evaluation.steps == config.checkpoint.steps == 90
    assert config.prompt.path.endswith("bill_of_lading_relation_explicit_v3_table_view.txt")


def test_followup_partition_configuration_is_content_pinned() -> None:
    config = load_dataset_partition_config(FOLLOWUP_SPLIT_CONFIG_PATH)

    assert config.source.records == 381
    assert config.source.sha256 == (
        "b5fdf1874f297bf86adad2e128e91ece04d1dcc4ceb1eddeb430f4f7f385ee46"
    )
    assert config.seed == 42
    assert config.validation_records == 57
    assert config.output_dir == ("artifacts/kie-training/datasets/mpci-bl-followup381-seed42-v1")


def test_combined_validation_partition_configurations_are_content_pinned() -> None:
    pilot = load_dataset_partition_config(PILOT_VAL10_SPLIT_CONFIG_PATH)
    followup = load_dataset_partition_config(FOLLOWUP_VAL50_SPLIT_CONFIG_PATH)

    assert pilot.seed == followup.seed == 42
    assert pilot.validation_records == 10
    assert followup.validation_records == 50
    assert pilot.output_dir == ("artifacts/kie-training/datasets/mpci-bl-pilot106-seed42-val10-v1")
    assert followup.output_dir == (
        "artifacts/kie-training/datasets/mpci-bl-followup381-seed42-val50-v1"
    )


def test_training_yaml_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")

    with pytest.raises(ConstructorError, match="duplicate key"):
        load_training_config(path)


def test_training_configuration_rejects_unknown_fields() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["runtime"]["silent_precision_fallback"] = True

    with pytest.raises(ValidationError, match="silent_precision_fallback"):
        TrainingConfig.model_validate(raw, strict=True)


def test_training_configuration_accepts_one_runtime_partition_source() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    source = raw["dataset"].pop("splits")["train"][0]
    raw["dataset"].update(
        {
            "source": source,
            "partition": {
                "algorithm": "seeded_sha256_rank_v1",
                "seed": 42,
                "validation_size": {"kind": "records", "value": 10},
                "coverage_policy": "retain_each_target_leaf_in_train",
            },
        }
    )

    config = TrainingConfig.model_validate(raw, strict=True)

    assert config.dataset.input_mode == "runtime_partition"
    assert config.dataset.splits is None
    assert config.dataset.source is not None
    assert config.dataset.partition is not None
    assert config.dataset.partition.seed == 42


def test_training_configuration_accepts_fractional_runtime_partition() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    source = raw["dataset"].pop("splits")["train"][0]
    raw["dataset"].update(
        {
            "source": source,
            "partition": {
                "algorithm": "seeded_sha256_rank_v1",
                "seed": 42,
                "validation_size": {
                    "kind": "fraction",
                    "value": 0.125,
                    "rounding": "half_up",
                },
                "coverage_policy": "retain_each_target_leaf_in_train",
            },
        }
    )

    config = TrainingConfig.model_validate(raw, strict=True)

    assert config.dataset.partition is not None
    assert config.dataset.partition.validation_size.kind == "fraction"


def test_training_configuration_requires_exactly_one_dataset_input_mode() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["dataset"].update(
        {
            "source": raw["dataset"]["splits"]["train"][0],
            "partition": {
                "algorithm": "seeded_sha256_rank_v1",
                "seed": 42,
                "validation_size": {"kind": "records", "value": 10},
                "coverage_policy": "retain_each_target_leaf_in_train",
            },
        }
    )

    with pytest.raises(ValidationError, match="exactly one input mode"):
        TrainingConfig.model_validate(raw, strict=True)

    raw["dataset"].pop("splits")
    raw["dataset"].pop("partition")
    with pytest.raises(ValidationError, match="requires both source and partition"):
        TrainingConfig.model_validate(raw, strict=True)


def test_training_configuration_rejects_prediction_without_generation() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["evaluation"]["predict_with_generate"] = False

    with pytest.raises(ValidationError, match="structured evaluation and prediction"):
        TrainingConfig.model_validate(raw, strict=True)


def test_training_configuration_rejects_dropped_prediction_rows() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["dataloader"]["drop_last"] = True

    with pytest.raises(ValidationError, match="prediction publication requires"):
        TrainingConfig.model_validate(raw, strict=True)


def test_generated_evaluation_rejects_busy_polling_memory_tracker() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["runtime"]["skip_memory_metrics"] = False

    with pytest.raises(ValidationError, match="busy-polls host memory"):
        TrainingConfig.model_validate(raw, strict=True)


def test_training_configuration_rejects_absent_selected_evaluation_split() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["dataset"]["splits"]["validation"] = []

    with pytest.raises(ValidationError, match="requires at least one validation file"):
        TrainingConfig.model_validate(raw, strict=True)


def test_training_configuration_requires_run_relative_log_paths() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["logging"]["structured_log"] = "/tmp/training-events.jsonl"

    with pytest.raises(ValidationError, match="relative to the run directory"):
        TrainingConfig.model_validate(raw, strict=True)


def test_training_configuration_requires_mlflow_and_checkpoint_resume_together() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["checkpoint"]["resume_from_checkpoint"] = "checkpoints/checkpoint-1"

    with pytest.raises(ValidationError, match="MLflow resume_run_id"):
        TrainingConfig.model_validate(raw, strict=True)


def test_training_configuration_requires_resume_for_source_code_drift() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["checkpoint"]["allow_resume_source_code_drift"] = True

    with pytest.raises(ValidationError, match="requires an explicit resume checkpoint"):
        TrainingConfig.model_validate(raw, strict=True)


def test_training_configuration_requires_best_model_loading_for_early_stopping() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["checkpoint"].update(
        {
            "load_best_model_at_end": False,
            "metric_for_best_model": None,
            "greater_is_better": None,
        }
    )

    with pytest.raises(ValidationError, match="early stopping requires"):
        TrainingConfig.model_validate(raw, strict=True)


def test_training_configuration_requires_complete_early_stopping_pair() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["evaluation"]["early_stopping_threshold"] = None

    with pytest.raises(ValidationError, match="must be set together"):
        TrainingConfig.model_validate(raw, strict=True)


def test_training_arguments_preserve_configured_warmup_ratio(tmp_path: Path) -> None:
    config = load_training_config(CONFIG_PATH)

    arguments = build_training_arguments(config, tmp_path / "run")

    assert arguments.warmup_steps == config.optimization.warmup_ratio
    assert arguments.report_to == []
    assert arguments.do_eval is True
    assert arguments.eval_strategy.value == "steps"
    assert arguments.eval_steps == 16
    assert arguments.save_steps == 16
    assert arguments.load_best_model_at_end is True
    assert arguments.metric_for_best_model == "field_value_f1"
    assert arguments.dataloader_in_order is True


def test_early_stopping_callback_matches_yaml_configuration() -> None:
    callback = _early_stopping_callback(load_training_config(CONFIG_PATH))

    assert callback is not None
    assert callback.early_stopping_patience == 3
    assert callback.early_stopping_threshold == 0.0


def test_cuda_allocator_rejects_conflicting_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_training_config(CONFIG_PATH)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync")

    with pytest.raises(RuntimeError, match=r"differs from runtime\.cuda_allocator_conf"):
        _configure_cuda_allocator(config)


def test_prompt_is_literal_single_placeholder_template(tmp_path: Path) -> None:
    config = load_training_config(CONFIG_PATH)
    task = get_training_task(config.task)
    prompt = load_prompt(PROJECT_ROOT, config.prompt, task)

    rendered = prompt.render("--- PAGE 1 ---\nOCR")
    assert "{{document_text}}" not in rendered
    assert "{{output_schema}}" not in rendered
    assert "--- PAGE 1 ---\nOCR" in rendered
    assert '"schemaVersion":{"const":"2.0.0"' in rendered
    assert "```plaintext\n--- PAGE 1 ---\nOCR\n```" in rendered

    bad_prompt = tmp_path / "bad.txt"
    bad_prompt.write_text(
        "{{output_schema}} and {{document_text}} and {{unknown}}", encoding="utf-8"
    )
    bad_config = config.prompt.model_copy(update={"path": str(bad_prompt)})
    with pytest.raises(ValueError, match="unsupported template expression"):
        load_prompt(PROJECT_ROOT, bad_config, task)


def test_relation_v5_prompt_encodes_independent_dg_and_equipment_contracts() -> None:
    baseline = load_training_config(CONFIG_PATH)
    prompt_config = baseline.prompt.model_copy(
        update={"path": "prompts/kie/bill_of_lading_relation_explicit_v5.txt"}
    )
    task = get_training_task("bill_of_lading_relation_explicit_v5")

    prompt = load_prompt(PROJECT_ROOT, prompt_config, task)
    rendered = prompt.render("--- PAGE 1 ---\nOCR")

    assert '"const":"5.0.0-experimental"' in rendered
    assert "`sizeCategory` and `typeCategory` together" in rendered
    assert "`packingGroupCategory` is independent of `flashPoint`" in rendered
    assert rendered.count("--- PAGE 1 ---\nOCR") == 1


def test_prompt_schema_is_sparse_and_derived_from_latest_target_model() -> None:
    task = get_training_task("bill_of_lading_semantic_v2")

    schema_text = task.prompt_schema_json()
    schema = json.loads(schema_text)
    raw_schema = task.target_model.model_json_schema(mode="serialization")

    def property_key_sets(value: object) -> list[frozenset[str]]:
        if isinstance(value, list):
            return [item for child in value for item in property_key_sets(child)]
        if not isinstance(value, dict):
            return []
        found = []
        properties = value.get("properties")
        if isinstance(properties, dict):
            found.append(frozenset(properties))
        return found + [item for child in value.values() for item in property_key_sets(child)]

    assert schema["required"] == ["schemaVersion", "documentPatch"]
    assert schema["properties"]["schemaVersion"] == {
        "const": "2.0.0",
        "type": "string",
    }
    assert schema["$defs"]["BillOfLadingDocumentPatch"]["properties"]["route"] == {
        "$ref": "#/$defs/BillOfLadingRoute"
    }
    assert schema["$defs"]["Mass"]["required"] == ["value", "unit"]
    assert "description" in schema["$defs"]["BillOfLadingGoodsItem"]["properties"]
    assert sorted(tuple(sorted(item)) for item in property_key_sets(schema)) == sorted(
        tuple(sorted(item)) for item in property_key_sets(raw_schema)
    )
    assert '"type":"null"' not in schema_text
    assert '"default"' not in schema_text
    assert '"description":"' not in schema_text
    assert '"title"' not in schema_text
