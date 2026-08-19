from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from yaml.constructor import ConstructorError

from document_ocr.training.config import TrainingConfig, load_training_config
from document_ocr.training.prompting import load_prompt
from document_ocr.training.runtime import (
    _configure_cuda_allocator,
    _early_stopping_callback,
    build_training_arguments,
)
from document_ocr.training.tasks import get_training_task

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "t5gemma2_270m_lora.pilot106.yaml"


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


def test_training_configuration_rejects_prediction_without_generation() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["evaluation"]["predict_with_generate"] = False

    with pytest.raises(ValidationError, match="structured evaluation and prediction"):
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
        return found + [
            item for child in value.values() for item in property_key_sets(child)
        ]

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
