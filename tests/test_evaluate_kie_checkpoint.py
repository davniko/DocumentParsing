from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from document_ocr.training.config import load_training_config


def _load_tool() -> ModuleType:
    path = Path(__file__).parents[1] / "tools" / "evaluate_kie_checkpoint.py"
    spec = importlib.util.spec_from_file_location("evaluate_kie_checkpoint", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load tool: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_TOOL = _load_tool()


def _checkpoint(tmp_path: Path, step: int = 900) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding="utf-8"
    )
    return run_dir, checkpoint


def test_resolve_checkpoint_accepts_matching_direct_child(tmp_path: Path) -> None:
    run_dir, checkpoint = _checkpoint(tmp_path)

    resolved, step = _TOOL._resolve_checkpoint(run_dir, checkpoint)

    assert resolved == checkpoint.resolve()
    assert step == 900


def test_resolve_checkpoint_rejects_step_drift(tmp_path: Path) -> None:
    run_dir, checkpoint = _checkpoint(tmp_path)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 899}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="differs"):
        _TOOL._resolve_checkpoint(run_dir, checkpoint)


def test_resolve_checkpoint_rejects_path_outside_run(tmp_path: Path) -> None:
    run_dir, _ = _checkpoint(tmp_path)
    outside = tmp_path / "outside" / "checkpoint-900"
    outside.mkdir(parents=True)
    (outside / "trainer_state.json").write_text(json.dumps({"global_step": 900}), encoding="utf-8")

    with pytest.raises(ValueError, match="direct child"):
        _TOOL._resolve_checkpoint(run_dir, outside)


def test_recorded_adapter_matches_training_contract() -> None:
    project_root = Path(__file__).parents[1]
    config = load_training_config(
        project_root
        / "configs"
        / "training"
        / "t5gemma2_270m_lora.mpci_bl_combined1157_task_facing.yaml"
    )
    adapter_dir = (
        project_root
        / "artifacts"
        / "kie-training"
        / "t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-v2"
        / "checkpoints"
        / "checkpoint-900"
        / config.peft.adapter_name
    )

    provenance = _TOOL._validate_adapter_contract(adapter_dir, config)

    assert provenance["adapter_weights_bytes"] == 60_825_184
    assert len(provenance["adapter_weights_sha256"]) == 64


@pytest.mark.parametrize("ordered", [["b", "a"], ["a", "b"]])
def test_recorded_evaluation_preserves_exact_order(ordered) -> None:
    assert _TOOL._evaluation_indices({"document_ids": ordered}, ["a", "b"]) == tuple(
        ["a", "b"].index(value) for value in ordered
    )


@pytest.mark.parametrize("ordered", [["a", "a"], ["a"], ["a", "c"], ["a", {}], ["a", True]])
def test_recorded_evaluation_rejects_incomplete_or_invalid_order(ordered) -> None:
    with pytest.raises(ValueError, match="complete source"):
        _TOOL._evaluation_indices({"document_ids": ordered}, ["a", "b"])


def test_replay_requires_exact_metrics() -> None:
    recorded = {"metrics": {"eval_loss": 0.125, "eval_field_value_f1": 0.9}}
    _TOOL._verify_recorded_metrics({**recorded["metrics"], "eval_runtime": 1.0}, recorded)
    with pytest.raises(RuntimeError, match="did not reproduce"):
        _TOOL._verify_recorded_metrics(
            {"eval_loss": 0.125, "eval_field_value_f1": 0.899999}, recorded
        )
    with pytest.raises(RuntimeError, match="did not reproduce"):
        _TOOL._verify_recorded_metrics({"eval_loss": 0.125}, recorded)
    with pytest.raises(ValueError, match="nonempty"):
        _TOOL._verify_recorded_metrics({"eval_loss": 0.125}, {"metrics": {}})
