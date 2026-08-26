from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from document_ocr.training.runtime import (
    _base_model_on_start_evaluation_callback,
    _cumulative_loss_callback,
    _evaluation_memory_cleanup_callback,
    _evaluation_metrics_at_step,
    _final_evaluation_policy_callback,
)


class _AdapterStateModel:
    def __init__(self) -> None:
        self.adapter_enabled = True
        self.disable_calls = 0
        self.restore_calls = 0

    @contextmanager
    def disable_adapter(self) -> Any:
        if not self.adapter_enabled:
            raise RuntimeError("adapter was already disabled")
        self.disable_calls += 1
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = True
            self.restore_calls += 1


def test_on_start_baseline_disables_adapter_and_restores_before_training() -> None:
    callback = _base_model_on_start_evaluation_callback(enabled=True)
    model = _AdapterStateModel()
    state = SimpleNamespace(global_step=0, log_history=[])

    callback.on_train_begin(None, state, None, model=model)

    assert model.adapter_enabled is False
    assert model.disable_calls == 1
    state.log_history.append({"eval_runtime": 2.0, "step": 0})
    logs = {"eval_runtime": 2.0}
    callback.on_log(None, state, None, logs=logs)
    assert logs["eval_is_base_model"] == 1.0
    assert state.log_history[-1]["eval_is_base_model"] == 1.0

    callback.on_evaluate(None, state, None, metrics=logs)
    callback.on_epoch_begin(None, state, None)

    assert model.adapter_enabled is True
    assert model.restore_calls == 1


def test_disabled_on_start_baseline_does_not_touch_adapter() -> None:
    callback = _base_model_on_start_evaluation_callback(enabled=False)
    model = _AdapterStateModel()
    state = SimpleNamespace(global_step=12, log_history=[])

    callback.on_train_begin(None, state, None, model=model)
    callback.on_epoch_begin(None, state, None)

    assert model.adapter_enabled is True
    assert model.disable_calls == 0
    assert model.restore_calls == 0


def test_cumulative_loss_is_weighted_by_optimizer_step_intervals() -> None:
    state = SimpleNamespace(
        global_step=10,
        log_history=[
            {"loss": 0.5, "step": 1},
            {"loss": 0.25, "step": 5},
            {"eval_loss": 0.4, "step": 5},
            {"loss": 0.1, "step": 10},
        ],
    )
    logs = {"loss": 0.1}

    _cumulative_loss_callback().on_log(None, state, None, logs=logs)

    assert logs["train_cumulative_loss"] == pytest.approx(0.2)
    assert state.log_history[-1]["train_cumulative_loss"] == pytest.approx(0.2)


def test_evaluation_cleanup_releases_cache_and_publishes_memory_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gibibyte = 1024**3
    reservations = iter((8 * gibibyte, 6 * gibibyte))
    cleanup_calls = 0

    def empty_cache() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda: (10 * gibibyte, 20 * gibibyte),
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 5 * gibibyte)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: next(reservations))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 7 * gibibyte)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 9 * gibibyte)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    state = SimpleNamespace(
        global_step=40,
        log_history=[{"eval_runtime": 100.0, "step": 40}],
    )
    logs = {"eval_runtime": 100.0}

    _evaluation_memory_cleanup_callback().on_log(None, state, None, logs=logs)

    assert cleanup_calls == 1
    assert logs["eval_cuda_allocated_gib"] == 5.0
    assert logs["eval_cuda_peak_allocated_gib"] == 7.0
    assert logs["eval_cuda_reserved_before_cleanup_gib"] == 8.0
    assert logs["eval_cuda_peak_reserved_gib"] == 9.0
    assert logs["eval_cuda_reserved_after_cleanup_gib"] == 6.0
    assert logs["eval_cuda_reserved_released_gib"] == 2.0
    assert state.log_history[-1]["eval_cuda_reserved_released_gib"] == 2.0


def test_evaluation_cleanup_ignores_training_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: pytest.fail("training log triggered evaluation cleanup"),
    )
    state = SimpleNamespace(global_step=5, log_history=[{"loss": 0.2, "step": 5}])
    logs = {"loss": 0.2}

    _evaluation_memory_cleanup_callback().on_log(None, state, None, logs=logs)

    assert logs == {"loss": 0.2}


def test_evaluation_cleanup_tracks_driver_memory_from_scheduled_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gibibyte = 1024**3
    memory = iter(
        (
            (12 * gibibyte, 20 * gibibyte),
            (8 * gibibyte, 20 * gibibyte),
            (10 * gibibyte, 20 * gibibyte),
        )
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: next(memory))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 4 * gibibyte)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 4 * gibibyte)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 6 * gibibyte)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 7 * gibibyte)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    callback = _evaluation_memory_cleanup_callback()
    control = SimpleNamespace(should_evaluate=True)
    callback.on_step_end(None, None, control)
    state = SimpleNamespace(
        global_step=40,
        log_history=[{"eval_runtime": 100.0, "step": 40}],
    )
    logs = {"eval_runtime": 100.0}

    callback.on_log(None, state, control, logs=logs)

    assert logs["eval_cuda_driver_used_before_gib"] == 8.0
    assert logs["eval_cuda_driver_used_before_cleanup_gib"] == 12.0
    assert logs["eval_cuda_driver_used_after_cleanup_gib"] == 10.0


def test_final_evaluation_policy_suppresses_only_implicit_unaligned_eval() -> None:
    callback = _final_evaluation_policy_callback(run_final_evaluation=False)
    args = SimpleNamespace(eval_strategy=SimpleNamespace(value="steps"))

    unaligned = SimpleNamespace(should_evaluate=True)
    callback.on_step_end(
        args,
        SimpleNamespace(global_step=9, max_steps=9, eval_steps=5),
        unaligned,
    )
    assert unaligned.should_evaluate is False

    aligned = SimpleNamespace(should_evaluate=True)
    callback.on_step_end(
        args,
        SimpleNamespace(global_step=10, max_steps=10, eval_steps=5),
        aligned,
    )
    assert aligned.should_evaluate is True


def test_final_evaluation_policy_preserves_requested_final_eval() -> None:
    callback = _final_evaluation_policy_callback(run_final_evaluation=True)
    control = SimpleNamespace(should_evaluate=True)

    callback.on_step_end(
        SimpleNamespace(eval_strategy=SimpleNamespace(value="steps")),
        SimpleNamespace(global_step=9, max_steps=9, eval_steps=5),
        control,
    )

    assert control.should_evaluate is True


def test_evaluation_metrics_at_step_selects_latest_matching_eval() -> None:
    state = SimpleNamespace(
        log_history=[
            {"eval_loss": 1.0, "eval_runtime": 2.0, "step": 5},
            {"loss": 0.5, "step": 9},
            {"eval_loss": 0.4, "eval_runtime": 3.0, "step": 9},
        ]
    )

    assert _evaluation_metrics_at_step(state, 9) == {
        "eval_loss": 0.4,
        "eval_runtime": 3.0,
    }
    assert _evaluation_metrics_at_step(state, 8) is None
