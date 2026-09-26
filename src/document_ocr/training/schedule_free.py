"""Schedule-free optimizer configuration and checkpoint lifecycle for Trainer."""

from __future__ import annotations

from typing import Any

from document_ocr.training.config import ScheduleFreeAdamWConfig


class ScheduleFreeTrainerMixin:
    """Keep published weights and optimizer state at the same evaluation point."""

    schedule_free_adamw: ScheduleFreeAdamWConfig | None = None

    def get_optimizer_cls_and_kwargs(
        self, args: Any, model: Any = None
    ) -> tuple[Any, dict[str, Any]]:
        optimizer_cls, kwargs = super().get_optimizer_cls_and_kwargs(args, model)  # type: ignore[misc]
        settings = self.schedule_free_adamw
        if args.optim != "schedule_free_adamw" or settings is None:
            raise RuntimeError("schedule-free Trainer requires schedule_free_adamw settings")
        kwargs.update(
            warmup_steps=settings.warmup_steps,
            r=settings.r,
            weight_lr_power=settings.weight_lr_power,
            foreach=settings.foreach,
        )
        return optimizer_cls, kwargs

    def _schedule_free_eval(self) -> None:
        if self.schedule_free_adamw is None:
            return
        optimizer = self.optimizer  # type: ignore[attr-defined]
        if optimizer is None:
            raise RuntimeError("schedule-free model save requires an initialized optimizer")
        eval_mode = getattr(optimizer, "eval", None)
        if not callable(eval_mode):
            raise RuntimeError("schedule-free optimizer does not expose eval()")
        eval_mode()

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        # Trainer saves model weights before optimizer state within a checkpoint.
        # Leave both in eval mode; its next training_step restores train mode.
        self._schedule_free_eval()
        super().save_model(output_dir, _internal_call=_internal_call)  # type: ignore[misc]

    def _load_best_model(self) -> None:
        # Convert the last training iterate before loading the saved best weights.
        # Otherwise a later eval() would extrapolate the best adapter using stale z.
        self._schedule_free_eval()
        super()._load_best_model()  # type: ignore[misc]
