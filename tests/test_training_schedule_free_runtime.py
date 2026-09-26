"""Small CPU integration check using the pinned training dependency group."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires train dependencies")
class ScheduleFreeRuntimeTest(unittest.TestCase):
    def test_trainer_checkpoint_best_model_and_resume(self) -> None:
        import torch
        from safetensors.torch import load_file
        from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator

        from document_ocr.training.config import ScheduleFreeAdamWConfig
        from document_ocr.training.schedule_free import ScheduleFreeTrainerMixin

        class TinyDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return 4

            def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
                return {
                    "input_ids": torch.tensor([float(index + 1)]),
                    "labels": torch.tensor([0.0]),
                }

        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(
                self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
            ) -> dict[str, torch.Tensor]:
                logits = input_ids * self.weight
                assert labels is not None
                return {"loss": ((logits - labels) ** 2).mean(), "logits": logits}

        class TinyTrainer(ScheduleFreeTrainerMixin, Seq2SeqTrainer):
            schedule_free_adamw = ScheduleFreeAdamWConfig(
                warmup_steps=1, r=0.0, weight_lr_power=2.0, foreach=True
            )

        def arguments(directory: Path, max_steps: int) -> Seq2SeqTrainingArguments:
            return Seq2SeqTrainingArguments(
                output_dir=str(directory),
                max_steps=max_steps,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                learning_rate=0.1,
                optim="schedule_free_adamw",
                lr_scheduler_type="constant",
                warmup_steps=1,
                eval_strategy="steps",
                eval_steps=1,
                save_strategy="steps",
                save_steps=1,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                logging_strategy="no",
                report_to=[],
                use_cpu=True,
            )

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            trainer = TinyTrainer(
                model=TinyModel(),
                args=arguments(directory, 2),
                train_dataset=TinyDataset(),
                eval_dataset=TinyDataset(),
                data_collator=default_data_collator,
            )
            result = trainer.train()
            self.assertEqual(result.global_step, 2)
            self.assertFalse(trainer.optimizer.param_groups[0]["train_mode"])

            best_checkpoint = Path(trainer.state.best_model_checkpoint)
            best_weight = load_file(best_checkpoint / "model.safetensors")["weight"]
            torch.testing.assert_close(trainer.model.weight.detach().cpu(), best_weight)

            checkpoint = directory / "checkpoint-1"
            optimizer_state = torch.load(checkpoint / "optimizer.pt", weights_only=False)
            self.assertFalse(optimizer_state["param_groups"][0]["train_mode"])

            final_directory = directory / "final"
            trainer.save_model(str(final_directory))
            final_weight = load_file(final_directory / "model.safetensors")["weight"]
            torch.testing.assert_close(final_weight, best_weight)

            earlier_weight = load_file(checkpoint / "model.safetensors")["weight"]
            trainer.optimizer.train()
            trainer.state.best_model_checkpoint = str(checkpoint)
            trainer._load_best_model()
            reloaded_directory = directory / "earlier-best"
            trainer.save_model(str(reloaded_directory))
            reloaded_weight = load_file(reloaded_directory / "model.safetensors")["weight"]
            torch.testing.assert_close(reloaded_weight, earlier_weight)

            resumed = TinyTrainer(
                model=TinyModel(),
                args=arguments(directory, 3),
                train_dataset=TinyDataset(),
                eval_dataset=TinyDataset(),
                data_collator=default_data_collator,
            )
            resumed_result = resumed.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(resumed_result.global_step, 3)
            self.assertFalse(resumed.optimizer.param_groups[0]["train_mode"])

            save_only_directory = directory / "save-without-eval"
            save_only_args = Seq2SeqTrainingArguments(
                output_dir=str(save_only_directory),
                max_steps=2,
                per_device_train_batch_size=1,
                learning_rate=0.1,
                optim="schedule_free_adamw",
                lr_scheduler_type="constant",
                warmup_steps=1,
                eval_strategy="no",
                save_strategy="steps",
                save_steps=1,
                logging_strategy="no",
                report_to=[],
                use_cpu=True,
            )
            save_only = TinyTrainer(
                model=TinyModel(),
                args=save_only_args,
                train_dataset=TinyDataset(),
                data_collator=default_data_collator,
            )
            save_only.train()
            unpaired_state = torch.load(
                save_only_directory / "checkpoint-1" / "optimizer.pt", weights_only=False
            )
            self.assertFalse(unpaired_state["param_groups"][0]["train_mode"])


if __name__ == "__main__":
    unittest.main()
