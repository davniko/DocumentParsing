"""Exercise actual Trainer/Accelerate epoch order without training a model.

Run in the pinned trainer image. Only model/device/reporting are replaced by a
tiny CPU fixture; sampling, microbatch, accumulation, workers and seeds come
from the actual RunPod config. Lengths are the measured R30 token lengths.
"""

import hashlib
import json
import tempfile
import time
from itertools import pairwise
from pathlib import Path

import torch
import transformers
from datasets import Dataset
from transformers import Seq2SeqTrainer, T5Config, T5ForConditionalGeneration, set_seed

from document_ocr.training.config import load_training_config
from document_ocr.training.prediction import SamplerAwarePredictionMixin
from document_ocr.training.runtime import build_training_arguments

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/analysis/r30-party-boundary-audit-20261108"


class ProbeTrainer(SamplerAwarePredictionMixin, Seq2SeqTrainer):
    pass


def collate(rows):
    return {"sample_id": torch.tensor([r["sample_id"] for r in rows])}


def main():
    started = time.perf_counter()
    config = load_training_config(
        ROOT
        / "configs/training/production"
        / (
            "t5gemma2_270m_lora.mpci_bl_real600_synthetic9400_r30_paddle_centroids_"
            "inputonly_v7_e10_compact_eva_a32_r64_runpod_schedulefree_v1.yaml"
        )
    )
    config.runtime.bf16 = False
    config.runtime.tf32 = False
    config.runtime.require_cuda = False
    config.logging.report_to = []
    lengths = []
    for split in ("synthetic-train", "real-train"):
        data = json.loads(
            (ROOT / f"docs/analysis/r30-training-readiness-20261106/r30-{split}.json").read_text()
        )
        lengths.extend(r["input_tokens"] for r in data["records"])
    assert len(lengths) == 10000
    dataset = Dataset.from_dict({"sample_id": list(range(len(lengths))), "input_length": lengths})
    with tempfile.TemporaryDirectory(prefix="r30-sampler-") as temporary:
        args = build_training_arguments(config, Path(temporary))
        args.use_cpu = True
        args.remove_unused_columns = False
        args.dataloader_pin_memory = False
        set_seed(config.optimization.seed)
        model = T5ForConditionalGeneration(
            T5Config(
                vocab_size=8, d_model=8, d_ff=16, num_layers=1, num_decoder_layers=1, num_heads=1
            )
        )
        trainer = ProbeTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            eval_dataset=dataset,
            data_collator=collate,
        )
        sampler = trainer._get_train_sampler()
        loader = trainer.get_train_dataloader()
        orders = []
        for epoch in range(3):
            loader.set_epoch(epoch)
            order = [int(i) for batch in loader for i in batch["sample_id"]]
            assert sorted(order) == list(range(10000))
            orders.append(order)
        assert len({tuple(x) for x in orders}) == 3
        evaluation1 = list(trainer._get_eval_sampler(dataset))
        evaluation2 = list(trainer._get_eval_sampler(dataset))
        assert evaluation1 == evaluation2
        result = {
            "transformers": transformers.__version__,
            "sampler": type(sampler).__name__,
            "loader": type(loader).__name__,
            "strategy": args.train_sampling_strategy,
            "microbatch": args.per_device_train_batch_size,
            "accumulation": args.gradient_accumulation_steps,
            "workers": args.dataloader_num_workers,
            "persistentWorkers": args.dataloader_persistent_workers,
            "seed": args.seed,
            "dataSeed": args.data_seed,
            "epochOrders": [
                {
                    "records": len(o),
                    "unique": len(set(o)),
                    "sha256": hashlib.sha256(json.dumps(o).encode()).hexdigest(),
                    "first10": o[:10],
                }
                for o in orders
            ],
            "samePositionAcrossConsecutiveEpochs": [
                sum(x == y for x, y in zip(a, b, strict=True)) for a, b in pairwise(orders)
            ],
            "evaluationOrderStable": True,
            "seconds": time.perf_counter() - started,
            "modelForwardCalls": 0,
            "optimizerSteps": 0,
        }
        (OUT / "epoch-shuffle.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
