"""Offline R30 dataset and disposable GPU checks; never start or save a training run."""

import argparse
import copy
import json
import os
import time
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.training.config import load_training_config
from document_ocr.training.data import prepare_datasets
from document_ocr.training.prompting import load_prompt
from document_ocr.training.runtime import (
    _configure_cuda_allocator,
    load_lora_model,
    load_tokenizer,
    validate_training_hardware,
)
from document_ocr.training.tasks import load_training_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.environ["HF_HOME"] = str(root / ".cache/huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config = load_training_config(root / args.config)
    _configure_cuda_allocator(config)
    import torch
    from transformers import DataCollatorForSeq2Seq, set_seed

    hardware = validate_training_hardware(config)
    set_seed(config.optimization.seed)
    task = load_training_task(root, config)
    prompt = load_prompt(root, config.prompt, task)
    tokenizer = load_tokenizer(config)
    started = time.perf_counter()
    prepared = prepare_datasets(
        project_root=root, config=config, prompt=prompt, task=task, tokenizer=tokenizer
    )
    preparation_seconds = time.perf_counter() - started
    print(json.dumps({"stage": "dataset_prepared", "report": prepared.report()}), flush=True)
    model, model_facts = load_lora_model(config)
    model.to("cuda")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": config.runtime.gradient_checkpointing_use_reentrant
        }
    )
    model.enable_input_require_grads()
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=config.dataset.preprocessing.pad_to_multiple_of,
    )

    def batch_for(dataset, indices):
        rows = [
            {key: dataset[i][key] for key in ("input_ids", "attention_mask", "labels")}
            for i in indices
        ]
        return {key: value.to("cuda") for key, value in collator(rows).items()}

    controls = []
    dataset = prepared.datasets["train"]
    for dimension in ("input_length", "target_length"):
        lengths = list(dataset[dimension])
        indices = sorted(range(len(dataset)), key=lengths.__getitem__, reverse=True)[
            : config.optimization.per_device_train_batch_size
        ]
        model.train()
        model.zero_grad(set_to_none=True)
        batch = batch_for(dataset, indices)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**batch)
        if not torch.isfinite(output.loss).item():
            raise RuntimeError("Non-finite preflight loss")
        output.loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
            raise RuntimeError("Missing or non-finite preflight gradients")
        torch.cuda.synchronize()
        control = {
            "tail": dimension,
            "document_ids": [dataset[i]["document_id"] for i in indices],
            "input_shape": list(batch["input_ids"].shape),
            "label_shape": list(batch["labels"].shape),
            "loss": output.loss.item(),
            "seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "gradient_tensors": len(gradients),
        }
        controls.append(control)
        print(json.dumps({"stage": "forward_backward_passed", **control}), flush=True)
        del output, batch, gradients
    model.zero_grad(set_to_none=True)
    model.eval()
    model.gradient_checkpointing_disable()
    validation = prepared.datasets["validation"]
    lengths = list(validation["input_length"])
    indices = sorted(range(len(validation)), key=lengths.__getitem__, reverse=True)[
        : config.evaluation.per_device_batch_size
    ]
    batch = batch_for(validation, indices)
    generation = copy.deepcopy(model.generation_config)
    generation.max_new_tokens = 8
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            generation_config=generation,
        )
    if output.shape[0] != config.evaluation.per_device_batch_size or output.shape[1] < 2:
        raise RuntimeError("Invalid generation smoke result")
    report = {
        "status": "PASSED",
        "hardware": hardware,
        "dataset": prepared.report(),
        "dataset_preparation_seconds": preparation_seconds,
        "model": model_facts,
        "training_tail_controls": controls,
        "generation_shape": list(output.shape),
        "optimizer_updates": 0,
        "eva_calibration_performed": False,
        "checkpoints_created": False,
        "scope": "Dataset validation, tail microbatch forward/backward and 8-token generation only",
    }
    destination = root / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_publish_json(destination, report)
    print(json.dumps({"status": "PASSED", "report": str(destination)}), flush=True)


if __name__ == "__main__":
    main()
