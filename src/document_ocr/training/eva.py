"""Train-only EVA initialization for T5Gemma2's shared self/cross-attention.

The decoder invokes the same K/V projection twice, on different token streams.
Feed PEFT one concatenated observation per projection and forward pass, so its
input-equality optimization cannot confuse decoder Q with decoder+encoder K/V.
Only activation rows are subsampled; model inputs/labels are never truncated.
"""

from __future__ import annotations

import json
import math
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from document_ocr.atomic import atomic_publish_json, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.training.config import EvaInitializationConfig, LoraConfig

_TEXT_LAYER = re.compile(
    r"base_model\.model\.model\.(encoder\.text_model|decoder)\.layers\.\d+\."
    r"(self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
)


def input_streams(module_name: str) -> tuple[str, ...]:
    match = _TEXT_LAYER.fullmatch(module_name)
    if match is None:
        raise ValueError(f"EVA has no token-stream contract for {module_name}")
    if match[1] == "encoder.text_model":
        return ("encoder",)
    if match[2] in ("self_attn.k_proj", "self_attn.v_proj"):
        return ("decoder", "encoder")
    return ("decoder",)


def activation_indices(mask: Any, limit: int) -> Any:
    """Spread a bounded number of actual nonpadding rows across each document."""
    import torch

    if mask.ndim != 2 or limit < 1:
        raise ValueError("EVA requires a two-dimensional token mask and positive row limit")
    rows = []
    for batch_index, row in enumerate(mask):
        positions = row.nonzero().flatten()
        if positions.numel() == 0:
            raise ValueError("EVA calibration document has no usable tokens")
        if positions.numel() > limit:
            positions = positions[
                torch.linspace(0, positions.numel() - 1, limit, device=mask.device).long()
            ]
        rows.append(torch.stack((torch.full_like(positions, batch_index), positions), dim=1))
    return torch.cat(rows)


def initialize_eva(
    *,
    model: Any,
    train_dataset: Any,
    collator: Any,
    peft: LoraConfig,
    dataset_identity: str,
    device: Any,
) -> dict[str, Any]:
    """Run bounded native PEFT EVA before optimizer construction; fail on nonconvergence."""
    import torch
    from peft import get_eva_state_dict, initialize_lora_eva_weights
    from peft.tuners.lora.layer import Linear
    from torch.utils.data import DataLoader, Subset

    candidate_settings = peft.eva
    if candidate_settings is None or peft.init_lora_weights != "eva":
        raise ValueError("EVA was not explicitly configured")
    settings: EvaInitializationConfig = candidate_settings
    if model.config.model_type != "t5gemma2":
        raise ValueError("EVA token-stream integration currently supports T5Gemma2 only")
    if len(train_dataset) < settings.sample_count:
        raise ValueError("EVA sample_count exceeds the available training partition")
    generator = torch.Generator().manual_seed(settings.seed)
    selection = torch.randperm(len(train_dataset), generator=generator)[
        : settings.sample_count
    ].tolist()
    sample_ids = [train_dataset[i]["document_id"] for i in selection]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("EVA calibration sample identities are not unique")
    dataloader = DataLoader(
        Subset(train_dataset, selection),
        batch_size=settings.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    adapted = {
        name: module
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and peft.adapter_name in module.lora_A
    }
    if not adapted or any(not isinstance(m, Linear) for m in adapted.values()):
        raise ValueError("EVA requires a nonempty set of supported linear adapters")
    if any(peft.rank > m.in_features for m in adapted.values()):
        raise ValueError("EVA rank exceeds an adapted input dimension")
    streams = {name: input_streams(name) for name in adapted}
    model.to(device)
    if any(
        bool(m.lora_B[peft.adapter_name].weight.detach().count_nonzero()) for m in adapted.values()
    ):
        raise ValueError("EVA must initialize fresh zero-update adapters, not trained weights")

    class ActivationTap(torch.nn.Linear):
        """A lightweight Linear-typed observation point for PEFT's public SVD API."""

        def forward(self, input: Any) -> Any:
            return input

    class CalibrationModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = model
            self.taps = torch.nn.ModuleDict(
                {
                    f"layer_{i}": ActivationTap(m.in_features, 1, bias=False, device=device)
                    for i, m in enumerate(adapted.values())
                }
            )
            for parameter in self.taps.parameters():
                parameter.requires_grad_(False)
            self.forward_passes = 0
            self.token_rows = {name: {s: 0 for s in streams[name]} for name in adapted}
            self.batches: dict[str, list[Any]] = {}
            self.masks: dict[str, Any] = {}
            self.indices: dict[str, Any] = {}

        def capture(self, name: str, args: tuple[Any, ...]) -> None:
            calls = self.batches[name]
            if len(calls) >= len(streams[name]):
                raise ValueError(f"Unexpected extra EVA projection invocation: {name}")
            stream = streams[name][len(calls)]
            states = args[0]
            if states.ndim != 3 or tuple(states.shape[:2]) != tuple(self.masks[stream].shape):
                raise ValueError(f"EVA activation/mask shape mismatch for {name}/{stream}")
            indices = self.indices[stream]
            calls.append(states.detach()[indices[:, 0], indices[:, 1]])
            self.token_rows[name][stream] += len(indices)

        def forward(self, **batch: Any) -> None:
            if self.forward_passes >= settings.max_forward_passes:
                raise RuntimeError(
                    f"EVA did not converge within {settings.max_forward_passes} forward passes; "
                    "training was not started. Increase the explicit calibration "
                    "budget after review."
                )
            self.forward_passes += 1
            if (
                "decoder_input_ids" not in batch
                or "labels" not in batch
                or "attention_mask" not in batch
            ):
                raise ValueError("EVA requires teacher-forced decoder inputs and both token masks")
            self.masks = {
                "encoder": batch["attention_mask"].bool(),
                "decoder": batch["labels"] != -100,
            }
            self.indices = {
                name: activation_indices(mask, settings.tokens_per_stream)
                for name, mask in self.masks.items()
            }
            self.batches = {name: [] for name in adapted}
            # Bypass the vocabulary head: EVA needs layer inputs, not logits/loss.
            self.base.get_base_model().model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
                decoder_attention_mask=self.masks["decoder"],
                use_cache=False,
                return_dict=True,
            )
            for index, name in enumerate(adapted):
                values = self.batches.pop(name)
                if len(values) != len(streams[name]):
                    raise ValueError(f"Missing EVA projection invocation: {name}")
                self.taps[f"layer_{index}"](torch.cat(values))
            if self.forward_passes == 1 or self.forward_passes % 16 == 0:
                print(
                    f"EVA calibration: {self.forward_passes} forward passes "
                    f"(limit {settings.max_forward_passes})",
                    flush=True,
                )

    previous_mode = model.training
    handles = []
    start = time.perf_counter()
    cuda_devices = [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            calibration = CalibrationModel()
            model.eval()
            for name, module in adapted.items():

                def capture(module: Any, args: tuple[Any, ...], name: str = name) -> None:
                    calibration.capture(name, args)

                handles.append(module.register_forward_pre_hook(capture))
            proxy_config = deepcopy(model.peft_config[peft.adapter_name])
            proxy_config.target_modules = r"^taps\.layer_[0-9]+$"
            with model.disable_adapter():
                state = get_eva_state_dict(
                    calibration,
                    dataloader,
                    peft_config=proxy_config,
                    prepare_model_inputs_fn=lambda inputs, config: None,
                    prepare_layer_inputs_fn=None,
                    gather_distributed_inputs=False,
                    show_progress_bar=False,
                )
            expected = {f"taps.layer_{i}" for i in range(len(adapted))}
            if state.keys() != expected:
                raise ValueError("EVA returned incomplete or unexpected projection coverage")
            mapped = {}
            for i, (name, module) in enumerate(adapted.items()):
                value = state[f"taps.layer_{i}"]
                if (
                    value.shape != (peft.rank, module.in_features)
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError(f"Invalid fixed-rank EVA initialization for {name}")
                mapped[name] = value
            # Complete coverage was checked above; PEFT cannot silently randomly
            # initialize missing layers on this path.
            initialize_lora_eva_weights(
                model, eva_state_dict=mapped, adapter_name=peft.adapter_name
            )
            remaining = {
                n
                for n, m in model.named_modules()
                if hasattr(m, "lora_A") and peft.adapter_name in m.lora_A
            }
            if remaining != adapted.keys():
                raise ValueError("Fixed-rank EVA unexpectedly changed adapter module coverage")
            # Native EVA rewrites target_modules to a list even without rank
            # changes. Retain the proved original regex for checkpoint/eval parity.
            model.peft_config[peft.adapter_name].target_modules = peft.target_modules_regex
            facts = []
            expected_scaling = peft.alpha / (math.sqrt(peft.rank) if peft.use_rslora else peft.rank)
            for name, module in adapted.items():
                a = module.lora_A[peft.adapter_name].weight
                b = module.lora_B[peft.adapter_name].weight
                if not torch.isfinite(a).all() or bool(b.detach().count_nonzero()):
                    raise ValueError(f"EVA failed its finite/no-op initialization contract: {name}")
                if not math.isclose(module.scaling[peft.adapter_name], expected_scaling):
                    raise ValueError("EVA unexpectedly changed rsLoRA scaling")
                facts.append(
                    dict(
                        module=name,
                        rank=module.r[peft.adapter_name],
                        scaling=module.scaling[peft.adapter_name],
                        activation_rows=calibration.token_rows[name],
                    )
                )
            if cuda_devices:
                torch.cuda.synchronize(device)
            return dict(
                status="complete",
                method="peft_eva_fixed_rank_t5gemma2_dual_stream_v1",
                settings=settings.model_dump(mode="json"),
                dataset_identity=dataset_identity,
                sample_document_ids=sample_ids,
                sample_ids_sha256=sha256_bytes(canonical_json_bytes(sample_ids)),
                forward_passes=calibration.forward_passes,
                distinct_documents_forwarded=min(
                    settings.sample_count,
                    max(1, calibration.forward_passes - 1) * settings.batch_size,
                ),
                modules=facts,
                seconds=time.perf_counter() - start,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device)
                if cuda_devices
                else None,
                rng_preserved=True,
                training_partition_only=True,
                input_truncation=False,
            )
    finally:
        for handle in handles:
            handle.remove()
        model.train(previous_mode)


def initialize_eva_for_run(
    *,
    path: Path,
    resuming: bool,
    model: Any,
    train_dataset: Any,
    collator: Any,
    peft: LoraConfig,
    dataset_identity: str,
    device: Any,
) -> None:
    """Publish initial calibration once; resume must load trained checkpoint weights."""
    if peft.eva is None:
        raise ValueError("EVA was not explicitly configured")
    if resuming:
        report = json.loads(read_regular_file_bytes(path))
        if (
            report["status"] != "complete"
            or report["settings"] != peft.eva.model_dump(mode="json")
            or report["dataset_identity"] != dataset_identity
        ):
            raise RuntimeError("resumed EVA initialization differs from its training contract")
        return
    report = initialize_eva(
        model=model,
        train_dataset=train_dataset,
        collator=collator,
        peft=peft,
        dataset_identity=dataset_identity,
        device=device,
    )
    atomic_publish_json(path, report)
