"""CPU-only spectra of actual LoRA updates B@A, without loading the base model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import time
from pathlib import Path

import torch
from safetensors import safe_open

from document_ocr.hashing import sha256_file


def update_singular_values(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact nonzero spectrum via a rank-sized core, not separate factor spectra."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
        raise ValueError("incompatible LoRA A/B shapes")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("adapter contains non-finite values")
    _, ra = torch.linalg.qr(a.double().T, mode="reduced")
    _, rb = torch.linalg.qr(b.double(), mode="reduced")
    return torch.linalg.svdvals(rb @ ra.T)


def spectrum_metrics(values: torch.Tensor) -> dict[str, float | int]:
    energy = values.square()
    total = energy.sum()
    if total <= 0:
        raise ValueError("zero update has no interpretable normalized spectrum")
    fractions = energy / total
    cumulative = fractions.cumsum(0)
    metrics: dict[str, float | int] = {
        "stable_rank": float(total / energy[0]),
        "energy_entropy_rank": float(
            torch.exp(-(fractions * fractions.clamp_min(1e-300).log()).sum())
        ),
        "top1_energy": float(fractions[0]),
        "top4_energy": float(fractions[:4].sum()),
        "bottom_quarter_energy": float(fractions[-max(1, len(values) // 4) :].sum()),
        "sigma_last_over_first": float(values[-1] / values[0]),
        "frobenius_norm": float(total.sqrt()),
    }
    for percent in (90, 95, 99):
        metrics[f"rank_{percent}"] = min(
            len(values), int(torch.searchsorted(cumulative, percent / 100)) + 1
        )
    return metrics


def analyze(adapter: Path) -> dict:
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_bytes())
    if config["use_dora"] or config["rank_pattern"] or config["alpha_pattern"]:
        raise ValueError(
            "this diagnostic requires a uniform-rank, uniform-alpha ordinary/rsLoRA adapter"
        )
    weights = adapter / "adapter_model.safetensors"
    scale = config["lora_alpha"] / (math.sqrt(config["r"]) if config["use_rslora"] else config["r"])
    rows = []
    with safe_open(weights, framework="pt", device="cpu") as tensors:
        keys = tensors.keys()
        a_keys = [k for k in keys if k.endswith(".lora_A.weight")]
        b_keys = {k for k in keys if k.endswith(".lora_B.weight")}
        if not a_keys or {k.replace(".lora_A.", ".lora_B.") for k in a_keys} != b_keys:
            raise ValueError("missing or unpaired adapter factors")
        for key in a_keys:
            module = key.removesuffix(".lora_A.weight")
            a = tensors.get_tensor(key)
            b = tensors.get_tensor(key.replace(".lora_A.", ".lora_B."))
            if a.shape[0] != config["r"]:
                raise ValueError("adapter tensor rank disagrees with its config")
            values = update_singular_values(a, b) * scale
            rows.append(
                dict(
                    module=module,
                    branch="encoder" if ".encoder." in module else "decoder",
                    projection=module.rsplit(".", 1)[-1],
                    rank=config["r"],
                    scale=scale,
                    **spectrum_metrics(values),
                    singular_values=values.tolist(),
                )
            )
    return dict(
        adapter=str(adapter),
        weights_sha256=sha256_file(weights),
        config_sha256=sha256_file(config_path),
        rank=config["r"],
        alpha=config["lora_alpha"],
        use_rslora=config["use_rslora"],
        modules=rows,
        interpretation=(
            "Describes the trained update within its imposed rank. Cannot prove that "
            "unseen higher-rank directions would not improve task quality."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    start = time.perf_counter()
    reports = [analyze(p) for p in args.adapter]
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        adapters=reports,
        seconds=time.perf_counter() - start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    (args.output / "spectra.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "modules.csv").open("w", newline="") as stream:
        first = {k: v for k, v in reports[0]["modules"][0].items() if k != "singular_values"}
        writer = csv.DictWriter(stream, fieldnames=["adapter", *first])
        writer.writeheader()
        for r in reports:
            for row in r["modules"]:
                writer.writerow(
                    dict(
                        adapter=r["adapter"],
                        **{k: v for k, v in row.items() if k != "singular_values"},
                    )
                )
    print(json.dumps({k: v for k, v in report.items() if k != "adapters"}))


if __name__ == "__main__":
    main()
