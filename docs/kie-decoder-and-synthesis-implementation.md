# Decoder training and synthesis foundation

This implementation keeps both new dependency stacks independent from the existing T5Gemma
environment. The images and CPU-side execution paths were validated on 2026-08-29; no model
weights were loaded and no training run was started.

## Decoder training

The decoder runtime supports Qwen3.5/Qwen3 BF16 LoRA SFT and SFT-initialized GRPO/Dr. GRPO. It uses
completion-only loss, direct non-thinking JSON, unconstrained generated evaluation, the existing
task canonicalizer/metrics, MLflow, immutable run contracts, and resumable TRL checkpoints. Invalid
JSON or schema output receives zero RL reward; valid output receives deterministic exact leaf F1.
An on-start evaluation runs before SFT attaches LoRA; GRPO instead evaluates the loaded SFT adapter
before its first policy update. Both are saved with predictions and logged to MLflow at step zero.

The executable first baseline is
`configs/decoder_training/qwen35_08b_lora_sft.mpci_bl_combined1157.yaml`. Its sequence limits were
measured with Qwen/Qwen3.5-0.8B at commit `2fc06364715b967f1860aea9cf38778875588b17` over all 1,157
records: 12,226 prompt tokens, 3,630 completion tokens, and 14,024 combined tokens. Configured limits
are 13,312 / 4,096 / 17,408 and overflow is fatal.

Validation does not load a tokenizer or model:

```bash
docker compose --profile decoder-training run --rm --no-deps decoder-tools \
  validate-config \
  --config configs/decoder_training/qwen35_08b_lora_sft.mpci_bl_combined1157.yaml \
  --project-root /workspace
```

Dataset inspection validates hashes, schemas, and the deterministic split without a model:

```bash
docker compose --profile decoder-training run --rm --no-deps decoder-tools \
  inspect-dataset \
  --config configs/decoder_training/qwen35_08b_lora_sft.mpci_bl_combined1157.yaml \
  --project-root /workspace
```

Tokenizer inspection downloads only the immutable tokenizer revision and proves every record fits
before any model or GPU is used:

```bash
docker compose --profile decoder-training run --rm --no-deps decoder-tools \
  inspect-tokens \
  --config configs/decoder_training/qwen35_08b_lora_sft.mpci_bl_combined1157.yaml \
  --project-root /workspace
```

The measured container result was 12,226 / 3,630 / 14,024 maximum prompt, completion, and combined
tokens. Full dataset inspection verified 1,157 records and the 1,057/100 seeded split at 1,489
records/second. The exact CUDA import probe resolved `UnslothSFTTrainer` and
`UnslothGRPOTrainer` on the RTX 4090. Unsloth must import before TRL; the runtime enforces that order.

Do not launch SFT until the longest-sequence memory benchmark chooses physical batch,
accumulation, and whether packing is beneficial. Batch one is a conservative starting point, not a
measured throughput optimum. Qwen3.5 may also incur a one-time Mamba kernel compile on its first
model run.

The Dr-GRPO config
`configs/decoder_training/qwen35_08b_lora_dr_grpo_probe.mpci_bl_combined1157.yaml` is a 20-step
readiness probe, not a full RL recipe. It is fail-closed on the SFT adapter path and must run only
after SFT evaluation demonstrates adequate schema-valid output and within-prompt reward variance.
It uses two generations, schema-invalid reward zero, exact field/value F1 otherwise, and unscaled
Dr-GRPO loss. The SFT adapter is loaded as the trainable policy and is not wrapped in a second LoRA
adapter; this follows Unsloth's documented continued-finetuning path.

References: [Unsloth Qwen3.5 fine-tuning](https://unsloth.ai/docs/models/qwen3.5/fine-tune),
[TRL SFTTrainer](https://huggingface.co/docs/trl/v0.24.0/sft_trainer), and
[TRL GRPOTrainer](https://huggingface.co/docs/trl/v0.24.0/grpo_trainer).

## Synthesis foundation

The current synthesis command is intentionally non-generative. It verifies every immutable source
row, validates the task schema, normalizes each target into a deterministic relational value graph,
reconstructs it, revalidates it, and requires byte-semantic equality. It publishes `documents` and
`nodes` JSONL tables plus validated SDV V1 metadata and an immutable manifest.

```bash
docker compose --profile synthesis run --rm synthesis-tools \
  prepare-foundation \
  --config configs/synthesis/mpci_bl_combined1157_foundation.yaml \
  --project-root /workspace
```

SDV and PydanticAI are pinned in `environments/synthesis/uv.lock`. No PydanticAI call is made by the
foundation stage: mutation ownership, target cohorts, template anchoring, relation-aware
cardinality changes, and LLM rendering remain decision gates described in the research board.

The validated foundation is published under
`artifacts/kie-synthesis/mpci-bl-combined1157-synthesis-foundation-v1/`. It contains 1,157 document
rows and 101,203 value-graph nodes. SDV 1.38.2 validated the metadata and both tables in 4.07 seconds;
an immediate repeat completed in 3.69 seconds with the identical persisted manifest hash. The
synthesis lock deliberately uses CPU-only Torch, avoiding CUDA/NVIDIA packages for this CPU-side
workflow. A subsequent instrumented full pass took 4.24 seconds with 512.6 MiB peak process RSS.

References: [SDV metadata API](https://docs.sdv.dev/sdv/concepts/metadata/metadata-api) and
[PydanticAI models](https://ai.pydantic.dev/models/).
