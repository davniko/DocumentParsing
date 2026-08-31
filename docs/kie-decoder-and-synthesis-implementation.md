# Decoder training and synthesis foundation

This implementation keeps both new dependency stacks independent from the existing T5Gemma
environment. The images and CPU-side execution paths were validated on 2026-08-29; no model
weights were loaded and no training run was started.

## Decoder training

The decoder runtime supports Qwen3.5/Qwen3 BF16 LoRA SFT and GRPO/Dr. GRPO initialized either from
the configured base model or an existing adapter. It uses completion-only SFT loss, direct JSON or
optional native Qwen thinking for GRPO, unconstrained generated evaluation, the existing task
canonicalizer/metrics, MLflow, immutable run contracts, and resumable TRL checkpoints. Invalid JSON,
invalid schema, or an invalid native-thinking boundary receives zero RL reward; valid final JSON
receives deterministic exact leaf F1.

An on-start evaluation runs before SFT or direct GRPO attaches fresh LoRA; adapter-initialized GRPO
instead evaluates the loaded adapter before its first policy update. Both are saved with predictions
and logged to MLflow at step zero.

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
readiness probe, not a full RL recipe. `grpo.initialize_from` is an explicit, required nullable
field: `null` loads `model.name_or_path` and creates a fresh LoRA adapter, while an adapter directory
loads and continues that adapter. The latter is not wrapped in a second LoRA adapter and follows
Unsloth's documented continued-finetuning path. Both initialization modes otherwise use the same
GRPO/Dr-GRPO trainer, reward, evaluation, checkpoint, and logging path. The supplied probe points to
the SFT result and should run only after that adapter demonstrates adequate schema-valid output and
within-prompt reward variance. It uses two generations, schema-invalid reward zero, exact
field/value F1 otherwise, and unscaled Dr-GRPO loss.

The direct-JSON probe keeps `sequence.thinking: disabled` as the latency/reliability control. The
separate
`configs/decoder_training/qwen35_08b_lora_dr_grpo_reasoning_probe.mpci_bl_combined1157.yaml`
sets `sequence.thinking: enabled`. This uses the pinned tokenizer's official chat-template contract:
the prompt ends with `<think>`, the generated continuation contains the trace followed by one
`</think>` boundary, and only the subsequent final JSON is passed to schema validation, reward, and
field metrics. Evaluation artifacts retain the full generated text, separated reasoning trace,
final answer, and boundary diagnostics. Missing/duplicate boundaries and thinking-only output fail
closed; the code never searches for a JSON-looking substring.

The reasoning probe reserves 6,144 completion tokens versus the measured 3,630-token maximum direct
target, leaving over 2,500 tokens for reasoning and the boundary. Exact-container inspection of all
1,157 rows measured maxima of 12,224 prompt, 3,632 minimal boundary-plus-final completion, and
14,024 combined tokens in 22.90 seconds (50.51 records/second), all within its 13,312 / 6,144 /
19,456 limits. This is not yet a claim that two stochastic generations fit or are
throughput-optimal on the GPU. Run the same longest-sequence memory benchmark before launching this
arm. SFT remains direct JSON because the dataset contains no auditable supervised reasoning traces.
The reasoning rollout also pins Qwen's recommended text-thinking values `temperature: 1.0`,
`top_p: 0.95`, `top_k: 20`, and `repetition_penalty: 1.0`; the 0.8B model card warns that thinking
can loop, so EOS, truncation, reasoning length, and boundary validity remain explicit go/no-go
metrics.

## T5Gemma 2 attention backend

The existing T5Gemma 2 runtime uses `attn_implementation: sdpa`. FlashAttention-2 is deliberately
not accepted by its strict configuration because Transformers 5.15 reports
`T5Gemma2ForConditionalGeneration._supports_flash_attn = False`. T5Gemma 2 merges decoder self- and
cross-attention and requires custom masks that are currently incompatible with the Transformers
FlashAttention path. Adding the `flash-attn` package or bypassing dispatch checks would therefore
not produce a supported implementation. Keep SDPA until upstream T5Gemma 2 support and equivalence
tests exist; do not silently substitute eager attention.

This does not mean training falls back entirely to unfused quadratic eager kernels. A BF16
forward/backward profiler probe of the T5Gemma 2 architecture in the pinned PyTorch 2.13/CUDA 13
trainer recorded PyTorch SDPA flash kernels for compatible attention calls and memory-efficient
SDPA kernels for the custom-mask calls. That is the supported optimized path currently available;
it is distinct from selecting Transformers' external `flash_attention_2` backend.

References: [Qwen3.5 0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B),
[Unsloth Qwen3.5 fine-tuning](https://unsloth.ai/docs/models/qwen3.5/fine-tune),
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

### Domain and synthesis-readiness preparation

The next non-generative stage is now implemented for the exact 1,157-row corpus. It refreshes the
provenance-aware EDA, projects every relation-v3 target into 17 B/L-specific tables, proves exact
inverse projection, validates the resulting graph and data with SDV 1.38.2, resolves immutable
annotation evidence into role-aware OCR anchors/format profiles, and publishes cohort/template
support. It also exposes tested deterministic primitives for ISO 6346 containers, scalar seals,
joint date shifts, quantities, masses, and allocation reconciliation. It does not generate or
publish a synthetic training row.

The full design, completed measurements, remaining decisions, and recommended configurable
generation architecture are recorded in
[`kie-synthesis-preparation-and-generation-plan-2026-08-30.md`](kie-synthesis-preparation-and-generation-plan-2026-08-30.md).

Refresh the EDA on the host with:

```bash
TMPDIR=/tmp UV_CACHE_DIR=/tmp/documentparsing-uv-cache \
uv run --frozen --group analysis document-kie-dataset-eda \
  --config configs/analysis/mpci_bl_combined1157_synthesis_eda.yaml
```

Rebuild the synthesis image whenever `src/` or the synthesis lock changes, then validate and run
the preparation contract:

```bash
docker compose --profile synthesis build synthesis-tools

docker compose --profile synthesis run --rm synthesis-tools \
  validate-preparation-config \
  --config configs/synthesis/mpci_bl_combined1157_preparation.yaml \
  --project-root /workspace

docker compose --profile synthesis run --rm synthesis-tools \
  prepare-corpus \
  --config configs/synthesis/mpci_bl_combined1157_preparation.yaml \
  --project-root /workspace
```

The final preparation publication is
`artifacts/kie-synthesis/mpci-bl-combined1157-synthesis-preparation-v3/`: 27,625 domain rows across
17 tables, 54,462 audited anchor rows, 13,201 role-aware format profiles, six
matplotlib/seaborn plots, complete support tables, and an immutable manifest. All 1,157 targets
round-trip exactly and every source-fact leaf has evidence. Unique or unique-within-excerpt anchors
cover 49,991/54,462 evidence rows; 4,471 repeated values remain explicitly ambiguous rather than
being assigned by guesswork.

### Structured synthesis baseline

The next two non-agentic passes are implemented behind `run-structured-baseline`. They construct
complete inspection-grade structured proposals while preserving source language, party and goods
facts, package categories, container types, and relation cardinality. They do not patch raw OCR or
publish training rows.

The runner first freezes a template-isolated train scope and solves the exact configured 50-source
selection with SciPy MILP. It then benchmarks profile-specific SDV candidates on template-grouped
folds. Statistical models propose cargo-group driver quantities and present per-driver measures;
task-owned deterministic code derives totals, every retained package level, allocations, identifiers,
and dates. Global identifier reservations exclude the complete real corpus. Every accepted proposal
must pass schema, relational inverse, exact-diff, arithmetic, collision, package-context, and
type-aware equipment-capacity checks.

Rebuild the image after any synthesis source change, copy the resulting image ID into both the
Compose environment and YAML runtime receipt, then run:

```bash
DOCKER_CONFIG=/tmp/documentparsing-docker-config \
docker compose --profile synthesis build synthesis-tools

DOCKER_CONFIG=/tmp/documentparsing-docker-config \
docker compose --profile synthesis run --rm synthesis-tools \
  validate-structured-baseline-config \
  --config configs/synthesis/mpci_bl_combined1157_structured_baseline50.yaml \
  --project-root /workspace

DOCKER_CONFIG=/tmp/documentparsing-docker-config \
docker compose --profile synthesis run --rm synthesis-tools \
  run-structured-baseline \
  --config configs/synthesis/mpci_bl_combined1157_structured_baseline50.yaml \
  --project-root /workspace
```

The accepted publication is
`artifacts/kie-synthesis/mpci-bl-combined1157-structured-baseline50-v8/`. It contains 50 distinct
templates across 20 carrier strata (19 named plus one missing-value stratum), 93 cargo-group
proposals, 105 changed task-facing package quantities,
92 valid unique ISO 6346 containers, complete CSV/JSON receipts, and 13 matplotlib/seaborn plots.
All 50 scenarios pass the hard gates and zero records are training eligible. Runtime was 197.48
seconds with 832.59 MiB peak Python RSS; the immutable committed-run fast path completed in 4.72
seconds without changing any artifact byte.

The statistical proposal layer explicitly separates model-routing support from package-plausibility
support. Role-wide rows can support a fit but never a plausibility decision. Accepted values must
lie inside an exact-identity or meaningful semantic-family support envelope. The generic
`UNREGISTERED_PRINTED_PACKAGE` and `UNTYPED_PACKAGE` labels cannot serve as semantic families.

The next stage is full party/goods/auxiliary-sensitive anonymization plus text-only realization.
PDF generation or editing remains outside scope. Until that stage proves complete source-sensitive
value removal and target-to-text grounding, these structured proposals must remain inspection-only.
