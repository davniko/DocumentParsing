# T5Gemma2 training and evaluation kernel efficiency

Date: 2026-10-08. Status: investigated, **not implemented or GPU-benchmarked**.

## Purpose and decision

Investigate faster, lower-memory execution of the existing T5Gemma2 extraction model without changing its architecture, training objective, labels, or attention semantics. The investigation was read-only while training occupied the GPU. No kernels were installed, no GPU benchmarks were launched, and no training configuration or running process was changed.

The strongest custom-attention candidate is **PyTorch FlexAttention with a T5Gemma2-specific mask integration**. Encoder-only FlashAttention-2 is a narrower alternative. Independently, fused linear cross-entropy deserves profiling because this small model has a very large vocabulary. FlashInfer and the supplied vLLM implementation are not drop-in training replacements.

These are candidates, not measured improvements. Profile the actual workload before choosing an implementation. Do not infer training throughput from inference microbenchmarks or equate `sdpa` with unfused attention.

## 1. Verified project execution path

The inspected run is:

`artifacts/kie-training/t5gemma2-270m-mpci-bl-real600-synthetic1500-positions-inputonly-v7-e10-compact-eva-a32-r32-local-sfadamw-v1/`

Its configuration is [the 600-real + 1,500-synthetic local configuration](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic1500_positions_inputonly_v7_e10_compact_eva_a32_r32_local_schedulefree_v1.yaml).

| Property | Verified value |
|---|---|
| GPU | NVIDIA RTX 4090, compute capability 8.9, 24 GiB |
| PyTorch | 2.13.0+cu130 |
| Transformers | 5.15.0 |
| Base checkpoint | `google/t5gemma-2-270m-270m` |
| Model revision | `7c38f16641f455ef0685b18431faf1b17722d5a1` |
| Precision / attention | BF16 / SDPA |
| Training microbatch / accumulation | 1 / 32 |
| Evaluation batch | 2 |
| LoRA | rank 32, alpha 32, rsLoRA, EVA, dropout 0.05 |
| Attention dropout | **0.0**, distinct from LoRA dropout |
| Gradient checkpointing | Enabled, reentrant |
| Whole-model compilation | Disabled |
| Training cache | Disabled |
| Generation cache | Explicitly enabled |

The model loader in [runtime.py](../src/document_ocr/training/runtime.py) passes `attn_implementation` to `AutoModelForSeq2SeqLM.from_pretrained`. [config.py](../src/document_ocr/training/config.py) currently accepts only `sdpa` or `eager`. There is no custom attention/backend configuration in our project yet.

The installed model implementation in the actual container and the inspected local Transformers installation had identical SHA-256:

`48abb27b88ae68ef67a08f7382ff63efc918655f4f9100524271f26661b34ef0`

Container path: `/opt/document-ocr-venv/lib/python3.12/site-packages/transformers/models/t5gemma2/modeling_t5gemma2.py`. Container inspection used read-only shell commands, not model loading or GPU execution. Recheck this identity before a later implementation: the Docker image copies source into `/opt/document-ocr`; a workspace bind mount does not replace those imports.

### Actual checkpoint dimensions

The cached configuration for the pinned checkpoint establishes:

- 18 text-encoder layers and 18 decoder layers.
- 15 sliding-attention layers and three full-attention layers in each stack.
- Sliding-window setting 512.
- Four query heads, one KV head, head dimension 256.
- Hidden size 640, intermediate size 2,048, `gelu_pytorch_tanh` activation.
- Vocabulary size 262,144.
- No attention-logit or final-logit softcapping in this checkpoint.
- Gemma-style RMSNorm: normalize in FP32, multiply by `1 + weight`, then cast back.

These are the actual small-checkpoint values, not the larger-model defaults appearing in some documentation or issues.

### Real sequence lengths

The run's [dataset report](../artifacts/kie-training/t5gemma2-270m-mpci-bl-real600-synthetic1500-positions-inputonly-v7-e10-compact-eva-a32-r32-local-sfadamw-v1/dataset-report.json) records:

| Training sequence | Median | p95 | Maximum |
|---|---:|---:|---:|
| Input | 2,113 | 4,840 | 8,066 |
| Target | 688 | 1,027 | 3,139 |

Configured caps of 19,200/5,500 are not typical executed lengths. Benchmark actual paired input/target shapes, including long outliers, rather than just those caps. Marginal maxima need not occur in the same document.

## 2. What makes this attention different

### Encoder

Encoder attention is bidirectional, with full and local layers. Q/K normalization and rotary position embedding precede attention. For local layers, preserve the installed helper's exact left/right window bounds; a generic `window_size=(W,W)` is not equivalent.

### Decoder

The decoder creates Q and self-K/V from decoder states and cross-K/V from encoder states. It normalizes Q/K, applies RoPE to Q and self-K, **not cross-K**, and concatenates keys/values in `[self, cross]` order.

The operation has **one softmax over both regions**:

```text
decoder query q
  self keys: valid decoder positions <= q, locally restricted in sliding layers
  cross keys: all valid encoder positions
  normalize jointly across both sets
```

Two independently normalized attention calls followed by addition are not equivalent. A partitioned implementation would need exact log-sum-exp weighting and correct backward propagation through that combination. That is not our recommended first implementation.

The current decoder constructs dense self and cross masks separately and concatenates them. Full cross attention remains necessary even in sliding decoder layers. Applying one sliding window over the combined key sequence would wrongly hide source tokens.

During generation, self-K/V evolves while encoder K/V is cached independently. Any replacement must preserve logical query offsets and local-cache offsets, not assume every call starts at token zero.

## 3. Existing SDPA is already a dispatcher

PyTorch SDPA may select Flash, memory-efficient, cuDNN or math implementations depending on inputs and support. We did not profile which backend this run actually executes.

However, PyTorch 2.13's Flash backend includes `check_for_attn_mask`, which rejects non-null masks. The explicit merged decoder masks therefore exclude that backend. This is **not proof that those calls use the math backend**; other fused implementations may support them. Sources: [CUDA dispatcher](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp), [mask gate](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/transformers/sdp_utils_cpp.h#L259-L267).

The installed Transformers SDPA wrapper also repeats KV heads when a mask is present instead of passing native grouped-query attention. This is a potential memory/traffic opportunity, not a demonstrated bottleneck. Inspect `transformers/integrations/sdpa_attention.py` in the benchmark environment.

T5Gemma2 explicitly declares `_supports_flash_attn=False`, `_supports_flex_attn=False`, `_supports_sdpa=True`, and `_supports_attention_backend=True`. The same Flash/Flex restrictions were verified in 5.19.0 and upstream `main` during research. Merely upgrading Transformers or changing the support flags is not a correct integration. [Upstream model source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/t5gemma2/modeling_t5gemma2.py).

## 4. Review of the supplied vLLM project

Repository: [ddickmann/vllm-factory T5Gemma2](https://github.com/ddickmann/vllm-factory/blob/main/models/t5gemma2/README.md). Inspected commit: `7d6ff68ce68f9f7c0a9d72f9645bcf6d335d02f0`.

The [attention kernel](https://github.com/ddickmann/vllm-factory/blob/7d6ff68ce68f9f7c0a9d72f9645bcf6d335d02f0/kernels/flash_t5gemma2_attention.py) implements online-softmax merged attention, grouped-query heads and masking without a full score matrix. It is useful design/reference code.

It is not a training drop-in: raw Triton launches write into allocated outputs, without backward kernels or autograd registration. The [fused Q/K normalization and RoPE kernel](https://github.com/ddickmann/vllm-factory/blob/7d6ff68ce68f9f7c0a9d72f9645bcf6d335d02f0/kernels/fused_qk_norm_rope.py) has the same limitation. Frozen base weights do not remove the need to differentiate through attention and normalization to train upstream LoRA adapters.

The implementation also lacks an attention-dropout argument and the cached query-offset handling required for our incremental decoder. Its [decoder initialization](https://github.com/ddickmann/vllm-factory/blob/7d6ff68ce68f9f7c0a9d72f9645bcf6d335d02f0/models/t5gemma2/t5gemma2_model.py#L468-L479) discards cache configuration for full-sequence parity.

The advertised ~1.43–1.51x throughput is not comparable to our training: encoder length 14, decoder length 8, FP32, `eval()`/`no_grad()`, repeated reference inputs, and warmup excluded. Moreover, `run_forward` calls the encoder and discards its result, then invokes the model without those encoder outputs, causing a second encoder execution. See [benchmark](https://github.com/ddickmann/vllm-factory/blob/7d6ff68ce68f9f7c0a9d72f9645bcf6d335d02f0/models/t5gemma2/throughput_benchmark.py#L46-L105) and [model forward](https://github.com/ddickmann/vllm-factory/blob/7d6ff68ce68f9f7c0a9d72f9645bcf6d335d02f0/models/t5gemma2/t5gemma2_model.py#L700-L729). Its vision-path results are not applicable to our text-only inputs.

Reuse is possible under its [Apache-2.0 license](https://github.com/ddickmann/vllm-factory/blob/7d6ff68ce68f9f7c0a9d72f9645bcf6d335d02f0/LICENSE), preserving notices. That does not license model weights separately or make the kernels training-safe.

## 5. Candidate implementations

### A. FlexAttention — preferred custom attention candidate

[FlexAttention](https://pytorch.org/blog/flexattention/) compiles Python mask/score definitions into fused kernels and generates backward computation. Its `BlockMask` can express the decoder's two-region mask and skip masked blocks in local encoder attention.

Implement a narrow T5Gemma2 backend and mask adapter while preserving projections, parameter names and checkpoint layout. Build combined masks directly in the backend representation; do not attempt to `torch.cat` independently constructed BlockMasks. Reuse mask metadata across layers with the same pattern. Preserve native grouped-query heads if the backend and actual hardware pass parity/performance tests.

The installed Transformers Flex wrapper rejects nonzero attention dropout; our checkpoint uses zero, so no training-policy change is needed. LoRA dropout remains untouched. Compile only the attention path initially; enabling whole-model compilation is a separate experiment.

Potential benefit: less mask allocation, efficient local attention, fewer repeated KV tensors. Constraints: head dimension 256 on Ada, backward resource usage, mask-build cost, variable-length recompilation, and cached decoding must all be tested. Full encoder layers and all encoder cross keys remain computational work; block sparsity does not eliminate them.

The newer [FlexAttention FlashAttention-4 backend](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/) targets Hopper/Blackwell. Do not assume its reported speedups apply to this Ada RTX 4090; start with the compatible Triton path.

### B. Encoder-only FlashAttention-2

A smaller integration can optimize bidirectional full/local encoder attention while retaining decoder SDPA. The [official FA2 implementation](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support) supports Ada, BF16 and dimension-256 backward with zero attention dropout.

Exact bidirectional window bounds, variable-length unpadding, rotary positions and gradient parity are still mandatory. This is useful if profiling identifies the encoder as the dominant attention cost. Do not copy an entire alternate model implementation to replace a small attention seam.

### C. FlashInfer — consider separately for generation

[FlashInfer](https://github.com/flashinfer-ai/flashinfer) is primarily a serving kernel library. Its [attention API](https://docs.flashinfer.ai/api/attention.html) supports custom masks and prefill/decode execution. The inspected standard softmax-attention [prefill path](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/prefill.py) does not supply the backward/autograd path required by our training. Specialized training APIs elsewhere in the library do not establish support for this operation.

There is no current project configuration switch that makes this a Transformers T5Gemma2 backend. Consider it only if profiling justifies a separate generation integration, including encoder/decoder cache layout and exact masking. Do not transfer inference benchmark speedups to forward-plus-backward training.

### D. Fused output projection and cross-entropy

The installed forward materializes `[batch,target_length,262144]` logits. `ForMaskedLMLoss` upcasts logits to FP32 before cross-entropy; decoder inputs are already shifted, so another causal-label shift would be wrong.

Calculated allocations at batch one, excluding padding/other buffers:

| Target length | BF16 logits | FP32 logits copy |
|---|---:|---:|
| Median 688 | 344 MiB | 688 MiB |
| Maximum 3,139 | 1,569.5 MiB | 3,139 MiB |

These are shape-derived sizes, not measured memory savings or proof that loss dominates runtime.

[Liger](https://github.com/linkedin/Liger-Kernel) fused linear cross-entropy and [Cut Cross-Entropy](https://github.com/apple-aiml-research/ml-cross-entropy) offer training-capable approaches that avoid materializing all logits. T5Gemma2 is absent from the inspected Liger model-patching registry; `use_liger_kernel=True` is not an established one-line solution. Use a narrow compatible integration if profiling warrants it.

Preserve ignored labels, loss normalization across accumulated batches, label alignment, tied/frozen classifier weights and hidden-state gradients. Generation and any consumer requiring logits must retain the appropriate logits path. Test a non-gradient-filtering reference configuration first if evaluating CCE; approximate gradient filtering is a separate numerical decision.

### E. Smaller fused operations and compilation

[Transformers kernel integration](https://huggingface.co/docs/transformers/kernels) supports selected Hub kernels, and Liger supplies normalization/activation/loss primitives. Our model already contains a rotary-function kernel hook, but it does not mean every layer is automatically covered.

Any RMSNorm replacement must preserve Gemma's weight offset and casting order. The MLP uses GELU-based gating, not generic SwiGLU. Avoid selecting kernels just because their names look similar.

`torch.compile` is another controlled candidate for reducing surrounding launch overhead. Measure compilation/recompilation and checkpointing interactions. Treat it separately from replacing attention so the source of any gain is identifiable.

## 6. Integration contract

[Transformers AttentionInterface and AttentionMaskInterface](https://huggingface.co/docs/transformers/attention_interface) provide supported registration seams. A custom attention function needs a matching mask implementation; otherwise constraints can be omitted. T5Gemma2's explicit merged-mask construction additionally needs adaptation.

Requirements:

1. Opt-in backend configuration with explicit capability checks; unsupported combinations error clearly.
2. Same model architecture, weight names, PEFT projection targets and adapter serialization.
3. Correct differentiable paths to encoder and decoder LoRA parameters.
4. Preserve full/local masks, padding, RoPE asymmetry and generation cache offsets.
5. Keep a selectable reference SDPA implementation for comparison; no silent backend substitution.
6. Pin compatible dependencies and validate in the same rebuilt training image, not only the host environment.
7. Do not change precision, optimizer, batch size, dataset or target policy at the same time as the kernel experiment.

## 7. Bounded validation and benchmark plan

### Establish the baseline first

- Profile actual short, median, p95 and longest paired sequences, using the real collator, BF16, LoRA and checkpointing.
- Separate encoder, merged decoder, masks/KV concatenation, MLP, output projection/loss, backward and optimizer time.
- Record selected SDPA kernels rather than infer from configuration.
- Profile cached autoregressive generation separately: encoder pass, first decoder call and incremental tokens.

### Correctness gates before speed claims

- Small FP32 forward/gradient comparisons against the existing implementation, followed by realistic BF16 tolerance checks.
- Full/local encoder and decoder layers; padded/unpadded batches; uneven lengths; window boundaries; short/long sequences; cached query offsets.
- Explicit future-token and padded-token perturbation tests to catch visibility leakage.
- Q/K/V and encoder-state gradients, LoRA gradients, checkpoint recomputation and one optimizer-step comparison.
- Loss alignment, ignore-index and accumulation normalization tests for fused loss.
- Compare cached and full-prefix logits and inspect greedy outputs; explain numerical differences rather than silently accepting changed extraction behavior.

### Performance gates

- Benchmark forward plus backward and complete training steps, not forward-only attention.
- Report wall-clock/token throughput and peak allocated/reserved VRAM across representative shapes.
- Separate cold compilation from warm throughput; count recompilations and include their amortized cost.
- Use repeated measurements; retain unchanged settings and disclose numerical tolerances.
- Run a short fixed-batch training sanity check before a full experiment, with finite losses/gradients and no systematic divergence.
- Verify evaluation correctness and latency independently; training and generation may need different backends.

If attention dominates, prioritize FlexAttention or the narrower encoder FA2 path. If logits/loss dominates, prioritize fused linear loss. If launch overhead dominates, evaluate compilation and small-op fusion. No credible end-to-end speed estimate exists until this baseline profile is available.

## 8. Boundaries and follow-up status

This document records a completed source-level investigation and a proposed implementation/validation plan. No kernel is certified for this project yet, no dependency or production code changed, and no throughput/memory improvement has been measured. The running training job was left untouched. GPU availability later removes the benchmarking limitation but does not itself authorize a new training experiment.
