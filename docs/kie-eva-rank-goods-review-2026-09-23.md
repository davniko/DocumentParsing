# EVA initialization, learned-adapter rank, and goods coverage

Date: 2026-09-23. Scope: configure the next run, implement and test initialization, analyze existing adapters, and explain registry/EDA coverage. No production training or production-model calibration was launched. Existing datasets, adapters, accepted EDA, and the earlier training config are unchanged.

## Prepared training configuration

Use [the new alpha-32/EVA configuration](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real1057_synthetic29910_recovered_v5_e5_eva_a32_v1.yaml).

| Setting | Selected value | Rationale |
| --- | --- | --- |
| rsLoRA rank / alpha | 32 / 32 | Requested alpha reduction; preserve rank and parameter budget. |
| Initialization | EVA | Training-input activation directions rather than random A initialization. |
| `rho` | 1.0 | Fixed rank in every projection; no redistribution or deleted adapters. |
| `tau` | 0.99 | PEFT's convergence threshold, retained rather than relaxed. |
| Whitening | false | Keep the usual unwhitened EVA directions. |
| Calibration pool | 512 train documents | Seeded sample across the merged training data, not the first file block. |
| Calibration batch | 1 | Accommodate this dataset's long encoder sequences. |
| SVD rows | Up to 512 nonpadding tokens per document per stream | Bound activation/SVD work while forwarding complete sequences. |
| Forward budget | 512 | Explicit nonconvergence error rather than an unbounded initialization loop. |
| Calibration seed | 20260923 | Reproducible pool; does not consume training RNG state. |

The nominal rsLoRA multiplier changes from `64 / sqrt(32) = 11.3137` to `32 / sqrt(32) = 5.65685`. This halves the multiplier, not necessarily the eventual update magnitude: changing A initialization also changes optimization. The documented rsLoRA convention and EVA configuration are in the [PEFT LoRA reference](https://huggingface.co/docs/peft/en/package_reference/lora).

EVA obtains principal directions from layer-input activations and uses them to initialize A, with B initially zero. It can also redistribute rank; that is optional and is deliberately disabled in this first fixed-rank configuration. Activation variance is a useful initialization signal, not a guarantee of better extraction accuracy. See the [EVA paper](https://arxiv.org/html/2410.07170v3) and [official PEFT example](https://github.com/huggingface/peft/blob/main/examples/eva_finetuning/README.md).

These are reasoned starting settings, not hyperparameters empirically optimized on this task. The paper's batch-size experiments do not establish an optimal calibration size for long, structured T5Gemma2 extraction. The selected pool contains 11 real, 176 Egypt-focused, 155 diversified, and 170 recovered-template documents; 386 ordinary, 72 thermal, and 54 DG examples. This approximates the actual training mixture. The algorithm can converge before consuming the entire pool; the runtime receipt records actual forwards and document exposure.

Other optimization settings are preserved: 5 epochs, rank 32, learning rate 1e-4, accumulation 32, and `on_start: false`. Dataset: 29,910 synthetic + 1,057 real training rows; the same separate 100 validation rows. This run changes alpha and initialization together, so any eventual validation improvement cannot be attributed to EVA alone.

Per the follow-up request, evaluation and checkpoint intervals are now **484 optimizer steps**. The installed Trainer rounds the final partial accumulation group up, giving `ceil(30,967 / 32) = 968` updates per epoch and 4,840 over five epochs. Validation occurs at steps **484, 968, 1452, 1936, 2420, 2904, 3388, 3872, 4356, 4840**: halfway through and at the end of each epoch. The separate final evaluation and prediction export are disabled to avoid an eleventh validation pass. Scheduled metrics, best-checkpoint selection, and the final best adapter remain enabled; detailed prediction export can be run separately if wanted. No training-loop implementation was changed for this schedule.

The earlier config remains byte-identical, SHA256 `551b5c85c204e8877a6fffc88412a6cee28800b4080207c3ab6f408bc1c53977`. The new config has a distinct run/adapter name, SHA256 `3fa28296f7e84308944622c4464fd4b282375bf7e594478467972ec30749410b`.

## Why EVA needed runtime integration, not just a YAML string

Inspection of the installed PEFT 0.19.1 and Transformers 5.15.0 execution path identified two encoder-decoder issues:

1. Encoder activations and teacher-forced decoder activations need different padding masks. Their lengths can coincide; length-based guessing is not a valid contract.
2. T5Gemma2 calls each decoder K/V projection twice: once on decoder states, then on encoder states. PEFT's first-input equivalence optimization could otherwise group those projections with decoder-only Q and miss the encoder contribution.

The integration collects both actual K/V streams, then gives PEFT one combined observation per projection per forward. It uses the native EVA incremental SVD implementation, not a replacement optimizer. Other decoder projections use decoder tokens; encoder projections use encoder tokens. Up to 512 valid rows are spread across each stream. Complete source and target sequences still reach the backbone. The vocabulary head is bypassed because initialization needs activations, not logits or loss.

Safety and lifecycle contracts:

- Calibration draws from the training partition only; no validation examples.
- Frozen base weights remain unchanged. B stays zero, so the initial adapter is a no-op.
- Every adapted projection must have a finite, correctly shaped EVA result. Missing coverage is an error, not PEFT's random-initialization fallback.
- Fixed rank and rsLoRA scaling are verified. The original target-module regex is restored after native EVA rewrites its representation; checkpoint evaluation remains compatible.
- Hooks are removed after success or failure, and model mode/RNG are restored. There are no calibration hooks in the training hot path.
- Initialization runs before Trainer/optimizer construction. Resume checks the existing receipt and loads trained checkpoint weights; it does not recalibrate them.
- `eva-initialization.json` records settings, training-cache identity, selected IDs, token exposure, runtime, and CUDA peak allocation.

Files changed for this task:

- [config.py](../src/document_ocr/training/config.py): explicit, validated EVA settings.
- [eva.py](../src/document_ocr/training/eva.py): fixed-rank, dual-stream calibration and resume contract.
- [runtime.py](../src/document_ocr/training/runtime.py): native EVA configuration, startup integration, provenance.
- [test_training_eva.py](../tests/test_training_eva.py): real-PEFT tiny-T5Gemma2 coverage and failure tests.
- [analyze_lora_spectrum.py](../tools/analyze_lora_spectrum.py) and [tests](../tests/test_lora_spectrum.py): independent, adapter-only rank diagnostic.
- The separate new production YAML linked above.

PEFT emits an advisory about adapter `low_cpu_mem_usage=False`. This path intentionally initializes allocated, small adapters rather than meta-device adapters, including for normal checkpoint restoration; the model loader's existing low-memory setting is unchanged. The advisory is not a failed calibration or random fallback.

## Learned-update SVD results

The Claude share URL returned HTTP 403; its full conversation could not be inspected. The user's supplied excerpt establishes the intended diagnostic. That diagnostic was applied to the actual combined update `Delta W = scale * B @ A`, not to A/B separately.

The latest 10k-run artifacts available locally belong to `t5gemma2-270m-lora-mpci-bl-real1057-complete-synth10000-exact-id-v5-e5-v1`. Its status records KeyboardInterrupt on 2026-09-21. The latest retained checkpoint is 1557, with recorded best field-value F1 0.834095; checkpoint 1384 was analyzed as a consistency check. These are not described as a newly completed training run.

Each checkpoint contains 252 adapted projections, all rank 32, alpha 64, rsLoRA enabled. Results below describe checkpoint 1557; checkpoint 1384 is almost identical.

| Per-projection statistic | Median | Range |
| --- | --- | --- |
| Directions for 90% squared singular-value energy | 25 | 19–26 |
| Directions for 95% energy | 28 | 23–29 |
| Directions for 99% energy | 31 | 29–32 |
| Stable rank | 11.67 | 5.57–17.32 |
| Energy in the last eight directions | 10.03% | 3.98–13.76% |
| Smallest/largest singular value | 0.321 | 0.121–0.484 |

251/252 projections require at least 30 directions to retain 99% of the weight-update energy. Keeping only 16 retains a median 72.74% in encoder projections and 72.98% in decoder projections. There is no sharp collapse at 8–16 directions.

Interpretation:

- There is no support here for reducing rank to 8 or 16 as a nearly lossless compression.
- Stable rank is `sum(sigma**2) / sigma_max**2`; 11.67 does **not** mean rank 12 is adequate. The observed 95%/99% energy ranks demonstrate the distinction.
- Appreciable energy near the rank-32 boundary makes capacity pressure plausible. It does not prove that rank 64 improves validation F1: weight-space energy is not task importance, and the optimization trajectory also affects the spectrum.
- This was a rank-32 run, not the deliberately generous rank-64/128 run described in the excerpt. Directions excluded by that rank constraint cannot be observed afterward. Even small discarded energy would not prove unchanged extraction quality.
- A later controlled rank-64 comparison is motivated, but was not launched or substituted into the requested rank-32 configuration. EVA's pre-training activation spectrum and this post-training weight-update spectrum answer different questions.

Method: reduced QR factorizations of A-transpose and B yield the exact nonzero spectrum through a rank-sized core, computed in float64. Both checkpoints together took 1.387 seconds, peak process RSS 580,668 KiB (567 MiB); the base model was not loaded. On a representative 640x2048 update, median core computation was 0.665 ms versus 58.884 ms for dense SVD (~88.5x faster), with numerical agreement checked. This is a diagnostic speedup, not a training-throughput claim.

Evidence: [spectrum plot](../artifacts/kie-training/analysis/lora-eva-preflight-20260923/adapter-spectrum.png), [every module's metrics](../artifacts/kie-training/analysis/lora-eva-preflight-20260923/spectrum/modules.csv), [full spectra and input hashes](../artifacts/kie-training/analysis/lora-eva-preflight-20260923/spectrum/spectra.json).

## What the HS6, HS-chapter, and UN plots mean

The existing EDA chooses the 16 most frequent pooled categories for display. Those are **not the only sampled codes**. Bar denominators are all field occurrences in each population, not just those 16 categories and not documents. There is no aggregated Other bar. `hs_chapter` groups six-digit prefixes by their first two digits.

Counts from the published dataset histograms:

| Population | Documents | Distinct HS6 prefixes | Chapters | Distinct UN numbers |
| --- | ---: | ---: | ---: | ---: |
| Real training data | 1,057 | 550 | 71 | 22 |
| Egypt-focused synthetic | 9,980 | 896 | 69 | 23 |
| Diversified synthetic | 9,930 | 1,357 | 46 | 512 |
| Recovered-template synthetic | 10,000 | 1,554 | 60 | 467 |
| Combined synthetic | 29,910 | 2,075 | 72 | 751 |

The newest set has 11,831 HS observations and 1,000 UN observations. Its own most common 16 HS6 values account for 14.76% of HS observations; its most common 16 UNs account for 25.8%. Distinct UN counts of 467 versus 512 in two 1,000-observation batches are not, by themselves, evidence of a sampling regression.

The older Egypt set really does have narrow DG diversity: 23 UN identities. Its 1,524 UN occurrences contribute heavily to the concentration visible in the combined plot. The newer two sets are not restricted to that small list.

Rebuilt support from the actual reviewed synthesis configuration:

| Registry/support level | Distinct identities |
| --- | ---: |
| Full pinned HS6 registry | 5,612 across 96 chapters |
| Ambient goods support | 3,574 HS6 |
| Ambient HS6 with observed equipment-heading support | 1,788 |
| Frozen / chilled support | 98 / 116 HS6 |
| Upstream HMT | 2,323 UN numbers; 2,305 maritime-eligible |
| Supported DG non-bulk packaging envelope | 1,203 UN numbers in 1,449 HMT records |
| Supported exact chemical-to-HS linkage subset | 384 UN numbers, 138 HS6, 386 links |

These pools are not interchangeable denominators, and thermal/ambient counts should not simply be added. Each template further restricts eligible goods according to its fields and physical/packaging contracts.

Sampling is randomized but **conditional and weighted**, not uniform across all registry rows:

- Ambient goods: select a compatible observed package signature and weighted HS heading, then an identity from the compatible heading. Equipment and capacity constraints also apply.
- Thermal goods: configured supported frozen/chilled profile, then a compatible identity and temperature regime.
- DG: eligible hazard category, then UN identity, then a coherent record/link. Exact HS linkage, packing groups, subsidiary hazards, and printed template fields can narrow candidates further. The current packaging envelope excludes unsupported specialized classes and entries needing unrepresented technical-name contracts.

Therefore registry sampling does not imply every global code can appear in every template, or that all codes have equal probability. The newest data demonstrates broad sampling within supported contracts. A uniform full-registry draw would violate those compatibility constraints and would also abandon the requested resemblance to real-data distributions.

One qualification: these EDA HS6 counts describe normalized label prefixes, not a certification that every older source code is in today's pinned registry. An additional membership check found zero out-of-registry prefixes in either newer synthetic set. The Egypt set has 12 such prefixes across 255 observations (2.48% of its HS occurrences), also present in the combined set. Some source/historical codes can fall outside a current registry; that alone is not proof of an extraction error. The out-of-registry values are `002023`, `030613`, `080800`, `340213`, `391991`, `731400`, `731500`, `732699`, `840899`, `842190`, `853699`, and `940540`. This investigation did not silently rewrite those samples or equate registry membership with whole-document quality.

New supplementary plots show every sampled code, not only the head: [HS6 full tail](../artifacts/kie-training/analysis/lora-eva-preflight-20260923/hs6-full-tail.png), [UN full tail](../artifacts/kie-training/analysis/lora-eva-preflight-20260923/un-full-tail.png). Detailed counts, registry support, and calibration IDs: [review.json](../artifacts/kie-training/analysis/lora-eva-preflight-20260923/review.json). The original accepted EDA remains unchanged.

## Validation and launch boundary

- 56 targeted tests passed in 28.94 seconds using the actual trainer dependency environment. Tests cover config validation, actual PEFT EVA on a tiny T5Gemma2, both K/V streams even with equal source/target lengths (including different Q versus joint-K/V subspaces), padding, base-weight/no-op preservation, checkpoint evaluator compatibility, timeout cleanup, resume behavior, SVD equivalence, and existing training preparation/contracts.
- Ruff passed on the six changed/new Python files. Mypy passed on training config, EVA, and runtime.
- Independent tiny CPU-model calibration completed in 0.213 seconds: 11 forwards, 14 projections, 801,424 KiB process RSS including imports; zero CUDA allocation. This is integration evidence, not a full-size performance estimate.
- Rebuilt the actual trainer image and verified its installed config/EVA/runtime sources match the workspace byte-for-byte. Its final-config `prepare-dataset` command completed in 32.784 seconds: 30,967 training + 100 validation rows, source maximum 19,194 tokens, target maximum 8,753, configured limits 19,200/8,960, and zero input truncation. Existing training/validation hashes were preserved.
- A model-free replay of the installed Trainer's step accounting and evaluation callbacks confirmed exactly ten scheduled evaluations and matching saves: [schedule receipt](../artifacts/kie-training/analysis/lora-eva-preflight-20260923/schedule-check.json), [reproducible probe](../artifacts/kie-training/analysis/lora-eva-preflight-20260923/check_schedule.py).
- Full-size EVA wall time, peak VRAM, convergence within the selected budget, and downstream training accuracy remain unmeasured. Initialization is additional one-time startup work, with no calibration hooks left during training. No throughput-parity or accuracy-improvement claim is made without that run.

Logs: [tests](../artifacts/kie-training/analysis/lora-eva-preflight-20260923/validation-tests.log), [mypy](../artifacts/kie-training/analysis/lora-eva-preflight-20260923/validation-mypy.log), [dataset preparation](../artifacts/kie-training/analysis/lora-eva-preflight-20260923/dataset-preparation.log), [tiny-model probe](../artifacts/kie-training/analysis/lora-eva-preflight-20260923/tiny-eva-probe.log).

Run from the project root when ready:

```bash
docker compose --profile training run --rm --build kie-trainer train \
  --config configs/training/production/t5gemma2_270m_lora.mpci_bl_real1057_synthetic29910_recovered_v5_e5_eva_a32_v1.yaml \
  --project-root /workspace
```

Expect an explicit EVA calibration phase before optimizer updates. A nonconvergence or coverage error stops before training; there is no silent substitution of random initialization.
