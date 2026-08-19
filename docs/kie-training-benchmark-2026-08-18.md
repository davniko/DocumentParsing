# T5Gemma 2 KIE training acceptance and benchmark — 2026-08-18

This report records the first real-GPU validation of the Bill-of-Lading semantic-v2 LoRA pipeline.
It is an infrastructure/performance acceptance result, not a model-quality or generalization claim.
All runs used the same frozen 106-document OCR-conditioned dataset and exact model/tokenizer commit.
The exercised microbatch sizes were 1 (smoke), 3, and 4; no batch size above 4 was tested.

## Fixed test surface

- Hardware: NVIDIA GeForce RTX 4090, 24,564 MiB, driver 581.57, compute capability 8.9.
- Container: digest-pinned `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime`.
- Runtime: PyTorch 2.13.0+cu130, Transformers 5.15.0, BF16, TF32, SDPA, fused AdamW.
- Model: `google/t5gemma-2-270m-270m` at
  `7c38f16641f455ef0685b18431faf1b17722d5a1`.
- Parameters: 793,623,280 total; 7,593,984 trainable LoRA parameters (0.9569%).
- LoRA surface: 252 text encoder/decoder attention and MLP modules; vision/projector excluded.
- Dataset: 106 documents; 179,169 non-padding source tokens in one epoch.
- Tokenizer profile: source max 4,364, P95 3,274, P99 4,071; target max 1,441,
  P95 777, P99 977. Configured 4,608/1,536 limits truncate zero records.
- MLflow: 3.15.1 tracking server with one-second host/NVIDIA system telemetry.

## Runtime results

| Profile | Result | Runtime | Docs/s | Input tokens/s | PyTorch training peak delta | MLflow GPU utilization | MLflow max GPU memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| Eager, checkpointing, batch 4, accumulation 2 | 106 docs / 14 updates | 233.923 s | 0.453 | 765.944 | 18,876,853,760 B | 43.94% mean / 100% max | 99.6% |
| Eager, checkpointing, batch 3, accumulation 3 | 106 docs / 12 updates | 249.686 s | 0.433 | 717.588 | 14,095,606,784 B | 20.59% mean / 100% max | 99.7% |
| Inductor `reduce-overhead`, checkpointing, batch 4 | no optimizer step | >314 s before interrupt | — | — | — | 9.32% mean | 48.1% |

The batch-4 profile completed the same epoch 6.31% faster and processed 6.74% more non-padding
tokens per second than batch 3. Its measured model allocation plus Trainer peak was approximately
19.16 GiB, leaving approximately 4.83 GiB of allocation headroom on this card. NVIDIA's device-level
reading approached the full card in both runs because the CUDA allocator retained reusable cache;
the lower batch still reached 99.7%, so that number alone does not represent live tensor allocation.

The compile run spent more than 314 seconds compiling on the CPU without completing one optimizer
step. It was deliberately interrupted through the normal failure path, and MLflow recorded terminal
status `FAILED`. This workload has too few updates to amortize that startup cost; `torch_compile`
therefore remains disabled.

## Checkpointing isolation probe

A warmed manual forward/backward microbenchmark used the four longest source documents
([4,364, 4,071, 3,714, 3,401] tokens) to isolate activation checkpointing:

| Gradient checkpointing | Mean batch time | Docs/s | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|
| enabled, non-reentrant | 1.661 s | 2.409 | 13,426,907,648 B | 15,743,320,064 B |
| disabled | 74.345 s | 0.0538 | 43,408,560,128 B | 45,600,473,088 B |

Without checkpointing, the workload oversubscribed the physical GPU and paged through WSL unified
memory. The resulting 44.8x latency regression makes non-reentrant checkpointing mandatory for this
long-input RTX 4090 profile.

## MLflow acceptance evidence

- One worst-case 4,364-token smoke update completed and its MLflow run finished successfully.
- The full batch-4 and batch-3 runs both finished successfully and retained 229 and 244 system
  metric samples respectively.
- Metrics include Trainer loss/learning rate/gradient norm/tokens/runtime plus CPU, RAM, disk,
  network, GPU utilization, GPU memory, and GPU power.
- Runtime-generated provenance tags bind the exact run ID, task, dataset-cache identity, and model
  revision. Each local completion manifest embeds the immutable MLflow experiment/run reference.
- Failure handling was exercised by the interrupted compile trial; local status and MLflow both
  recorded failure instead of publishing a completed manifest.

MLflow run IDs:

- full epoch, batch 4: `740b637e55a6445ab34d641c8c2ab559`
- full epoch, batch 3: `aeab4d7d12c84fde958237b3adf8e85b`
- compile trial, intentionally failed: `9b4fb4b95b834553be9ebbf26bc64560`
- worst-case smoke: `857c25e5a9024511b38862592edcace2`

## Original pre-schema pilot profile

Before schema injection, batch 4 with accumulation 2 was the fastest measured profile. Batch 3 with
accumulation 3 cost 6.31% in epoch time while reducing live allocation. The schema-prompt follow-up
below supersedes this selection because it materially changes source lengths and activation memory.

The remaining gate is model-quality acceptance: overfit a frozen 32–64-document slice and evaluate
generated structured outputs against the same references. Falling teacher-forced loss during these
throughput runs proves optimization is active, but it does not prove generation memorization.

## Generated-evaluation regression and fix

The first epoch-end generated evaluation exposed a separate Trainer-side bottleneck. With
`runtime.skip_memory_metrics: false`, the Transformers detailed memory tracker ran a daemon thread
that called `psutil.Process.memory_info()` continuously without sleeping. The training kernels were
large enough to hide most of that contention, but the Python-driven autoregressive loop was starved.
The exact longest validation pair needed 606 seconds to reach its first batch boundary.

An isolated exact-Trainer A/B forced the same two examples to 512 generated tokens:

| Trainer detailed memory tracker | Runtime | Relative |
|---|---:|---:|
| enabled | 200.081 s | 1.0x |
| disabled | 19.747 s | 10.13x faster |

With the tracker disabled, a full 1,535-token decode of that pair completed in 54.575 seconds.
MLflow's one-second process/GPU telemetry remains enabled, so the fix removes duplicate intrusive
polling rather than removing operational memory observability. Configuration validation now rejects
generated evaluation when the detailed Trainer tracker is enabled.

The fixed path was then profiled at larger evaluation batches on the maximum 5,267-token validation
source and maximum 865-token validation target:

| Eval batch | 256-token docs/s | Peak allocated | Full 1,535-token result |
|---:|---:|---:|---:|
| 2 | 0.204 | 5.81 GiB | measured separately at 54.575 s/pair |
| 4 | 0.437 | 10.09 GiB | not required after batch 8 passed |
| 8 | 0.764 | 18.63 GiB | 52.428 s for eight documents |

The batch-8 full-ceiling probe reserved 20.81 GiB and completed without unified-memory
oversubscription. The pilot therefore uses evaluation batch 8; this selection is specific to the
fixed 16-record validation surface and does not change the separately measured training microbatch
of 3.

## Schema-prompt and seeded-split follow-up

The later prompt revision injects the current compact sparse JSON Schema derived from the Pydantic
target model. A seed-42, coverage-guarded pre-training partition now yields 90 training and 16
held-out validation documents. The exact tokenizer profile increased the source maximum from 4,364
to 6,043 tokens; the 6,400-token ceiling still truncates zero records.

Because this changes the activation surface, batches 3, 4, 6, and 8 were probed again on a
deliberately conservative batch containing both the 6,043-token encoder maximum and the
1,441-token decoder maximum:

| Per-device batch | Step time | Peak allocated | Tracked headroom | Device free after | Result |
|---:|---:|---:|---:|---:|---|
| 3 | 3.316 s | 18.468 GB | 7.289 GB | 3.925 GB | safe |
| 4 | 85.511 s | 24.079 GB | 1.678 GB | 0 GB | spill-prone |
| 6 | 16.990 s | 35.295 GB | -9.538 GB | 0 GB | oversubscribed |
| 8 | 91.748 s | 46.515 GB | -20.758 GB | 0 GB | oversubscribed |

The fixed allocation was approximately 1.64 GB and the worst-shape slope approximately 5.61 GB per
sample, placing the mathematical device-capacity boundary near batch 4.3. CUDA workspaces, allocator
cache, and the display process make batch 4 operationally unsafe despite its positive tracked-only
margin. The current pilot therefore uses batch 3. The completed v3 experiment used accumulation 8
(effective batch 24), which remains the v4 configuration; batch 4 remains only a historical
pre-schema throughput result.

## In-training generated-evaluation allocator regression

The first 25-epoch pilot exposed a separate WSL residency failure after generated evaluation. The
first 40 optimizer steps ran in roughly 2–4 seconds each and batch-8 evaluation completed in
117.6 seconds. After evaluation, five-step windows expanded first to 129 seconds and then to 409
seconds. MLflow showed roughly 25.5 GB resident, and the WSL kernel recorded
`dxgkio_make_resident: Ioctl failed: -12`. CPU utilization remained about 3–4%, while the GPU stayed
near 100% utilization at unusually low power: the process was paging GPU residency rather than
waiting on the dataloader.

Reducing evaluation batch size did not fix the transition. Full 16-record train/eval/train probes
at batches 4, 3, and 2 all generated to the 1,536-token ceiling, subsequently filled the card, and
emitted a new residency ENOMEM. A matched nine-step control with generated evaluation disabled
completed in 28.0 seconds; steps 8 and 9 remained 2.7 and 3.0 seconds. This isolates the regression
to allocation-order fragmentation across autoregressive evaluation and resumed training, not to
later sampler batches.

PyTorch expandable CUDA segments were then bound before PyTorch import. The full batch-4 probe
completed evaluation in 229.1 seconds and kept all four following steps between 2.7 and 3.8 seconds.
The final batch-8 probe was both stable and faster:

| Measurement | Batch-8 expandable result |
|---|---:|
| Validation documents / batches | 16 / 2 |
| Generated evaluation runtime | 119.686 s |
| Generated tokens, mean / max | 1,536 / 1,536 |
| EOS completion fraction | 0.0 |
| Peak PyTorch allocated / reserved | 18.770 / 18.961 GiB |
| Post-eval step durations | 2.381, 3.338, 2.363, 3.867 s |
| New WSL residency errors | 0 |

The CUDA 13 `cudaMallocAsync` backend was also measured rather than dismissed by assumption:

| Allocator | Nine-step training-only | Batch-8 evaluation | Mixed run total | Transition |
|---|---:|---:|---:|---|
| native | 27.997 s | 117.558 s in the pilot | did not remain tractable | residency ENOMEM |
| expandable segments | 29.388 s | 119.686 s | 148.749 s | stable |
| `cudaMallocAsync` | 28.707 s | 124.748 s | 153.869 s | stable |

`cudaMallocAsync` reduced the training-only allocator cost but made generation slow enough that the
mixed workload was 5.12 seconds (3.44%) slower than expandable segments. Expandable segments are
therefore the measured end-to-end selection, not merely the first allocator that avoided failure.

The 0% EOS result explains the early-run evaluation latency: every prediction paid the full
autoregressive ceiling. A post-run audit established that this was not merely undertraining: all
106 old labels contained BOS and none contained EOS. The v4 decoder contract instead tokenizes
target content without tokenizer special tokens, forbids reserved IDs in content, and appends
exactly one EOS. Batch 8 is retained because it is twice as fast as batch 4 on the fixed validation
set and passed the complete post-evaluation training transition. The allocator mode is part of the
strict YAML/environment contract and is recorded in `environment.json`.

MLflow evidence:

- no-evaluation control: `3774559660a84258b88dea83e20f024d`
- expandable batch 4: `0bbf264cc16a4db4bcff441b51ee3fc4`
- expandable batch 8: `5857534f4d274318a4b8c206c3d1042b`
- `cudaMallocAsync` training-only: `dfaa97f4935d4252ac61a79a85e8c0fd`
- `cudaMallocAsync` batch 8: `127cce541012435cbbb8a854803b852f`

## Exact field-value metric pass

The original structured evaluator already compared complete scalar leaves, but it included the
constant `schemaVersion`, exposed no accuracy score, and discarded all otherwise correct fields
when any value made a parseable prediction schema-invalid. The revised metrics operate only on
scalar paths below `documentPatch`. A path and its complete canonical JSON value must both match;
wrong, missing, and extra values receive no token-level partial credit. Accuracy uses exact matches
over the union of predicted and reference field paths, while precision and recall use predicted and
reference field-value counts.

On the frozen 16-document validation targets, 100 complete metric passes measured 6.279 ms/pass
before and 6.362 ms/pass after the additional accuracy and independent partial-credit accounting.
The 0.083 ms absolute delta is negligible beside the measured 52.428-second generated-evaluation
batch and does not add model, GPU, or retained-array memory.

## Repository and container verification

- `pytest`: 452 passed in 62.37 seconds after the allocator/telemetry pass.
- Ruff: all checks passed.
- mypy strict mode: no issues in the three changed training modules. The repository-wide command
  reports one pre-existing unused `import-not-found` suppression in `vllm_contract.py` because the
  training dependency group supplies Starlette; no training module errors are present.
- Compose resolution: valid with the `training` profile.
- CPU dataset inspection: all 106 split records schema/hash-valid at 1,568.35 records/s.
- Rebuilt training image: environment verifier passed and `python -m pip check` reported no broken
  requirements.
- MLflow server: healthy on the localhost-only endpoint, with successful client reads of every run
  and metric history cited above.
