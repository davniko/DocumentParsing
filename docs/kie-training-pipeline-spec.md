# Configurable KIE training pipeline specification

Status: implemented and GPU-runtime validated on RTX 4090, 2026-08-18; the explicit small-corpus
memorization-quality gate remains before the first full pilot.

This document defines the first training path from page-ordered GLM-OCR text to a sparse semantic
document label. The first concrete model/task pair is T5Gemma 2 270M-270M with the semantic-v2
Bill-of-Lading target, trained with PEFT LoRA. The boundaries deliberately support later document
tasks and dataset lineages without adding speculative augmentation or full-parameter training code
now.

No command described as a validation or inspection command may load model weights, initialize CUDA,
or train. The GPU smoke test and all throughput tuning remain explicit later gates.

## 1. Current upstream facts

The implementation is based on the following primary sources, checked on 2026-08-18:

- The [Google T5Gemma 2 model card][t5gemma-model] identifies
  `google/t5gemma-2-270m-270m` as an encoder-decoder checkpoint adapted with UL2. It accepts text
  input and emits text, with advertised input/output ceilings of 128K/32K tokens. The repository is
  gated by the Gemma license. The exact revision used here is
  `7c38f16641f455ef0685b18431faf1b17722d5a1`, returned by the Hugging Face model API on
  2026-08-18.
- The [Transformers T5Gemma 2 documentation][t5gemma-transformers] exposes
  `AutoModelForSeq2SeqLM`/`T5Gemma2ForConditionalGeneration` and marks the implementation as
  supporting SDPA and Flash Attention. The model is a multimodal encoder plus text decoder; text-only
  calls do not execute the vision tower, but its base weights remain part of the checkpoint.
- The installed Transformers 5.15.0 implementation names text attention projections `q_proj`,
  `k_proj`, `v_proj`, and `o_proj`, and feed-forward projections `gate_proj`, `up_proj`, and
  `down_proj`. The vision tower has overlapping projection suffixes, so a bare suffix list would
  accidentally adapt vision modules. The initial LoRA regex therefore names only
  `model.encoder.text_model.layers.*` and `model.decoder.layers.*`.
- [PEFT's LoRA reference][peft-lora] defines rank, alpha, dropout, target modules, bias policy,
  rank-stabilized LoRA, and weight tying. [PEFT's Transformers integration][peft-transformers]
  confirms that Trainer updates only adapter parameters and saves adapter-only checkpoints.
- [Transformers Trainer][trainer] and `Seq2SeqTrainer` provide distributed execution, mixed
  precision, checkpointing, generation-based evaluation, logging, and resumption. This task uses
  `Seq2SeqTrainer` because it is a text-to-text model with teacher-forced decoder labels.
- [DataCollatorForSeq2Seq][collator] dynamically pads encoder and decoder batches, pads labels with
  `-100`, and can pad to a Tensor-Core-friendly multiple. Fixed global padding would waste most of
  the compute on this variable-length OCR corpus.
- [Hugging Face Datasets][datasets] uses Arrow-backed data, cached transforms, multiprocessing, and
  memory mapping. Tokenization is therefore a cached preprocessing transform rather than repeated
  work in each training epoch.
- [PyTorch `torch.compile`][torch-compile] supports training and captures backward graphs, but it can
  recompile on changing shapes and has a non-zero warm-up cost. It is exposed as a measured toggle,
  not enabled without a workload benchmark.
- [Transformers' MLflow integration][transformers-mlflow] records Trainer parameters and metrics,
  while [MLflow system metrics][mlflow-system-metrics] adds host and NVIDIA GPU telemetry when
  `nvidia-ml-py` is installed. The runtime owns one MLflow run from startup through final evaluation
  and prediction publication rather than allowing the integration to create an implicit run.
- The [MLflow tracking server][mlflow-tracking-server] is deployed as a digest-pinned service with
  a persistent SQLite backend/artifact volume and localhost-only host exposure. Allowed hosts and
  CORS origins are explicit per [MLflow's network security controls][mlflow-network-security].
- The selected container base is the official
  `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime` image, pinned to
  `sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407`.
  Its published image configuration contains Python 3.12 and PyTorch 2.13.0.

[t5gemma-model]: https://huggingface.co/google/t5gemma-2-270m-270m
[t5gemma-transformers]: https://huggingface.co/docs/transformers/model_doc/t5gemma2
[peft-lora]: https://huggingface.co/docs/peft/package_reference/lora
[peft-transformers]: https://huggingface.co/docs/transformers/peft
[trainer]: https://huggingface.co/docs/transformers/main_classes/trainer
[collator]: https://huggingface.co/docs/transformers/main_classes/data_collator
[datasets]: https://huggingface.co/docs/datasets/process
[torch-compile]: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
[transformers-mlflow]: https://huggingface.co/docs/transformers/main_classes/callback#transformers.integrations.MLflowCallback
[mlflow-system-metrics]: https://mlflow.org/docs/latest/ml/tracking/system-metrics/
[mlflow-tracking-server]: https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/
[mlflow-network-security]: https://mlflow.org/docs/latest/self-hosting/security/network/

## 2. Training contract

One model example is exactly:

```text
encoder input = prompt template with one page-ordered raw-OCR document injected
decoder target = canonical compact JSON for one sparse semantic document patch
```

The source image, PDF, evidence sidecar, warnings, MPCI projection, and application defaults never
enter the target. The input remains the exact `joinedRawText` already published by the label run,
including its explicit page markers. The target remains `target`, serialized with sorted keys,
compact separators, UTF-8 characters, and no null scaffolding.

For the Bill-of-Lading task, every source target is strictly validated as `BillOfLadingLabel`
before tokenization. A later invoice, packing-list, COO, or other task registers a separate target
validator/canonicalizer under a new task name. Dataset loading, prompting, tokenization, collation,
training, logging, and publication stay shared.

## 3. Configuration boundary

Every reproducibility- or performance-relevant value is represented in strict YAML:

- run identity and output directory;
- task and objective;
- base model/tokenizer repository and exact revisions;
- gated-model token environment-variable name and offline policy;
- dtype, attention backend, cache policy, and remote-code policy;
- prompt file and required document/schema placeholders;
- exactly one dataset-input mode: explicit hashed train/validation/test JSONLs, or one hashed
  source JSONL plus a deterministic runtime train/validation partition declaration;
- source record field mapping;
- tokenizer limits, overflow policy, multiprocessing, and cache location;
- LoRA rank, alpha, dropout, bias, rank stabilization, initialization, and exact target regex;
- batch sizes, gradient accumulation, epochs/steps, optimizer, scheduler, learning rate, warm-up,
  weight decay, gradient clipping, label smoothing, and seeds;
- BF16/FP16/TF32, gradient checkpointing, `torch.compile`, and compiler backend/mode;
- data-loader workers, pinning, persistence, prefetch, and length-grouped sampling;
- evaluation cadence/generation settings, checkpoint cadence/retention/resumption, and best-model
  selection;
- structured JSONL and MLflow tracking/server settings, tags, system telemetry, and resume identity.

For a fresh run with `evaluation.on_start: true`, epoch-zero evaluation is an explicit base-model
baseline: the runtime disables the PEFT adapter for the complete generated evaluation, logs
`eval_is_base_model=1`, and restores the adapter before the first training microbatch. A resumed
run does not repeat this baseline because its on-start state is a trained checkpoint, not the
original base model.

Unknown keys, YAML duplicate keys, implicit coercions, non-finite numbers, unsafe output paths,
incompatible strategies, invalid hashes, overlapping IDs, hash mismatches, schema-invalid targets,
and target overflow are errors. Source overflow is also an error unless truncation is explicitly
selected. There is no automatic batch-size reduction, precision fallback, target truncation, or
model revision drift.

The first pilot configuration uses a deterministic seed-42 partition of a provenance-linked
correction publication derived from the published 106-row artifact: 90 training records and 16
held-out validation records. Thirteen explicitly documented ASCII transliterations across five
documents are restored to their OCR-printed Latin Unicode values without mutating the original
labeling publication or changing split membership. The partition refuses exact input
duplicates and skips a validation candidate if selecting it would remove the last training example
of any target leaf path. Production configs must additionally provide an independently frozen test
split; descendants and non-exact duplicate groups must remain in one fold upstream.

Runtime-partitioned configs use the same content-pinned source and coverage policy without first
publishing derivative JSONLs. They declare `seeded_sha256_rank_v1`, a non-negative seed, and a
validation size as either an integer record count or a fraction with explicit `half_up` rounding.
Inspection validates every source row before selecting membership, rejects duplicate document or
input identities, preserves source order within both folds, and reports every selected ID,
membership hash, resolved count, and coverage-skipped candidate. Valid empty target collections
are coverage leaves; null target values remain forbidden. The resolved membership participates in
the tokenized-cache identity and immutable run report. A resumed run fails if recomputation differs
from that report.

## 4. Prompt contract

A prompt is a UTF-8 text file containing exactly one `{{document_text}}` placeholder and one
`{{output_schema}}` placeholder, with no other template expressions. The registered task derives a
compact sparse JSON Schema from its current Pydantic target model and binds it before document-text
injection. Tasks with registry-owned categoricals additionally bind a canonical, hash-pinned
task-constraints artifact; its exact category tokens become schema enums and the canonical target
validator rejects out-of-vocabulary values. Relation-explicit B/L training refuses to start without
that artifact. Rendering is literal replacement, not Jinja or Python formatting. The resolved
prompt hash therefore changes with template wording, target schema, or bound vocabulary and
participates in the tokenized-cache identity.

The initial B/L prompt shows that schema and requests only compact JSON grounded in the supplied
OCR. It reiterates that missing values must be omitted, printed country/locality text must remain
literal, pages belong to one document, addresses remain one logical value, flavor text is excluded,
and no explanation or Markdown is allowed.

## 5. Dataset preparation and collation

The loader scans every explicit split JSONL, or the single runtime-partition source, once and
validates:

1. the whole-file SHA-256 and declared record count;
2. unique non-empty document IDs across all splits;
3. non-empty input text and, when configured, its stored SHA-256;
4. object-valued, null-free targets;
5. task-specific strict schema validity and canonical equality.

It then builds Arrow datasets containing only normalized strings and provenance fields. A batched,
optionally multiprocess tokenization transform emits `input_ids`, `attention_mask`, `labels`,
`input_length`, `input_original_length`, `source_truncated`, and `target_length`. The cache identity
binds source hashes, field mapping, resolved partition membership when applicable, task,
prompt hash, tokenizer identity/revision, preprocessing settings, and the decoder-target contract.
Source tokenization uses its explicitly configured special-token behavior. Decoder targets are
tokenized with tokenizer special tokens disabled, are rejected if their content resolves to a
reserved BOS/EOS/PAD ID, and receive exactly one terminal EOS. Target length checks include that
EOS. Cached files are reused only under that exact identity.

The collator dynamically pads each batch and uses `-100` for label padding; the model creates
shifted decoder inputs and supplies decoder BOS itself. Training samples are
grouped approximately by input length to reduce padding. The report records length percentiles and
the amount of padding can later be measured from structured logs before changing bucket strategy.

## 6. Model and LoRA construction

The runtime loads the configured tokenizer and `AutoModelForSeq2SeqLM` from exact revisions with
`trust_remote_code=False`. No training `device_map="auto"` is used; Trainer/Accelerate owns device
placement. The model must report `is_encoder_decoder=True` and a matching configured model type.

The LoRA adapter uses PEFT task type `SEQ_2_SEQ_LM`. After insertion, the runtime enumerates every
adapted module and fails unless:

- at least one module was adapted;
- every adapted module matches the configured regex; and
- no adapted module lives under `vision_tower` or `multi_modal_projector`.

Base weights remain frozen. The run metadata records total/trainable parameters and their ratio.
The default target surface includes all seven attention/MLP projections in both text encoder and
decoder. This is a baseline capacity choice, not a claim of optimal rank or target surface.

## 7. Performance policy

The initial fast path uses SDPA, BF16, TF32, fused AdamW, dynamic padding to a multiple of eight,
pinned/persistent loader workers, cached Arrow tokenization, and length-grouped sampling. These
settings are subject to runtime hardware checks and fail rather than degrade silently.

Gradient checkpointing is enabled with non-reentrant checkpointing in the initial long-input
configuration to lower activation memory. It trades compute for memory and must be benchmarked
against the disabled case once the GPU is available. `torch.compile` starts disabled for the same
reason: variable sequence shapes and PEFT wrappers can erase its benefit through compilation and
recompilation overhead. The config supports `inductor` modes for an explicit later A/B run.

The GPU benchmark matrix should hold data, seed, effective global batch size, and evaluation cadence
fixed while varying only:

1. gradient checkpointing on/off;
2. compile off versus `inductor` default/reduce-overhead;
3. data-loader worker/prefetch settings;
4. maximum source length after the real tokenizer profile;
5. LoRA target surface/rank only as a quality-capacity experiment, not a throughput-only toggle.

Record examples/s, non-padding tokens/s, step p50/p95, peak allocated/reserved VRAM, host RAM, data
wait time, compile warm-up, and validation metrics. A setting is promoted only from measured steady
state and must not regress target completeness.

## 8. Evaluation and logging

Generated evaluation output is scored with:

- canonical JSON document exact match;
- JSON parse validity;
- task-schema validity;
- micro exact extracted field-value accuracy, precision, recall, and F1;
- for relation-explicit tasks, micro cargo-relation precision, recall, and F1 plus per-document
  relation exact match and support fraction; and
- for registry-backed categorical tasks, identifier-anchored category precision, recall, and F1
  plus per-document category exact match and support fraction.

An extracted field-value is one scalar leaf below `documentPatch`, identified by its full JSON
path (including list indexes) and compared as one canonical JSON value. It receives credit only
when both its path and complete value match exactly; token overlap receives no partial credit.
`schemaVersion` is structural metadata and is excluded. Field-value accuracy is exact matches over
the union of predicted and reference field paths, while precision and recall use predicted and
reference field-value counts respectively. Canonical JSON exact match remains the stricter
whole-document score. A parseable `documentPatch` retains per-field partial credit even when an
unrelated value makes the overall prediction schema-invalid; JSON and schema validity remain
separate metrics rather than silently erasing correctly extracted fields.

The relation-explicit metrics project package membership, allocation coverage, package-scope links,
container membership, allocation quantities, and direct container/package links into document-local
graph facts keyed by `groupId`, `packageId`, and container number. Category metrics key package
categories by group/package identity and container categories by container number. Both are set
comparisons, so a semantically identical array permutation does not masquerade as a relationship
error. Strict leaf/path F1 remains index-sensitive and is reported alongside them.

Loss is still logged, but loss alone is insufficient for structured extraction. Optional final
prediction JSONL files retain document ID, generated text, reference text, parse/schema status, and
exact-match status for selected splits.

The pilot evaluates and checkpoints on the same optimizer steps, selects the best checkpoint by
exact field-value F1, and uses YAML-configured early stopping. When final validation predictions are
requested, one generated prediction pass both scores the reloaded best checkpoint and publishes
the per-document JSONL; metrics from a later stopping checkpoint are never mislabeled as best-model
metrics, and validation generation is not duplicated.

The interval `loss` emitted by Trainer covers only optimizer steps since the preceding log. The
runtime additionally logs `train_cumulative_loss`, reconstructed as an optimizer-step-weighted
average over all interval losses through the current global step. Generated evaluation logs mean
and maximum output-token counts, EOS completion fraction, PyTorch allocated/reserved peaks, and
CUDA driver residency before and after evaluation cleanup.

Trainer logs go to normal progress output, an append-only structured `events.jsonl` file, and one
runtime-owned MLflow run. Automatic Trainer integration construction is disabled; the runtime
creates the MLflow run first, binds the explicit Transformers callback to it, and keeps it active
through training, final evaluation, and prediction publication. Startup verifies the tracking
server before model work. Completion checks the terminal MLflow status, failures are marked
`FAILED`, and checkpoint resume requires the exact immutable MLflow run ID so history cannot fork
silently.

MLflow logs flattened Trainer parameters, scalar train/eval metrics, configured provenance tags,
and one-second host/GPU system metrics. The server listens only on `127.0.0.1:5000` at the host,
uses an explicit allowed-host/CORS policy, and persists its SQLite database and proxied artifacts in
the `mlflow-data` named volume. Heavy checkpoint/model artifact upload is disabled by default because
the local manifest already content-addresses those files; enabling it is an explicit YAML decision.

Generated evaluation requires `runtime.skip_memory_metrics: true`. In the pinned Transformers
runtime, the detailed Trainer memory tracker busy-polls process RSS in a no-sleep thread. That
thread contends with the Python-driven autoregressive decoding loop and caused a measured 10.1x
regression on the longest validation batch. MLflow system metrics remain enabled at a one-second
interval, so disabling the Trainer tracker removes duplicate intrusive polling without losing the
host/GPU memory time series.

The pilot binds `runtime.cuda_allocator_conf: expandable_segments:True` before PyTorch import and
rejects a conflicting process environment. Autoregressive evaluation and checkpointed training
have different allocation shapes; the native allocator's independent fixed segments fragmented
under their transition on WSL and drove later optimizer steps into GPU residency paging. Expandable
segments keep the allocator's virtual range mergeable while physical pages are mapped on demand.
The exact train/eval/train acceptance benchmark must still pass whenever the model, sequence
ceilings, batch sizes, or CUDA/PyTorch stack changes.

In pinned Transformers 5.15, a fractional warm-up is represented by the float form of
`warmup_steps`; the runtime maps YAML `warmup_ratio` to that field. Local run artifacts include the
frozen YAML, prompt, source hashes, environment/package versions, model/adapter facts,
dataset/token-length report, train/eval metrics, checkpoints, immutable `mlflow-run.json`, final
adapter, tokenizer, predictions, and a manifest published last.

## 9. Extension seams and deliberate non-features

Concrete extension points are:

- add a task entry with a Pydantic target model/canonicalizer;
- provide a new prompt file;
- provide explicit hashed split JSONLs, or one hashed source with a runtime partition, and field
  mapping;
- later add another tuning strategy beside `lora` while retaining the same data contract;
- later add synthetic JSONLs carrying parent IDs, operations, seeds, and generator versions before
  they enter frozen splits.

Not implemented now:

- synthetic/augmentation generation;
- full-parameter fine-tuning;
- QLoRA/bitsandbytes;
- TRL chat-style `SFTTrainer`;
- automatic test-set construction;
- automatic truncation or batch-size search;
- removal/offload of the unused T5Gemma vision tower;
- automatic model merge or deployment.

Those are separate measured changes. In particular, the multimodal base includes a vision tower
that is unused by text-only forward calls but still occupies base-weight memory. Removing it could
save memory, but the upstream API does not expose a supported text-only checkpoint load. That
optimization remains deferred until it can be tested on the real checkpoint and adapter
save/resume path.

## 10. GPU acceptance sequence and current evidence

The acceptance sequence is:

1. Accept the Gemma license for the intended Hugging Face account and expose `HF_TOKEN` only at
   runtime.
2. Run configuration and dataset inspection; confirm exact hashes and split counts.
3. Run tokenizer preparation/profile and set source/target limits from observed P95/P99/max lengths.
4. Run one forward/backward LoRA smoke batch and verify non-zero finite gradients only on intended
   text adapters.
5. Run a 32-64-example memorization experiment. Failure to approach near-perfect training
   extraction blocks scaling and triggers pipeline/label/truncation diagnosis.
6. Benchmark the optimization matrix above.
7. Only then start a larger real-data run with a reproducible validation partition and an
   independently frozen test split.

Steps 1-4 and the controlled runtime benchmark in step 6 were executed on 2026-08-18 on an RTX
4090. The exact tokenizer profile across all 106 rows observed source max 4,364 and target max
1,441, so the configured 4,608/1,536 caps preserve every token with no truncation. The exact model
revision loaded as an encoder-decoder model with 793,623,280 total parameters and 7,593,984 trainable
LoRA parameters across 252 intended text modules; the vision/projector surface remained excluded.
A worst-case 4,364-token forward/backward smoke step completed with finite training output.

The compile trial produced no optimizer step after 314 seconds of CPU-side compilation and was
intentionally terminated; its MLflow run is correctly marked `FAILED`. Eager execution with
non-reentrant gradient checkpointing is therefore the selected pilot path. Exact full-epoch
throughput and device telemetry are recorded in the
[dated benchmark report](kie-training-benchmark-2026-08-18.md). Step 5 remains a
quality gate: infrastructure validation and falling teacher-forced loss are not a substitute for a
32-64-example generation memorization check.

The 106-document pilot is sufficient for pipeline/memorization checks, not a production-quality or
generalization claim.

## 11. Implemented components and fixed invariants

The implementation is split by responsibility:

- `training/config.py`: strict duplicate-key-rejecting YAML and cross-field validation;
- `training/splitting.py`: deterministic coverage-guarded train/validation selection and optional
  standalone publication;
- `training/prompting.py`: immutable UTF-8 prompt loading and literal injection;
- `training/tasks.py`: task registry and strict Pydantic target canonicalization;
- `training/data.py`: content verification, normalized examples, Arrow caching, and token limits;
- `training/collator.py`: cached-length sampler metadata removal before model forward;
- `training/metrics.py`: JSON/schema/exact/leaf metrics and prediction assessments;
- `training/runtime.py`: hardware checks, exact model loading, proven LoRA insertion, Trainer,
  structured progress, explicit resume, and manifest-last publication;
- `training/cli.py`: separate validate, inspect, tokenizer-prepare, and train commands;
- `docker/training/`: digest-pinned, lock-verified CUDA environment;
- `prompts/kie/` and `configs/training/`: task prompt and complete pilot declaration.

The first implementation is deliberately single-process/single-GPU. The CPU-only `kie-tools`
Compose service cannot reserve a GPU; `kie-trainer` explicitly reserves one. Multi-GPU execution is
not implied by Trainer support and requires a separately designed publication/barrier contract.

The following are fixed safety/contract invariants rather than hidden hyperparameters: safetensors
model loading, no remote code, no Hub push, no automatic batch-size search, no target truncation,
no sampled evaluation generation, no model `device_map`, and no Trainer unused-column removal.
The last item keeps precomputed lengths available to the length-grouped sampler; the dedicated
collator removes document IDs and length metadata before `model.forward`.

Every resume checkpoint must be named explicitly in YAML and live inside the run's checkpoint
directory. The initial training contract remains immutable; the resume pointer is the only field
allowed to differ, and each resume YAML is retained under `resume-invocations/`. A completed
manifest prevents any overwrite or further resume.

Current CPU evidence is intentionally narrower than GPU acceptance: all 106 records pass file,
record, OCR hash, uniqueness, sparse-null, and strict target-schema checks; the real PEFT library
adapts exactly the 14 intended projections on a one-layer in-memory T5Gemma2 probe and a CPU
forward/backward produces gradients only on adapter parameters. The actual checkpoint, tokenizer,
CUDA memory, throughput, and quality have not yet been exercised.
