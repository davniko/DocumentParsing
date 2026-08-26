# Document OCR dataset pipeline

This repository builds an immutable, page-level OCR corpus from multi-page PDFs. It is the first
stage of a planned two-model document system:

```text
PDF -> page raster -> GLM-OCR raw page text -> versioned Parquet corpus
    -> later page joining -> T5Gemma 2 270M key-information extraction
```

The pages dataset contains exactly one successful extraction row per page, with enough source,
renderer, model, request, and code provenance to trace the row to exact PDF bytes and a page within
that PDF. Page joining and text normalization are deliberately later, versioned transforms.

The project decisions extracted from the planning conversation are in
[`docs/project-direction-from-shared-conversation.md`](docs/project-direction-from-shared-conversation.md).

## Safety boundary

No repository setup, configuration validation, source inventory, test, or CPU renderer probe starts
vLLM, downloads GLM-OCR, or uses a GPU. The benchmark command also never starts a server; it only
contacts the endpoint in the selected configuration. A model download and GPU allocation happen
only after an operator explicitly runs `docker compose up`. Do not run that command while other
scheduled GPU work is active.

Real-document access and permission to send document content to any service remain separate
governance gates. The extraction path targets a self-hosted endpoint. AWS and vLLM credentials are
read from the environment and are not written to configuration or published dataset artifacts.

## Why this shape

- [GLM-OCR](https://github.com/zai-org/GLM-OCR) accepts a page image with the exact prompt
  `Text Recognition:` and has a native Multi-Token Prediction (MTP) head.
- The [official vLLM GLM-OCR recipe](https://docs.vllm.ai/projects/recipes/en/stable/GLM/GLM-OCR.html)
  demonstrates one speculative token, while the current
  [GLM-OCR self-hosting example](https://github.com/zai-org/GLM-OCR#deployment) uses three. The
  [vLLM MTP guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/) documents the
  server configuration and benchmarking direction. A controlled target-GPU benchmark selected
  depth 3 for this repository's executable extraction contract; see the
  [remaining-corpus throughput report](artifacts/glm-ocr/benchmarks/remaining-throughput-2026-08-23/REPORT.md).
- vLLM engine batching is not client admission control. The pipeline separately bounds active
  documents, render processes, global inference requests, and inference requests per PDF.
- PDFium work is isolated in processes, with a bounded process-local document cache.
- Prefect is not in the page hot path. It can later orchestrate coarse extraction and training jobs
  without changing the deterministic core pipeline.

## Prerequisites

- `uv`
- Python 3.12 (installed by `uv` if needed)
- Docker Compose and NVIDIA Container Toolkit only when the GPU server is intentionally started
- AWS identity and permissions only for an S3 source

Install the locked CPU pipeline and development tools:

```bash
uv sync --frozen
```

This resolves Python packages only; it does not install vLLM or download the model.

## Configure and inventory

The current NEW/OLD S3 corpus audits, classifier selections, completed local snapshots, combined
BLC corpus, complete classification catalog, and one explicit malformed-PDF quarantine are
recorded in [`docs/s3-input-audit.md`](docs/s3-input-audit.md).

The three classifier lineages and all meaningful NEW/OLD raw PDFs are joined in a deterministic,
document-level catalog. Re-materialization validates the live unversioned S3 inputs against pinned
hashes; verification is fully offline:

```bash
uv run document-ocr-classification-catalog materialize \
  --config configs/classification_catalog.yaml
uv run document-ocr-classification-catalog verify \
  --config configs/classification_catalog.yaml
```

The catalog has 4,035 logical rows: 3,578 retained final classifications, 434 reported exclusions,
8 final-manifest omissions without a drop report, and 15 raw PDFs that were never classified. It
keeps the original page-manifest line locations, S3 aliases, and local content-addressed snapshot
aliases without copying PDF payloads again.

The selected originals are retained locally rather than streamed and deleted. Recreate or resume
that manifest-last snapshot with conditional S3 GETs:

```bash
uv run document-ocr-snapshot materialize \
  --config configs/s3_snapshot.blc_swb.yaml
```

If `snapshot.json` already exists, `materialize` performs the same complete local verification and
does not contact S3. The explicitly offline form is:

```bash
uv run document-ocr-snapshot verify \
  --config configs/s3_snapshot.blc_swb.yaml
```

The completed snapshot retains 2,729 originals. Its extraction trees contain 2,060 BLCs (4,857
pages) and 668 SWBs (2,158 pages); one malformed BLC remains available under `files/` but is
content-pinned and quarantined from extraction.

The curated OLD PDFs and the separate unprocessed-BLC prefix are retained in two additional
manifest-last snapshots:

```bash
uv run document-ocr-snapshot materialize \
  --config configs/s3_snapshot.old_classified.yaml
uv run document-ocr-snapshot materialize \
  --config configs/s3_snapshot.old_blc_unprocessed.yaml
```

The active BLC input is a content-pinned, hard-linked union of all three snapshots. Exact filename
duplicates retain every source alias but occupy one corpus entry:

```bash
uv run document-ocr-corpus materialize --config configs/corpus.blc.local.yaml
uv run document-ocr-corpus verify --config configs/corpus.blc.local.yaml
```

It contains 2,483 BLC PDFs and 5,611 pages. The corpus view does not copy PDF payload bytes.

For the first bounded OCR run, use the catalog-derived 150-document BLC pilot rather than the
provisional combined BLC label tree:

```bash
uv run document-ocr-pilot plan --config configs/pilot.blc150.yaml
uv run document-ocr-pilot materialize --config configs/pilot.blc150.yaml
uv run document-ocr-pilot verify --config configs/pilot.blc150.yaml
uv run document-ocr validate-config --config configs/glm_ocr.blc150.local.yaml
```

The committed pilot contains 150 unique-content PDFs and 286 pages (123,491,086 logical source
bytes). Every row is a locally present, final-label `blc` with explicit `dummy.is_dummy=false`,
`triage.requires_augmentation=false`, `readability=fully_readable`, and
`augmentation_need=use_as_is` evidence. Documents are capped at four pages and stratified across
the three classifier lineages and photo/rendered/scanned categories. The separate pilot directory
is an immutable hard-linked view of the verified snapshot originals, so it adds provenance and a
clean test boundary without duplicating PDF payloads.

Copy one fully explicit example:

```bash
cp configs/glm_ocr.local.example.yaml configs/glm_ocr.yaml
# or
cp configs/glm_ocr.s3.example.yaml configs/glm_ocr.yaml
```

Edit every machine- or dataset-specific value. Local source and output paths must be absolute.
Duplicate or unknown keys, implicit type coercions, non-finite numbers, overlapping local
source/output roots, and unsafe root output paths are rejected.

Validate the file without contacting S3 or vLLM:

```bash
uv run document-ocr validate-config --config configs/glm_ocr.yaml
```

Freeze and inspect the exact source inventory without running OCR:

```bash
uv run document-ocr inventory --config configs/glm_ocr.yaml
```

The inventory command accesses the configured source. For S3, each object must have an exact
`VersionId`; ETags are retained only as metadata. The pipeline also computes a complete SHA-256
from the locally materialized PDF bytes before extraction.

The ready local configurations and their already-frozen inventories are:

```bash
uv run document-ocr inventory --config configs/glm_ocr.blc.local.yaml
uv run document-ocr inventory --config configs/glm_ocr.blc150.local.yaml
uv run document-ocr inventory --config configs/glm_ocr.swb.local.yaml
uv run document-ocr inventory --config configs/glm_ocr.awbc.local.yaml
uv run document-ocr inventory --config configs/glm_ocr.coo.local.yaml
uv run document-ocr inventory --config configs/glm_ocr.inv.local.yaml
uv run document-ocr inventory --config configs/glm_ocr.pl.local.yaml
```

They use separate source roots, run IDs, and output roots, so every document type remains distinct.
Each dataset-version string binds its local snapshot or combined-corpus manifest SHA-256. The OLD
image/GIF/TIFF and XLSX files are not silently passed through the PDF renderer; their separate
input/page semantics must be defined before they are added.

## Validate and start the GPU service later

The Compose service follows the
[official vLLM Docker deployment shape](https://docs.vllm.ai/en/latest/deployment/docker/) and
builds a narrow derived image from a digest-pinned vLLM base. vLLM 0.26.0 has a confirmed
[GLM-OCR MTP prefix regression](https://github.com/vllm-project/vllm/issues/49856), so the build
applies the exact classifier fix from
[vLLM PR #49869](https://github.com/vllm-project/vllm/pull/49869). It also applies the exact fix
from [vLLM PR #51966](https://github.com/vllm-project/vllm/pull/51966) for GLM-OCR MTP's unsafe
boolean-index update during CUDA-graph capture. The Docker build verifies every base target, patch,
patched target, and the build manifest by SHA-256, then runs focused static probes without loading
the model or allocating a GPU. The GLM-OCR Hugging Face revision remains independently pinned.

Copy `.env.example` to the standard project-root `.env`, replace the API key, and validate the
Compose structure without printing the resolved configuration or secret. Compose loads that
`.env` file automatically:

```bash
cp .env.example .env
docker compose config --quiet
docker compose build glm-ocr-vllm
```

The model and scheduler settings are intentionally literal values in `compose.yaml`:

- `max-model-len=32768`
- `max-num-batched-tokens=8192`
- `max-num-seqs=16`
- `gpu-memory-utilization=0.90`
- MTP with three speculative tokens
- one image per request

The service also enables `VLLM_WSL2_ENABLE_PIN_MEMORY=1`. vLLM's V2 model runner requires Unified
Virtual Addressing, while vLLM disables the prerequisite pinned-memory path on WSL2 unless this
supported runtime flag is set.

They must match the extraction YAML. A benchmark variant requires an intentional paired edit to
the server definition and its versioned configuration; environment-variable overrides cannot
silently change this contract. The strict configuration schema accepts only the two explicitly
supported benchmark depths, 1 and 3; the running server contract must still match the selected
configuration exactly.

Only when the target GPU is free, start the service explicitly:

```bash
docker compose up --wait --wait-timeout 3600 glm-ocr-vllm
```

Add `--build` when intentionally rebuilding the derived image; Compose also builds it when it is
missing locally. Startup may pull the pinned base, download the pinned model revision, and allocate
the GPU.
The service binds only to loopback. Before extraction, the client requires the configured API-key
environment variable and verifies `/health`, the exact `/version`, the unique served-model alias,
the underlying model repository, and `max_model_len` exposed by `/v1/models`. An authenticated,
hashed `/document-ocr/server-contract` response is derived from the running process's resolved
vLLM state and must exactly attest the model revision, scheduler concurrency, GPU-memory fraction,
generation-config policy, MTP method/depth, one-image limit, digest-pinned base image, and the baked
build-manifest SHA-256. The endpoint also verifies every installed patched vLLM file against that
manifest before returning a claim.

## Extract, resume, and inspect

Load the API key into the current shell, then run or resume one immutable document-type run:

```bash
set -a
. ./.env
set +a
uv run document-ocr run --config configs/glm_ocr.blc.local.yaml
# After that run completes, or as a separately scheduled run:
uv run document-ocr run --config configs/glm_ocr.swb.local.yaml
```

`run` writes canonical JSONL progress events to stderr after server-contract checking and after
each page and document completion. Events include document counts, page counts, successful and
failed page outcomes for the invocation, elapsed time, pages/second, pages/hour, and an ETA. Set
`run.expected_pages` to the exact frozen-corpus page count to enable the page denominator and ETA;
use an explicit `null` only when the source page count is genuinely unknown. The final result
remains a single JSON object on stdout. Page outcomes and inference attempts are committed
continuously to `state.sqlite3`, so progress is resumable even if the terminal output itself is not
retained. Redirect or tee stderr when an operator log file is required.

The remaining B/L and SWB surface is frozen by
`configs/catalog_selection.blc_swb_remaining.yaml`. Its configured 15-page cap retains 2,164 of
2,174 otherwise eligible documents (99.54%) and 5,351 pages. The ten excluded PDFs contain 452
pages and remain enumerated in the selection artifact rather than being silently dropped. The raw
and table-recognition configurations both consume this same immutable surface, and table
recognition reuses the retained 200-DPI page rasters from raw OCR.

Start or resume the frozen remaining-corpus raw run with:

```bash
docker compose up --wait --wait-timeout 3600 glm-ocr-vllm
set -a
. ./.env
set +a
mkdir -p artifacts/glm-ocr/remaining/blc-swb-max15/operator-logs
set -o pipefail
uv run --frozen document-ocr run \
  --config configs/glm_ocr.blc_swb_remaining.local.yaml \
  2>&1 | tee -a artifacts/glm-ocr/remaining/blc-swb-max15/operator-logs/raw-r2.log
```

After raw OCR publishes successfully, run the page-aligned table view with
`configs/glm_ocr.table.blc_swb_remaining.local.yaml`. The profiling evidence and precision sweep
behind these settings are recorded in
[`artifacts/glm-ocr/benchmarks/remaining-throughput-2026-08-23/REPORT.md`](artifacts/glm-ocr/benchmarks/remaining-throughput-2026-08-23/REPORT.md).

Inspect completion without contacting vLLM:

```bash
uv run document-ocr status --config configs/glm_ocr.blc.local.yaml
uv run document-ocr status --config configs/glm_ocr.swb.local.yaml
```

Each run owns this structure under the configured output root:

```text
<output-root>/runs/<run-id>/
  inventory.jsonl
  inventory.jsonl.sha256
  run-provenance.json
  state.sqlite3
  raw-responses/<document-id>/<page-id>/<raw-response-sha256>.json
  page-images/<document-id>/<page-id>/<raster-sha256>.png
  dataset/pages-<16-char-sha256-prefix>.parquet
  dataset/attempts-<16-char-sha256-prefix>.parquet
  dataset/manifest.json
  scratch/
    sources/
    rasters/
```

The raw-response artifacts hold the exact vLLM response bytes. When
`output.retain_page_images=true`, the exact encoded image sent to vLLM is hard-linked into the
content-addressed `page-images` tree before inference. The pages Parquet contains the verbatim
`raw_ocr_text`, `raw_response_path`, and `raster_path`; publication re-hashes both artifacts and
checks the raster byte count. The same row already repeats the original `source_uri`, canonical
local path/relative key, source SHA-256, document identity, and page position, so each extraction
points to both the original PDF and its exact page image. `dataset/manifest.json` reports aggregate
page-image count/bytes and is published last. Setting retention to `false` leaves `raster_path`
null and deletes the temporary raster after its request.

Request attempts are a separate typed Parquet dataset. Terminal page/document failures remain in
the diagnostic SQLite ledger and prevent publication, so a completed corpus never contains an
always-empty or misleading failure file. Parquet artifacts are content-addressed and are reopened
and validated before the manifest is committed.

`document_id` identifies an exact source-object occurrence/version, while `source_sha256` identifies
the exact PDF content. Every page row repeats `document_id`, zero-based `page_index`, one-based
`page_number`, and `document_page_count`. Downstream code must group by `document_id` and order by
`page_index`; physical Parquet row adjacency is not a document boundary.

SQLite WAL state is the resume ledger. A page is skipped only after the same extraction identity has
a durable successful record. Request-attempt sequences and their resulting page success or failure
commit in one transaction, eliminating crash windows that could otherwise strand request history.
The frozen inventory uses a validated digest commit marker; an inventory-only hard-crash state is
recommitted only after its exact bytes pass typed and canonical validation. An orphaned S3 scratch
PDF is reused only after re-downloading and hashing the same immutable `VersionId`. Every rendered
page must also match the scratch-file identity captured during initial PDF inspection before it can
reach inference. Conflicting identities or outputs stop the run. Pending or failed pages prevent the
manifest from being committed, so downstream consumers cannot mistake a partial corpus for a
completed one.

## Concurrency

The extraction configuration exposes four independent limits:

- `max_active_documents`: concurrently materialized PDFs and document workers;
- `renderer_processes`: isolated PDFium worker processes;
- `max_inflight_pages_per_document`: per-PDF page workers, enforcing fairness and bounded work;
- `max_inflight_pages_global`: HTTP admission across all PDFs.

Each PDF can have at most the configured per-document number of pages in inference work, while the
global semaphore caps all PDFs together. Rendering occurs before that semaphore, so bounded page
workers naturally pre-render while earlier pages are in inference; there is no serial
convert-then-infer bottleneck or unbounded prefetch queue. On this host, a CPU-only probe of 43 real
eligible BLC pages at the configured 200-DPI PNG settings measured 2.0, 3.9, 6.5, and 8.7 pages/s
with 1, 2, 4, and 8 renderer processes. Eight renderers are the pilot starting point for the first
experiment. The page-level `render_duration_ms`, queue time, and
inference time will show whether the full run needs more renderer processes.

## Benchmark a running server

The benchmark isolates the HTTP/model path from PDF rendering by consuming a fixed JSONL manifest
of already-rendered pages. It validates every raster before the sweep and around every request; the
client then re-hashes the exact bytes it base64-encodes for submission. One strict JSON object is
required per line:

```json
{"page_id":"page-0001","document_id":"document-0001","raster_path":"/absolute/path/page-0001.png","mime_type":"image/png","raster_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}
```

Required constraints:

- `raster_path` is an absolute, canonical regular-file path with no symbolic-link traversal;
- `mime_type` is `image/png` or `image/jpeg` and must match the actual encoding;
- `raster_sha256` is the lowercase SHA-256 of the encoded image bytes;
- `page_id` and canonical raster path are each unique; `document_id` may repeat across pages.

With an already-running server and `VLLM_API_KEY` loaded:

```bash
uv run document-ocr benchmark \
  --config configs/glm_ocr.yaml \
  --raster-manifest /absolute/path/benchmark-rasters.jsonl \
  --report /absolute/path/glm-ocr-benchmark.json
```

The YAML `benchmark` block defines warm-up pages, measured pages, repetitions, and global/per-document
concurrency points. The atomic JSON report records wall-clock and request-window pages/s, completion
tokens/s, request latency minimum/mean/p50/p95/p99/maximum, attempts and retries, exact config/server/
raster identities, per-request hashes, and OCR-text hash consistency across every point and
repetition in the sweep. A text-hash change aborts the sweep instead of reporting a misleading
speedup. The benchmark identity also binds the exact benchmark/client source files, relevant
installed package versions, and Python runtime, and the sweep aborts if those source files change
while it is running.

The command does **not** start or reconfigure vLLM, render PDFs, vary MTP or server scheduler flags,
or collect GPU utilization, VRAM, host CPU, or host RAM. Capture hardware telemetry alongside each
run with the platform's normal monitoring tools. Comparing MTP-off/depth-1/depth-3 or different
vLLM scheduler settings therefore requires separately identified server/config variants. vLLM does
not promise bitwise batch-invariant output, so cross-concurrency comparisons use separate fresh
server runs rather than treating output hashes across batch shapes as a correctness oracle. The
current production extraction contract is BF16 weights and KV cache, the measured 2,048-token
scheduler budget, 16 global page requests, and GLM-OCR MTP depth 1. The 2,048/depth-1 and
8,192/depth-3 points were statistically indistinguishable in the controlled BF16 sweep, while the
former is also the exact contract of the completed 893-page run that sustained about 806 pages/hour.
FP8 is deliberately not enabled: on the same 32-page slice it halved throughput, caused three
timeouts/retries, and produced material text omissions.

## Prepare KIE adapter training

The first training path maps one page-ordered raw-OCR document to one compact semantic-v2 JSON
target. It is supervised encoder-decoder teacher forcing with PEFT LoRA updates; it does not use a
chat formatter or TRL `SFTTrainer`. PDFs, page images, evidence sidecars, and downstream MPCI form
scaffolding are not model inputs or decoder targets.

The sourced architecture and acceptance gates are in
[`docs/kie-training-pipeline-spec.md`](docs/kie-training-pipeline-spec.md). The concrete pilot is
fully declared in
[`configs/training/t5gemma2_270m_lora.pilot106.yaml`](configs/training/t5gemma2_270m_lora.pilot106.yaml),
including the exact model/tokenizer commits, source hash/count, prompt, LoRA surface, optimizer,
precision, batching, checkpointing, evaluation, and logging settings. This 106-row config is a
pipeline/pilot experiment, not a production generalization benchmark. A deterministic seed-42
pre-training partition retains every target leaf type in the 90-record training fold and publishes
a disjoint 16-record validation fold for step-scheduled and final generated evaluation. A real run
still needs a larger independently frozen test JSONL, with duplicates and descendants kept in one
fold.
The current partition is a provenance-linked v2 publication: it preserves the original split
membership and replaces 13 explicitly audited ASCII transliterations in five documents with their
cited printed Latin Unicode values. The completed labeling artifact remains untouched.
The pilot disables Transformers' intrusive detailed memory tracker during generation; MLflow keeps
the one-second host/GPU memory and utilization time series.
Generated evaluations log exact `eval_field_value_accuracy`, `eval_field_value_precision`,
`eval_field_value_recall`, and `eval_field_value_f1` metrics; complete scalar values must match and
receive no token-level partial credit. They also log generated-token length, EOS completion, and
PyTorch/driver CUDA peaks so a generation ceiling or memory transition is visible in MLflow.

Install the locked host training group when a compatible local PyTorch/CUDA environment is desired:

```bash
uv sync --frozen --group train
```

The preferred reproducible GPU environment uses the digest-pinned PyTorch 2.13.0 CUDA 13.0 image.
It installs every non-PyTorch dependency from `uv.lock` with hashes and verifies the environment at
build time. Building does not download T5Gemma, start MLflow, or allocate a GPU:

```bash
docker compose --profile training build kie-trainer
```

The CPU-only partition, configuration, and inspection gates neither need `HF_TOKEN` nor import
model weights:

```bash
uv run --frozen document-kie-train split-dataset \
  --config configs/training/mpci_bl_pilot106_split.seed42.yaml \
  --project-root .
uv run --frozen document-kie-train validate-config \
  --config configs/training/t5gemma2_270m_lora.pilot106.yaml
uv run --frozen document-kie-train inspect-dataset \
  --config configs/training/t5gemma2_270m_lora.pilot106.yaml
```

Before the first GPU run, accept the Gemma license for the Hugging Face account, put its read token
in the standard project-root `.env` as `HF_TOKEN`, and run the tokenizer-only preparation gate. It
downloads the exact tokenizer revision, validates all target lengths without truncation, and builds
the cache whose identity binds data, the resolved task-schema prompt, task, tokenizer, versions, and
preprocessing settings:

```bash
docker compose --profile training run --rm kie-tools prepare-dataset \
  --config configs/training/t5gemma2_270m_lora.pilot106.yaml \
  --project-root /workspace
```

Review the reported token P95/P99/max and adjust the provisional source/target ceilings before any
model load. The configured default fails on source overflow and always fails on target overflow;
there is no silent truncation, precision fallback, or automatic batch-size reduction.
Decoder targets are tokenized without tokenizer-added special tokens and receive exactly one
terminal EOS token. BOS, EOS, or PAD occurring in target content is rejected before training.

MLflow is the monitoring and experiment registry. Its digest-pinned server stores run metadata in
SQLite and artifacts in the persistent `mlflow-data` Docker volume. Start it independently when the
UI is wanted without a training process, then open <http://localhost:5000>:

```bash
docker compose --profile training up -d --wait mlflow-server
```

Only in a free GPU window, start the explicit run. Compose starts the MLflow dependency when needed
and waits for its health check before the trainer is created:

```bash
docker compose --profile training run --rm kie-trainer train \
  --config configs/training/t5gemma2_270m_lora.pilot106.yaml \
  --project-root /workspace
```

Progress is visible in the terminal, MLflow, and line-buffered structured JSONL. MLflow captures
Trainer parameters/metrics plus one-second CPU, RAM, disk, network, GPU utilization, GPU memory, and
GPU power samples. `loss` is the average since the previous log event;
`train_cumulative_loss` is the optimizer-step-weighted average over the run so far. Run outputs live
under `artifacts/kie-training/<run_id>/`: frozen config/prompt,
dataset and environment reports, checkpoints, logs, metrics, the immutable MLflow run reference,
optional prediction JSONL, the final adapter and tokenizer, and a completion manifest written last.
On a fresh run, `evaluation.on_start: true` disables PEFT for the epoch-zero base-model baseline,
records `eval_is_base_model=1`, and restores PEFT before the first training microbatch. Resume runs
do not repeat that base-model baseline.
Resume is never guessed: set `checkpoint.resume_from_checkpoint` to an exact checkpoint inside that
run and `logging.mlflow.resume_run_id` to its recorded MLflow run ID. A completed local run cannot
be overwritten, and a resume cannot silently create a second MLflow run.

`docker compose down` preserves MLflow history. Do not use `docker compose down -v` unless deleting
the `mlflow-data` volume and all locally tracked experiments is intentional.

The initial fast path uses BF16, TF32, SDPA, fused AdamW, cached batched tokenization, precomputed
length-grouped sampling, dynamic padding to a multiple of eight, pinned persistent loader workers,
non-reentrant gradient checkpointing, and PyTorch expandable CUDA segments. The allocator setting
is bound by YAML and Compose before PyTorch import; this preserves stable train/eval/train latency
under the materially different allocation shapes of long autoregressive evaluation. The
schema-expanded worst-shape probes reject per-device
batches 4, 6, and 8 and select batch 3. The current pilot uses accumulation 8 (effective batch 24);
the mathematical capacity edge is near 4.3, but batch 4 exhausted device-free memory. The measured
compile path failed to complete one
optimizer step within 314 seconds, so `torch.compile` remains off. The tokenizer, backward, and
controlled throughput/VRAM gates are recorded in the
[`2026-08-18 benchmark report`](docs/kie-training-benchmark-2026-08-18.md); the remaining pre-pilot
gate is a 32–64-example generation memorization check.

New invoice, COO, packing-list, or synthetic lineages use new task schema registrations, prompt
templates, and content-pinned split JSONLs. Each task's current Pydantic target model supplies the
compact sparse JSON Schema injected into its prompt; relation-explicit categorical tasks additionally
require a hash-pinned vocabulary artifact that narrows category fields to exact enums. They reuse the loader, cache, collator,
Trainer, logging, and publication contract without coupling future augmentation generation to
training.

## Prepare the next MPCI B/L experiment

The next experiment remains 270M-only and combines a semantic instructor, an explicit cargo-relation
target, and readable registry-backed package categories. Container type stays printed text/code
because the platform has no stable semantic category registry for its generated accepted codes. The
complete final handoff is in
[`docs/mpci-bl-dual-cargo-pdf-categorical-training-handoff-2026-08-22.md`](docs/mpci-bl-dual-cargo-pdf-categorical-training-handoff-2026-08-22.md).

The auxiliary table path reuses accepted 200-DPI rasters and calls GLM-OCR directly with the exact
`Table Recognition:` task prompt. It never invokes DocLayoutV3. The completed corpus contains
881/881 successful page views across 487 documents; attach its manifest-last output to cloned rows
with:

```bash
uv run --frozen document-ocr-table-view prepare \
  --config configs/glm_ocr.table.mpci_bl_combined487.yaml
uv run --frozen document-ocr-table-view run \
  --config configs/glm_ocr.table.mpci_bl_combined487.yaml
uv run --frozen document-ocr-table-view join \
  --config configs/glm_ocr.table.mpci_bl_combined487.yaml
uv run --frozen document-ocr-table-view verify-join \
  --config configs/glm_ocr.table.mpci_bl_combined487.yaml
```

The clone stores table pages under `auxiliaryViews.glmOcrTableRecognition`. The trainer still reads
only `joinedRawText`, so table content cannot enter a prompt without a later explicit task change.

Publish the exact reviewed source-key map, audit, and run the reversible semantic-v2 to
relation-explicit-v3 transform:

```bash
uv run --frozen document-kie-semantic-v3 categories \
  --config configs/transforms/mpci_bl_semantic_v3_relation_explicit_table_view.yaml \
  --review-config configs/transforms/mpci_bl_semantic_v3_category_review_v1.yaml
uv run --frozen document-kie-semantic-v3 audit \
  --config configs/transforms/mpci_bl_semantic_v3_relation_explicit_table_view.yaml
uv run --frozen document-kie-semantic-v3 transform \
  --config configs/transforms/mpci_bl_semantic_v3_relation_explicit_table_view.yaml
```

The 34 reviewed cargo relations and 204 exact category source keys are hash-bound to the table-view
inventory. Package mapping covers 636/658 occurrences; 22 ambiguous occurrences remain verbatim.
All container values remain printed fallbacks. The transform never edits semantic-v2 and publishes
483 examples plus a runnable, hash-pinned training YAML. Its `task-constraints.json` is mandatory for
relation-explicit training and evaluation, so out-of-vocabulary package categories are rejected
rather than merely pattern-valid.
Relation-explicit evaluation additionally logs identifier-anchored cargo-relation and category
precision/recall/F1 plus document exactness. These facts are compared independently of array
position, while the original strict leaf metrics remain available to expose ordering mistakes.

Install the optional PydanticAI labeling runtime and prepare the deterministic Luna Max cost/quality
sample without making an API call:

```bash
uv sync --frozen --group labeling
uv run --frozen --group labeling document-kie-label-agents prepare \
  --config configs/labeling_agents/mpci_bl_dual_cargo_v3_luna_cost50.yaml
```

The paid `run` action uses `OPENAI_API_KEY` from the standard project-root `.env`, keeps per-response
and aggregate token/cost receipts, performs mandatory independent review, and sends a hash-verified,
requested-page PDF only after an explicit layout-ambiguity request. The PDF can inform grouping but
cannot supply label values absent from raw OCR. Publication also compares outcomes with the hash-pinned
accepted targets without exposing those references to either agent. The Ollama example config
supports an exact operator-supplied local model tag with no automatic cloud fallback.

## Development validation

```bash
uv run --frozen pytest
uv run --frozen ruff check .
uv run --frozen mypy src
```

These checks and CPU renderer probes do not invoke either model. The KIE training image, exact
T5Gemma revision, LoRA surface, worst-case backward path, MLflow lifecycle/system telemetry, and
controlled eager/compile throughput profiles were GPU-validated on 2026-08-18. Starting an OCR or
training workload remains an explicit operator action; repository validation never reserves a GPU.
