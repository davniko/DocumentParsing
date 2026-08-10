# Document OCR dataset pipeline

This repository builds an immutable, page-level OCR corpus from multi-page PDFs. It is the first
stage of a planned two-model document system:

```text
PDF -> page raster -> GLM-OCR raw page text -> versioned Parquet corpus
    -> later page joining -> T5Gemma 2 270M-270M key-information extraction
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
  enables the MTP head with one speculative token. This repository's executable extraction
  contract is therefore MTP depth 1. The
  [vLLM MTP guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/) documents the
  server configuration and benchmarking direction.
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

Copy one fully explicit example:

```bash
cp configs/glm_ocr.local.example.yaml configs/glm_ocr.yaml
# or
cp configs/glm_ocr.s3.example.yaml configs/glm_ocr.yaml
```

Edit every machine- or dataset-specific value. Local source and output paths must be absolute.
Unknown keys and implicit type coercions are rejected.

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

## Validate and start the GPU service later

The Compose service follows the
[official vLLM Docker deployment shape](https://docs.vllm.ai/en/latest/deployment/docker/) and pins
both the vLLM image digest and GLM-OCR Hugging Face revision. Copy `.env.example` to `.env`, replace
the API key, and validate the Compose structure without printing the resolved configuration or
secret:

```bash
cp .env.example .env
docker compose --env-file .env config --quiet
```

The model and scheduler settings are intentionally literal values in `compose.yaml`:

- `max-model-len=32768`
- `max-num-seqs=16`
- `gpu-memory-utilization=0.90`
- MTP with one speculative token
- one image per request

They must match the extraction YAML. A benchmark variant requires an intentional paired edit to
the server definition and its versioned configuration; environment-variable overrides cannot
silently change this contract. The current strict extraction schema accepts MTP depth 1 only.

Only when the target GPU is free, start the service explicitly:

```bash
docker compose --env-file .env up glm-ocr-vllm
```

That command may pull the pinned image, download the pinned model revision, and allocate the GPU.
The service binds only to loopback. Before extraction, the client requires the configured API-key
environment variable and verifies `/health`, the exact `/version`, the unique served-model alias,
the underlying model repository, and `max_model_len` exposed by `/v1/models`. An authenticated,
hashed `/document-ocr/server-contract` response is derived from the running process's resolved
vLLM state and must exactly attest the model revision, scheduler concurrency, GPU-memory fraction,
generation-config policy, MTP method/depth, one-image limit, and digest-pinned container claim.

## Extract, resume, and inspect

Load the API key into the current shell, then run or resume the immutable run ID in the YAML:

```bash
set -a
. ./.env
set +a
uv run document-ocr run --config configs/glm_ocr.yaml
```

Inspect completion without contacting vLLM:

```bash
uv run document-ocr status --config configs/glm_ocr.yaml
```

Each run owns this structure under the configured output root:

```text
<output-root>/runs/<run-id>/
  inventory.jsonl
  inventory.jsonl.sha256
  run-provenance.json
  state.sqlite3
  raw-responses/<document-id>/<page-id>/<raw-response-sha256>.json
  dataset/pages-<16-char-sha256-prefix>.parquet
  dataset/attempts-<16-char-sha256-prefix>.parquet
  dataset/manifest.json
  scratch/
    sources/
    rasters/
```

The raw-response artifacts hold the exact vLLM response bytes. The pages Parquet contains the
verbatim `raw_ocr_text` plus its hash and `raw_response_path`; request attempts are a separate typed
Parquet dataset. Terminal page/document failures remain in the diagnostic SQLite ledger and prevent
publication, so a completed corpus never contains an always-empty or misleading failure file.
Parquet artifacts are content-addressed and `dataset/manifest.json` is published last, after
reopening and validating every file.

`document_id` identifies an exact source-object occurrence/version, while `source_sha256` identifies
the exact PDF content. Every page row repeats `document_id`, zero-based `page_index`, one-based
`page_number`, and `document_page_count`. Downstream code must group by `document_id` and order by
`page_index`; physical Parquet row adjacency is not a document boundary.

SQLite WAL state is the resume ledger. A page is skipped only after the same extraction identity has
a durable successful record. Every successful request-attempt sequence and its page result commit
in one transaction, eliminating a crash window that could otherwise strand a successful attempt.
Conflicting identities or outputs stop the run. Pending or failed pages prevent the manifest from
being committed, so downstream consumers cannot mistake a partial corpus for a completed one.

## Concurrency

The extraction configuration exposes four independent limits:

- `max_active_documents`: concurrently materialized PDFs and document workers;
- `renderer_processes`: isolated PDFium worker processes;
- `max_inflight_pages_per_document`: per-PDF page workers, enforcing fairness and bounded work;
- `max_inflight_pages_global`: HTTP admission across all PDFs.

Each PDF can have at most the configured per-document number of pages in inference work, while the
global semaphore caps all PDFs together. The supplied values are measured starting points, not a
claim of optimal GPU throughput.

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
speedup.

The command does **not** start or reconfigure vLLM, render PDFs, vary MTP or server scheduler flags,
or collect GPU utilization, VRAM, host CPU, or host RAM. Capture hardware telemetry alongside each
run with the platform's normal monitoring tools. Comparing MTP-off/depth-1/depth-3 or different
vLLM scheduler settings therefore requires separately identified server/config variants; the current
production-oriented extraction contract remains MTP depth 1.

## Development validation

```bash
uv run --frozen pytest
uv run --frozen ruff check .
uv run --frozen mypy src
```

These checks and CPU renderer probes do not invoke the model. GPU inference validation and actual
throughput tuning remain deferred until an operator provides a free GPU window and intentionally
starts the pinned service.
