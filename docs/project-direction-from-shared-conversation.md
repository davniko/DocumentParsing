# Project direction extracted from the shared planning conversation

Status: working project brief  
Source reviewed: [AI MPCI Project Strategy](https://chatgpt.com/share/6a72e2ee-39b8-83eb-874d-f9f394c0e805)  
Conversation reviewed in full: 2026-08-05

This document records the project-relevant conclusions from the shared conversation. It
separates settled direction from earlier hypotheses so that superseded ideas do not quietly
become implementation requirements. It deliberately excludes credentials and attached source
documents.

## 1. Problem and product context

- The product is an AI-assisted MPCI/CUSCAR filing workflow for UAE imports. Users upload Bills
  of Lading (B/Ls), and the system should populate a complex, nested filing form.
- The previous MVP used a fine-tuned DONUT model. It handled only the first page and reached
  approximately 0.85 field/tag F1 under the historical evaluation definition.
- The previous serving path was highly optimized: TensorRT, Triton, a C++ autoregressive loop,
  constrained XML decoding, custom CUDA work, field uncertainty signals, RabbitMQ integration,
  and an `ai-genie` semantic mapping layer.
- That image-to-schema approach depended on a document corpus whose training use later became a
  contractual/legal problem. The replacement architecture therefore separates general OCR from
  the company-specific semantic extraction task.
- The current planning premise is that approximately 3,500 documents have explicit approval for
  hypothesis testing and validation. That approval must not be assumed to cover every later use,
  external API transfer, derived dataset, or production training run.

## 2. Operational constraints

- The intended production hardware is currently a single 24 GB NVIDIA GPU: an AWS G5 A10G or G6
  L4 class instance.
- OCR and the downstream extractor may need to coexist on that GPU, so small models and bounded
  memory use matter.
- A few seconds per document is acceptable; correctness and completeness matter more than
  matching the old approximately 0.5 second single-page latency.
- The established inference boundary is Triton. The first research pipeline can call a separate
  vLLM service, but deployment artifacts must remain compatible with a later Triton-orchestrated
  system.
- Multi-page documents are a first-class requirement.
- The project uses `uv` for Python dependency and lockfile management. GPU serving is isolated in
  a container so CUDA/vLLM dependencies do not contaminate the CPU data pipeline.

## 3. Selected two-model architecture

The working architecture is:

```text
multi-page PDF
  -> deterministic page rasterization
  -> GLM-OCR whole-page text recognition, one request per page
  -> immutable raw page extraction dataset
  -> deterministic page joining/preprocessing for training
  -> one fine-tuned T5Gemma 2 270M-270M extractor
  -> sparse, canonical MPCI document patch in JSON
  -> narrow deterministic validation/assembly layer
  -> complete platform form / CUSCAR submission path
```

The responsibilities are deliberately separated:

- GLM-OCR transcribes each page.
- T5Gemma 2 performs document-level semantic extraction and mapping to canonical MPCI paths.
- Deterministic code validates, sanitizes, resolves mutable external references, merges with the
  current form, and supplies system-owned scaffolding.
- No model is asked to invent platform state or values absent from its input.

## 4. Settled OCR decision: raw page text first

Several representations were considered: native PDF text, layout regions and coordinates,
table HTML, and lossless fusion of whole-page and region OCR. The eventual first implementation
decision was intentionally simpler:

- Use GLM-OCR whole-page `Text Recognition:` output unchanged.
- Keep one raw extraction row per PDF page.
- Preserve explicit document identity, page index, page number, and total page count.
- Do not join pages in the raw extraction layer.
- Do not add layout tags, coordinates, inferred sections, or normalized text to the raw row.
- Join and preprocess pages later when constructing T5Gemma inputs.

This is supported by the sample B/L transcripts in the conversation: section headings,
field/value adjacency, parties, routes, B/L identifiers, goods descriptions, HS codes,
containers, seals, package counts, weights, and visible page reading order generally survived
whole-page transcription.

### Layout experiment and why it was deferred

Seven representative pages were compared using raw whole-page GLM-OCR and the hosted full
GLM-OCR layout pipeline:

- Raw OCR retained usable values in 84 of 91 manually audited field groups.
- Layout parsing retained 62 of 91.
- Raw OCR was better in 24 groups, layout was better in 1, and 66 were tied.
- Five pages were treated mostly as one large table, which preserved some HTML structure but
  silently omitted important headers, parties, identifiers, or route fields.
- The hosted layout path averaged approximately 9.79 seconds per page in that small experiment.

Therefore layout is an ablation, not a prerequisite. If downstream error analysis later proves
that raw text repeatedly loses relationships, the smallest escalation is:

```text
PP-DocLayoutV3 -> region crops -> batched GLM-OCR -> ordered typed blocks
```

The complete whole-page transcript must remain available even in that branch. Layout detector
boxes alone cannot reliably map substrings in a separate whole-page transcript to regions.

### Errors that would justify revisiting layout

- Multi-column reading-order interleaving
- Shipper/consignee/notify boundary confusion
- Container-to-seal association errors
- Goods-to-HS-code association errors
- Package/weight/container relationship errors
- Dense table continuation across pages

Coordinates should be added only after block grouping demonstrates insufficient accuracy.

## 5. GLM-OCR serving contract for this repository

- Model artifact: [`zai-org/GLM-OCR`](https://github.com/zai-org/GLM-OCR) (approximately 0.9B
  parameters), pinned by immutable Hugging Face revision in configuration and Compose.
- Task prompt: exactly `Text Recognition:`.
- One rendered page image per request.
- Temperature zero; no sampling.
- The full response text is stored verbatim before any trimming or normalization.
- A response ending because of the output-token limit is a failure, not a successful extraction.
- The model and tokenizer revision, prompt, sampling settings, renderer settings, serving engine,
  speculative-decoding settings, digest-pinned vLLM base image, and patched-image build manifest
  are part of the extraction provenance.

GLM-OCR includes a native Multi-Token Prediction (MTP) head. vLLM can use that head for speculative
decoding without a separate draft model. The
[official vLLM GLM-OCR recipe](https://docs.vllm.ai/projects/recipes/en/stable/GLM/GLM-OCR.html)
uses one speculative token, and the
[vLLM MTP guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/) documents the
resolved MTP configuration and treats speculative depth as a performance-tuning parameter. The
executable extraction contract in this repository follows the vLLM GLM-OCR recipe at MTP depth 1,
not a runtime default that an environment variable can change. Disabled and depth-3 remain research
comparisons that would require separately versioned server and configuration variants before the
strict schema can accept them.

Speculation must be measured rather than assumed beneficial. Vision encoding and long OCR
prefill/output patterns may dominate on the target GPU.

## 6. Raw page dataset contract

The raw OCR dataset is evidence, not a convenience cache. Every successful row must make the
extraction reproducible and auditable.

### Identity and grouping

- `document_id` identifies one exact source-object occurrence/version.
- `page_id` and `extraction_id` are stable hashes derived from the document, page index, and
  pipeline fingerprint.
- `page_index` is zero-based for machine use.
- `page_number` is one-based for human use.
- `document_page_count` is repeated on every page row.
- `source_sha256` supports exact-content deduplication without conflating different S3 object
  occurrences.

Consumers group pages by `document_id` and order by `page_index`; physical Parquet row adjacency
is never the grouping contract.

### Source provenance

For S3, retain:

- Bucket and exact key
- Object version ID when available
- ETag as metadata, never as an assumed content hash
- Returned S3 checksum metadata and checksum type
- Size and last-modified timestamp
- Locally computed full-file SHA-256

For local files, retain the canonical URI/path, size, modification time, and full SHA-256. The
source must be checked for mutation while it is being hashed/read.

### Raster provenance

Retain the renderer and version, PDFium version, page size, intrinsic rotation, requested DPI,
effective scale, long-edge/pixel limits, annotation/form behavior, output format, dimensions,
encoded byte count, and raster SHA-256.

The initial renderer defaults are informed by the
[official GLM-OCR implementation](https://github.com/zai-org/GLM-OCR) (200 DPI, RGB, long edge
capped at 3,500 pixels) but remain configuration and benchmark parameters. The raw pipeline uses
PDFium through `pypdfium2`; the local CPU probe supported that choice. PDFium work runs in isolated
processes rather than concurrent threads.

### Inference provenance

Retain:

- Model ID and immutable model revision
- Served model name
- vLLM and container identity
- Prompt and generation settings
- Speculative-decoding method and depth
- Server and client request IDs
- Finish reason and token usage
- Attempt count and start/end/latency measurements
- Raw response artifact hash/pointer
- Exact OCR text and its UTF-8 SHA-256

Failures are stored in a separate attempt/error ledger. A training Parquet row with null OCR text
must never masquerade as a completed page.

### Completeness and publication

- Every expected page index from `0` through `document_page_count - 1` must have exactly one durable,
  successful result before a document is publishable.
- Retries apply only to classified transient transport/overload failures.
- Conflicting successful outputs for the same extraction ID are surfaced rather than resolved by
  "last write wins."
- Parquet files are written to a unique staging location, reopened and verified, hashed, and
  listed in a commit manifest written last.
- Dataset consumers read only files named by the committed manifest.

## 7. Concurrency and performance direction

The client and server controls are separate:

- Maximum active documents controls source and scratch pressure.
- Renderer process count controls CPU rasterization.
- Maximum pages in flight per document prevents a long PDF monopolizing inference.
- Global maximum pages in flight controls outstanding HTTP requests.
- vLLM `max-num-seqs` and token budgets govern continuous batching inside the server.

The pipeline uses bounded queues/semaphores and a persistent async HTTP connection pool. It does
not create an unbounded task for every corpus page.

The GPU research matrix should eventually cover:

- Client concurrency: 1, 2, 4, 8, 16, then higher only if useful
- MTP disabled versus depth 1 (and depth 3 as the upstream GLM comparison)
- vLLM sequence/token scheduler limits
- Base64 versus narrowly mounted local-file image transport, if both are operationally acceptable
- Render DPI and PNG/JPEG only with an OCR-quality comparison

Record pages/second, output tokens/second, p50/p95/p99 latency, queue time, errors, GPU utilization,
VRAM, host CPU/RAM, and output-hash consistency.

The implemented `document-ocr benchmark` command is narrower and deliberately reproducible: it
holds a pre-rendered raster manifest and one running server configuration fixed while sweeping the
configured client global/per-document concurrency points. Its report includes wall-clock and
request-window pages/second, completion tokens/second, request latency percentiles, attempts,
retries, server/config/raster identities, and exact text-hash consistency. It does not start or
reconfigure vLLM, create the raster manifest, vary MTP or scheduler flags, or collect hardware
telemetry. GPU/VRAM/CPU/RAM measurements must be captured alongside the report, and each MTP or
scheduler comparison must use an explicitly identified server/config variant.

## 8. Downstream model direction

The selected research candidate is `google/t5gemma-2-270m-270m`.

- It is an encoder-decoder text-to-text checkpoint rather than a chat model.
- Text-only fine-tuning uses a tokenizer, plain encoder input, a separate decoder target, and
  teacher-forced sequence-to-sequence loss.
- No chat template is needed.
- UL2 was a pretraining objective; downstream supervised extraction does not require UL2 mode
  tokens or denoising objectives.
- The non-vision parameter count is approximately 370M because embeddings are tied.
- Its advertised maximum context is not a reason to train at extreme length. Actual OCR and
  target P95/P99 tokenizer lengths should determine encoder and decoder limits.

The starting training pair is conceptually:

```text
encoder:
extract_mpci_v0_1:
<page_1>
raw GLM-OCR text
</page_1>
<page_2>
raw GLM-OCR text
</page_2>

decoder:
{"canonicalSparseMpciPatch":"..."}
```

One complete fine-tune is preferred over one LoRA per field group. Parties, routing, equipment,
goods, and measurements constrain each other; splitting them would repeat document encoding and
create reconciliation problems.

## 9. Target contract and deterministic ownership

The model target is sparse, canonical JSON—not a fully hydrated frontend form and not the
historical ai-genie intermediate format.

The model should own document-derived values:

- Parties and contacts
- B/L identifiers and dates
- Vessel, voyage, and route evidence
- Freight terms present in the document
- Containers, seals, equipment details, temperatures where explicit
- Goods, descriptions, HS codes, packages, and measurements
- Dangerous-goods fields where explicit and supported
- Explicit goods-to-container relationships
- Small, stable, document-supported categorical values as canonical schema codes

The model should not own:

- User/system-selected filing settings
- Platform identifiers
- Message/test flags
- UI state or validation errors
- Mutable external lookup results
- Empty/null scaffolding
- Allocations or relationships not stated by the document

The deterministic layer retains JSON parsing/schema validation, types, date/number/unit handling,
forbidden-character sanitation, fixed qualifiers, array numbering, reference integrity, mutable
lookups, merge precedence, and final business validation. That is a typed assembler and guardrail,
not a semantic remapping layer.

## 10. Grounded labeling policy

Two truths must remain separate:

- `ocr_conditioned_label`: everything the downstream text model can reasonably recover from its
  actual OCR input.
- `document_truth`: everything a human can read from the source image, including facts GLM-OCR
  omitted or corrupted.

Only OCR-conditioned truth belongs in the T5Gemma training target. Document truth is useful for
end-to-end OCR-plus-extractor evaluation.

A target value is admissible when it is copied from OCR, deterministically normalized, mapped to
a stable code from explicit evidence, derived from an explicit relation such as "same as
consignee," or uniquely recoverable under an approved rule. It is omitted when it is image-only,
absent, ambiguous, an unsupported allocation, a platform default, or an external lookup result.

Each candidate label has two artifacts:

1. The sparse JSON training target.
2. An audit sidecar with canonical path, value, OCR character spans, evidence text, derivation
   class, uncertainty, and annotator/prompt/model versions.

Useful derivation classes include `copy`, `normalize_date`, `normalize_number`, `map_enum`,
`map_static_code`, `explicit_relation`, `deterministic_correction`, `unsupported`, and
`ambiguous`.

## 11. Later corpus-processing plan

The broader dataset pipeline should remain three distinct systems:

1. Corpus profiling, deduplication, template clustering, and split assignment
2. Grounded label production, deterministic validation, and selective review
3. Training-only augmentation

Profile eligibility, exclusion reasons, B/L kind, template family and revision, OCR condition,
page and entity structure, reefer/dangerous-goods features, target coverage, language, difficulty,
and exact/near-duplicate clusters. Template identity is normally associated with the carrier,
NVOCC, freight forwarder, or form issuer—not simply the shipper.

Freeze original-document splits before augmentation. Keep all pages, duplicate scans, near
duplicates, and synthetic descendants of a document in one fold. Maintain natural-distribution,
template-held-out, challenge, and rare-feature test views.

The first baseline contains no synthetic data. Later augmentation is label-first and transactional:
update the canonical record, rewrite all affected OCR spans consistently, validate grounding and
cross-field constraints, and record parent ID, operations, seed, and generator version. Reefer and
dangerous-goods synthesis comes only after lower-risk transformations and authoritative semantic
rules exist.

## 12. Training and evaluation sequence

The conversation's provisional proof-of-concept sequence is:

1. OCR and profile the authorized corpus.
2. Double-review 50-100 calibration annotations spanning templates and difficult structures.
3. Micro-overfit 32-64 examples; failure to approach perfect training extraction is a pipeline,
   truncation, label-noise, or capability blocker.
4. Run a real-only pilot with roughly 500-800 training documents, about 100 validation documents,
   and 150-250 fully audited test documents.
5. Add absent-field/hard-negative examples if hallucination is a dominant error.
6. Try substructure curriculum only if full-patch learning fails.
7. Add a controlled synthetic tranche only after a real-only baseline.
8. Compare full fine-tuning with LoRA; full fine-tuning is plausible at this model size if measured
   sequence lengths fit the 24 GB GPU.

JSON is the default target because it is canonical, typed, sparse, directly schema-validatable,
and natural for nested arrays and relations. XML remains only a measured control if JSON validity
or token efficiency proves problematic.

Evaluation must include schema validity, field presence/value F1, exact identifier accuracy,
categorical accuracy, grouping and relationship F1, hallucinated/unsupported field rates,
template-held-out performance, document-level exactness, latency, output tokens, and VRAM.

## 13. Known semantic and governance gates

Before production label generation or schema freeze, resolve:

- The exact model-owned sparse patch schema versus the complete form schema
- Date representation
- Form mode variants and categorical code coverage
- Response envelope and diagnostics contract
- User-value merge precedence and concurrent form edits
- Cargo gross weight versus container verified gross mass
- Aggregate versus per-goods quantities and package allocation
- House versus master B/L classification
- Goods-to-container placement semantics
- Freight charge scope, marks/numbers, "same as consignee," and dangerous-goods scope

Permission to use a corpus internally does not automatically permit sending it to an external
teacher-LLM API. Dataset, model, prompt, retention, geography, logging, derived-artifact, and
production-use permissions need an explicit matrix.

## 14. Scope of this repository's first implementation

This repository starts only with the raw OCR stage:

- Discover local or versioned S3 PDFs.
- Freeze source inventory and provenance.
- Render PDFs page-by-page under explicit limits.
- Send bounded concurrent per-page requests to a separately managed vLLM GLM-OCR server with MTP
  enabled.
- Persist attempts and successes durably for safe resume.
- Publish a validated one-row-per-page Parquet dataset and commit manifest.
- Benchmark a separately started server against an immutable, pre-rendered raster JSONL manifest.

The setup completed here has not pulled the private corpus, started the server, invoked the model,
generated labels, synthesized examples, or trained T5Gemma. Those actions require source
configuration, an available GPU window, and the remaining authorization/contract decisions.

### Implemented operator and artifact contract

The current commands are `validate-config`, `inventory` (alias `prepare-inventory`), `run`,
`status`, and `benchmark`. None implicitly starts Docker or a model server. `run` and `benchmark`
contact the endpoint already named in configuration; the other commands do not contact vLLM.

Every extraction run is rooted at `<output-root>/runs/<run-id>/`. Its source inventory is
`inventory.jsonl` with `inventory.jsonl.sha256`; runtime provenance is `run-provenance.json`; and
the durable resume ledger is `state.sqlite3`. Exact vLLM response bytes are stored at
`raw-responses/<document-id>/<page-id>/<raw-response-sha256>.json`. Dataset publication creates
content-addressed `pages-<16-char-sha256-prefix>.parquet`,
`attempts-<16-char-sha256-prefix>.parquet`, then commits `dataset/manifest.json` last. Terminal
failure diagnostics remain in SQLite and prevent a completed manifest from being published.

Hard-crash recovery preserves the same evidence boundaries: an inventory missing only its digest
commit marker is accepted only after typed canonical revalidation; an S3 scratch PDF whose hash was
not yet returned to the caller is re-proven against the same exact object `VersionId`; inference
attempts and their page outcome are committed atomically; and each rendered page is bound to the
scratch-file identity observed during the document inspection that established its page count.

The page dataset repeats the full provenance contract on each row. Pages belonging to one source
document are grouped by `document_id`, ordered by `page_index`, and checked against
`document_page_count`; file order or Parquet row adjacency never defines a document. The exact raw
response pointer and hash accompany the verbatim OCR text and its UTF-8 hash.

Compose pins the vLLM container digest, model revision, MTP depth, one-image limit,
`max-model-len`, `max-num-seqs`, and GPU-memory utilization. These scheduler/model values are
literal in `compose.yaml` and must match the extraction YAML. The client verifies an authenticated,
hashed contract generated from the running server's resolved vLLM state before extraction.
Structural validation uses the automatically loaded project-root `.env` with
`docker compose config --quiet` so the resolved API key is not printed. Neither
that validation command nor any repository test starts the server, loads GLM-OCR, or uses a GPU.
