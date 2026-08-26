# MPCI Bill-of-Lading next-experiment preparation

Date: 2026-08-22  
Status: implemented, categorically reviewed, transformed, and locally validated

> Final implementation and handoff details superseding the earlier blocked/image-helper notes in
> this preparation log are in
> `docs/mpci-bl-dual-cargo-pdf-categorical-training-handoff-2026-08-22.md`.

## Fixed decisions

- The serving and training target remains **T5Gemma 2 270M**. No 1B model is part of the current or
  planned experiment matrix. Improvements must preserve the latency/throughput reason for choosing
  270M.
- The next controlled experiment combines:
  1. the detailed semantic instructor;
  2. a relation-explicit cargo target; and
  3. readable, registry-backed package categories. Container categories are omitted because the
     platform has no stable semantic registry for its generated four-character code set.
- GLM-OCR table recognition is prepared as a second, page-aligned source view using the exact
  predefined prompt `Table Recognition:`. It is attached to cloned rows for inspection, but is
  deliberately **not** injected into the next training prompt yet.
- Countries, localities, and ports remain printed source text. ISO-2 and UN/LOCODE resolution stays
  downstream.
- Original semantic-v2 rows and labeling artifacts are immutable. Every experiment is a clone with
  source hashes and lineage; no in-place migration is permitted.

## What is implemented

| Workstream | Executable/config | Current state |
|---|---|---|
| Table-view OCR | `document-ocr-table-view`; `configs/glm_ocr.table.mpci_bl_combined487.yaml` | Complete: 487 documents / 881 pages published, joined, and independently verified |
| Semantic-v3 readiness and transform | `document-kie-semantic-v3`; `configs/transforms/mpci_bl_semantic_v3_relation_explicit_table_view.yaml` | Complete: exact package mapping and 34 reviewed relations bound; 483 rows published |
| 270M semantic-v3 training task | `bill_of_lading_relation_explicit_v3`; generated `training.yaml` in the v3 dataset root | Validated: 423 train / 60 validation; prompt injects schema plus `joinedRawText` only |
| PydanticAI labeling | `document-kie-label-agents`; dual-cargo Luna and Ollama YAMLs under `configs/labeling_agents/` | Dual normal/relation output and page-scoped PDF assistance prepared; no paid Luna request made |

## Table-recognition path

### Contract

The table path uses the same retained 200-DPI page rasters that produced the accepted raw text. It
does not rerender PDFs and does not use PP-DocLayoutV3, DocLayoutV3, or any third-party table model.
Each request sends one page image to the pinned GLM-OCR/vLLM service with:

```yaml
prompt: "Table Recognition:"
sampling:
  temperature: 0.0
  top_p: 0.00001
  top_k: 1
  repetition_penalty: 1.1
  max_tokens: 8192
  seed: 0
```

Those generation values and the 200-DPI source raster policy match the official GLM-OCR task
configuration. The current service contract retains the already-patched, pinned vLLM 0.26.0 MTP
path with one speculative token. It does not introduce the GLM-OCR SDK layout detector.

Before inference, the pipeline validates all of the following:

- exact accepted-label record hashes;
- validated annotation and raw-text hashes;
- source extraction-run and page identities;
- page count, index, number, and source order;
- retained raster path, MIME type, dimensions, byte length, and SHA-256; and
- the exact model, revision, prompt, sampling, speculative-decoding, and server contracts.

The first pass bounds eight active documents, sixteen global page requests, and four pages per
document. Pages that already have a committed failed attempt additionally pass through a global
recovery semaphore on resume. Its deterministic ceiling is one fair active-document share of the
global batch (`min(per_document, max(1, global // active_documents))`), which is two pages for this
16-page/eight-document config. This retains first-pass throughput without recreating a saturated
batch for the deterministic timeout tail. Every response, attempt, usage count, latency,
finish reason, and source identity is committed atomically. A completion manifest is written last.
Interrupted runs resume from page commits; a page can make at most eight total attempts across
resumptions.

Publication reports empty, complete-HTML, incomplete-HTML, and other nonempty page outputs without
discarding any of them. It also records finish-reason counts, completion-token/raw-character
distributions, retry incidence, attempt outcomes/timing, and a page-level quality-flags JSONL for
repetition-stopped or structurally suspicious outputs. These flags remain auxiliary audit metadata;
they cannot change the raw-text training input or semantic-v2 labels.

The completed run contains 881/881 successful pages across all 487 documents. It required 1,021
request attempts: 881 successes and 140 retryable transport timeouts. Sixty-eight pages required
more than one attempt. The final outputs contain 832 complete HTML tables, six incomplete HTML
tables, and 43 nonempty non-HTML responses; 874 stopped normally and seven stopped through the
model's repetition detector. Forty-nine pages are retained in `quality-flags.jsonl` for downstream
inspection rather than filtered or rewritten. At least one complete HTML table appears in 484/487
documents.

### Commands

CPU-only preparation and provenance verification:

```bash
uv run --frozen document-ocr-table-view prepare \
  --config configs/glm_ocr.table.mpci_bl_combined487.yaml
```

Only while the GPU is free:

```bash
docker compose up --wait --wait-timeout 3600 glm-ocr-vllm
set -a
. ./.env
set +a
uv run --frozen document-ocr-table-view run \
  --config configs/glm_ocr.table.mpci_bl_combined487.yaml
```

Inspect/resume and then clone the training JSONL rows:

```bash
uv run --frozen document-ocr-table-view status \
  --config configs/glm_ocr.table.mpci_bl_combined487.yaml
uv run --frozen document-ocr-table-view join \
  --config configs/glm_ocr.table.mpci_bl_combined487.yaml
uv run --frozen document-ocr-table-view verify-join \
  --config configs/glm_ocr.table.mpci_bl_combined487.yaml
```

### Non-injection guarantee

The cloned row adds only:

```text
auxiliaryViews.glmOcrTableRecognition.pages[]
```

The join verifies byte-for-byte preservation of `joinedRawText`, `joinedRawTextSha256`, `target`,
and every other source field. Each auxiliary page carries its table text, page/raster/result
identities, `finishReason`, and completion-token count, so repetition-stopped responses remain
filterable without dereferencing a sidecar. The trainer still formats `joinedRawText` only. Merely
attaching the table view cannot alter the current prompt. A later text-plus-table experiment must
introduce a new task/input contract and cache identity explicitly.

### Measured preparation and join cost

The real 487-document/881-page preparation gate was rerun on the WSL-mounted dataset:

- wall time: 72.29 seconds;
- user/system CPU: 3.30/4.90 seconds;
- peak RSS: 67,336 KiB;
- input-page JSONL SHA-256:
  `5ea5bf3319ec849c5ad01313f47b1b750915c7889e6ba1f119b21de6205b83a7`.

The difference between wall and CPU time is deliberate full-raster disk verification, not a large
in-memory data structure.

The first correct join implementation re-opened mutable run state and revalidated the 2.01 GiB
raster corpus. On the WSL-mounted drive that took 563.65 seconds and 104,392 KiB peak RSS. The final
join instead consumes the manifest-last, SHA-256-verified published table dataset, verifies its four
declared files and the frozen input-page identities, and never reopens mutable inference state.
Real-corpus join time is now 2.71 seconds with 101,664 KiB peak RSS: about 208x faster. The separate
independent `verify-join` pass takes 1.95 seconds and 83,788 KiB peak RSS. Source/raster validation
remains mandatory in preparation, inference, and table-dataset publication; it was moved to the
correct lifecycle boundary rather than removed.

The joined clone is:

```text
artifacts/kie-training/datasets/mpci-bl-combined487-table-view-v1/
```

Its train/validation files contain 427/60 documents with SHA-256 values
`386d2dcc9a9f094537a49eebc1edb60c7ee374b76cdb4d60ffe12f22d7166784` and
`7fd68ed24db8d750d05a1b20b487270231cfec83109d5b4b2b4c87c029866085`.
The joined manifest SHA-256 is
`8c237c3d682d8aebfe9976a07fabaf31d7395508b0aa4472d547df9ddf7ec3c1`.

## Semantic-v3 target and reversible transform

### Why a new target

The prior error audit showed that most cargo errors were group/list shifts, omissions, and ambiguous
aggregate-versus-nested package relationships. The model was being asked to infer list position as
an implicit relation. Semantic-v3 makes the relation explicit:

- `cargoGroups`: source-ordered semantic cargo rows (`g1`, `g2`, ...);
- `cargoPackages`: source-ordered package facts (`p1`, `p2`, ...) linked to one cargo group;
- `cargoAllocationGroups`: container membership/allocation linked to a group and, where supported,
  the exact package level(s); and
- `containers`: printed container facts. A printed code is retained as type text when no description
  exists; no container category is trained in this experiment.

Allocation coverage is one of:

- `one_to_one_package_allocations`;
- `single_package_level`;
- `all_package_levels_combined`;
- `unlinked_package_quantities`; or
- `container_membership_only`.

Pydantic validators enforce contiguous identifiers, valid references, existing containers, unique
ownership, coverage shape, and exact allocation sums. They reject a guessed relationship instead of
silently choosing an array index.

### Categorical ownership

Semantic-v3 uses model-readable tokens rather than opaque wire identifiers:

| Family | Model target | Downstream projection |
|---|---|---|
| Package type | observed `PACKAGE_*` token from the frozen platform registry | exact token to the frozen two-character application code |
| Container type | printed `typeDescription` or printed code copied as text | versioned downstream resolver; no learned category without a semantic platform registry |
| IMDG hazard class | `FLAMMABLE_LIQUIDS`, `CORROSIVE_SUBSTANCES`, ... | fixed readable category to class `1`...`9` |
| Dangerous-goods packing group | `HIGH_DANGER`, `MEDIUM_DANGER`, `LOW_DANGER` | fixed readable category to `I`, `II`, `III` |
| Country/locality/port | text printed in OCR | versioned ISO-2 / UNLOCODE resolver |
| Negotiability, freight, units | existing readable small enum | fixed wire-code mapping |
| Party/route/text roles | semantic JSON path | fixed MPCI qualifier |
| Identifiers | supported source string | format validation only |

Raw package wording remains in the immutable semantic-v2 source, evidence artifacts, and transform
lineage. It is removed from the normal v3 target so competing aliases do not become separate package
classes, but is retained as `typeDescription` only when a reviewer records that the printed wording
cannot support one exact registry category. Container `typeDescription` remains in the v3 target;
code-only inputs are copied there verbatim. Container `typeCategory` is omitted from both the prompt
schema and every target because the platform research found no stable semantic identity for the
56,100 generated accepted codes.
Dangerous-goods UN numbers and HS codes remain source identifiers. The transform replaces only the
categorical IMDG class, subsidiary class, and packing-group codes with readable semantics and emits
their complete fixed reverse maps in the dataset manifest.

The category transform has two independent frozen inputs:

1. a registry where every readable token maps to exactly one application code and is bound to the
   authority, revision, path, and source-file SHA-256; and
2. a reviewed assignment covering every exact observed `(description, printedCode)` source key,
   bound to the 487-row inventory and both registry-file hashes.

Every assignment records occurrence count, review basis, reviewer, and date. A reviewed null category
must use `insufficient_source_specificity`; it cannot be silently treated as a generic code. Package
nulls require printed text so the model target can retain the exact fallback, while container nulls
retain the existing printed description. Transform publication fails if coverage is incomplete, a
count differs, a key is duplicated, a non-null token/code is absent, or a source hash changed.

On publication, the transform also writes `task-constraints.json`. It contains the sorted category
tokens actually assigned in this dataset (31 package, zero container) plus the registry, base
prompt-schema, and target-schema hashes. The training loader requires this frozen artifact for
`bill_of_lading_relation_explicit_v3`, injects those exact tokens as JSON-schema enums, and rejects
both reference/generated targets containing a token outside the vocabulary. A relation-explicit
run therefore cannot silently fall back to the broad token lexical pattern used by the reusable
Pydantic data model.

Generated v3 evaluation retains strict leaf/path metrics and adds two index-independent views:

- cargo-graph precision/recall/F1 and document exactness over package membership, allocation
  coverage, package-scope links, container membership, allocation quantities, and direct
  container/package links; and
- category precision/recall/F1 and document exactness, anchored by package ID or container number.

This directly distinguishes a list-order miss from a wrong cargo relationship or wrong canonical
category.

### Real-data audit

The final table-bound readiness artifact is:

```text
artifacts/kie-training/datasets/
  mpci-bl-semantic-v3-relation-explicit-table-view-readiness-v3/
```

It reports:

| Measure | Count |
|---|---:|
| Source documents | 487 |
| Train / validation | 427 / 60 |
| Package category occurrences / exact source variants | 658 / 99 |
| Container category occurrences / exact source variants | 905 / 105 |
| One-to-one package/container allocations | 289 |
| Single-package-level distributed allocations | 93 |
| All-package-levels-combined allocations | 6 |
| Container-membership-only allocations | 22 |
| Unlinked allocation quantities | 1 |
| Goods groups without allocations | 207 |
| Ambiguous duplicate package quantity | 3 documents |
| Incomplete allocation quantities | 1 document |

The category inventory also shows why the model-facing target should not be the application wire
code. All 658 package occurrences (99 exact spelling variants) currently carry printed package text
and no printed two-character code. For containers, 812/905 occurrences (85 variants) are text-only,
37 occurrences (10 variants) carry both text and a code, and 56 occurrences (10 variants) carry a
code only. In other words, most supervision must consolidate printed aliases such as `CARTONS`,
`CTNS`, `40HC`, and `40' HIGH CUBE` into readable semantic categories; asking the 270M model to emit
opaque downstream identifiers would make that same semantic problem harder without adding source
evidence. The frozen registry and reviewed source-key map make that consolidation explicit and
auditable.

The returned platform audit proved all 405 package entries and published a fail-closed empty
container semantic registry. Exact source-key review safely maps 88/99 package variants and 636/658
occurrences (96.66%) to 31 observed readable tokens. Eleven variants / 22 occurrences remain
verbatim. All 105 container variants remain printed text/code by design.

The audit emits a separate 34-row cargo-relation review queue for every nontrivial transformable
goods item with multiple package facts plus allocations. A reviewed assignment is permitted to
override the arithmetic candidate only with another arithmetically valid relation. This matters for
seven rows where one source-ordered package fact maps directly to each container: those are not
collective sums. The remaining nontrivial queue contains 21 unique-level and six genuine combined
level cases. Publication requires exact queue coverage and binds every decision to the source
relation hash and reviewer metadata.

The four relation-blocked documents are explicitly listed in the YAML and readiness manifest. The
other 483 documents are structurally transformable. The table-bound source inventory SHA-256 is
`331baf9324f6cd16a4c19014ab8c1fd8b27128dd2e9082b3e996561f1045d21c`.
Its 34-row review queue was compared to the pre-table queue after projecting document/goods
identity, OCR hash, packages, allocations, candidate class, and source-relation hash. Both
projections have SHA-256
`acc7a5d1e7b532826f1c24efdd5cfd93b7458d2f16a85054911f921c40ffc3c3`.
The reviewed 21 single-level, seven one-to-one, and six combined decisions were therefore carried
without modification and rebound only to the new source inventory. The strict semantic-v3 audit
then proved exact 34/34 coverage and arithmetic validity.

### Commands and current gate

```bash
uv run --frozen document-kie-semantic-v3 categories \
  --config configs/transforms/mpci_bl_semantic_v3_relation_explicit_table_view.yaml \
  --review-config configs/transforms/mpci_bl_semantic_v3_category_review_v1.yaml
uv run --frozen document-kie-semantic-v3 audit \
  --config configs/transforms/mpci_bl_semantic_v3_relation_explicit_table_view.yaml
uv run --frozen document-kie-semantic-v3 transform \
  --config configs/transforms/mpci_bl_semantic_v3_relation_explicit_table_view.yaml
```

All three commands now pass. The transform publishes 423 train and 60 validation rows, four declared
exclusions, lineage, task constraints, a token audit, and a runnable training YAML under
`artifacts/kie-training/datasets/mpci-bl-semantic-v3-relation-explicit-table-view-v1/`. The package
registry is pinned to platform revision `72f9aff735c2581f2ccff74e7fde34c2aea9b265`; confirmation that
this tag was deployed to the target tenant/data period remains a release-provenance check. No source
semantic-v2 record is edited.

## PydanticAI label-and-review flow

The first 50-document Luna measurement emits one evidence-rich dual annotation: the normal
semantic-v2-compatible label and the relation-explicit cargo label. This keeps direct comparison
with existing accepted artifacts while testing whether agents can produce the new cargo ontology
without a second labeling pass. Category canonization remains a deterministic frozen transform, not
an ungrounded agent judgment.

### Architecture

The new runner uses reusable, stateless PydanticAI agents with native Pydantic output schemas:

```text
validated work item + page-ordered OCR
  -> extractor (text only)
  -> deterministic schema/evidence checks
  -> independent fresh-context reviewer (text only)
  -> accept OR a fresh extraction attempt
  -> optional requested-page PDF layout helper only on explicit ambiguity
  -> immutable outcome + usage/cost receipt
```

Concurrency is bounded independently at the document and model-request layers. The Luna pilot is
configured for eight documents and eight model requests at once, with three candidate attempts,
mandatory independent review, no silent provider fallback, and at most one PDF escalation. HTTP
timeouts, 408/409/429 responses, and 5xx responses use the OpenAI SDK's bounded two-retry transport
policy; schema retries and fresh semantic attempts remain separately bounded and attributable.

The PDF helper may resolve only layout, heading scope, row/column association, continuation, or
document boundaries. It cannot contribute or correct a value absent from raw OCR. A hash-verified,
in-memory PDF containing only explicitly requested source pages is not constructed or transmitted
unless an extractor/reviewer returns the typed document request. If an assigned model lacks PDF-file
support, the document is published as needing review rather than being silently rerouted.

Each response records provider/model, request ID, latency, input/cached/cache-write/output/reasoning
tokens, response finish metadata, and calculated price. Receipts are retained per call, per attempt,
per document, and for the complete run. Responses consumed by a failed structured-output run are
also receipted and priced before the retry or fail-closed decision. Candidate/review history is
immutable; a failed reviewer never edits the rejected attempt.

Publication then compares each outcome with the hash-pinned previously accepted semantic-v2 target.
Those references are never copied into prepared work items or supplied to a model. The run publishes
per-document and aggregate JSON/schema validity, canonical exact match, and exact field/value
precision, recall, and F1 for both all selected documents and the validated-output subset. Exclusions
and needs-review outcomes count as missing predictions in the all-selected view. This is a
reproducibility/quality comparison against the current accepted corpus, not a claim that those
existing labels are independent ground truth.

Per-document and run summaries distinguish priced OpenAI calls, unpriced OpenAI failures, and local
Ollama calls. `openAiCostComplete`/`openai_cost_complete` is false whenever any OpenAI call lacks a
provider response receipt, so the priced subtotal cannot be mistaken for a complete observed cost.

### Luna configuration and cost accounting

The primary config is:

```text
configs/labeling_agents/mpci_bl_dual_cargo_v3_luna_cost50.yaml
```

It fixes `gpt-5.6-luna`, Responses API, `reasoning_effort: max`, `store: false`, native structured
output, and a deterministic 50-document selection. The OpenAI model page currently lists $0.20 per
million input tokens, $0.02 cached input, and $1.20 output; cache writes are 1.25 times input price,
and requests over 272K input tokens receive the documented 2x input/1.5x output multipliers. These
rates and their effective date are explicit configuration data rather than constants hidden in
code.

For each response, the implementation calculates:

```text
uncached_input = input_tokens - cached_input_tokens - cache_write_tokens
cost = uncached_input * input_rate
     + cached_input * cached_rate
     + cache_write * input_rate * cache_write_multiplier
     + output_tokens * output_rate
```

The long-context multipliers apply to the complete request when input exceeds the configured
threshold. Reasoning tokens are reported but not double-counted when the provider includes them in
output tokens.

The prepared selection is 50 documents / 100 pages with inventory SHA-256
`d733d6595a64b7e5978388a7a0d4a0088166bfb3ab000277065c9a7b5c54a60e`. A repeated real preparation
took 13.58 seconds and 103,716 KiB peak RSS. The publication-only 50-reference comparison probe took
1.30 seconds and 66,276 KiB peak RSS, and produced exact/F1 = 1.0 for an identity comparison. No
model request has been made, so measured API cost is still exactly absent rather than estimated as
an observed result.

The selected raw OCR documents contain 1,432 / 4,473 / 19,771 characters at minimum, median, and
maximum. The extractor decision schemas total 16,422 compact JSON characters and the reviewer
decision schemas total 4,062. These are character counts rather than claimed model token counts,
but they confirm that even the largest selected text/schema request is nowhere near Luna's 272K
long-context pricing threshold or the configured per-agent input guard. Provider-reported token
usage remains the only value used for final cost accounting.

### Local Ollama option

`configs/labeling_agents/mpci_bl_dual_cargo_v3_ollama.example.yaml` accepts an exact arbitrary Ollama
model tag. It does not assume that the informal name “Qwen-3.8 27B” corresponds to a published tag.
The portable Ollama configuration is text-only; arbitrary local tags do not establish a PDF-file
input contract.

The adapter uses PydanticAI's first-class `OllamaModel` and `OllamaProvider`, not a hand-declared
generic model profile. Preflight checks the local server version, exact installed model tag,
and native JSON-schema support. Self-hosted Ollama 0.5+ supports
grammar-constrained native JSON schema through its OpenAI-compatible endpoint. Arbitrary Ollama
models do not share a portable protocol-level reasoning switch, so the runner records and preserves
the selected model's provider default rather than pretending to disable or force hidden reasoning.
There is no automatic Luna fallback: a local-model experiment remains attributable and
independently measurable.

### Installation and commands

```bash
uv sync --frozen --group labeling
```

Add the key to the standard project-root `.env` without committing it:

```dotenv
OPENAI_API_KEY=replace-with-a-real-key
```

Then prepare and inspect without a paid request:

```bash
uv run --frozen --group labeling document-kie-label-agents inventory \
  --config configs/labeling_agents/mpci_bl_dual_cargo_v3_luna_cost50.yaml
uv run --frozen --group labeling document-kie-label-agents prepare \
  --config configs/labeling_agents/mpci_bl_dual_cargo_v3_luna_cost50.yaml
uv run --frozen --group labeling document-kie-label-agents preflight \
  --config configs/labeling_agents/mpci_bl_dual_cargo_v3_luna_cost50.yaml
```

The paid 50-document quality/cost experiment is intentionally a separate operator action:

```bash
uv run --frozen --group labeling document-kie-label-agents run \
  --config configs/labeling_agents/mpci_bl_dual_cargo_v3_luna_cost50.yaml
uv run --frozen --group labeling document-kie-label-agents publish \
  --config configs/labeling_agents/mpci_bl_dual_cargo_v3_luna_cost50.yaml
```

## Next controlled training experiment

The first v3 experiment changes the prompt and target together because the semantic instructor
defines the relation-explicit target. It holds the model/revision, optimizer, LoRA setup, effective
batch, epochs, and evaluation procedure fixed against the best semantic-v2 baseline. Source/target
ceilings increase only because the exact token audit proves the new schema/targets need them.

The table field should be present in the cloned input JSONLs but excluded from prompt formatting.
This produces the requested prepared table corpus without confounding the v3 semantic/categorical
test. After v3 is validated, run a distinct text-only versus text-plus-table branch with the same
v3 labels and hyperparameters.

Required pre-training gates:

1. package registry and all 99 package source variants reviewed — complete;
2. container no-category policy and all 105 printed fallbacks frozen — complete;
3. 483-row transform, exact four exclusions, and manifest verification — complete;
4. schema/prompt/dataset validation and exact tokenizer-length audit — complete;
5. containerized tokenizer/cache preparation with zero truncations — complete;
6. smoke/benchmark the conservative batch-1, accumulation-24 GPU path at the larger ceilings; and
7. freeze the resulting run config before the complete 25-epoch experiment.

Evaluation must include JSON/schema validity, exact field/value metrics, group and relation metrics
that are insensitive to array index, projected MPCI field metrics, and results by each published
cargo-relation cohort. A v3 result should not be declared better merely because its JSON is easier
to generate.

## Official references

- Z.ai, [GLM-OCR model and deployment](https://github.com/zai-org/GLM-OCR).
- Z.ai, [official task prompts, sampling, and 200-DPI configuration](https://github.com/zai-org/GLM-OCR/blob/main/glmocr/config.yaml).
- Z.ai, [GLM-OCR table-recognition skill](https://github.com/zai-org/GLM-OCR/blob/main/skills/glmocr-table/SKILL.md).
- Pydantic, [PydanticAI agents, structured outputs, usage limits, and concurrency](https://pydantic.dev/docs/ai/core-concepts/agent/).
- Pydantic, [PydanticAI document and binary inputs](https://pydantic.dev/docs/ai/core-concepts/input/).
- Pydantic, [PydanticAI output types](https://pydantic.dev/docs/ai/core-concepts/output/).
- Pydantic, [OpenAI provider](https://pydantic.dev/docs/ai/models/openai/).
- Pydantic, [Ollama provider and native JSON-schema requirements](https://pydantic.dev/docs/ai/models/ollama/).
- OpenAI, [GPT-5.6 Luna model capabilities and pricing](https://developers.openai.com/api/docs/models/gpt-5.6-luna).
- OpenAI, [file inputs and PDF behavior](https://developers.openai.com/api/docs/guides/file-inputs).
- UNECE, [Recommendation 21 package/cargo code lists](https://unece.org/code-list-recommendations).
- BIC, [ISO container size/type code reference](https://www.bic-code.org/size-type-code/).
