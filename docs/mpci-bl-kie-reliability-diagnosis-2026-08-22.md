# MPCI Bill-of-Lading KIE reliability diagnosis

- Date: 2026-08-22
- Status: diagnosis and experiment proposal; no runtime, schema, label, or training configuration was changed
- Run under review: `t5gemma2-270m-lora-mpci-bl-combined487-v2`

## Purpose and evidence boundary

This report preserves the diagnosis behind the next Bill-of-Lading KIE experiments. It answers four
questions:

1. Why the current model remains below the desired 0.90 reliability level.
2. Whether the nested `goodsItems` target should be represented differently.
3. Whether a second GLM-OCR table view should be supplied alongside whole-page OCR.
4. Which values should be copied from the document and which should be resolved to MPCI/CUSCAR
   categoricals outside the model.

The report distinguishes measured evidence from proposed experiments. It does **not** select a new
production contract yet. The existing semantic-v2 labels, the completed training run, and the
immutable reliability publication remain unchanged.

Primary local evidence:

- [completed reliability audit](../artifacts/kie-training/analysis/t5gemma2-270m-lora-mpci-bl-combined487-v2-reliability-audit-v1/REPORT.md)
- [machine-readable audit summary](../artifacts/kie-training/analysis/t5gemma2-270m-lora-mpci-bl-combined487-v2-reliability-audit-v1/summary.json)
- [semantic-v2 design](mpci-kie-semantic-schema-v2-design.md)
- [labeling semantics](../BILL_OF_LADING_LABELING_REFERENCE_V2.md)
- [current runtime prompt](../prompts/kie/bill_of_lading_semantic_v2.txt)
- [semantic target model](../src/document_ocr/label_schemas/bill_of_lading.py)
- [MPCI projector](../src/document_ocr/label_schemas/mpci_projection.py)
- [MPCI application-schema investigation](../artifacts/mpci-ai-schema/README.md)

## Executive diagnosis

The current approach is learning the task, but its `0.7934` exact leaf micro-F1 and `0.6894` exact
path-union accuracy do not support a highly reliable deployment. The primary deficit is semantic
recall and cargo-relationship reconstruction, not unconstrained invention of values.

The strongest measured facts are:

- precision `0.8551`, recall `0.7399`, with `1,889` true positives, `320` false positives, and
  `664` false negatives;
- `54/60` JSON-valid outputs, `48/60` schema-valid outputs, and only `1/60` exact whole-document
  outputs;
- `531` omissions, `187` additions, and `133` substitutions;
- `130/133` unmatched predictions in the audited cargo fields occur somewhere in the raw OCR;
- package type, package quantity, and goods description are common in training, yet remain weak;
- the complex-goods validation cohort reaches `0.753` F1 versus `0.834` for simple goods; and
- making the goods section perfect would yield about `+0.100` global F1, the largest section-level
  oracle gain.

This leads to four conclusions.

1. **The model is usually choosing or associating the wrong OCR-grounded fact.** It is not mainly
   fabricating cargo vocabulary or numbers.
2. **JSON arrays are not intrinsically the problem, but positional array identity is.** A missed or
   merged goods row shifts several otherwise correct values to different indexed paths.
3. **The prompt is materially under-specified.** It conveys types and nesting but omits the semantic
   rules the human labelers used.
4. **More data is necessary but should follow a contract experiment.** Scaling a weak ontology or
   ambiguous prompt to 10,000 examples would make the same ambiguity more consistent rather than
   resolve it.

The recommended sequence is therefore:

1. add a concise semantic schema instructor while keeping the current target and input fixed;
2. test an auxiliary GLM-OCR table view while keeping the target fixed;
3. test a compact, relation-explicit cargo intermediate target while keeping the improved prompt and
   input fixed; and
4. only then scale real labeling, capacity, and synthetic augmentation around the winning contract.

## Measured error anatomy

### Headline metrics

| Metric | Result |
|---|---:|
| Validation documents | 60 |
| Train documents | 427 |
| JSON validity | 0.900 |
| Schema validity | 0.800 |
| Exact leaf precision | 0.855 |
| Exact leaf recall | 0.740 |
| Exact leaf micro-F1 | 0.793 |
| Exact path-union accuracy | 0.689 |
| Whole-document exact match | 1/60 |

The document-bootstrap 95% interval in the audit is approximately `0.735–0.847` for F1 and
`0.607–0.771` for path-union accuracy. The point estimate is therefore not merely a little below
0.90; the uncertainty interval is also far below the intended acceptance threshold.

### Cargo field distance

| Field | Train docs | Train values | Validation values | P | R | F1 | Correct value at wrong index | Omitted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Package type | 396 | 575 | 83 | 0.652 | 0.542 | 0.592 | 15 | 13 |
| Package quantity | 405 | 580 | 81 | 0.681 | 0.580 | 0.627 | 10 | 10 |
| Goods description | 408 | 497 | 67 | 0.815 | 0.657 | 0.727 | 3 | 7 |
| Marks and numbers | 215 | 427 | 75 | 0.553 | 0.280 | 0.372 | 1 | 9 |
| Additional information | 85 | 228 | 26 | 0.125 | 0.038 | 0.059 | 0 | 24 |

The full field-distance audit sharpens the diagnosis:

- **package type:** 38 misses comprise 15 correct values at another index, 13 omissions, and 10
  other selections; all 25 non-absent candidate values are present in raw OCR;
- **package quantity:** 34 misses comprise 10 wrong-index values, 10 omissions, 13 other
  OCR-grounded numbers, and one number within 5% of the reference; and
- **description:** all 23 misses are row shifts, omissions, or partial/contaminated copies; 15 of 16
  non-absent candidate strings are raw-OCR grounded.

The detailed examples are preserved in:

- [package type errors](../artifacts/kie-training/analysis/t5gemma2-270m-lora-mpci-bl-combined487-v2-reliability-audit-v1/tables/package_type_errors.csv)
- [package quantity errors](../artifacts/kie-training/analysis/t5gemma2-270m-lora-mpci-bl-combined487-v2-reliability-audit-v1/tables/package_quantity_errors.csv)
- [goods-description errors](../artifacts/kie-training/analysis/t5gemma2-270m-lora-mpci-bl-combined487-v2-reliability-audit-v1/tables/goods_description_errors.csv)
- [human-readable error gallery](../artifacts/kie-training/analysis/t5gemma2-270m-lora-mpci-bl-combined487-v2-reliability-audit-v1/examples/focus_error_gallery.md)

### What positional repair can and cannot solve

If every correct package or description value at the wrong list index were placed perfectly, the
field-level optimistic ceilings would be approximately:

| Field | Current F1 | Perfect wrong-index repair |
|---|---:|---:|
| Package type | 0.592 | 0.789 |
| Package quantity | 0.627 | 0.760 |
| Goods description | 0.727 | 0.777 |

This is important: a relation-aware representation can recover a meaningful portion of the error,
but **representation alone cannot reach 0.90**. Omissions, incorrect aggregate-versus-nested
selection, partial copies, and target ambiguity remain.

### Structural support, not only scalar support

The training set includes:

- 39 documents with multiple goods items;
- 55 with multiple package levels under one goods item;
- 76 with multiple container allocations under one goods item;
- 138 with multiple containers; and
- 76 with three or more pages.

The overlapping complex-goods cohort contains 152 train and 25 validation documents. That is enough
to reveal a real structural weakness, but not enough to densely cover carriers, table layouts,
aggregate/nested combinations, and allocation patterns. Counts of package scalar values alone hide
this combinatorial sparsity.

## Root causes, ranked by evidence

### 1. Semantic prompt mismatch — confirmed

The human annotators followed the detailed rules in
[the semantic-v2 reference](../BILL_OF_LADING_LABELING_REFERENCE_V2.md). The runtime model sees the
JSON structure and a generic instruction to extract a sparse object, but it is not told:

- when aggregate and nested package levels should both be retained;
- what constitutes one goods item;
- how descriptions, quantities, package types, weights, and allocations form a row/group;
- which marks and additional-information strings are eligible;
- how repeated data across pages should be reconciled; or
- how party, carrier, delivery-agent, and other role headings differ.

The current schema-rendering code removes `title`, `description`, and `default`. The measured prompt
is 6,546 bytes and contains zero field-description annotations. Consequently, adding Pydantic field
descriptions alone would **not** repair the runtime instruction. The fix needs to be a task-owned,
versioned prompt section that is identical in training and inference.

This diagnosis is consistent with [UIE](https://aclanthology.org/2022.acl-long.395/), which uses a
structural schema instructor rather than field types alone. [TRUE-UIE](https://aclanthology.org/2024.naacl-long.103/)
also makes grouping relations explicit instead of relying on sequence position to imply them.
[PARSE](https://aclanthology.org/2025.emnlp-industry.184/) independently identifies ambiguous or
incomplete developer-oriented JSON schemas as a structured-extraction reliability problem. These
works support the proposed ablation; their reported results are not assumed to transfer unchanged to
this smaller model or OCR-conditioned B/L dataset.

### 2. Flattened OCR obscures row and hierarchy relations — confirmed

The current input is page-ordered, whole-page `Text Recognition:` output. It preserves value content
well, but serializing a two-dimensional goods table into one text stream can interleave headers,
descriptions, package levels, container rows, and totals. The cargo misses overwhelmingly contain
valid source values, while the wrong relationship is selected.

This is the failure mode addressed by document-structure work that treats documents as relations or
graphs rather than plain sequences. Examples include
[Spatial Dependency Parsing](https://aclanthology.org/2021.findings-acl.28/),
[DocStruct](https://arxiv.org/abs/2010.11685), and
[Graph Convolution for visually rich documents](https://aclanthology.org/N19-2005/).
These papers do not prove that this repository needs a graph neural network; they support the narrower
claim that layout-derived relationships can be lost in one-dimensional OCR.

### 3. Positional arrays couple entity extraction to row alignment — confirmed

The final MPCI form is array-shaped, and the semantic-v2 target mirrors that where semantically
necessary. During strict evaluation, however, array position acts as entity identity. If the model
omits `goodsItems[0]`, correct values for the next item become errors at every downstream indexed
path. The global index-insensitive F1 gain is only about `+0.016`, so this is not the entire problem,
but the per-cargo error audit shows that it is concentrated where it matters.

### 4. Output validity — confirmed secondary cause

Six outputs are invalid JSON and twelve are schema-invalid. Constrained decoding should make syntax
and schema validity deterministic. Even a deliberately impossible oracle that replaces all
schema-invalid documents with perfect labels reaches only about `0.893` F1, so validity constraints
are a required reliability floor, not the semantic solution.

### 5. Model/data capacity — plausible but not isolated

T5Gemma 2 270M-270M is a pretrained encoder-decoder rather than an instruction-tuned extractor.
Supervised examples and the exact task prompt must teach the output policy. The run does demonstrate
learning, so LoRA is not invalid for the task; it does **not** establish that 270M capacity or the
current LoRA surface is sufficient for 0.90 on complex cargo relationships.

This checkpoint status is explicit in the
[official 270M-270M model card](https://huggingface.co/google/t5gemma-2-270m-270m); the
[T5Gemma 2 technical report](https://arxiv.org/abs/2512.14856) provides the family and scale context.

The current run also does not support treating Adam betas, more epochs, or sampling values as the
primary lever. The error shape is semantic and structural, and the epoch curve was already close to
saturation. Capacity and adaptation should be tested after the target/input contract is improved.

## Question 1: should the nested goods target change?

### Short answer

Keep the final semantic-v2/MPCI projection as the application contract, but test a **compact,
relation-explicit intermediate cargo target** for model supervision. Do not replace all JSON objects
and arrays merely because cargo arrays are weak.

Lists and dictionaries are not inherently unlearnable. Parties, route, identifiers, and other object
sections perform materially better. The specific problem is that cargo array index simultaneously
means:

- entity identity;
- row order;
- parent goods-item membership;
- package-level membership; and
- sometimes container allocation membership.

A single missing item therefore creates several correlated errors. The recommended experiment makes
entity and relation identity explicit, scores them order-independently, and deterministically projects
them back into the current nested form.

### Literature precedents

| Precedent | Useful idea for this project | What should not be copied blindly |
|---|---|---|
| [DocILE](https://arxiv.org/abs/2302.05658) | Separates key information localization/extraction from line-item recognition; line items are grouped field tuples and evaluated with group matching rather than literal list order. | Its invoices and spatial annotations are not identical to B/L cargo or raw GLM-OCR text. |
| [UIE](https://aclanthology.org/2022.acl-long.395/) | Represents entities and associations with a structural schema instructor and a structured extraction language. | Its serialization would require a new parser and would give up some existing JSON/Pydantic tooling. |
| [TRUE-UIE](https://aclanthology.org/2024.naacl-long.103/) | Makes membership/order-like relations explicit through a small relation vocabulary. | Its universal relation language is broader than the concrete MPCI task requires. |
| [Spatial Dependency Parsing](https://aclanthology.org/2021.findings-acl.28/) | Treats key-value/group relations as a first-class parsing target. | The project currently trains from OCR text, not token bounding boxes. |
| [DocStruct](https://arxiv.org/abs/2010.11685) | Models document form structure as a hierarchy/graph of fragments. | A graph model is not yet justified when a simpler table view plus relation-explicit JSON can be tested first. |

[DocILE's official implementation](https://github.com/rossumai/docile) also provides a concrete
precedent for order-insensitive line-item matching. That is more relevant to the immediate evaluation
problem than changing braces, key names, or JSON formatting.

### Schema options

| Option | Advantage | Cost/risk | Recommendation |
|---|---|---|---|
| Current nested semantic-v2 JSON | Already implemented, Pydantic-valid, close to MPCI projection. | Positional coupling; long target; grouping rules implicit. | Retain as baseline and final projection contract. |
| Current JSON plus semantic instructor | Isolates whether missing semantics, not shape, are the main issue. | Still positional. | **Run first.** |
| Flat relation-explicit cargo facts | Separates entity extraction from grouping; compact; order-insensitive scoring. | Requires a versioned projector and new validators. | **Recommended target experiment.** |
| One object per table row | Natural when the source has a clean table. | B/Ls often mix aggregate totals and nested packaging, so a physical row is not always a semantic goods item. | Use only as an input view or intermediate observation, not universal truth. |
| UIE-style structural language | Compact entity/relation representation with literature precedent. | New grammar/parser and weaker reuse of current JSON tooling. | Consider only if flat JSON remains inadequate. |
| Separate section decodes | Reduces output length and field competition. | More inference calls and deterministic merge requirements. | Test independently after prompt repair. |

### Proposed experimental cargo ontology

The following is an **illustrative v3 intermediate**, not an adopted schema:

```json
{
  "cargoFacts": [
    {
      "groupId": "g1",
      "kind": "description",
      "text": "MACHINERY PARTS"
    },
    {
      "groupId": "g1",
      "kind": "package",
      "packageId": "p1",
      "quantity": 60,
      "typeText": "PALLETS"
    },
    {
      "groupId": "g1",
      "kind": "package",
      "packageId": "p2",
      "quantity": 600,
      "typeText": "CARTONS"
    },
    {
      "groupId": "g1",
      "kind": "grossWeight",
      "value": 12000,
      "unit": "kilogram"
    },
    {
      "groupId": "g1",
      "kind": "allocation",
      "containerNumber": "MSCU1234567",
      "packageId": "p1",
      "packageQuantity": 30
    }
  ]
}
```

Contract rules:

- `groupId` and `packageId` are document-local relation identifiers, not facts printed on the B/L.
- Their literal spelling is not scored; entities are aligned by optimal matching.
- Every scalar fact remains grounded in the configured input view.
- A deterministic projector assembles the facts into `goodsItems`, `packages`, measurements, marks,
  and container allocations.
- The projector may validate relationships and order output deterministically, but may not infer an
  unprinted package level, allocate an ambiguous total, or rewrite OCR values.
- If a final MPCI path cannot preserve `packageId`, the projector can discard the identifier only
  after using it to assemble and validate the relationship.

The current `ContainerAllocation` relates a container number and quantity to a goods item, but has no
explicit link to a particular package level. That becomes ambiguous when one goods item contains both
an aggregate and a nested package count. An intermediate `packageId` would make the annotation and
modeling rule testable without changing the final MPCI form.

### Why not switch immediately?

Changing the prompt, input representation, target ontology, model size, and data volume at once would
make improvement uninterpretable. The current field-level index-repair ceilings also show that the
new ontology cannot solve omissions and semantic selection by itself. The causal order should be:

1. current input + current target + semantic instructor;
2. text-plus-table input + current target + the same instructor; and
3. text-plus-table input + relation-explicit cargo target + the same instructor.

### Evaluation for the alternate target

Training serialization and application evaluation should be separated:

1. Parse and validate the intermediate output.
2. Match predicted/reference groups and facts order-independently, using an optimal bipartite
   assignment rather than array index.
3. Measure entity/fact F1 and relation/grouping F1 separately.
4. Project both predictions and references to semantic-v2/MPCI shape.
5. Retain the existing exact projected field/value metrics as the deployment-facing result.

This prevents a serialization improvement from being mistaken for application correctness.

## Question 2: should GLM-OCR table output become a second input view?

### Short answer

Yes, as a controlled **auxiliary view**, not as a replacement for whole-page OCR. Reuse the existing
page rasters, run GLM-OCR with `Table Recognition:`, preserve the result as a separate immutable
artifact, and measure whether it improves cargo grouping.

The official [GLM-OCR model card](https://huggingface.co/zai-org/GLM-OCR) explicitly defines
`Text Recognition:`, `Formula Recognition:`, and `Table Recognition:` prompts. The official
[configuration](https://github.com/zai-org/GLM-OCR/blob/main/glmocr/config.yaml) and
[fine-tuning guide](https://github.com/zai-org/GLM-OCR/blob/main/examples/finetune/README.md) use the
same task distinction. The official
[table skill](https://github.com/zai-org/GLM-OCR/blob/main/skills/glmocr-table/SKILL.md) describes
structured table output, and the [technical report](https://arxiv.org/abs/2603.10910) evaluates table
structure recovery.

This is not yet proof that full-page `Table Recognition:` reliably isolates the furnished-goods
section on this corpus. The official full parsing pipeline commonly combines layout detection with
task-specific recognition. The user's observation that full-page table prompting often targets the
goods section is a strong experiment hypothesis, not a production guarantee.

### Why the table view is relevant to the measured failures

The current weak fields are usually readable but associated with the wrong row or package level.
A table representation may restore:

- column headings;
- row boundaries;
- aggregate versus nested package levels;
- adjacency between description, quantity, package type, and weight; and
- container-to-cargo row relationships.

This is a better match to the evidence than re-running the same text prompt at a higher DPI or tuning
generation sampling. The values already appear in raw OCR in nearly every audited non-omission case.

### It must remain a second view

An earlier seven-page layout experiment found usable values in `84/91` manually audited field groups
for raw whole-page OCR, while the hosted layout representation retained `62/91`; raw was better in
24 groups, layout in one, with 66 ties. That experiment is recorded in
[the project-direction report](project-direction-from-shared-conversation.md). It rules out replacing
whole-page text with a broad layout transformation. It does **not** rule out an auxiliary cargo-table
view that leaves raw text untouched.

Recommended model input shape:

```text
=== PAGE 1 / TEXT VIEW ===
<verbatim Text Recognition output>

=== PAGE 1 / TABLE VIEW ===
<verbatim or losslessly normalized Table Recognition output>

=== PAGE 2 / TEXT VIEW ===
...

=== PAGE 2 / TABLE VIEW ===
...
```

Page order and view identity must be explicit. Text output must not be deleted, normalized in place,
or replaced by the table result.

### Repository implications

The current YAML field is technically passed to the OpenAI-compatible request, but the durable
configuration and provenance contracts restrict the prompt to literal `Text Recognition:`. In
particular, the Pydantic configuration and `InferenceMetadata` encode that literal. A table branch is
therefore **not** a safe one-line YAML edit. It needs a new versioned extraction-view contract.

Each table record should include at least:

- document ID, page ID, page number, and page count;
- source PDF identity and raster path/hash;
- `viewType: table`;
- exact prompt and prompt hash;
- GLM-OCR model and tokenizer revision;
- vLLM/build and sampling metadata;
- raw response, finish reason, token counts, and latency; and
- success/failure state independent of the text-view state.

The existing page images are already persisted, so the pilot does not need to render PDFs again.
The combined 487-document dataset contains 881 pages; its 60-document validation split contains 111
pages. A validation-only table probe is therefore small enough to diagnose before processing all
training pages.

### OCR-conditioned truth boundary

This is the most important data-contract caveat.

Current labels are grounded in the `Text Recognition:` output. For the first table ablation, the
table view should be used only to improve grouping of values already present in the text view. If a
table response contains a value missing from text OCR, that value must not silently become a target.

If text plus table becomes the production input contract, the target truth boundary changes to the
union of both versioned views. Existing labels must then be re-audited or rebuilt so that values
visible only in the table view are not falsely treated as omissions by the model. Image-only values
remain outside the target unless the production model also receives images.

### Table-view experiment

#### Phase 1 — diagnostic extraction, no training

Run `Table Recognition:` on the 111 already-rendered validation page images and publish a separate
view manifest. Measure:

- table-output success and malformed-output rates;
- reference cargo-value coverage in text, table, and their union;
- duplicated or conflicting cells;
- preservation of row, column, and package-level relationships;
- output token count, latency, peak GPU memory, and pages/second; and
- behavior on pages without a meaningful cargo table.

For a manually adjudicated table-heavy subset, record whether each reference relation is preserved.
If gold table structure is authored, use a structure-aware measure such as TEDS from
[PubTabNet](https://arxiv.org/abs/1911.10683); do not use TEDS against unreviewed generated Markdown.

#### Phase 2 — controlled training ablation

Use the same 427/60 split, same target, same model/training settings, and same semantic instructor:

| Branch | Input | Target | Purpose |
|---|---|---|---|
| A | text | current nested semantic-v2 | Reproduced baseline |
| B | text | current nested semantic-v2 | Semantic-instructor effect |
| C | text + table | current nested semantic-v2 | Incremental table-view effect over B |
| D | text + table | relation-explicit cargo intermediate | Target-ontology effect over C |

Branch A and B differ only in the prompt. Branch B and C differ only in the input view. Branch C and
D differ only in the cargo target. This makes every result attributable.

#### Phase 3 — optional region/table gate

Only if the full-page table probe is frequently empty, contaminated, or wasteful should a layout or
page-selection gate be tested. That branch needs its own recall benchmark because a detector can omit
an unusual but valid goods region. No DocLayoutV3 wiring should be reintroduced merely because it is
part of the official full parsing stack; the earlier local evidence does not justify it as the first
experiment.

## Prompt/schema instructor proposal

The prompt should still include the machine-readable schema, but precede it with a compact,
versioned semantic contract. At minimum it should define:

- one multi-page source equals one transport document;
- sparse omission and no outside knowledge;
- repeated-page reconciliation using only values present in an input view;
- party-role rules and explicit `sameAs` handling;
- one logical address rather than OCR-line arrays;
- goods-item grouping and when a new item begins;
- aggregate versus nested package-level retention;
- the relationship between descriptions, package levels, measures, and allocations;
- marks versus boilerplate and additional information versus description;
- value-only extraction without headings, tax/contact flavor text, or neighboring fields; and
- copied-text versus readable-enum versus downstream-resolver ownership.

The instructor should be generated or hashed as a task asset and used byte-identically in training
and serving. Pydantic descriptions may remain valuable for developers, but they are not a substitute
while the runtime schema compactor removes them.

## Base model, real-data scale, and synthesis

### What the current run establishes

- The pretrained T5Gemma 2 model can learn substantial Bill-of-Lading KIE behavior from 427 examples.
- LoRA is a viable initial adaptation method; this run does not prove it is optimal.
- The current contract is not reliable enough to justify scaling it unchanged.
- More epochs on the same data are unlikely to address the dominant structural/semantic misses.

### Recommended real-data learning curve

After selecting the prompt/input/target contract, add the next 500 adjudicated real documents by
structure and template rather than random volume. Pre-register learning-curve checkpoints at roughly
`+125`, `+250`, and `+500` real examples. Selection should deliberately increase:

- multiple goods items;
- aggregate plus nested package levels;
- multiple per-item container allocations;
- multiple containers and multi-page continuations;
- marks and eligible additional information;
- uncommon carriers/templates; and
- difficult but usable OCR.

Only after the contract-level curve is improving should a 270M-only capacity matrix covering LoRA
rank/target surface and selective or full tuning be run. No 1B model is part of the current or
planned system: the 270M serving target is a deliberate latency and throughput constraint. This
still separates data/ontology failure from optimization or adapter-capacity failure without changing
the deployment class.

### Path to 10,000 examples

Synthetic expansion is appropriate after the real-data contract is frozen. It should vary cargo
structure, layout, carrier template, lexical surface, package hierarchy, and OCR noise while keeping
facts internally consistent. Required controls:

- generated descendants of a real document stay in the same split as their source;
- the external acceptance test remains real-only and template/carrier disjoint;
- every synthetic target round-trips through schema and deterministic projectors;
- rendered inputs are OCR-grounded, with no target value absent from the configured input view;
- exact/near duplicate checks run before publication; and
- synthetic mixtures are ablated against a same-size real-only baseline.

[DocILE](https://arxiv.org/abs/2302.05658) is a useful scale precedent—its benchmark combines about
6.7K annotated business documents with a much larger synthetic set and tests new layouts—but it is
not evidence that synthetic B/Ls will automatically transfer. The real-only test remains the gate.

## Categorical ownership audit

### Container type: does `typeDescription` match MPCI?

Yes. It is not an accidental label. The investigated MPCI/CUSCAR form has two distinct concepts:

- `containerSizeAndType.containerCode`: a required platform four-character categorical code; and
- `equipmentDescription`: optional source/free-text equipment description.

The semantic-v2 target accordingly has `containers[].typeCode` and
`containers[].typeDescription`, and the projector maps them to those separate application fields.
The source analysis records this in
[the MPCI field catalog](../artifacts/mpci-ai-schema/field-catalog.md) and
[code-list audit](../artifacts/mpci-ai-schema/selects-and-code-lists.md).

However, the current training distribution exposes a schema-quality issue:

| Container target presence across 487 labels | Occurrences |
|---|---:|
| Description only | 812 |
| Code only | 56 |
| Both description and code | 37 |
| Neither | 53 |

`typeDescription` appears in 315 training documents with 760 training values and 85 exact variants;
its validation F1 is about `0.767`. `typeCode` appears in only 45 training documents with 89 values
and ten exact variants; only three validation documents contain it (four values), and its observed
F1 is `0.400`.
Observed code-like values include several conventions and aliases such as `HC40`, `40HQ`, `20GP`,
`40HC`, and `20DV`.

The next experimental target deliberately replaces these inconsistent source aliases with one
human-readable category token, for example:

```json
{
  "containerNumber": "TGHU1234567",
  "typeCategory": "FORTY_FOOT_HIGH_CUBE_DRY"
}
```

This is a model-facing categorical, not the application code and not a verbatim-copy field. A frozen
registry gives every readable token exactly one four-character application code. A separately
reviewed assignment maps every exact observed `(typeDescription, typeCode)` source key to one token.
The raw wording remains in the immutable source/evidence lineage rather than becoming a second target
that competes with the category. The transform must fail closed on an unknown or ambiguous source
key. This is the semantic-v3 experiment contract and must be compared with semantic-v2 before it is
adopted.

### Package type

The final form similarly has:

- `typeOfPackages`: free-text package type; and
- `packageTypeDescriptionCode`: UNECE Recommendation 21 code.

The current 487 semantic labels contain **no** `packages[].typeCode` values. They contain many raw
types—`PACKAGES`, `CARTONS`, `PALLETS`, `BAGS`, and numerous variants—so semantic-v2 learns wording,
not the form categorical. Semantic-v3 instead teaches readable closed categories such as `PACKAGE`,
`CARTON`, `PALLET`, or `DRUM`. Each token maps one-to-one to a frozen UNECE Recommendation 21 code.
Every one of the 99 observed exact source variants must receive a human-reviewed assignment before
publication; unknown or ambiguous variants are rejected rather than guessed. The source text remains
traceable in the original row and transformation lineage.

### Full ownership matrix

| Field family | Model target | Deterministic layer | Reason |
|---|---|---|---|
| Country/locality | Printed name/text | ISO-2 and UN/LOCODE resolver | Large/mutable lookup and ambiguity policy. |
| Container type | Readable registry token such as `FORTY_FOOT_HIGH_CUBE_DRY` | Exact token-to-platform-code map | Easier to infer than an opaque code; one target replaces inconsistent aliases. |
| Package type | Readable registry token such as `CARTON` or `DRUM` | Exact token-to-UNECE-Rec21-code map | Context-readable while remaining one-to-one with the form categorical. |
| Party role | Named semantic path such as `shipper` | Fixed MPCI party-function code | Role is explicit/contextual; wire code follows path. |
| Route role | Named semantic path such as `portOfLoading` | Fixed location qualifier | Qualifier follows path. |
| Freight arrangement | Readable closed enum | Fixed MPCI arrangement code | Small semantic decision with explicit wording. |
| Negotiability | Readable closed enum | `NEG`/`NON` mapping | Small semantic decision. |
| Mass/volume/temperature units | Readable closed enum | MPCI unit code | Small stable enum, source-supported. |
| Contact means | Semantic phone/email/website fields | Fixed contact wrapper/means codes | Type follows target path. |
| Goods text qualifier | `description` or `additionalInformation` path | `AAA`/`AAI` | Code follows path. |
| Container, seal, HS, UN identifiers | Source-supported identifier with frozen lexical normalization | Validation and sanitation only | Identifiers are not categoricals to infer. |
| System/user/form defaults | Absent from model | Application-owned | Not document facts. |

### Other potential categorical mismatches

- Handling instructions currently preserve source text while the form also permits coded handling
  descriptions. Do not add code supervision until the product-required code list and source evidence
  rules are frozen.
- Dangerous-goods hazard class and packing group are already small, document-supported semantic
  fields; their wire conversion can remain deterministic.
- Country and port names must continue to be copied as printed. The current semantic-v2 decision to
  resolve ISO-2/UNLOCODE after inference is correct and should not be reversed.
- HS codes, UN numbers, B/L numbers, container numbers, seals, IMO numbers, and dates are identifiers
  or normalized scalars, not categorical classification targets.
- Fixed MPCI fields such as party codes, location qualifiers, text qualifiers, and measurement
  attribute codes should remain outside model output.

## Registered experiment program

### Stage 0 — repair measurement infrastructure

Before comparing model branches:

1. ensure prediction publication retains original `documentId` under length-grouped evaluation;
2. add grammar/schema-constrained decoding;
3. retain exact path and field metrics, validity, and document-bootstrap confidence intervals;
4. add order-insensitive cargo fact/group metrics; and
5. regenerate the five reviewed sidecars whose targets were subsequently corrected.

### Stage 1 — semantic instructor

Keep model, split, target, OCR input, and training hyperparameters fixed. Compare the current generic
prompt with the compact semantic instructor. Primary endpoint: projected goods-section exact F1.

Retain the branch only if the paired document-bootstrap 95% interval for the goods-F1 delta is above
zero, the point improvement is at least `+0.03`, and non-goods F1 falls by no more than `0.01`.

### Stage 2 — table view

First run the 111-page diagnostic. If table output is structurally usable and the values/relations are
grounded, train text-only versus text-plus-table with the same instructor and target. Retain the table
view only if the paired goods-F1 delta is statistically positive, at least `+0.03`, and latency/storage
cost is measured and acceptable for the intended throughput.

### Stage 3 — cargo ontology

Compare the current nested target against the relation-explicit intermediate. Require:

- statistically positive projected goods-F1 delta of at least `+0.03`;
- no regression in package type or quantity;
- relation/grouping F1 improvement of at least `+0.05`;
- JSON/schema validity at least as high as the constrained baseline; and
- a deterministic, lossless projection for every representable reference label.

### Stage 4 — data and capacity

Using the winning contract:

1. add targeted real examples in `+125/+250/+500` increments;
2. compare learning curves by structural cohort and carrier/template novelty;
3. compare 270M LoRA ranks/target surfaces and selective or full 270M tuning only after the curve is
   known;
4. freeze a selected configuration before opening the external acceptance test; and
5. introduce synthetic data only as a separate ablation.

## Reliability acceptance contract

The eventual 0.90 claim should require all of the following on a frozen, real-only,
carrier/template-disjoint test:

- JSON validity `1.000` and schema validity `1.000`;
- exact field/value micro-F1 and exact path-union accuracy at least `0.90`;
- document-bootstrap 95% lower confidence bounds at least `0.90` for both headline metrics;
- predeclared floors for B/L number, dates, parties, route, package type, package quantity, goods
  description, weights, container number, and container allocation;
- explicit results by page count, goods count, package-level count, carrier/template novelty, and OCR
  quality; and
- calibrated abstention/review coverage if some documents cannot meet those floors automatically.

Whole-document exact match should continue to be reported even if it is not the primary model-
selection metric. A production workflow also needs fail-closed review for low-confidence or
structurally inconsistent outputs; aggregate F1 alone cannot guarantee every filing.

## Decisions and non-decisions

### Supported by current evidence

- Add field semantics to the runtime prompt and keep it identical in training and serving.
- Preserve whole-page text and test table output as an auxiliary page-aligned view.
- Evaluate cargo entities and relations independently of array index.
- Keep current semantic-v2/MPCI projection as the deployment-facing comparison contract.
- Teach readable, registry-backed categorical tokens while keeping their application-code projection
  deterministic and versioned outside the model.
- Acquire structurally targeted real data before broad synthetic multiplication.

### Not yet supported

- Replacing whole-page OCR with table or layout OCR.
- Reintroducing DocLayoutV3 before a measured full-page table diagnostic.
- Migrating all sections away from JSON objects/arrays.
- Claiming that full fine-tuning, LoRA rank, Adam betas, or more epochs is the root solution.
- Training ISO-2 or UN/LOCODE guesses from source text, or publishing container/package category
  targets without a frozen registry and fully reviewed source-key assignments.
- Scaling directly to 10,000 labels before the prompt/input/target contract is validated.

## Primary external sources

- Duan et al. (2026), [GLM-OCR Technical Report](https://arxiv.org/abs/2603.10910).
- Z.ai, [official GLM-OCR model card and prompt contract](https://huggingface.co/zai-org/GLM-OCR).
- Z.ai, [official GLM-OCR task configuration](https://github.com/zai-org/GLM-OCR/blob/main/glmocr/config.yaml).
- Z.ai, [official GLM-OCR table skill](https://github.com/zai-org/GLM-OCR/blob/main/skills/glmocr-table/SKILL.md).
- Lu et al. (2022), [Unified Structure Generation for Universal Information Extraction](https://aclanthology.org/2022.acl-long.395/).
- Zhang et al. (2024), [TRUE-UIE: Two Universal Relations Unify Information Extraction Tasks](https://aclanthology.org/2024.naacl-long.103/).
- Shrimal et al. (2025), [PARSE: LLM Driven Schema Optimization for Reliable Entity Extraction](https://aclanthology.org/2025.emnlp-industry.184/).
- Šimsa et al. (2023), [DocILE Benchmark for Document Information Localization and Extraction](https://arxiv.org/abs/2302.05658).
- Hwang et al. (2021), [Spatial Dependency Parsing for Semi-Structured Document Information Extraction](https://aclanthology.org/2021.findings-acl.28/).
- Wang et al. (2020), [DocStruct](https://arxiv.org/abs/2010.11685).
- Liu et al. (2019), [Graph Convolution for Multimodal Information Extraction from Visually Rich Documents](https://aclanthology.org/N19-2005/).
- Zhong et al. (2019), [PubTabNet and TEDS](https://arxiv.org/abs/1911.10683).
- Borchmann et al. (2022), [Business Document Information Extraction: Towards Practical Benchmarks](https://arxiv.org/abs/2206.11229).
- Google DeepMind, [official T5Gemma 2 270M-270M model card](https://huggingface.co/google/t5gemma-2-270m-270m).
- Steiner et al. (2025), [T5Gemma 2: Seeing, Reading, and Understanding Longer](https://arxiv.org/abs/2512.14856).

## Local artifact map for follow-up

- Headline and gap arithmetic:
  [summary.json](../artifacts/kie-training/analysis/t5gemma2-270m-lora-mpci-bl-combined487-v2-reliability-audit-v1/summary.json)
- Field support and performance:
  [field_support_performance.csv](../artifacts/kie-training/analysis/t5gemma2-270m-lora-mpci-bl-combined487-v2-reliability-audit-v1/tables/field_support_performance.csv)
- Structural cohorts:
  [structural_cohort_performance.csv](../artifacts/kie-training/analysis/t5gemma2-270m-lora-mpci-bl-combined487-v2-reliability-audit-v1/tables/structural_cohort_performance.csv)
- Error budget:
  [counterfactual_error_budget.csv](../artifacts/kie-training/analysis/t5gemma2-270m-lora-mpci-bl-combined487-v2-reliability-audit-v1/tables/counterfactual_error_budget.csv)
- Template proximity:
  [template_nearest_train.csv](../artifacts/kie-training/analysis/t5gemma2-270m-lora-mpci-bl-combined487-v2-reliability-audit-v1/tables/template_nearest_train.csv)
- Label/evidence revision audit:
  [label_sidecar_provenance.csv](../artifacts/kie-training/analysis/t5gemma2-270m-lora-mpci-bl-combined487-v2-reliability-audit-v1/tables/label_sidecar_provenance.csv)
- Current GLM-OCR provenance contract:
  [project direction](project-direction-from-shared-conversation.md)
- MPCI field/categorical ownership:
  [field catalog](../artifacts/mpci-ai-schema/field-catalog.md),
  [code lists](../artifacts/mpci-ai-schema/selects-and-code-lists.md), and
  [dataset contract](../artifacts/mpci-ai-schema/dataset-contract.md)
