# KIE synthesis preparation and generation plan

Date: 2026-08-30  
Status: preparation plus structured non-linguistic Passes 1-2 implemented and validated

This document records what is now proven for the exact 1,157-document MPCI bill-of-lading corpus,
what the preparation artifacts mean, and how the subsequent augmentation system should be built.
It deliberately separates implemented structured generation from the still-pending linguistic and
raw-text realization stages. No synthetic training row and no paid PydanticAI call has been
produced.

The generation scope is text-only: construct a synthetic structured target, then apply a validated
set of exact edits to the selected source document's page-ordered raw OCR while preserving its
textual format. Generating, editing, or rerendering PDFs is outside this pipeline.

## 1. Completed implementation

### 1.1 Exact source refresh and EDA

The refreshed EDA is pinned to the task-facing 1,157-row corpus, the immutable lineage and annotation
manifests, all source PDFs, and all 2,452 page rasters. It publishes 50 plots and complete tabular
artifacts under:

```text
artifacts/kie-synthesis/mpci-bl-combined1157-synthesis-readiness-eda-v1/
```

The machine-readable configuration is:

```text
configs/analysis/mpci_bl_combined1157_synthesis_eda.yaml
```

The source contains 1,157 unique documents and source-PDF hashes, 826 bills of lading and 331 sea
waybills. The median document has two pages, one container, one cargo group, and one package fact.
The weak cohorts are not merely small marginal categories: many important intersections are even
smaller.

| Cohort | Documents | Share |
|---|---:|---:|
| Multiple cargo groups | 73 | 6.3% |
| Multiple package levels | 66 | 5.7% |
| Temperature setting | 45 | 3.9% |
| Dangerous goods | 33 | 2.9% |
| Forwarding agent | 83 | 7.2% |
| Delivery agent | 376 | 32.5% |
| Multiple cargo plus multiple package levels | 2 | 0.2% |
| Temperature plus multiple containers | 6 | 0.5% |
| Forwarding plus delivery agents | 24 | 2.1% |

The conservative template proxy produces 682 groups; 570 are singletons and the largest group has
44 documents. This is a proxy inventory, not a ground-truth carrier-template registry. Its safe use
is source grouping, leakage protection, and minimum-support checks—not asserting that two visually
similar forms are the same template.

The proxy is constructed deterministically. Documents are first partitioned by normalized carrier
family and document type. Inside each partition, a greedy representative-based clustering pass uses
the first-page 20 x 20 normalized ink-layout vector and the ordered first occurrence of known OCR
headings. A document can join a representative only when visual similarity is at least 0.90, OCR
heading-order similarity is at least 0.70, and `0.60 * visual + 0.40 * OCR` is at least 0.88. The
proxy ID hashes the carrier/type and representative identity. These deliberately conservative
groups organize sources; the individual source document, not the proxy representative, supplies the
raw-text skeleton for a synthetic descendant.

### 1.2 Domain-specific relational projection

`BillOfLadingRelationDomainAdapter` now projects the task-facing relation-v3 target into 17 explicit,
ordered tables and reconstructs the exact original target. The all-corpus proof is 1,157/1,157 exact
reconstructions followed by task canonicalization and Pydantic validation.

| Table | Rows |
|---|---:|
| `documents` | 1,157 |
| `document_locations` | 4,558 |
| `document_references` | 627 |
| `parties` | 4,542 |
| `party_contacts` | 2,731 |
| `containers` | 2,115 |
| `container_seals` | 2,156 |
| `cargo_groups` | 1,360 |
| `cargo_additional_information` | 534 |
| `cargo_marks_numbers` | 1,349 |
| `cargo_hs_codes` | 1,204 |
| `cargo_handling_instructions` | 84 |
| `packages` | 1,423 |
| `allocation_groups` | 1,023 |
| `allocation_group_packages` | 871 |
| `allocations` | 1,853 |
| `dangerous_goods` | 38 |

This 27,625-row domain representation is the synthesis/modeling surface. The previously published
generic value graph remains an independent byte-semantic losslessness guard. Neither replaces the
training target.

The fixed SDV Metadata V1 graph declares every primary key and the parent-child relationships that
SDV can model without guessing types from data. SDV 1.38.2 validates both metadata and all 17 real
tables. Document-local `gN` and `pN` values are identities reconstructed deterministically; they are
not learned categorical data.

### 1.3 Evidence-backed OCR anchor and format inventory

The preparation runner resolves each task-facing relation leaf through the immutable normal-label
evidence and projection lineage. It handles the two audited retargeting cases explicitly:

- legacy container `typeCode` evidence projected to the readable task-facing description; and
- retained package levels whose source package index changed after outer-package policy projection.

The result contains 54,462 evidence rows and 13,201 role-aware format profiles:

| Location outcome | Anchors |
|---|---:|
| Exact unique in page | 34,326 |
| Unique inside a unique evidence excerpt | 15,665 |
| Repeated/ambiguous in page | 4,471 |

Every source-fact leaf in every document has audited evidence. The median document has 98.89% of
its leaves at a uniquely located span; 576 documents have 100% uniquely located leaves. Repeated
values remain explicit and fail closed for direct scalar patching.

Each anchor records document and page identity, task-facing and normal paths, normalized role path,
target value, audited raw value and excerpt, evidence kind, normalization rule, exact location
status, and surface family/pattern. Format profiles group by role, surface family, and presentation
pattern, so a date, quantity, seal, or country is rendered according to its semantic role rather
than by a global string replacement.

An anchor locates audited evidence; it does **not** assert that the entire `rawValue` is a scalar
replacement span. A composite raw value such as a container plus seal needs a bounded field/block
renderer. The preparation artifact therefore does not claim that end-to-end raw-OCR synthesis is
already implemented.

### 1.4 Deterministic generator contracts

The following reusable primitives are implemented and tested:

- HMAC-SHA256 counter streams keyed by seed, namespace, document, and semantic field path. Output is
  independent of worker scheduling and concurrency.
- ISO 6346 container generation using an explicitly supplied observed owner/category prefix, a
  generated six-digit serial, the shared production check-digit implementation, uniqueness checks,
  and known regression vectors.
- Whitespace-free scalar seal generation that preserves character class and punctuation. The 14
  observed whitespace-delimited/compound seal surfaces are deliberately rejected until a typed
  block policy exists.
- Joint bounded date shifting that preserves field missingness and the exact interval between issue
  and shipped-on-board dates.
- Integer quantity generation from explicit support and largest-remainder allocation, with exact
  total preservation.
- Allocation reconciliation for every current relation coverage mode.
- Decimal measure scaling with explicit rounding and cross-unit gross-weight >= net-weight checks.

These functions own formal validity and arithmetic. They do not choose business distributions; a
future proposal stage must supply source-supported bounds, categories, scale factors, and scenario
facts.

A 10,000-value collision probe first confirmed that independent six-digit draws are not a valid
batch uniqueness strategy. The final allocator resolves collisions in sorted semantic-identity
order, making assignments independent of request order and worker scheduling. The corrected probe
produced 10,000/10,000 unique valid containers at 146,394 values/second and 10,000/10,000 unique
shape-preserving scalar seals at 37,455 values/second on the current host.

### 1.5 Support and readiness publication

The immutable preparation output is:

```text
artifacts/kie-synthesis/mpci-bl-combined1157-synthesis-preparation-v3/
```

It includes the 17 domain tables, SDV metadata, full OCR anchors, document coverage, format profiles,
cohort and template support tables, a Markdown report, six matplotlib/seaborn plots, and a hashed
manifest. The generation configuration is:

```text
configs/synthesis/mpci_bl_combined1157_preparation.yaml
```

Current semantic-generator support is deliberately narrower than evidence support:

| Cohort | Source docs | Evidence-anchor ready | Semantic-generator ready | Remaining gate |
|---|---:|---:|---:|---|
| All documents | 1,157 | 1,156 | 1,156 | field/block renderer |
| Multi-page | 765 | 765 | 765 | field/block renderer |
| Multi-container | 353 | 351 | 351 | field/block renderer |
| Multi-cargo | 73 | 56 | 56 | field/block renderer |
| Multiple package levels | 66 | 62 | 62 | field/block renderer |
| Container allocations | 861 | 859 | 859 | field/block renderer |
| Refrigerated | 45 | 40 | 0 | temperature scenario generator and renderer |
| Dangerous goods | 33 | 31 | 0 | pinned DG registry and coherent generator |
| Delivery agent | 376 | 333 | 0 | seeded linguistic/party generator |
| Forwarding agent | 83 | 79 | 0 | seeded linguistic/party generator |
| HS codes | 751 | 722 | 0 | pinned HS/product registry and generator |

These counts are capability diagnostics, not a promise that all listed source documents can be
published synthetically today.

### 1.6 Route dependencies and auxiliary-text gap

The current relational projection represents route locations and party localities, but it does not
yet encode their dependency edges or generate them hierarchically. Corpus evidence supports adding
that layer. Among documents where both values are printed, shipper country equals port-of-loading
country in 359/436 cases, consignee country equals port-of-discharge country in 375/417,
notify-party country equals port-of-discharge country in 332/391, delivery-agent country equals
port-of-discharge country in 140/149, place-of-receipt country equals port-of-loading country in
191/196, and place-of-delivery country equals port-of-discharge country in 237/242. These are strong
conditional tendencies, not universal rules. Party city names only rarely equal port names, so the
dependency belongs at jurisdiction/role level rather than literal city equality. Port-of-discharge
country is Egypt in 482/520 documents that print it, so fitting an uncontrolled destination model
to the source corpus would reproduce the acquisition bias.

The anchor inventory also covers label facts, not unlabeled identifying material around them. A
conservative syntax-and-party-proximity scan finds at least 2,042 VAT, tax, TIN/GST, CNPJ, ACID,
EORI, registration, import/export-ID, or fax surfaces in 767/1,157 documents. This is a heuristic
lower bound, not yet a published party-block annotation, but inspected examples confirm that these
identifiers frequently sit inside shipper, consignee, or notify blocks while intentionally remaining
outside the KIE target.
The text-only renderer therefore needs a party-block residual inventory: known target spans are
protected, residual sensitive identifiers are replaced with format-preserving synthetic distractors,
and non-sensitive headings/flavor text remain unchanged. The generated distractors remain absent
from the label so they continue teaching the model what to ignore. Existing label-validation and
annotation policies already recognize tax, VAT, ACID, customs, CNPJ, and registration metadata;
those audited rules should seed the residual classifier instead of duplicating a new regex policy.

### 1.7 Structured non-linguistic Passes 1-2

The first two structured-generation passes are now implemented. They deliberately stop before
party/goods linguistic generation or raw-OCR realization and therefore publish inspection plans,
not training rows. The accepted 50-scenario baseline is:

```text
artifacts/kie-synthesis/mpci-bl-combined1157-structured-baseline50-v8/
```

Its strict configuration is:

```text
configs/synthesis/mpci_bl_combined1157_structured_baseline50.yaml
```

Pass 1 provides the fail-closed execution and semantic layer:

- a template-isolated train-only source scope and receipts;
- an exact-quota SciPy MILP selector with template and carrier caps;
- immutable staged publication with behavior, environment, image, source, fit, and transaction
  hashes;
- run-global container, seal, document-reference, and voyage reservations against the complete
  real corpus;
- deterministic HMAC-keyed streams whose results do not depend on worker or request order;
- coherent date, identifier, package-quantity, measure, allocation, and relation updates; and
- exact schema, canonicalization, relational-inverse, change-ledger, collision, arithmetic, and
  equipment-capacity gates.

Pass 2 supplies statistically generated cargo quantities and measures. The complete 17-table graph
remains the semantic authority; SDV models only compact, dense cargo-group profiles. Gross, net,
and volume missingness are hard routing dimensions. A profile selects the narrowest train-supported
route from exact package identity and role, semantic package family and role, then role-wide fit
support. Role-wide rows may fit a proposal model but can never define plausibility. Every accepted
proposal must also lie inside the raw marginal bounds and log-scaled nearest-neighbor envelope of
an exact identity or a meaningful package family. Generic `UNREGISTERED_PRINTED_PACKAGE` and
`UNTYPED_PACKAGE` labels are explicitly forbidden as semantic-family fallbacks.

For each supported profile, empirical resampling and four selectable Gaussian Copula variants are
evaluated over five template-grouped folds and five seeds. CTGAN and TVAE are configured but remain
ineligible below the declared 1,000-row/100-template support threshold. Candidate selection requires
perfect diagnostic validity, bounded paired quality deficit versus the empirical baseline, at least
90% novel proposals, at least 25% post-projection proposal yield, and all task-owned business-rule
gates. Statistical output proposes only a driver quantity and present per-driver-package measures;
deterministic projection derives totals and all dependent package/allocation values.

The accepted run produced 50 distinct templates across 20 carrier strata (19 named carrier
families plus one missing-value stratum), 93 cargo-group proposals, and 105 changed task-facing
package quantities. It generated 92 collision-free,
ISO-6346-valid containers. All 50 targets passed strict schema, canonical, inverse, exact-ledger,
allocation, and capacity validation, with no exact train-row copies. The generated cargo-group
proposal yield was 92.09%. Pipeline runtime was 197.48 seconds with 832.59 MiB peak Python RSS.
An immediate repeat returned the committed artifact in 4.72 seconds and left its complete content
tree byte-identical.

This baseline remains `trainingEligible=false`. It preserves source language, party identities,
goods descriptions, package categories, container types, and relation cardinality. Those retained
facts are useful template context but are not anonymous synthetic training data. Seven retained
metadata-only package-hierarchy facts also remain explicitly pending text realization.

## 2. Recommended generation architecture

The pipeline should be a staged compiler from a pinned real document to a constrained synthetic
plan, not a single generative model call.

```text
immutable real corpus and source-grouped split
  -> target cohort deficit planner
  -> eligible base/template selector
  -> typed scenario proposal
  -> deterministic semantic reconciliation
  -> optional bounded PydanticAI linguistic value realization
  -> one-call PydanticAI typed text-edit plan from whole-document context
  -> deterministic raw-text patch executor
  -> target projection and evidence regeneration
  -> structural, semantic, leakage, diversity, and utility gates
  -> immutable synthetic-only and real-plus-synthetic publications
```

### Stage 0 — freeze source, split, and registries

Inputs must be hashes, not mutable directories. Freeze the real train/eval/test assignment by
source PDF and conservative template proxy before fitting a synthesizer. Every descendant of one
real source belongs to the same split, and synthetic descendants are train-only. This prevents a
synthetic variant of a held-out layout from leaking into training.

Pin every external registry with authority, revision, license, path, SHA-256, and canonicalization
rules. At minimum this concerns countries/localities, ports, HS/product descriptions, UN dangerous
goods and packing groups, package vocabulary, and container categories. Unsupported values fail;
there is no silent generic fallback.

### Stage 1 — cohort deficit, template, and base-document selection

The scheduler takes target counts, minimums, exact quotas, and caps as distinct concepts. It solves
for source documents that jointly cover the requested cohort while respecting:

- template capability and minimum real support;
- maximum variants per real document and template;
- carrier/template concentration caps;
- source-group and split boundaries;
- required anchor/renderer capabilities; and
- requested marginal and joint deficits.

The recommended default is a constrained deficit optimizer, not independent weighted sampling.
Independent sampling will miss rare intersections and overuse the few eligible templates. If no
feasible plan exists, publish the unsatisfied constraints and support counts instead of weakening
them.

Implement the optimizer as a deterministic mixed-integer linear program using SciPy's HiGHS-backed
`milp`, which is already present in the isolated synthesis environment. Integer variables allocate
variant counts per eligible source; binary variables express whether a source/template is used.
Hard constraints own requested total, cohort minimum/exact/maximum counts, per-source/template/
carrier caps, split, and capability. A lexicographic objective first minimizes target-distribution
deviation and maximum concentration, then maximizes supported template/source/cohort diversity with
a deterministic document-ID tie break. Solver status, objective terms, achieved quotas, and any
infeasible constraints are published; a greedy fallback is not permitted.

Selection is cohort-first. For cardinality-preserving version 1, a requested feature must already
exist in the source document. The selector then applies renderer/anchor capability, split, total
template support, cohort-specific template support, carrier/template caps, and per-source reuse caps
before choosing one individual document. The selected document's complete raw OCR is the template;
the proxy only supplies grouping and diversity constraints. For example, a dangerous-goods request
starts from an actual dangerous-goods document rather than inserting a new DG block into a non-DG
sibling from the same proxy.

### Stage 2 — typed statistical proposals

Do not fit one synthesizer to the full 17-table graph. SDV documents public HMA as optimized for
roughly five tables and one relationship level, while this graph is deeper and contains bridge
relations. Use compact modeling views whose outputs are still only proposals:

1. document/scenario view: type, carrier/template family, route class, counts, feature flags;
2. route-scenario and party-role/locality view;
3. cargo/package/weight view;
4. container/type/temperature view; and
5. dangerous-goods/HS/product view once authoritative registries exist.

The optimal default should be `auto_benchmark`, not a permanently hard-coded easy baseline. For
each view, benchmark Gaussian Copula, CTGAN, and TVAE where data support is sufficient, using
source-grouped folds. Select by a configured composite of marginal fidelity, dependency fidelity,
target-condition yield, invalid-proposal rate, fit/sample latency, and downstream pilot utility.
Prefer the simplest candidate within a configured quality margin. With 1,157 rows and rare cohorts,
Gaussian Copula is the likely initial winner because it is efficient and supports mathematical
single-table conditioning; that expectation is not a substitute for the benchmark.

HMA is an explicit simplified-depth-one ablation. It is not the default for the complete graph.
Public multi-table targeted sampling is not assumed because SDV documents that capability as an
Enterprise feature. Community mode instead uses profile-specific models, eligible-template
selection, deterministic interventions, and bounded rejection with reported yield.

Route generation is hierarchical rather than a set of independent country columns. First sample a
typed shipment scenario: direct or transshipment route, origin/destination countries, seaports,
places of receipt/delivery/final destination, and issue-location role. Then sample party roles
conditioned on that scenario: shipper and origin forwarder usually belong to the origin side;
consignee, notify, and delivery agent usually belong to the destination side; explicit third-party
and `sameAs` relationships preserve real exceptions. Canonical registry identities remain internal
constraints while the task-facing label and raw text use locality/country spellings generated in the
selected source format. Destination targets must be configured explicitly rather than learned from
the Egypt-heavy marginal.

### Stage 3 — deterministic semantic generation and reconciliation

The constraint engine owns facts that must be correct 100% of the time:

- document-local identities, ordering, and foreign keys;
- container and seal identifiers and collisions;
- route/port/country compatibility;
- date chronology and base-template presentation;
- package quantities, net/gross weight, volume, and totals;
- package hierarchy and allocation coverage semantics;
- container/package/cargo graph construction;
- HS/DG/package/container registry membership; and
- categorical vocabulary and task-schema validation.

SDV's own documentation states that probabilistic models do not guarantee business rules by
themselves. Use SDV constraints where a supported single-table transformation exactly expresses a
rule; use project deterministic code for multi-table and domain graph invariants. Reject invalid
proposals rather than asking an LLM to repair arithmetic or relationships.

Version 1 should preserve party, container, cargo, package, and allocation cardinality and preserve
the relation topology/coverage class. This is the safest way to prove useful augmentation before
solving layout expansion.

Semantic completeness is a hard contract. Every source-present task fact must receive a declared
policy: regenerate, statistically resample, deterministically derive, or explicitly preserve as an
approved non-identifying categorical. A generator may not omit a field because its method is not
implemented. Null remains valid only when the selected scenario/source structure legitimately lacks
that field. Container size/type and similar non-identifying categoricals may initially be preserved
or resampled from compatible support; “exception” never means deleting them from the label.

Represent phase boundaries as distinct types rather than partially valid labels:

```text
DraftScenarioPlan (may contain typed PendingRealization tasks)
  -> ResolvedSemanticPlan (no pending task; all relations and values valid)
  -> TextPatchPlan (all target and auxiliary text operations located)
  -> PublishedSyntheticRecord (final OCR text, target, evidence, lineage, receipts)
```

Only `ResolvedSemanticPlan` can project into the task label, and only
`PublishedSyntheticRecord` can enter training. The initial non-linguistic implementation may emit
auditable `DraftScenarioPlan` artifacts, but it may not use placeholders, source PII, or missing
fields to masquerade as a completed synthetic label.

### Stage 4 — whole-document edit planning and deterministic raw-OCR rendering

The renderer consumes a validated semantic plan and the anchor/format inventory. The default is one
agent request for one document. That request receives:

- the complete page-ordered raw OCR with explicit page boundaries;
- the complete source task label and complete synthetic task label in compact JSON;
- a deterministic typed diff containing field path, entity/role identity, source label value,
  synthetic value, coupling group, and any source evidence or printable aliases;
- field semantics and the relevant categorical display mapping when a canonical label value differs
  from what a document would print; and
- the full-party anonymization and format-preservation rules.

Providing the whole document lets the model resolve repetitions, cross-page copies, composite rows,
and nearby flavor data in one coherent operation. The agent is not asked to rewrite the text. It
returns only a compact list of typed exact-context edits:

```text
TextEdit
  page_number
  kind: target_fact | auxiliary_sensitive
  target_paths[]
  entity_id / coupling_group
  exact_old_text
  replacement_text
  prefix_context / suffix_context
  expected_occurrences
```

Character offsets are deliberately not model output: asking a language model to count characters is
unnecessary and brittle. Deterministic code resolves each exact quote plus context to unique offsets,
records those resolved spans in the receipt, and applies non-overlapping edits from the end of each
page toward the beginning. A repeated semantic fact is represented by an explicit edit for each
intended surface or by an occurrence policy whose complete match set is proven locally.

The planner covers both task facts and residual party-sensitive flavor data in the same request.
For example, it can identify a VAT, tax, registration, account, or unlabeled contact value inside a
party block even though that value is intentionally absent from both task labels, and return a
shape-appropriate synthetic replacement that remains outside the target. Known deterministic
detectors seed this inventory; the model handles semantic residuals rather than receiving a second
routine review call.

The deterministic executor must:

1. recheck all source, target, evidence, and template hashes;
2. resolve every proposed quote and context uniquely, never call global `str.replace`;
3. reject missing, overlapping, protected, or unexpectedly repeated matches;
4. couple repeated role-equivalent surfaces so one semantic fact is consistent everywhere;
5. apply non-overlapping edits from the end of each page toward the beginning;
6. preserve page order, headings, separators, units, and OCR/template grammar;
7. require dependent totals to agree with the already reconciled semantic plan; and
8. fail on stale, missing, ungrounded, or unresolved repeated evidence.

The Pydantic output schema enforces shape and primitive constraints only. Requirements such as
complete diff coverage, identical coupled replacements, evidence grounding, absence of old party
data, and unchanged bytes outside edits are checked by deterministic validators. Encoding those
cross-record invariants as Pydantic model validators would turn otherwise valid structured output
into costly contract-repair loops.

The normal path uses provider-native structured output and no agent tools. All required context is
already present, so tool round trips add cost without adding information. One bounded repair request
is allowed only after a deterministic validator returns precise edit-level errors; a second failure
holds the document instead of starting an unbounded retry/review cascade. An occurrence-query or
patch-preview toolset may be enabled later for a measured class of ambiguous documents, but it is
an explicit escalation backend rather than the default.

The agent cannot write files, apply patches, change synthetic semantic values, or return an
unconstrained whole-document rewrite. It also returns no chain-of-thought or narrative rationale;
the edit list, validation report, local transcript, usage, and cost receipt are the audit record.

Rendering operates only on raw text. It does not generate or modify a PDF, and it does not rerun
OCR. Page separators, headings, line/block grammar, formatting conventions, and intended OCR noise
come from the selected real raw-text skeleton.

Party blocks have two coordinated edit classes. The semantic class replaces target facts such as
name, address, city, country, and labeled contacts. The anonymization class replaces residual
sensitive identifiers such as VAT, tax, registration, or unlabeled contact values while preserving
their labels and surface patterns. Publication fails if any known source-sensitive value remains or
if an auxiliary replacement accidentally appears in the target.

A deterministic pre-planner may still emit obvious scalar edits for identifiers, simple dates,
numeric values, and unambiguous fixed-layout rows. Those edits are supplied as fixed/protected input
to the same planner or bypass the model entirely; they do not justify a separate model call. The
unresolved 4,471 repeated anchors must be resolved by exact document context or remain ineligible;
ambiguity must not be resolved by occurrence number alone.

### Stage 5 — bounded PydanticAI linguistic realization

PydanticAI is reserved for genuinely linguistic work:

- controlled goods-description variation from fixed HS/product/package/DG facts;
- synthetic party/address realization from fixed locality and contact facts;
- realistic auxiliary-sensitive replacements that preserve a detected surface role and format; and
- diagnosis of a failed round trip, without permission to publish a correction.

Use provider-native `NativeOutput` when the configured provider/model supports it, otherwise strict
tool output; prompted JSON is not the default. Each output is a small Pydantic contract. Apply
request, input-token, output-token, and total-cost limits, bounded output retries, concurrency
limits, and immutable message/usage/cost receipts. A model may phrase text but may not choose codes,
relations, quantities, or authoritative facts. Static system instructions and schemas are kept
stable for provider prompt caching, while document-specific payloads are compact and contain no
duplicated prose.

Avoid repetitive names and descriptions through a seed-first process:

1. retrieve a source-supported semantic seed from a versioned catalog or real-train-only cluster;
2. reserve it without replacement within a configured reuse window;
3. give the agent the fixed structured facts, surface constraints, and diverse exemplars;
4. reject exact, normalized, n-gram, and embedding-near duplicates against real and accepted
   synthetic text;
5. maintain per-cluster and per-output frequency caps; and
6. back-extract/check required facts from the realized block before acceptance.

The agent should not be asked open-endedly for “50 company names” or “50 goods descriptions.” That
produces narrow model-prior repetition and weak provenance.

PydanticAI work is deliberately a later phase. The first implementation pass may publish complete
structured semantic plans for audit, but those intermediate plans are not training samples. A final
synthetic sample exists only after full party anonymization, raw-text rendering, regenerated
evidence, and all privacy/round-trip gates pass.

### Stage 6 — target, evidence, and lineage regeneration

Project the reconciled domain tables through the same task adapter to a relation-v3 target, then
reconstruct and validate it. Regenerate synthetic evidence against the final rendered OCR spans.
The synthetic target must be derived from the semantic plan, never from reparsing the model-written
text as the source of truth.

Every record receives:

- synthetic ID and raw-text hash;
- base document, template proxy, source manifest, and split;
- plan, generator, renderer, registry, and prompt versions;
- per-field old/new semantic and raw values;
- exact page spans and repeated-surface coupling groups;
- random-stream identities;
- agent receipts if any; and
- all gate results.

### Stage 7 — hard quality, privacy, and diversity gates

The following are 100% publication gates:

- JSON/task schema and exact domain inverse;
- primary/foreign/local identities and ordering;
- categorical/registry validity;
- chronology, allocation, and measure invariants;
- target-to-final-OCR evidence coverage;
- unchanged-field equality outside declared mutations;
- no stale source fact in a required replacement group;
- no duplicate synthetic ID or final OCR hash; and
- no source/eval/template leakage.

Measure statistical quality separately: marginal distributions, pair trends, relation cardinality,
target-condition yield, source/template concentration, duplicate and nearest-neighbor rates, agent
lexical/semantic diversity, and disclosure risk. Statistical similarity cannot override a hard
failure.

### Stage 8 — downstream utility and publication

Publish a synthetic-only dataset and a separate real-plus-synthetic train projection. Keep real
validation/test immutable. The decisive experiments are paired:

- real only;
- synthetic only, tested on real;
- real plus synthetic at 0.5x, 1x, 2x, and 5x ratios; and
- targeted-cohort ablations.

Report overall and per-field F1/precision/recall, cargo-relation and category scores, schema/JSON
validity, and cohort results for rare fields, complexity, carrier, and seen/unseen templates. Scale
only cohorts and ratios that improve untouched real evaluation.

## 3. Configuration and extension model

The configuration should expose policies and method selections without leaking implementation
internals. Every named method is a registered, versioned component with a strict Pydantic settings
model. Unknown fields and unsupported method/task combinations fail validation.

```yaml
schema_version: 1
task_adapter: mpci_bl_relation_v3

run:
  run_id: mpci-bl-synthetic-pilot-v1
  seed: 4242
  output_dir: artifacts/kie-synthesis

source:
  dataset_manifest: artifacts/.../manifest.json
  allowed_split: train
  template_inventory_manifest: artifacts/.../manifest.json

selection:
  method: scipy_milp_deficit_optimizer_v1
  requested_documents: 250
  maximum_variants_per_source: 2
  minimum_total_template_documents: 3
  minimum_cohort_documents_per_template: 1
  template_share_cap: 0.05
  carrier_share_cap: 0.10
  cardinality_policy: preserve
  cohorts:
    dangerous_goods: {minimum: 50}
    temperature_controlled: {minimum: 50}
    multi_cargo: {minimum: 75}
    multiple_package_levels: {minimum: 75}

modeling:
  default_method: auto_benchmark_v1
  default_candidates: [gaussian_copula, ctgan, tvae]
  selection:
    source_group_folds: 5
    minimum_rows_for_neural: 500
    simplest_within_quality_margin: 0.01
  views:
    scenario: {method: auto_benchmark_v1}
    route_party: {method: auto_benchmark_v1}
    cargo_package: {method: auto_benchmark_v1}
    hma_ablation: {enabled: false, method: hma_depth_one_v1}

generators:
  completeness:
    source_present_fields: require_declared_policy
    unsupported_policy: reject
    preserve_allowed: [container_size_type, non_identifying_categorical]
  identifiers: {method: deterministic_domain_v1}
  dates: {method: joint_bounded_shift_v1}
  quantities: {method: conditional_statistical_then_reconcile_v1}
  relations: {method: preserve_and_reconcile_v1}
  route_scenario:
    method: registry_backed_hierarchical_v1
    distribution:
      method: bilateral_trade_flow_v1
      dataset: {path: registries/trade-flows.jsonl, sha256: "..."}
      weight: trade_value
      year_window: {start: 2022, end: 2025}
      smoothing_temperature: 1.0
      empirical_corpus_mixture: 0.0
    country_eligibility:
      method: active_maritime_trade_v1
      require_positive_flow: true
      require_maritime_locode: true
      allow_domestic_route: false
    preserve_relation_class: true
  party_roles:
    method: route_conditioned_seeded_v1
  registries:
    ports: {path: registries/ports.jsonl, sha256: "..."}
    hs: {path: registries/hs.jsonl, sha256: "..."}
    dangerous_goods: {path: registries/dg.jsonl, sha256: "..."}

rendering:
  preplanner: deterministic_unambiguous_edits_v1
  planner:
    method: pydanticai_whole_document_edit_plan_v1
    input_scope: whole_page_ordered_raw_text
    include_source_label: true
    include_synthetic_label: true
    include_typed_diff: true
    default_tool_calls: 0
    repair_requests: 1
    whole_document_rewrite: forbidden
    execution: deterministic_only
  executor: exact_context_nonoverlap_v1
  repeated_surface_policy: require_complete_context_resolved_set
  unresolved_policy: reject
  output_mode: raw_ocr_text_only
  auxiliary_party_text:
    method: classify_then_format_preserving_replace_v1
    unknown_sensitive_policy: reject

agents:
  enabled_tasks: [linguistic_values, whole_document_edit_plan]
  output_mode: native
  concurrency: 16
  per_call_request_limit: 2
  per_call_output_tokens_limit: 1200
  run_cost_limit_usd: 20
  seed_catalog:
    reuse_window: 500
    maximum_normalized_duplicate_rate: 0.0

anonymization:
  party_target_fields: replace
  auxiliary_sensitive_fields: replace_preserve_surface
  non_sensitive_flavor_text: preserve
  unresolved_sensitive_span: reject

quality:
  hard_gate_policy: all
  nearest_neighbor_threshold: 0.90
  maximum_variants_per_template: 25
  require_human_audit_for_pilot: true

publication:
  publish_synthetic_only: true
  publish_real_plus_synthetic: true
```

These values illustrate the abstraction; quota, threshold, and cost values require the decisions in
the next section. `auto_benchmark_v1` is the recommended statistical default. Explicit method
overrides are for controlled ablations and operational tuning, and every run records the resolved
component choices and learned parameters.

The extension boundary is a `DocumentTaskAdapter`, not conditionals scattered through the engine.
Each task plugin owns:

- the canonical Pydantic target and relational projection/inverse;
- domain tables and modeling views;
- cohort predicates and capability inventory;
- field families, registries, dependency graph, and deterministic validators;
- anchor-path projection and renderers; and
- task-specific quality metrics.

The shared engine owns source/split integrity, configuration, scheduling, random streams, SDV model
selection, agent budgets/receipts, immutable publication, and generic quality/leakage reports. An
invoice plugin can therefore define seller/buyer/line/tax/payment relations; a packing-list plugin
can define package/carton/item/dimension relations without weakening the B/L ontology or forcing one
universal cargo schema.

## 4. Decisions required before generation implementation

### Decision 1 — real split and template leakage policy

**Recommended:** freeze template-grouped real train/validation/test before fitting anything; synthetic
records are train-only and inherit their source group. Decide whether the current 100-document eval
split is replaced by or supplemented with a template-held-out test set. This is a prerequisite, not
an optional quality setting.

### Decision 2 — first-pilot objective and quotas

**Recommended:** a 100–250-record cardinality-preserving pilot, emphasizing supported multi-cargo,
multi-package, multi-container, allocation, temperature, DG, and agent-role cohorts rather than
uniformly multiplying common documents. Exact quotas should follow feasibility output; zero-support
schema fields such as container VGM require real seeds before synthesis.

### Fixed policy — full party anonymization from pilot one

Every party name, address, locality, labeled contact, and auxiliary identifying value is replaced.
VAT/tax/registration and similar distractors retain their presence, label, and source-style surface
form but receive synthetic values and remain outside the KIE label. Every original party-sensitive
value must be absent before publication. This is empirical anonymization with measured leakage
gates, not a claim of formal differential privacy.

### Decision 4 — approved reference registries

Approve sources/licenses for port-country-locality, HS/product, UN/DG, package, container, and any
carrier/owner data. **Recommended:** no DG/HS/route synthesis until these are pinned. Existing
MPCI-derived readable categories remain the output vocabulary; numeric UI codes stay downstream.

Also approve the target destination prior. The source corpus cannot supply this automatically:
among printed port-of-discharge countries it is 92.7% Egypt. The recommended interface accepts a
pinned country/region distribution or explicit minimums and caps, then samples valid seaports and
dependent party localities from approved registries. Uniform-over-country is not an automatic
default because it is unlikely to reflect realistic trade volume. The production UN/LOCODE release
is the recommended route-location backbone because it supplies country, subdivision, location,
function, and status fields; restrict port roles to maritime-function entries and retain the pinned
release and SHA-256 in every run.

The initial distribution method is now defined as a configurable bilateral export-flow artifact,
not independent hard-coded country lists. Build joint origin/destination weights from a pinned
trade dataset, then intersect its positive-flow pairs with countries having eligible maritime
UN/LOCODE entries. This excludes non-commercial or unsupported destinations through data and
registry predicates rather than country-name exceptions. The year window, trade-value versus mass
weight, smoothing temperature, minimum observations/value, regional caps/floors, commodity scope,
and optional corpus mixture remain explicit settings. Publish the included/excluded country and
route audit for every run.

### Decision 5 — statistical model selection budget

The cargo-profile baseline now has a pinned five-fold/five-seed Community SDV benchmark and selects
among four Gaussian Copula representations. CTGAN and TVAE remain declared but fail closed below
the configured 1,000-row/100-template evidence threshold. For each additional modeling view, choose
and benchmark its own support thresholds and fit-time budget rather than inheriting the cargo result.
HMA remains only a possible simplified-view ablation. Enterprise multi-table targeting can be added
as a registered backend later; the core design must not depend on it.

### Decision 6 — linguistic provider, budget, and retention

Choose the primary PydanticAI model and local alternative, maximum cost per run/document, transcript
retention, and human audit rate. **Recommended:** provider-native structured output, all-attempt usage
and cost receipts, narrow prompts, a hard run budget, deterministic validators, and a stratified
human review of the first pilot. LLM output never bypasses hard gates.

### Decision 7 — seed catalogs and acceptable novelty

Choose whether party and goods seed catalogs may include licensed external data, only train-split
real clusters, or both. Define acceptable lexical/semantic distance from real text and maximum
reuse. **Recommended:** external/pinned names where licensing permits; real-train-only goods clusters
conditioned by HS/cargo facts; no source party reuse; without-replacement reservations plus n-gram
and embedding duplicate gates.

### Decision 8 — cardinality expansion

**Recommended:** defer adding/removing cargo rows, packages, or containers until replacement-only
synthesis demonstrates downstream utility. Expansion is a separate version and requires proven
repeat-unit spans, insertion/reflow rules, totals, new identities/relations, and likely template-
specific renderers. Start with one well-supported template family.

### Fixed scope — text-only synthesis

This is no longer an open decision. The pipeline publishes a structured synthetic label and a
page-ordered raw-OCR text variant created from an existing raw-text skeleton. PDF generation,
rerendering, and synthetic OCR passes are outside scope. Cardinality expansion, if later approved,
must be expressed through bounded text-block insertion while preserving the source's textual grammar.

## 5. Implementation sequence after decisions

1. Freeze the source/template split and implement strict generation configuration, task-adapter,
   semantic-completeness, and registry-provider contracts.
2. Implement the MILP cohort/template/base-document planner and infeasibility report.
3. Implement the bilateral trade-flow distribution loader, UN/LOCODE eligibility join, route
   scenario graph, and included/excluded-country audit. Production data paths remain required inputs.
4. Implement deterministic non-linguistic semantic generators and reconciliation, then publish
   typed `DraftScenarioPlan` artifacts with explicit pending linguistic tasks for audit only; they
   are neither task labels nor training rows.
5. Benchmark statistical candidates per modeling view and freeze model-selection receipts.
6. Complete registry-backed route, HS/DG, package/category, temperature, and seed-catalog-backed
   party/cargo generators until every source-present field has a declared complete policy.
7. Implement the semantic diff builder, deterministic obvious-edit pre-planner, and party-block
   residual-sensitive inventory.
8. Add the one-call PydanticAI whole-document edit planner and bounded linguistic realization using
   native/tool output, budgets, receipts, and novelty/back-extraction validation. Keep tools as an
   evidence-based escalation backend rather than the default path.
9. Implement exact-context resolution, the deterministic patch executor, full-party anonymization
   proof, evidence regeneration, and immutable final publication.
10. Run a stratified mutation probe and require 100% hard gates before generating the 100–250-record
    text-only pilot.
11. Run real-only versus mixed-ratio training ablations on untouched real evaluation and scale
    toward 10,000 only when the learning curve identifies useful cohorts/ratios.
12. Treat cardinality expansion and each new document type as separately approved task-plugin work.

### 5.1 Original Passes 1-2 contract and completion boundary

The following contract is retained as the implementation trace. The strict state/policy models,
train-isolated MILP selection, non-linguistic document mutation, profile-routed SDV benchmark,
global reservations, task projection, immutable CLI publication, and inspection reporting are now
complete. Route/locality generation, linguistic realization, and text patching remain intentionally
deferred. The original contract specified:

1. **Strict generation state and policy models.** Add `DraftScenarioPlan`,
   `PendingRealization`, `ResolvedSemanticPlan`, `SemanticChange`, `TextEdit`, `TextPatchPlan`, and
   `PublishedSyntheticRecord`. Add a task-owned policy registry that requires every B/L target path
   to declare one of regenerate, resample, derive, approved non-identifying preserve, or
   legitimately absent. Unknown paths and incomplete policies fail configuration.
2. **Cohort/template/base selection.** Implement the SciPy MILP selector with cardinality-preserving
   cohort eligibility, total-template and cohort-template support, source/template/carrier caps,
   exact quota accounting, deterministic tie-breaking, and an infeasibility report. The selected
   individual raw OCR—not the template-proxy representative—is always the text skeleton.
3. **Route and locality provider boundary.** Implement pinned bilateral-trade and UN/LOCODE provider
   schemas, loaders, hashes, maritime eligibility predicates, a hierarchical route graph, and
   included/excluded country/pair reports. Tests use pinned fixtures; a production run requires an
   explicit trade snapshot, UN/LOCODE snapshot, year window, weight, and thresholds, with no
   hard-coded country-name fallback.
4. **Complete deterministic draft semantics.** Extend the existing order-independent generators for
   identifiers, dates, quantities, weights, allocations, and compatible relations. Add route-first
   party-role dependencies and explicit pending tasks for linguistic party/goods values. A draft
   with pending work cannot project to a task label, so partial synthetic records cannot enter
   training.
5. **One-call edit-planner implementation.** Implement the compact source/synthetic diff, strict
   PydanticAI `TextPatchPlan` output, reusable system prompt, whole-document request builder,
   transcript/usage/cost receipts, and exactly one validator-driven repair allowance. The output
   contract has no rationale and no model-generated offsets. Tools are disabled by default.
6. **Deterministic execution and proof.** Resolve exact quotes plus context, apply non-overlapping
   edits, and prove diff coverage, coupled-value consistency, target grounding, source-party-value
   removal, auxiliary-value exclusion from labels, page-order preservation, and byte equality
   outside declared edits. Add fake-model tests and hand-authored real-document fixtures for target
   and auxiliary-sensitive edits.
7. **CLI, immutable artifacts, and measurements.** Add separate plan and render-pilot commands,
   strict YAML, atomic manifests, resumable per-document states, and reports for feasibility,
   generator support, country eligibility, planner acceptance, retry/hold taxonomy, latency,
   input/output tokens, and cost.

The completed structured pass fits only bounded cargo-profile SDV proposal models. It does **not**
invent a trade prior, generate PDFs, mutate the 1,157-row real corpus, or publish synthetic training
rows. The text-renderer-specific acceptance gates in the original contract remain future gates:

- all new config/error branches covered by targeted tests;
- exact relational projection/inverse retained on the pinned source corpus;
- deterministic output independent of concurrency and resume order;
- zero changed bytes outside declared edit spans in every render fixture;
- all old party target and detected auxiliary-sensitive values absent from rendered fixtures;
- benchmarked selector, generator, and patch-executor throughput and peak memory; and
- no network/API call in the default test suite.

After those remaining gates pass, the subsequent rendering pass may end with an explicitly
approved, hard-budgeted
eight-document agent probe: one simple document and one each stressing multi-page repetition,
multi-container, multi-cargo/package relations, dangerous goods, temperature, auxiliary party data,
and ambiguous repeated evidence. The probe defaults to one request per document, at most one repair,
and abort-before-request cost enforcement. It publishes single-pass acceptance, repair and held
rates, complete validator taxonomy, latency, tokens, and provider-receipted cost. It does not publish
the records as training data.

The current concrete end state is a runnable, audited structured semantic-planning engine with
measured statistical proposals and no training publication. The next boundary is to pin production
route/party/goods registries and seed catalogs, complete linguistic and raw-text realization, and
then run the first 100–250 fully anonymized text-only synthetic pilot.

## 6. Authoritative references used for the design

- [SDV multi-table synthesizer comparison](https://docs.sdv.dev/sdv/multi-table-data/modeling/synthesizers)
- [SDV HMA synthesizer scope and schema simplification](https://docs.sdv.dev/sdv/multi-table-data/modeling/synthesizers/hmasynthesizer)
- [SDV multi-table conditional sampling](https://docs.sdv.dev/sdv/multi-table-data/sampling/conditional-sampling)
- [SDV constraints and probabilistic-rule boundary](https://docs.sdv.dev/sdv/multi-table-data/modeling/customizations/constraints)
- [PydanticAI structured output modes](https://pydantic.dev/docs/ai/core-concepts/output/)
- [PydanticAI tools and toolsets](https://pydantic.dev/docs/ai/tools-toolsets/tools/)
- [PydanticAI agents, usage accounting, and limits](https://pydantic.dev/docs/ai/core-concepts/agent/)
- [PydanticAI durable execution options](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/)
- [UN Trade Statistics and UN Comtrade](https://unstats.un.org/unsd/trade/)
- [UN/LOCODE Recommendation 16 and maritime function codes](https://unlocode.unece.org/recommendation16/)
