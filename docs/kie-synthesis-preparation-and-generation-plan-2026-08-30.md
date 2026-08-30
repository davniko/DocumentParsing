# KIE synthesis preparation and generation plan

Date: 2026-08-30  
Status: non-generative preparation implemented and validated; generation design ready for decisions

This document records what is now proven for the exact 1,157-document MPCI bill-of-lading corpus,
what the preparation artifacts mean, and how the subsequent augmentation system should be built.
It deliberately separates implemented foundations from proposed generation behavior. No synthetic
training row and no paid PydanticAI call was produced in this pass.

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

## 2. Recommended generation architecture

The pipeline should be a staged compiler from a pinned real document to a constrained synthetic
plan, not a single generative model call.

```text
immutable real corpus and source-grouped split
  -> target cohort deficit planner
  -> eligible base/template selector
  -> typed scenario proposal
  -> deterministic semantic reconciliation
  -> role-aware surface renderer
  -> optional bounded PydanticAI linguistic realization
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

### Stage 1 — cohort deficit and template selection

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

### Stage 2 — typed statistical proposals

Do not fit one synthesizer to the full 17-table graph. SDV documents public HMA as optimized for
roughly five tables and one relationship level, while this graph is deeper and contains bridge
relations. Use compact modeling views whose outputs are still only proposals:

1. document/scenario view: type, carrier/template family, route class, counts, feature flags;
2. route/party-locality view;
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

### Stage 4 — role-aware OCR surface rendering

The renderer consumes a validated semantic plan and the anchor/format inventory. It must:

1. recheck all source, target, evidence, and template hashes;
2. identify a scalar field span or a bounded block span, never call global `str.replace`;
3. render according to the role-specific format profile;
4. couple repeated role-equivalent surfaces so one semantic fact is consistent everywhere;
5. apply non-overlapping edits from the end of each page toward the beginning;
6. preserve page order, headings, separators, units, and OCR/template grammar;
7. recompute and render dependent totals from semantic facts; and
8. fail on stale, overlapping, missing, or unresolved repeated evidence.

A deterministic renderer should cover identifiers, simple dates, simple numeric values, registry
tokens, and fixed-layout rows. Composite evidence needs an explicit field parser or block renderer.
The unresolved 4,471 repeated anchors should be handled by a unique role/order solution or remain
ineligible; ambiguity must not be resolved by occurrence number alone.

### Stage 5 — bounded PydanticAI linguistic realization

PydanticAI is reserved for genuinely linguistic work:

- controlled goods-description variation from fixed HS/product/package/DG facts;
- one bounded cargo/marks/handling block when deterministic formatting cannot express it;
- synthetic party/address realization from fixed locality and contact facts; and
- diagnosis of a failed bounded round trip, without permission to publish a correction.

Use provider-native `NativeOutput` when the configured provider/model supports it, otherwise strict
tool output; prompted JSON is not the default. Each output is a small Pydantic contract. Apply
request, input-token, output-token, and total-cost limits, bounded output retries, concurrency
limits, and immutable message/usage/cost receipts. A model may phrase text but may not choose codes,
relations, quantities, or authoritative facts.

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
  method: constrained_deficit_optimizer_v1
  requested_documents: 250
  maximum_variants_per_source: 2
  minimum_template_support: 3
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
  identifiers: {method: deterministic_domain_v1}
  dates: {method: joint_bounded_shift_v1}
  quantities: {method: conditional_statistical_then_reconcile_v1}
  relations: {method: preserve_and_reconcile_v1}
  registries:
    ports: {path: registries/ports.jsonl, sha256: "..."}
    hs: {path: registries/hs.jsonl, sha256: "..."}
    dangerous_goods: {path: registries/dg.jsonl, sha256: "..."}

rendering:
  scalar_method: role_aware_anchor_v1
  block_method: bounded_template_block_v1
  repeated_surface_policy: require_unique_role_solution
  unresolved_policy: reject

agents:
  enabled_tasks: [goods_description, party_block, bounded_cargo_block]
  output_mode: native
  concurrency: 16
  per_call_request_limit: 2
  per_call_output_tokens_limit: 1200
  run_cost_limit_usd: 20
  seed_catalog:
    reuse_window: 500
    maximum_normalized_duplicate_rate: 0.0

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

### Decision 3 — augmentation versus anonymization

Choose whether every party/contact/address must be replaced or whether the pilot only varies selected
training facts. **Recommended for eventual full synthesis:** replace all party and identifying values
and report empirical leakage, without claiming formal differential privacy. This materially expands
renderer and catalog work and should not be implied by the word “synthetic.”

### Decision 4 — approved reference registries

Approve sources/licenses for port-country-locality, HS/product, UN/DG, package, container, and any
carrier/owner data. **Recommended:** no DG/HS/route synthesis until these are pinned. Existing
MPCI-derived readable categories remain the output vocabulary; numeric UI codes stay downstream.

### Decision 5 — statistical model selection budget

Choose allowed fit time and benchmark candidates. **Recommended:** Community SDV with per-view
Gaussian Copula/CTGAN/TVAE benchmark and HMA simplified-view ablation. Enterprise multi-table
targeting can be added as a registered backend later; the core design must not depend on it.

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

### Decision 9 — OCR-text-only versus rendered-document variants

**Recommended for version 1:** patch audited raw OCR and preserve existing page/layout grammar. For
large repeated-block expansion, decide whether to render a PDF and pass it through GLM-OCR so OCR
noise and layout remain realistic. Do not mix those two provenance classes without labeling them.

## 5. Implementation sequence after decisions

1. Freeze the source/template split and approved registries.
2. Implement strict generation configuration and the task-adapter interfaces.
3. Implement the constrained cohort/template planner and feasibility report.
4. Implement field and bounded-block renderers, including repeated-surface coupling.
5. Benchmark statistical candidates per modeling view and freeze model-selection receipts.
6. Complete registry-backed route, HS/DG, category, and temperature generators.
7. Add bounded PydanticAI tasks with seed reservation, structured output, budgets, receipts, and
   novelty/back-extraction validation.
8. Run a deterministic mutation probe on stratified real documents and require 100% hard gates.
9. Generate the 100–250-record pilot, audit it, and publish synthetic-only data.
10. Run real-only versus mixed-ratio training ablations on untouched real evaluation.
11. Scale toward 10,000 only when the learning curve identifies useful cohorts/ratios.
12. Treat cardinality expansion and each new document type as separately approved task-plugin work.

## 6. Authoritative references used for the design

- [SDV multi-table synthesizer comparison](https://docs.sdv.dev/sdv/multi-table-data/modeling/synthesizers)
- [SDV HMA synthesizer scope and schema simplification](https://docs.sdv.dev/sdv/multi-table-data/modeling/synthesizers/hmasynthesizer)
- [SDV multi-table conditional sampling](https://docs.sdv.dev/sdv/multi-table-data/sampling/conditional-sampling)
- [SDV constraints and probabilistic-rule boundary](https://docs.sdv.dev/sdv/multi-table-data/modeling/customizations/constraints)
- [PydanticAI structured output modes](https://pydantic.dev/docs/ai/core-concepts/output/)
- [PydanticAI agents, usage accounting, and limits](https://pydantic.dev/docs/ai/core-concepts/agent/)
- [PydanticAI durable execution options](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/)
