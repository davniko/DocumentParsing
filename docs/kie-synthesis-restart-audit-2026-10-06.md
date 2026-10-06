# Synthesis restart: current-state audit and bounded restart plan

Date: 2026-10-06. Investigation only: no production source, template, dataset,
training configuration or provider setting was changed. No paid requests or
training were launched.

## Decision in brief

Reuse the byte renderer, deterministic identifiers/arithmetic, registries,
dependency checks and run/cost bookkeeping. Rebase a **selected set of current
training sources onto the current reduced V7 contract**, rather than restart the
old 1,507-template campaign or blindly convert its old generated targets.

There is a concrete starting pool: **66 lower-complexity candidates across 23
carrier-family labels**, with an explicit inventory and a proposed **24-source
first experiment**. These are candidates for rebinding/testing, not already
V7-ready templates. The next deliverable is a working, measured V7 synthesis
slice over that bounded set—not another broad historical repair campaign.

The existing synthesis entry point is not ready for the new dataset. Its
planner/publication contract is explicitly V5, and its address projection
implements the old city/country-stripping policy. This is a confirmed
compatibility problem, not a speculation about model performance.

## Evidence and reproduction

- [Inventory, per-source traits and exact loader outcomes](analysis/synthesis-restart-20261006/inventory.csv)
- [Summary and input hashes](analysis/synthesis-restart-20261006/summary.json)
- [Lower-complexity candidates](analysis/synthesis-restart-20261006/seed-candidates.json)
- [Focused probes, including a concrete address counterexample](analysis/synthesis-restart-20261006/probes.json)
- [Proposed 24-source experiment](analysis/synthesis-restart-20261006/proposed-pilot.json)
- [Inventory script](analysis/synthesis-restart-20261006/audit.py) and [probe script](analysis/synthesis-restart-20261006/probes.py)

Run from the repository root:

```bash
.venv/bin/python docs/analysis/synthesis-restart-20261006/audit.py
.venv/bin/python docs/analysis/synthesis-restart-20261006/probes.py
```

The scripts write investigation outputs only into their own directory. They
do not call providers. Source datasets/catalogs are read-only. Summary hashes
pin the inspected datasets/catalog inventories; the actual source loader also
checks individual source/template identities. Elapsed time and peak RSS for the
final inventory run were **38.627 seconds and 220.1 MiB** and are recorded in the summary. This is a CPU/filesystem audit, not a
throughput estimate for future LLM generation.

The focused existing suite passed **177 tests in 14.44 seconds**:

```bash
TMPDIR=/tmp .venv/bin/pytest -q \
  tests/test_synthesis_complete_pipeline.py \
  tests/test_synthesis_latest_template_target.py \
  tests/test_synthesis_party_address_roles.py \
  tests/test_synthesis_generation_realization_contract.py \
  tests/test_synthesis_cargo_description_role.py \
  tests/test_synthesis_equipment_projection.py
```

These tests establish working existing mechanisms, not a V7 end-to-end run.

## 1. Which assets are relevant now?

### Current authority

`data/curated/mpci-bl-real-v7-reviewed-r16-paddle-positions-660` contains
600 training and 60 validation records. The position-enriched variant retains
original OCR separately; labels are the R15 address-punctuation labels, with
the preceding casing and equipment repairs. Use `joinedRawText`, not text
containing `|| x,y`, when compiling semantic templates. Use the current
`target` plus the dataset's frozen reduced schema as the label authority.

Relevant policy/code:

- [Current baseline policy](kie-real-baseline-label-policy-2026-10-05.md)
- [V7 field models](../src/document_ocr/label_schemas/bill_of_lading_v7.py)
- [Task adapters and reduced projection](../src/document_ocr/training/tasks.py)

Current scope is one goods accounting group, possibly many products/HS codes
and many containers. It is not one product name per goods item. Addresses are
`addressLine` plus `country`, with no separate city target. Product wording,
including product qualifiers/package capacity, belongs in description; V7 has
no AAI/product-overflow field. Human-readable text uses the approved uppercase
policy. Addresses retain printed delimiters and join physical lines with spaces.
The later conversation left addresses alone: this audit makes no new address
format change.

The reduced model target omits carrier party, vessel flag, marks/numbers and
forwarding/export references. Those can still be printed context in inputs;
omitting a target field is not authorization to leave contradictory source text
behind when its related generated entity changes.

### Historical synthesis assets

Two principal catalogs were inspected in depth:

| Catalog | All entries | Current training sources | Current validation sources |
|---|---:|---:|---:|
| Full V34 discharge-country catalog | 1,507 | 439 | 40 |
| V38b provisional GROUND-015 subset | 1,212 | 357 | 25 |

Both live under `artifacts/kie-synthesis-production/template-base/catalogs/`.
V38b's own summary says `provisional_cargo_role_only_not_full_synthesis_ready`
and `partySourceRoleCertificationComplete: false`.

The union covers 479 current documents: 439 train and 40 validation. The
remaining 161 train/20 validation do not occur in these two catalogs. A metadata
scan of all 39 catalog indexes found one of those training sources in earlier
catalogs only (`doc_87465a0c…`); 160 train/20 validation are absent from all 39
scanned catalog indexes. This does not mean those documents cannot be compiled.
It means we should not assume an existing current catalog entry for them.

Historical material that remains useful, but is not a new launch instruction:

- [Template recovery handoff](kie-template-recovery-handoff-2026-09-22.md)
- [Recovered generation campaign](kie-recovered-synthesis-campaign-2026-09-22.md)
- [Equipment/numeric-sidecar handoff](kie-v31-equipment-and-numeric-sidecar-readiness-2026-09-26.md)
- [GROUND-015 address upstream requirements](../artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground015_cargo_text/ADDRESS_TEMPLATE_AND_SYNTHESIS_REPAIR_REQUIREMENTS.md)

The address handoff includes earlier address+city+country proposals. The current
addressLine+country decision supersedes those target definitions, while its
findings about split slots and duplicate locality generation remain relevant.

## 2. Actual production data flow

```text
source OCR + historical accepted labels
  -> compiler proposes owned byte spans, semantic bindings and dependencies
  -> deterministic host checks + compiler/critic review -> committed catalog
  -> source/validation selection, quotas and repeat plan
  -> route / party-geography sampling + joint cargo/equipment sampling
  -> deterministic identifiers, dates, quantities, masses and allocations
  -> host-owned lexical values + small linguistic generation requests
  -> frozen target + per-slot values + dependency receipts
  -> exact byte-span rendering + coherence/grounding checks
  -> old address/package projection -> V5 training records
```

Entry point: `document-kie-synthesis` in `pyproject.toml`, routed through
`src/document_ocr/synthesis/cli.py`. The relevant end-to-end generation command
is `run-complete-compiled-synthesis`; the older `run-raw-text-pipeline` is a
different historical workflow and should not be revived accidentally.

| Stage | Main implementation | What it owns |
|---|---|---|
| Compilation | `template_compiler/pipeline.py`, `agents.py`, `host.py`, `models.py` | Source ownership, literal/mutable classification, repeated/segmented bindings, dependencies, critic receipts |
| Catalog/selection | `production_catalog.py`, `production_synthesis.py` | Hash-pinned catalog, source exclusions, capability counts, reuse balancing, exact-ID or layout-proxy validation exclusion |
| Route/party geography | `route_plan.py`, `route_projection.py`, `shipment_scenarios.py` | Train-fitted origins/routes, UN/LOCODE/ports/localities, trade-flow support, parties' own localities, route constraints |
| Cargo scenario | `cargo_scenarios.py`, `transport_capacity.py`, package/equipment/DG registries | Coupled goods/HS/DG/thermal/equipment/packaging support; capacities, mass/volume, units and quantity constraints |
| Complete target | `complete_targets.py` | Deterministic identifiers and arithmetic, target proposal, lexical requests and finalization |
| Linguistic values | `complete_pipeline.py`, `lexical_facts.py`, `prompts/complete_generation.md` | Batched/cached provider values, controlled cargo wording, typed party slot generation, residual requests |
| Rendering/publication | `descendant.py`, `raw_text_template.py` | Replacements, unchanged literals, source format, repeated facts, frozen-target checks, final record/lineage publication |

Provider requests use PydanticAI, structured outputs, configurable concurrency
(current config maximum 16), batching, response caches, attempt limits and a
spending guard. These useful mechanisms need not be reinvented. Not all text
comes from an LLM: in the joint scenario branch, `lexical_facts.prepare` can
own descriptions before the linguistic request.

### Sampling is already richer than random field replacement

The route layer uses train observations plus registry exploration. Historical
production settings mixed observed exporter distributions with broader registry
sampling. Cargo generation combines registry identities with package/equipment
and physical constraints, rather than sampling HS, UN, temperature, package
quantity and weight independently. Identifiers and arithmetic are host-owned.

However, current route/cargo fit configs point to the old 1,057-source training
split and old validation split. Cargo fit code explicitly reads `cargoGroups`,
`cargoPackages`, `containers` and old allocation structures. Simply pointing
these configs at the new JSONL is not a supported V7 fit path. Refit on the 600
current training sources through an explicit, tested scenario-data interface.
Do not reuse historical priors containing now-held-out sources.

The amount of template reuse and all quotas should be explicit. Balanced use
of source templates and balanced use of generated goods/localities are separate
questions; many samples from a small set of layouts do not add layout diversity.

## 3. Template syntax and what its validation really proves

These are structured JSON templates, not loose Jinja/string replacement.
Each slot has an exact UTF-8 byte interval, original text/hash, role, target
paths and format envelope. Semantic bindings collect occurrences and specify
single/repeated/segmented/projected rendering, auxiliary facts or derivations.
Unbound bytes remain literal. Source-only facts and carrier text are represented
separately from training target fields.

The low-level renderer is deterministic. The compiler checks exact span content,
UTF-8 boundaries, disjoint slots and source round-trip; rendering rejects a
changed source hash or missing/extra slots. There are also semantic/coherence
guards above that layer.

**Observed:** all 479 matched V34 versions and all 382 matched V38b versions
passed exact identity replay and an independent literal-region/sentinel replay
in this audit: **861 versioned template checks across 479 distinct documents**.
The independent check compared actual output byte regions, not just the
renderer-returned proof flags.

This is valuable evidence that the text replacement machinery is reusable.
It does not prove that a region labeled “address” contains only an address, or
that a frozen literal remains true after cargo/country changes.

### Falsification probe: a byte-correct but semantically wrong address

In [the current OCR for `doc_00388da5…`](../data/curated/mpci-bl-real-v7-reviewed-r16-paddle-positions-660/inputs/original/doc_00388da5efb560ae60c645831f6b34db7049b9c0e0772d76eecda0d7387de900.txt),
the shipper block prints:

```text
UNIT NO. 201, 2ND FLR, APURUPA PCH COMPLEX,
8-2-293/A/A/1, ROAD, NO.2, BANJARA HILLS
HYDERABAD-34 TELANGANA INDIA.
```

The old address binding owns the first two lines **and `TELANGANA`**; the city
and country are separate bindings. A local lower-level probe supplied the new
full address `14 NEW BUSINESS PARK, INDUSTRIAL ZONE, MUMBAI 400001, MAHARASHTRA INDIA`
to the old generic segmented-address mechanism. It rendered:

```text
14 NEW BUSINESS PARK,
INDUSTRIAL ZONE, MUMBAI 400001, MAHARASHTRA
HYDERABAD-34 INDIA INDIA.
```

Literal/page/line/format checks passed. The source-sized word partition put
`INDIA` in the region slot. This deliberately incomplete lower-level mutation
**was not accepted or published by the production workflow**. The newer
`_require_party_address_role_pin` guard correctly rejected uncertified mutable
address generation in the negative probe. The lesson is precise: keep the byte
engine and the guard; replace this address contract, do not bypass the guard.

## 4. Measured compatibility and repair traits

The following are counts over the 357 current **training** sources in V38b.
Traits overlap and are not independent defect counts.

| Trait | Sources | Interpretation |
|---|---:|---|
| Exact same source OCR as current real dataset | 346 | Layout/span reuse is potentially straightforward |
| Source OCR differs | 11 | Rebase offsets and inspect differences; do not reuse old spans blindly |
| Current legacy source loader passes | 355 | Existing source contracts load, still against old labels |
| Loader rejects party evidence ownership | 2 | Specific old binding/label problem, not an irreparable current document |
| At least one multi-occurrence address binding | 238 | May be segmentation or repetition; needs postal ownership, not proportional word distribution |
| Old separate party city labels | 345 | Old-to-new address target change is widespread |
| Agent-residual bindings | 173 | Some rendering was not fully deterministic; inventory the reason before admission |
| Semantic-only facts | 152 | Often negotiability, not automatically wrong; confirm the source rule/dependency |
| Old description matches current description ignoring case | 299 | A useful reuse signal, not proof of complete cargo equivalence |
| Description differs beyond case | 58 | Includes formatting and semantic changes; rebase against current gold |
| Old multiple cargo groups, now one goods entry | 5 | Old group/placement topology cannot just be copied |
| Shared non-party scalar paths have differing values ignoring case | 165 | Additional reason not to trust old labels; not all differences are defects |
| Current DG sources | 4 | Broad old catalog had 10 of the current DG sources |
| Current temperature-bearing sources | 19 | Need coupled equipment/settings/handling validation |

The two source-loader failures are `doc_914681ee…` (consignee ownership) and
`doc_fabe32eb…` (delivery-agent contact ownership), with full errors in inventory.
Every matched V34 source fails the **current** source loader because it lacks
the later required `goods-role-certificate.json`. That is an admission/provenance
incompatibility; it is not evidence all 479 old byte templates are malformed.

V34 has AAI in 82 of the 439 matched training sources and 15 of 40 validation
sources. V38b has no source AAI in its matched subset. This shows some GROUND-015
cargo work reached templates; it does not establish that the later real V7
descriptions, package totals and relations have been adopted.

### Concrete label differences

1. **Postal order (`00388da5…`).** The old address ends in `BANJARA HILLS
   TELANGANA`, with separate city `HYDERABAD-34` and country `INDIA`. The current
   `addressLine` correctly orders `… BANJARA HILLS HYDERABAD-34 TELANGANA INDIA`.
   Concatenating old address+city+country would put the city in the wrong place.

2. **Simple-looking ice-cream source (`33546e11…`).** Description remains
   `ICE CREAM PRODUCTS`, but current goods include 11,870 kg gross weight and
   a handling instruction absent from the old cargo group. Old description
   equality is not sufficient to adopt its complete old target.

3. **Thermal source (`0951955d…`).** Current goods retain both the printed
   ventilation instruction (10 CBM/hour) and the carrying-temperature wording
   (-3°C), whereas the old goods group contains only the ventilation instruction.
   A shipment volume and a ventilation flow must not share a numeric dependency.

4. **DG/multi-container source (`7d44dcc4…`).** Current target is one
   `POTASSIUM AMYL XANTHATE` goods entry, UN3342, with 80 boxes, 77,600 kg,
   104 m³ and four 20-box placements. The old cargo group lacks those total
   mass/volume fields. These source-accounting differences must be resolved when
   rebuilding dependencies, not by only renaming `containerNumber`.

5. **Source-byte drift (`06984148…`).** The catalog split `LAPTOP SHOP LLC.`
   from the next street line, whereas current authentic OCR has
   `LAPTOP SHOP LLC.31 RICHAR RD.` on one line. Both can be interpreted, but the
   old byte offsets and repair provenance cannot be transplanted unchanged.

These source files are available under the V38b catalog `cases/<documentId>/`;
the current originals/labels are under the R16 dataset's `inputs/original/` and
`labels/`. Full IDs and paths are recoverable from the linked inventory.

## 5. Confirmed changes needed before another synthesis run

### A. Current schema boundary, not a superficial schema-version edit

`latest_target_from_source` accepts only V3/V5 source targets. A real current V7
record was explicitly rejected in the probe. `ProductionSynthesisPlanConfig`
pins V5, and complete-target validation/publication use the V5 task adapter.

Required: a supported current-contract compiler/plan/publisher path with nested
goods placements, the reduced field mask, current category vocabularies and
casing. Keep richer **private** scenario metadata (outer packages, source-only
totals, postal geography) where rendering needs it. Do not expose obsolete
coverage/group IDs/AAI/city fields to the model or silently discard facts during
a late V5-to-V7 conversion.

### B. Generate one postal bundle per party; render from its ownership map

The old publication projector still transforms:

```text
12 EL MAHATTA STREET, DAMIETTA 34516, EGYPT
-> 12 EL MAHATTA STREET, 34516
```

That was directly reproduced. It is the old contract working as written, and
must not sit on the new publication path.

For a contiguous postal region, one generated postal address is enough. For
separate city/region/postcode slots, request one coherent party bundle with the
components needed by that layout in **one request**, then render them to their
declared slots. There is no need for one LLM call per building/street component,
nor a universal fine-grained address ontology for a simple full-address block.

Construct `addressLine` from the final owned postal surfaces in source order,
join line breaks with spaces, preserve rendered delimiters, apply casing once.
Country remains separately labeled if printed. Do not append a second city or
country to a region already containing it. Repeated party copies render the
same bundle; they do not concatenate into the label. Contacts, tax IDs, party
names and captions remain separately owned and protected.

Where multiple postal surfaces are non-contiguous, the source-level ownership
decision is made once during template admission. Subsequent descendants should
not require another open-ended address-extraction/repair process. A doubtful
source can be left outside the selected generation pool without holding up the
other admitted sources.

### C. Goods wording and accounting must share one scenario

Retain the agreed single-goods accounting scope. Multiple HS codes are valid.
Repetitions across containers/pages do not create new goods; distinct
product-accounted totals require a different capability and stay outside this
first expansion. Preserve observed membership-only placements without inventing
container-specific quantities. Known complete splits must sum exactly.

The old host lexical path can append `; product reference <12 hex characters>`
and can force whole tariff definitions into descriptions. The generator prompt
also still describes AAI roles and carries a large set of old projection rules.
This may preserve registry identity while producing unnatural, repetitive
training text. It is a **distribution/learnability risk**, not proof of why a
past model stalled.

Use registry definitions as constraints on goods identity; produce commercial
wording with source-like detail and length. Product codes, brand/lot/grade and
capacity text should exist only as coherent product attributes, not padding to
fill old slots. Fixed product-specific text must either constrain the sampled
goods family or become a declared dependent slot. Do not insert an unrelated
commodity into a template still printing old product qualifiers.

For efficient initial generation, constrain each admitted layout to a compatible
goods family and vary valid product/party/numeric attributes. Broader registry
sampling can be enabled after its lexical and physical combinations pass the
same checks. This is preferable to unconstrained random HS selection followed
by repeated attempted repairs.

### D. Equipment, temperatures, units and numeric contracts

The recent alias work reached the shared equipment resolver and its tests.
Reuse it. However, V7 permits a known type without a printed size; old target
construction requires a complete pair before replacing `typeDescription`.
Private physical sampling can know dimensions without adding unprinted size to
the public target. Reefer equipment does not by itself authorize a temperature;
NOR/non-operating equipment and ventilation/settings must retain their distinct
meanings and dependencies.

The numeric-sidecar issue is more specific than the old handoff suggests: a
later `mpci-bl-numeric-contracts-discharge-country-v6/contracts.jsonl` now exists.
Against the 357 matched V38b training sources:

| Existing numeric-row identity | Sources |
|---|---:|
| Source/target/effective-template hashes match | 333 |
| Missing row | 13 |
| Target hash differs | 8 |
| Source, target and template hashes differ | 1 |
| Source loader blocked | 2 |

This probe checked identity, **not complete arithmetic replay or catalog commit
certification**. All rebased V7 sources need current pins and verified equations.
Reuse unchanged equations after evidence checks; only new/changed quantities
need fresh adjudication. Full 1,507-source restoration is unnecessary for a
selected campaign. Distinguish shipment totals, container splits, tare,
per-package capacity and ventilation; units and printed precision are part of
each equation. Exact derived totals remain allowed under the agreed policy
when all components and ownership are established.

### E. Split safety and sampling priors

Exclude the current 60 validation sources and their known duplicate shipments
from templates, prior fitting and caches of exemplar content. Historical train
membership is not sufficient: the validation split changed.

The lower-complexity shortlist also conservatively excludes known validation
layout-proxy overlaps using both catalogs. V38b has 80 matched training sources
with such a proxy overlap; this is **not proof of duplicate shipment leakage**.
Carrier/layout similarity and source identity are different. The 20 validation
sources lacking catalog metadata still require a duplicate/layout check against
the final proposed pool; the shortlist is not a complete layout-disjointness
certificate. Do not drop existing real training records based on this audit.

Refit quotas on current training data, then state intentional oversampling of
multi-container rows, long descriptions, DG/thermal or rare types. Inspect the
actual generated distributions and source-reuse concentration. Do not promise
that a certain synthetic sample count will achieve 0.95 model metrics.

### F. Positional inputs are a separate integration requirement

R16 alignment files contain per-line source text, page, region boxes and
coordinates. The synthesis renderer operates on text bytes; it currently does
not generate spatial inputs.

Do not realign new names/products against the original Paddle-recognized words
as if they were the same OCR, and do not blindly attach old centroids to text
whose row structure has changed. For the first spatial-capable subset, preserve
the source line/region structure and attach declared **synthetic layout anchors**
through the render receipt. These are borrowed layout positions, not measured
bounding boxes of nonexistent new document images. Compare this approximation
in a controlled pilot before using it at scale.

Changed wrapping, combined columns, new row counts or ambiguous source alignment
need explicit region-aware handling, or the existing empty-coordinate marker.
All plain and positioned variants must share identical labels and strip back to
identical text. Plain-text synthesis readiness and spatial-synthesis readiness
should be recorded separately, so this experiment does not silently delay or
contaminate all synthesis.

## 6. Bounded execution plan

### Stage 1 — one working V7 slice

Start with the named **24-source manifest**: 16 lower-complexity core sources,
then two DG, two thermal, two segmented-postal and two many-container stress
sources. Selection is from current training only; exact OCR and current loader
availability are checked. These labels/source layouts still require rebinding.

For each, take current gold as authority; use old slots as reusable hints, not
old labels as truth. Record owned postal/product regions, fixed context,
repetitions, quantity/placement equations, and allowed variation. Reuse exact
unchanged spans mechanically; resolve only changed/uncertain boundaries.
Implement the current target/scenario/publication seam and remove the obsolete
address projection from that path. No parallel legacy publication mode is
needed for the new experiment.

### Stage 2 — falsification before paying for bulk generation

For each admitted source, replay current source values and three deliberately
different valid scenarios: new locality/product identities, boundary-length
postal/product wording, and unequal container quantities where the source
supports them. Fixed equal-split sources retain their required equality.
This is **72 positive variant checks** over a bounded 24-source pilot, not a
manual review of thousands of descendants. Scenario selection remains subject
to each source's actual capabilities.

Inject invalid variants too: stale source/postcode, city duplicated in a second
postal slot, missing district digits, seal substituted for package count,
missing repeated container, bad allocation total, swapped HS/product,
temperature left stale, and an added unprinted label value. The appropriate
validator must reject each seeded defect. A test that only confirms output
schema or that numbers appear somewhere in text is insufficient.

### Stage 3 — measured linguistic pilot

Run the bounded 24-source/three-variant campaign only after authorization.
One cheap-model call can produce the coherent requested party/product bundles
for a sample; deterministic values stay out of the model's decision scope.
Supply relevant source blocks as text, structured sampled facts as context,
and concise output-field descriptions. Avoid full-document re-extraction,
per-scalar evidence essays and multi-round unconstrained rewriting.

Use existing concurrency/caching/spend receipts. Measure first-pass acceptance,
review count, total cost including retries, cost per accepted sample and wall
time. Review every pilot source and all failures, plus each distinct mutation
pattern. If a failure occurs, invalidate its affected source contract/descendants
by dependency, not the whole catalog and not an untracked blanket repair pass.

### Stage 4 — controlled expansion

Expand toward the 66 seed candidates plus needed DG/thermal/multi-container
families that pass the same admission. The 66 contain 38 single-container,
12 two-to-three-container, 10 four-to-six-container, four seven-plus-container
and two containerless sources; five are temperature-bearing and none are DG.
Thirty-six of the 66 are MAERSK/CMA CGM, so cap source/carrier concentration and
add capability coverage rather than treating 66 as a representative sample.
Only nine have exclusively single-occurrence address bindings: pretending the
rest are the same flat-address case would repeat the old error.

Publish independently admitted shards with explicit source/version/status.
Failed sources remain outside the selected pool, with a specific reason; passing
sources are not repeatedly held hostage by unrelated historical families.
Increase sample counts only after the actual distribution and acceptance/cost
measurements support doing so. Do not create another 10k first and discover
contract errors during training.

## 7. Definition of acceptance

| Layer | Required evidence | What it does not claim |
|---|---|---|
| Source admission | Current OCR/label identity; reviewed field ownership, omitted-field/context dependencies; validated grouping | Old compiler's `certified: true` alone is not enough |
| Rendering | Exact slot/literal accounting; complete repeated occurrences; protected non-target text | Byte fidelity alone is not semantic correctness |
| Label derivation | Labels from final owned surfaces; approved normalization only; all printed in-scope facts covered | Whole-document string presence is not ownership/completeness |
| Scenario | Joint goods/HS/DG/equipment/thermal validity; explicit unit/quantity equations; no unsupported allocations | Plausible independent random values are not a coherent shipment |
| Falsification | Known bad mutations rejected by the intended gate, plus valid awkward cases accepted | Passing a static schema is not sufficient |
| Dataset publication | Exact current reduced schema, case policy, no AAI/city/dropped targets, source/validation isolation, lineage, hashes, duplicate checks | Correct data does not guarantee a particular model score |
| Spatial variant | Explicit line/region mapping, coordinate provenance/unknowns, exact text stripping and shared labels | Borrowed source anchors are not fresh OCR measurements |

The semantic decision is concentrated at **source/capability admission**, once,
not delegated anew for every generated field in every descendant. Deterministic
receipts then make known relationships testable on every sample. Independent
pilot review must check omissions as well as invented values. This is a bounded,
testable quality process; it is not an assertion of universal certainty about
all future unseen templates.

## 8. Cost, performance and scope of this pass

This audit incurred **$0 provider cost**. It changed only audit scripts and
documentation. The existing 660 records, all historical catalogs, current
training configs and paid caches remain unchanged.

The historical recovered 10k campaign reported $6.9254 generation cost across
2,989 requests, but that excludes template recovery/compilation and subsequent
repair work and used different contracts. It is not a price quote for the next
campaign. Establish a new cost-per-accepted-sample figure on the bounded pilot;
include compile/review costs separately instead of hiding them in generation.

No registry update, geography/commodity resampling campaign, production refactor
or full-catalog recertification was performed here. The next authorized task
can be the single V7 slice and its 24-source validation, with the artifacts above
providing a concrete starting point and explicit exit criteria.
