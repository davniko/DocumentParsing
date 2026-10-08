# Next synthesis campaign: 1,000 fresh samples from 200 reviewed templates

**Follow-up maintenance:** the [old500 back-check and upstream integrity repair](kie-synthesis-followup-integrity-2026-10-08.md)
has now updated17 old and3 new inputs, and one old goods-description target.
The original byte-preservation statements and hashes below describe the initial
publication; current hashes are in the follow-up report and dataset manifests.
All real training and validation data remain unchanged.

## Completed publication

**Complete:** all 1,000 new samples were generated, repaired where necessary,
reviewed against their final full rendered OCR and targets, and published with
coordinates. No candidate is held and no review finding remains unresolved.
The training mixture and hash-pinned configuration are ready; training was not
started.

| Cohort | Existing synthetic | New synthetic | Final synthetic |
|---|---:|---:|---:|
| Original 100 templates | 500 | 300 (3 per template) | 800 |
| Newly admitted 100 templates | 0 | 700 (7 per template) | 700 |
| **Total** | **500** | **1,000** | **1,500** |

The new [dataset directory](../data/curated/mpci-bl-real600-synthetic1500-v7-positions-v1/README.md)
contains **2,100 training records (600 real + 1,500 synthetic)** and the same
**60 real validation records**. The old 500 synthetic records, real training
records, and validation records are byte-for-byte preserved. The separate
200-sample template-admission pilot is excluded.

Inspection artifacts:

- [All 1,000 sources and plain rendered samples](../artifacts/kie-synthesis-production/curated-v7-expansion1000-v1/samples.md).
- [All 1,000 positioned samples](../artifacts/kie-synthesis-production/curated-v7-expansion1000-v1/positions-reflow-v1/samples.md).
- [Geometry gallery: 20 documents / 49 pages](../artifacts/kie-synthesis-production/curated-v7-expansion1000-v1/positions-reflow-v1/audit/GALLERY.md).
- [Publication, review and preservation manifest](../data/curated/mpci-bl-real600-synthetic1500-v7-positions-v1/manifest.json).

Recorded API spend for this campaign, including failed/retried attempts, postal
repairs, full-text reviews and re-reviews: **USD1.098998860**, approximately
**USD0.00110 per accepted new sample**. This excludes engineering/manual-review
time and training compute. No additional paid requests are needed to use the
published dataset.

## Scope and dataset accounting

Run the user-approved new synthesis campaign, without launching training. Its executable
configuration is
[`mpci_bl_curated_v7_expansion1000.yaml`](../configs/synthesis/mpci_bl_curated_v7_expansion1000.yaml).

The requested increment is **1,000 fresh synthetic samples**. Following the user's
allocation change, use **three descendants from each original source (300)** and
**seven from each newly admitted source (700)**. Preserve the existing 500-sample
synthetic training collection and all real records. The final synthetic set will
therefore contain eight descendants per original family and seven per new family.

The most recent 200-sample template-admission pilot stays separately available
for inspection; it is **not automatically added to this training mixture**.
Thus the intended subsequent training set is **600 real + 1,500 synthetic =
2,100 training records**, with the same **60 real validation records**. Including
the admission pilot as well would instead yield 1,700 synthetic records; that is
not this plan. This supersedes the earlier optional suggestion to reuse those
200 admission examples and generate only 800 more.

No training mixture is rebuilt until the new generation and its audits pass.

## What the previous review established

All 200 admission descendants received complete rendered-text/label model review
and deterministic checks. Flagged findings were manually adjudicated against
the rendered sample and source evidence. This does **not** mean the primary
agent personally read every one of those 200 documents line by line. The prior
[admission report](kie-synthesis-expansion200-2026-10-08.md) retains the receipts,
resolved findings, exact replay checks and geometric audit. Every new descendant
in this campaign must receive its own review; past template admission does not
replace that review.

## Configuration

| Setting | Prepared value |
|---|---|
| Source catalog | `artifacts/synthesis-templates/mpci-bl-v7-reviewed` |
| Sources | All 200 admitted sources, exact catalog capabilities |
| Allocation | `template_sampling.samples: 1000`, explicit `source_counts`: 3 old / 7 new |
| Per-call batch | `variants_per_source: 2`, yielding batches 2 + 1 or 2 + 2 + 2 + 1 |
| Concurrency | 8 |
| Provider/model | Existing pinned OpenRouter Fireworks GLM-5.3-Flash configuration |
| Structured output/reasoning | Native structured output; low reasoning |
| API budget | USD 2.00, including the existing in-flight reservation guard |
| Campaign seed | `202610082` |
| Output | `artifacts/kie-synthesis-production/curated-v7-expansion1000-v1` |
| Positioned publication | `positions-reflow-v1` beneath the new output |
| Target casing | Uppercase human-readable fields, existing identifier/contact rules |
| Render casing | Existing uppercase/title variation policy |

Keep the established registry-based commodity, DG, thermal, package, equipment,
country and route sampling. Source contracts constrain which facts and structures
can vary. There is no new Egypt-only destination restriction. Negotiability and
notify references inherit the actual source instruction, including an explicit
unknown when appropriate; they are not manufactured to meet a quota.

No additional negotiability/notify oversampling is applied. Exact `source_counts`
cannot be combined with instruction-fraction allocation: that would give conflicting
instructions. All selected sources must be covered exactly once with positive
integer counts summing to `samples`; unknown/missing IDs and conflicting quotas
are rejected before generation. Source instructions remain unchanged.

| Cargo family | Templates | New samples |
|---|---:|---:|
| Ambient | 184 | 936 |
| Vehicle | 6 | 26 |
| Chilled | 4 | 12 |
| Frozen | 3 | 13 |
| DG chemical | 2 | 10 |
| DG vehicle | 1 | 3 |
| **Total** | **200** | **1,000** |

The source-derived instruction distribution is 80 negotiable, 896
non-negotiable, and 24 unknown; 278 samples have a notify reference. These counts
were confirmed in the final accepted labels. The new records contain 400 distinct
printed/public HS values, 23 public package categories, and seven equipment
size/type pairs. Sampled but unprinted HS facts are not added to labels. Printed
shipper and consignee country labels cover 176 and 178 countries respectively;
this campaign is not restricted to Egypt as destination.

Coordinates retain the tested measured-anchor transfer, joint elastic reflow
and coherent page scale/translation. Calibration and font content hashes remain
pinned. The new campaign uses a fresh deterministic coordinate seed, with the
same geometric bounds. Missing source anchors remain explicit missing
coordinates; generated text is never dropped to improve coverage.

## Setup defects caught before generation

The first full deterministic sweep passed 994/1,000 planned draws and rejected
six. The [initial inventory](../artifacts/kie-synthesis-production/curated-v7-expansion1000-v1/audit/preflight-initial.json)
records the exact sources, variants and errors. No paid requests were made.

1. **Two positive private gross-weight requirements were missing from catalog
   capabilities:** sources `0dbc75a0` and `68d08294`. Both templates own a printed
   gross total absent from their public goods labels. A replacement donor must
   provide that fact so it can be sampled jointly and checked against capacity.
   The latter failed one planned draw; the former was found by the complete
   typed-owner inventory, despite its five planned draws passing. Add
   `required_private_measures: [grossWeight]` to both catalog capabilities and
   the new campaign. Do not add a public label or retain unrelated source-product
   density. This is a source capability correction, not a seed-specific exception.
2. **A reviewed vehicle donor hash was stale in the shared catalog:** source
   `bef02d48` references donor `740670f6`. The current donor still has all the
   pinned source literals and the reviewed physical bundle. Its current target
   hash matches the already-updated expansion300 configuration. Align the shared
   capability and new campaign with that pin, retaining all semantic checks.

The catalog-wide inventory uses typed physical owners, not a regex search for
numbers. Positive source-only measures are required for profiles that may select
different donors. The existing `source_whole_units` source `3eaf543b` is a valid
exception: it deliberately retains its own physical bundle and has a uniquely
owned source total. Forcing a different-donor requirement there would invalidate
the intended profile.

The new config is checked for exact capability equality with the catalog.
Historical campaign configs and published samples are not rewritten. No
production Python changes are needed for these corrections: the existing
sampler/renderer already implement and enforce these declarations.

## Running the campaign

### Initial preparation validation (uniform five-per-source plan)

The [final preflight receipt](../artifacts/kie-synthesis-production/curated-v7-expansion1000-v1/audit/preflight-final.json)
records the exact config/catalog hashes and checksums of the preserved datasets.

- **1,000/1,000 actual planned draws pass**, with no source or sample-ID overlap
  with validation/existing records. Source B/L identifiers were checked against
  validation as well as document IDs and exact OCR identity.
- **300 additional stress draws pass** across the three corrected source
  capabilities, without changing the campaign seed.
- **200/200 capabilities match the shared catalog**; the complete typed physical
  ownership inventory has no undeclared positive private measures in profiles
  that can change donor. All 13 sources with positive private measures are covered,
  including the one explicitly source-only physical profile.
- **1,204 catalog/shared files verify against their hashes**. Registry and task
  constraints load through the production code's pin checks.
- **200/200 source-coordinate round trips are exact**, over 373 source pages;
  the actual Paddle data, calibration and font dependencies load successfully.
  This validates coordinate prerequisites, not the yet-ungenerated synthetic
  wording's final geometry.
- The planned draws contain **561 HS6 codes**, **23 package categories**,
  **179 origin and 180 destination country/territory values**, **275
  multi-container documents** and **30 transshipment documents**.
- **211 targeted tests pass:** 179 campaign/sampling/scenario/position/publication
  tests in 12.88 seconds, plus 32 physical-render tests in 7.46 seconds.
- Full preflight, geometry checks and extra stress draws took **38.304 seconds**,
  with **486.93 MiB peak RSS**. These are preparation measurements, not an estimate
  of LLM generation time or a claimed runtime speedup.
- Ten existing real, mixed-training and published synthetic JSONL files remain
  byte-identical. **Zero API calls, zero API spend.** `git diff --check` passes.

Those measurements describe the original uniform plan, before the user's 7/3
allocation change. Its receipts are retained rather than overwritten.

### Approved 7-new / 3-old allocation and execution

The [new preflight receipt](../artifacts/kie-synthesis-production/curated-v7-expansion1000-v1/audit/preflight-7new-3old.json)
confirms **1,000/1,000 approved draws pass**, with exactly 700 new-family and 300
old-family samples. No paid calls were made by preflight. It took 29.08 seconds,
with peak RSS 440.04 MiB. The exact-count allocation benchmark took 0.097 seconds
for 1,000 allocations, versus 0.172 seconds for uniform allocation on the same
200 sources. This measures allocation only, not LLM generation throughput.

The quota extension and existing campaign behavior passed 68 targeted tests in
8.81 seconds; Ruff passed. Explicit counts preserve deterministic source order,
reject bool/float/zero counts, require complete source coverage and exact totals,
and prohibit ambiguous instruction fractions.

Generation was then explicitly launched at the user's request. No training run
will be started by this task. Final publication requires the gates below.

From the repository root, launch generation explicitly:

```bash
.venv/bin/python -m document_ocr.synthesis.curated_campaign generate \
  --config configs/synthesis/mpci_bl_curated_v7_expansion1000.yaml \
  --project-root .
```

`generate` already includes company email/website generation. A separate
`contacts` invocation is only needed for deliberate resumption of that stage.
Review the generation summary before moving on; do not ignore per-source errors.

Then run the same command with these stage names, in order:

1. `render`: require all 1,000 candidates, with schema, source ownership,
   physical/route and policy checks passing.
2. `review`: inspect every rendered text and target. Process success is **not**
   equivalent to no findings: inspect the findings and adjudicate them.
3. Resolve genuine defects or reject false-positive findings with evidence.
   Changed candidates require fresh applicable reviews. The explicit
   `correct-postal` stage is available for actual postal failures, not as an
   unconditional extra pass.
4. `validate`: require the complete publication contract, current review hashes,
   no unresolved findings, exact replay, no missing/extra IDs and no duplicate OCR.
5. `publish`: create the immutable plain-text publication only after validation.
6. `positions`: produce the separate positioned publication from the accepted
   plain text, using measured source geometry and the configured reflow policy.
7. Run the same independent geometric/content audit as the admission pilot:
   text/label preservation, bounds, collisions, ordering, anchor retention,
   coverage by field and failure reason, plus representative diagnostic plots.
8. Only then assemble existing500 + new1000 with real600/validation60; recheck
   split identities, source leakage, duplicates, schema and hashes.

The `validate` CLI stage validates generated candidates; it is not a
pre-generation dry-run command. Preparation instead exercises the actual
`Campaign.plan`/`Campaign.preflight` path locally for every planned draw, plus
source geometry and typed-owner checks.

Do not enable retry switches indiscriminately or change the seed to bypass a
failed draw. Existing cached results and receipts support explicit resumption.
The USD2 reservation cap can pause work below USD2 billed spend when the remaining
budget cannot cover the next request's maximum reservation. Inspect the ledger
and remaining work before choosing whether to change that cap.

## Cost basis

The recent 200-sample admission campaign recorded USD0.216433335 in API spend,
including retries/corrections and complete content re-reviews. A direct linear
extrapolation is approximately **USD1.08 for 1,000 samples**, excluding engineering
and manual-review time. This is a planning estimate, not a price or completion
guarantee: source complexity, batching and correction counts can change it.

Preparation itself uses no API calls and incurs no model charges.

## Execution findings and resolutions

Generation used the exact requested seed and source quotas. Rejected calls or
candidates were repaired explicitly, not bypassed by changing the seed or dropping
families. Original paid-call results and exact before/after repair receipts are
retained under the campaign's `calls/` and `audit/` directories. Temporary,
self-contained execution/reconciliation scripts are under
`docs/analysis/synthesis-expansion1000-20261008/`, not the production package.

The initial full rendering admitted 751 records and held 249. A scoped postal
correction step processed 243 addresses; the following render admitted 957 and
held 43. Exact lexical/manual corrections and a Unicode comparison correction
resolved the remaining cases. This is intermediate history, not the final
publication count. Original errors remain inspectable in
`audit/render-initial.json` and `audit/render-after-postal.json`.

The full-text review was run on every candidate's complete OCR and target, not
only on sampled fields. Provider rate limits interrupted some early requests;
the explicit resume used concurrency two and kept successful content-addressed
results. Failed requests were not counted as completed reviews. Changed text or
targets were reviewed again, while unchanged inputs reused their exact receipts.
The primary agent and a bounded helper inspected the findings, their actual
sample text, source bindings, physical receipts and source layout where needed.
This is not a claim that a human opened all 1,000 unflagged pairs line by line.

### Defects corrected before publication

| Layer | What was wrong | Correction and scope |
|---|---|---|
| Equipment rendering | A length-only source receipt could hide a newly sampled high-cube/reefer type; one quote boundary and one private ISO summary were incomplete. | Curated physical rendering now requests explicit equipment-category support. Corrected source spans for `66dc4377`, `52f97e4d`, and `e7dd0529`; checked typed, mixed and reversed-count receipts. Existing 500 records were audited, not rewritten. |
| Unicode casing validation | Unicode upper/title rendering could encode an equivalent dotted-I sequence differently. | Normalize comparison operands to NFC. Preserve output text and target policy; do not strip letters or silently alter identifiers. |
| Nested auxiliary rendering | A dynamic destination placeholder inside a larger owned private fragment was copied literally. | Evaluate nested caption text through the same strict scenario interpolation as standalone fragments; unknown keys still raise. Full-campaign scan confirmed no remaining origin/destination placeholders. |
| Repeated quantities and issue/handling ownership | A repeated package count was not rebound; an issue-country owner was inside a carrier address; a transit clause's text varied while its label stayed at the source destination. | Corrected `291f67b3`, `73576135`, and `2d4248d3` source contracts/ownership. Handling text and target now derive from the same sampled final destination. |
| Address/contact boundaries | `058d92bb` retained source postcode/country tails and extra plus signs outside its mutable slots. | Widened the reviewed postal/phone ownership; all seven variants changed only the intended suffixes/prefixes, with targets and generated wording otherwise identical. |
| Country-dependent private text | Egypt-only terminal/customs/agent clauses and a fixed exporter country survived route variation. | Scoped neutral captions or sampled party/route-dependent values, including `2784e543`, `28f4afa5`, `5ffbf842`, `6ca7fa6f`, `81877adf`, `9d4a6b90`, `a5265d59`, `c06401a6`, `d684b168`, `d97350b6`, and `e9120c73`. No whole-document country substitution. |
| Repeated private identifiers/dates | Some repeated VAT/exporter IDs and invoice/shipping-bill dates stayed at the source value while related facts varied. | Bound exact repeated owners to coherent identifier/date recipes (`0d102efa`, `2784e543`, `68d08294`, `b27f5fad`, `d0d9e558`, `e9275395`, and `5ffbf842`). Date shifts preserve the source chronology, rather than guessing new offsets. |
| Obsolete private marks/contacts | Fixed case-number ranges encoded the old package count; a goods-area contact continuation retained unrelated original contact details. | Removed only the certified synthetic-only private blocks in `2784e543`, `7f8fafe7` and `873a72fe`. Original real OCR and all public facts remain preserved. |
| Generated wording | Drafting fragments, conflicting product attributes, invented packing/fill weights, chilled/frozen wording conflicts, and machine/boat dimensions or unit weights inconsistent with the host load. | Exact, receipted wording corrections; re-rendered both OCR and description labels together. Sampled container, package and shipment totals were preserved. The unit-weight sweep extended the fixes beyond the initial model findings. |
| Contact wording | One generated email domain belonged to a neighbouring synthetic company. | Corrected that contact and its generation receipt consistently, preserving the original response for audit. |

These changes update the reusable catalog and production rendering paths where
the problem originated. The manifest pins all 1,204 case/shared files and records
maintenance hash changes. Generated product wording remains subject to full-pair
review: a structured-output schema alone does not guarantee semantic or physical
consistency.

### Findings correctly rejected

Each rejection is bound to the exact finding, current candidate and review hashes,
with a reason and a literal evidence quote. Important categories were:

- **Inner/outer packages:** five or six source-supported outer pallets can coexist
  with varying inner bags/cartons. Our public target deliberately represents only
  the inner level; private pallet counts are not missing public labels.
- **Reading order versus layout:** interleaved OCR captions do not move vessel,
  receipt, payment or issue values between their independently owned table columns.
- **Private facts:** source-only tare, fax, customs references and pre-carriage
  vessels need not be public target fields. They must still be internally coherent.
- **Independent actors:** a carrier/signing agency, owner or `VIA` intermediary
  can remain fixed and need not share the sampled shipper's or consignee's country.
- **Classification versus extraction:** the agreed objective permits related
  expanded product wording without exact HS-to-product classification. This does
  not permit wrong printed HS labels or physically impossible load claims.
- **Review errors:** a few findings cited a sibling shipment, confused `ACID`
  customs references with chemicals, or mistook source `DRAFT`/legal boilerplate
  for generated drafting commentary. These were checked against the actual text.

Postal deliverability is not certified; printed party ownership and label
consistency are. Coordinate plausibility is checked as an input feature, not as
a claim to recreate a typeset PDF pixel for pixel.

### Code validation and measured cost of the fixes

The combined targeted suite passed **461 tests in 17.20 seconds**, and the final
post-publication rerun passed **461 tests in 17.83 seconds**. The auxiliary-only
follow-up passed **24 tests in 8.19 seconds**. Ruff and diff checks
passed. Template changes additionally ran real descendant renders with exact
text-difference and unchanged-target checks.

Nested dynamic-caption rendering measured **1.750 microseconds** versus **0.095
microseconds** for the incorrect literal substitution (100,000 iterations,
best of three). The approximately 1.65-microsecond cost is negligible per affected
fragment and necessary to compute the correct value. Already explicit equipment
receipts measured 6.23 versus 6.19 microseconds; the changed length-only path
measured 9.10 versus 5.47 microseconds. These are component probes, not claimed
end-to-end generation speedups. Complete 1,000-record deterministic rendering
took about **34 seconds** in the final review preparation.

The training image was rebuilt and its dependency verification passed. Training
itself has not been launched.

## Final content and geometry validation

All 1,000 current candidate hashes are covered by complete rendered-text/target
model reviews, across 200 source receipts. The final review contains 49 findings
that were individually inspected and rejected with exact evidence and reasons;
earlier genuine defects were corrected and their changed candidates re-reviewed.
There are **zero pending decisions**. This is model review plus deterministic
validation and targeted adjudication, not a claim of human line-by-line review
of every unflagged document.

The mixture builder independently verified all 1,500 synthetic review receipts,
candidate/publication hashes and geometric receipts, then validated all 2,160
targets against the active schema and casing policy. It checked exact source
quotas and seeded IDs, input uniqueness, and validation separation by document,
OCR, PDF and available annotated B/L identifiers. No overlaps were found.
Annotated B/L comparison covered 477 distinct real-training numbers, 51
validation numbers and 1,222 synthetic numbers; absent numbers were not invented
to claim universal shipment-identity coverage. All 1,204 reusable catalog file
hashes were also verified after repairs.

For the **new 1,000 records**, independent positional replay found:

| Line group | Lines | Anchor-only positioned | Final positioned |
|---|---:|---:|---:|
| All | 103,849 | 81,555 (78.53%) | 87,324 (84.09%) |
| Goods-description lines | 6,555 | 464 (7.08%) | 4,665 (71.17%) |
| Address lines | 9,403 | 4,680 (49.77%) | 6,205 (65.99%) |
| Other lines | 87,891 | 76,411 (86.94%) | 76,454 (86.99%) |

Reflow added 5,769 positions without losing a single existing position or
changing plain text/targets. Of 1,871 pages, 915 accepted joint reflow, 954 had
no eligible reflow, and two rejected a proposed reflow at the calibrated density
floor; those two retain the valid anchor-based representation. Remaining
coordinate gaps use explicit ` ||`, not invented positions or omitted text.
The main held geometry groups have incomplete source anchors (599 groups) or
foreign text inside an ownership envelope (420 groups). These are geometry
coverage limitations, not held documents or unresolved semantic findings.

The independent checker tests accepted-page bounds, envelope collisions,
ordering, coherent page transforms, unique line mapping, and exact text
preservation. Four deliberately corrupted controls all failed: changed text,
off-page coordinates, duplicated lines and inconsistent anchors.
Diagnostic plots cover high-gain, low-coverage, rejected-reflow and hash-selected
examples. Direct visual inspection included the 24-line goods expansion in
`syn_full_v7_c866de84e9a75c22cc0b6500` and the density-rejected example
`syn_full_v7_d4b8a89ec4daba89d450bd79`: added positions retain goods/address
regions, while rejected proposals do not partially alter a page.

Across all 1,500 synthetic records, 128,977 of 154,424 lines have coordinates
(83.52%), with zero lost source anchors. Geometry coverage is deliberately not
reported as 100%; the supported coordinates are the learning signal.

Measured local operations: plain publication plus positioned generation took
63.16 seconds with 629.45 MiB peak RSS; the independent 1,000-record adversarial
probe took 51.48 seconds including its anchor-only baseline, with 614.75 MiB peak
RSS; the full geometric audit and 49 plots took 15.51 seconds. Mixture validation
and assembly took 29.24 seconds with 555.04 MiB peak RSS. These are observed
execution costs, not comparisons with an equivalent earlier campaign.

## Training configuration and actual-container preflight

Configuration:
[`t5gemma2_270m_lora.mpci_bl_real600_synthetic1500_positions_inputonly_v7_e10_compact_eva_a32_r32_local_schedulefree_v1.yaml`](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic1500_positions_inputonly_v7_e10_compact_eva_a32_r32_local_schedulefree_v1.yaml).

The previous experiment's settings remain: input-only positioned text, compact
JSON, rank 32 / alpha 32 with rsLoRA and EVA, ScheduleFree AdamW at 0.0001,
microbatch 1 / accumulation 32, and 10 epochs. Evaluation/checkpoint interval is
165 updates, corresponding approximately to epochs 2.5, 5, 7.5 and 10 on the
expanded 2,100-row training set. The five-step optimizer warmup is unchanged.

The rebuilt Compose training image passed `validate-config` and `inspect-dataset`
using `/opt/document-ocr/src`, then the actual data preparation path with the
baseline's saved, revision-pinned tokenizer. No weights were loaded and no
training was started. Results:

| Split | Retained | Maximum input tokens | Maximum target tokens | Truncated / excluded |
|---|---:|---:|---:|---:|
| Training | 2,100 | 8,066 | 3,139 | 0 / 0 |
| Validation | 60 | 5,660 | 1,338 | 0 / 0 |

Configured source/target limits remain 19,200/5,500. The existing 3,072-token
evaluation generation limit also exceeds every validation target length.
CPU preflight took 37.69 seconds, with 1,589.96 MiB main-process peak RSS
(worker memory is not included in that RSS figure).
Full [tokenizer receipt](analysis/synthesis-expansion1000-20261008/tokenizer-preflight.json).

Pinned train SHA-256:
`70ad28d9992d18113fc2d965bba59b01601b84c6f8e3ea3228382fcbbb7d9592`.
Unchanged validation SHA-256:
`14d306aff64e2903a8496d5809279ec42f4e9672b6d50432e72b8d170ec50f68`.

Launch when ready:

```bash
docker compose --profile training run --rm --build kie-trainer train \
  --config configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic1500_positions_inputonly_v7_e10_compact_eva_a32_r32_local_schedulefree_v1.yaml \
  --project-root /workspace
```
