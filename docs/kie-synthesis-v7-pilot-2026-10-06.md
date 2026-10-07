# Curated V7 synthesis: implemented pilot and release audit

Date: 2026-10-06. Follow-up to the
[synthesis restart investigation](kie-synthesis-restart-audit-2026-10-06.md).

For the current implementation, stage commands, validation boundaries and
artifact map, see the [complete synthesis flow guide](kie-synthesis-flow.md)
(verified 2026-10-07).

## Current result: full-scenario restart published

The replacement full-scenario pilot is complete: **24 current training sources,
72 published variants, 72 successful authority replays, and no unresolved final
review findings**. It samples both ends of the route: **60 origin countries and
59 destination countries**, plus 55 HS6 identities, 14 package categories and
four size/type pairs. Egypt is not a destination constraint.

- [Current source/variant inspection gallery](../artifacts/kie-synthesis-production/curated-v7-full-pilot24-contacts-v2/samples.md)
- [Current 72-record dataset](../artifacts/kie-synthesis-production/curated-v7-full-pilot24-contacts-v2/dataset.jsonl)
- [Current publication manifest](../artifacts/kie-synthesis-production/curated-v7-full-pilot24-contacts-v2/manifest.json)
- [Position-enriched variants](../artifacts/kie-synthesis-production/curated-v7-full-pilot24-contacts-v2/positions/samples.md)
- [Contact and positional extension results](#11-emailwebsite-and-positional-extension--2026-10-07)
- [Final implementation, validation and scaling assessment](#10-final-full-scenario-results-and-scaling-boundary)

The real 600/60 files are unchanged. The contact revision and optional
source-anchored positional inputs are complete (section 11). This
is an inspected pilot, not an unattended 10,000-document campaign or a claim
that the entire historical catalog is recertified. Sections 1–8 preserve the
earlier pilot and its failure analysis; they are not the current release.
Section 10 records the pre-contact publication and its historical commands;
use the current flow guide's revision configuration for new commands.

## Historical assessment after the first variability audit

**The earlier readiness assessment was too broad.** The 72-record publication
below established current-schema mutation/replay, not the requested restoration
of full shipment sampling. The subsequent audit found that countries, routes,
HS/UN identities and equipment/package categories never vary, and that the
renderer introduces awkward leading commas in six samples. These records remain
unchanged for inspection; their previous mechanical publication status is not a
recommendation to scale this generator or merge the pilot into training.

[Section 8](#8-follow-up-audit-why-variability-and-rendering-fell-short) documents
the actual flow, measured omissions, rendering reproduction, historical cost
comparison, validation gaps and recommended implementation sequence. This
follow-up made no generator or dataset edits and no paid requests.

## Historical first-pilot outcome

The bounded **24-source × 3-variant execution published 72 synthetic records**,
using the current reduced V7 target directly. No V5 target conversion or old
address-stripping projection sits on this path. At that release, all records
received source/sample approval tied to the reviewed hashes. The subsequent
audit above supersedes the implication that this was a complete, sufficiently
varied and visually clean synthesis result.

The real 600-training/60-validation dataset was read-only. No training was
started, and these synthetic records have **not** been merged into that dataset.
The historical catalogs and their earlier generation path were not overwritten.

Primary deliverables:

- [Inspection gallery: all source OCR and rendered variants](kie-synthesis-v7-pilot-2026-10-06-samples.md)
- [Published dataset: 72 JSONL records](../artifacts/kie-synthesis-production/curated-v7-pilot24/dataset.jsonl)
- [Publication receipt](../artifacts/kie-synthesis-production/curated-v7-pilot24/publication.json)
- [Positive/negative validation results](../artifacts/kie-synthesis-production/curated-v7-pilot24/pilot-validation.json)
- [Final manual review, including reviewer false positives and numeric dispositions](../artifacts/kie-synthesis-production/curated-v7-pilot24/manual-review.md)
- [Coverage, timing and request-level cost accounting](../artifacts/kie-synthesis-production/curated-v7-pilot24/final-audit-statistics.json)
- [Exact lexical correction receipts](../artifacts/kie-synthesis-production/curated-v7-pilot24/lexical-corrections.json)

The approved sample snapshot is
`7b9070c44b2ab75c6299dacb3fab7362bbbc4efc865e9bfd54a15b80a36561a9`.
This is the digest of the complete ordered candidate records, including their
provenance, not the separate published JSONL byte digest.
The published JSONL SHA-256 is
`9e0527fbb8cdaa721bd97a778887c2b003979c952686b83dde762caa6aeb244c`;
the real CLI reports 72 expected, 72 valid, no failures and `published: true`.

## 1. Scope and actual variation

The selection contains **20 carrier-family labels** from the prior inventory:
16 core sources, two deliberately selected DG sources, two thermal sources,
two segmented-postal sources and two many-container sources. These selection
cohorts are exclusive; their actual content traits overlap.

| Final sample characteristic | Documents |
|---|---:|
| One source-supported goods accounting group | 72 |
| No labeled containers | 6 |
| One container | 42 |
| Two containers | 3 |
| Four containers | 9 |
| Five containers | 3 |
| Six containers | 3 |
| Seven containers | 6 |
| Dangerous-goods labels | 6 |
| Temperature labels | 12 |

There are **222 party address labels**. Repeated copies of a party are rendered
from the same variable, not generated independently. Source selection is from
current training only. Checks exclude exact validation OCR and matching B/L
identifiers; the shortlist also excludes its known catalog layout-proxy overlaps.
This is not a claim that all possible near-duplicate shipments have been solved
by a universal fuzzy detector.

What actually changes:

| Variable family | Documents with at least one changed value |
|---|---:|
| Party/company names | 72 |
| Postal surfaces | 72 |
| Compatible product wording | 66 |
| Container identifiers | 66 |
| Other contracted identifiers | 66 |
| Package counts | 48 |
| Masses | 51 |
| Volumes | 33 |

**This is source-compatible augmentation, not unrestricted new-commodity
generation.** Countries, routes, dates, HS/UN classifications, package types,
equipment categories, printed temperature/ventilation instructions and goods/
container membership topology remain source-constrained. Product wording varies
within that identity. Nonmutable contacts, tax fields and boilerplate stay fixed.
This was an implementation scope restriction, not a technical necessity for
all these sources. The request deferred positions, not the essential route/goods
sampling functionality. The historical samplers were not connected to this path.

Physical quantities for four vehicle/serial/chassis-bound source families remain
fixed where changing them could conflict with the retained equipment identity.
Other package counts use an exact integer lattice; mass/volume use a 60–99%
downscaling envelope subject to printed precision and existing sum equations.
The envelope avoids increasing source loads. It is a controlled augmentation
range, not a learned shipment distribution or universal physical-capacity proof.
Some unequal count splits have GCD 1 and cannot vary under ratio preservation;
their lack of count variation is recorded, not hidden by an invented split.

## 2. Implementation and entry points

| Component | Responsibility |
|---|---|
| [curated.py](../src/document_ocr/synthesis/curated.py) | Strict source contracts, deterministic scenario, exact rendering, V7 targets, provider calls, validation and hash-approved publication |
| [Pilot config](../configs/synthesis/mpci_bl_curated_v7_pilot24.yaml) | Dataset, frozen schema/vocabulary digest, source IDs, seed, count, concurrency, provider and spending ceiling |
| [Rebinding utility](../scripts/synthesis/rebase_curated_v7.py) | Reuse historical byte/numeric evidence only after matching current OCR and current labels |
| [Source rebinding declarations](../configs/synthesis/contracts/curated_v7_pilot24_rebinding.yaml) | Reviewed postal ownership, auxiliary occurrences, dependent targets, required anchors and frozen facts |
| [Lexical adjudications](../configs/synthesis/contracts/curated_v7_pilot24_adjudication.yaml) | Full variable corrections, applied through all occurrences and dependent labels |
| [Offline materializer/validation](../scripts/synthesis/validate_curated_pilot.py) | Replay saved generation receipts, apply reviewed amendments, run negative probes, approve only an explicitly supplied snapshot |
| [Numeric repetition screen](../scripts/synthesis/audit_curated_numeric.py) | Surface unbound equal-valued numbers for ownership review, never blanket replacement |
| [Audit statistics](../scripts/synthesis/summarize_curated_pilot.py) | Coverage, request-cost reconciliation and repeatable CPU timings |
| [Behavioral tests](../tests/test_synthesis_curated.py) | Source ownership, formatting, arithmetic, scenario integrity, API failures and publication gates |

This reuses the established UTF-8 byte renderer, deterministic identifier
generators/checksums, source-style numeric presentation and current task adapter.
It does not launch a parallel replacement for every historical synthesis module.
The old V5 production command is **not** the entry point for this V7 pilot.

### Flow

1. Match selected current training OCR and labels to reusable source evidence.
2. Establish disjoint mutable regions, repeats, auxiliary facts and all existing
   target dependencies. Current source text and labels must both replay exactly.
3. Generate three coherent lexical bundles per source in one PydanticAI call.
   Supply ordinary OCR text and a concise list of owned regions/constraints.
4. Generate IDs and coupled quantities on the host; render all occurrences from
   the same variable values. Preserve source line structure and protected text.
5. Derive the current target from those same values and its reviewed bindings.
6. Review lexical semantics separately from mechanical checks. Apply approved
   corrections at the variable level, not ad hoc text patches.
7. Validate, manually adjudicate the bounded set and publish only the exact
   approved source/contract/sample hashes.

PydanticAI uses native strict structured outputs, described fields and model
docstrings. Actual provider: OpenRouter **GLM-5.3-Flash**, Wafer, low reasoning,
concurrency 8, 300-second request timeout, zero automatic retries, explicit
provider routing and a $5 reservation ceiling. Failed responses are recorded
and are not approvals. The generation prompt does not ask for a new extraction,
unrestricted source rewrite, per-scalar rationale or span evidence output.

Source contracts remain structured data because byte ownership and dependency
bindings require structure; this is distinct from wrapping OCR into JSON for
the language model. The source OCR is sent as text.

## 3. What the pilot uncovered and corrected

### A. Postal ownership and repeated identity

Ten source families needed explicit rebinding beyond a simple whole-label text
match. Cases include separately printed country after contact/tax lines,
postcodes joined to locality text, delivery-agent continuation across pages,
party-name line wrapping, parenthetical person identity, and bank/caption text
interspersed with postal data. These are now source declarations reused by all
three descendants, not another per-descendant address extraction campaign.

Labels use complete owned postal surfaces in source order, physical line breaks
become spaces, rendered punctuation is retained, and casing is normalized once.
No city/country stripping or blind append is performed. Tax IDs, names, contacts
and captions are outside postal spans. Repeated consignee/notify copies share
one generated bundle.

Example: [segmented Indian postal source, generated variant 1](../artifacts/kie-synthesis-production/curated-v7-pilot24/previews/doc_00388da5efb560ae60c645831f6b34db7049b9c0e0772d76eecda0d7387de900/variant-1.txt)
prints its new building/street and `HYDERABAD-33 TELANGANA INDIA` in the correct
order; the address label contains that entire postal sequence without adding a
second city/country or swallowing the shipper name.

### B. Dependencies the old bindings did not fully cover

- `01d86535`: bind the second printed 13,800 kg / 68 m³ totals.
- `049ad12d`: bind the printed 24,040 kg gross total as well as its components.
- `00388da5`: bind the separate numeric repeat of the package quantity.
- `88bab156`: bind the attachment's 5,600-package total.
- `70efc721` and `1464f450`: connect existing placement quantities to their
  changing package totals. Genuine quantity variation exposed these omissions;
  baseline replay alone would not have caught them.
- `3dc8551d`: retain the buyer identity also printed in a shortened order
  heading; reconcile generated delivery-agent acronyms with the generated name.

The follow-up numeric screen left **24 equal-number flags across seven sources**.
All were reviewed: they are page/form/clause numbers, equipment dimensions,
fixed one-unit vehicle counts, or unchanged zero-volume facts—not omitted mutable
totals. Each disposition is in the linked manual review.

### C. Goods, equipment and allocations

Single goods accounting remains single goods even when there are several HS
codes or repeated product rows. Membership-only placements remain membership
only; equal numbers do not authorize invented container allocations. Complete
known allocations must sum to the corresponding new package total.

Example: [four-container DG variant 1](../artifacts/kie-synthesis-production/curated-v7-pilot24/previews/doc_7d44dcc4a4c621a115e1c3d4f8402e4d1b8edfd43d3fd15018a8cfe74d4f118d/variant-1.txt):

| Fact | Source | Generated |
|---|---:|---:|
| Boxes | 80 | 48 |
| Boxes per container | 20 × 4 | 12 × 4 |
| Gross mass | 77,600 kg | 46,560 kg |
| Volume | 104 m³ | 62.4 m³ |

POTASSIUM AMYL XANTHATE, UN3342, HS classification and equipment categories remain
compatible and unchanged; all four new container IDs occur in text and placements.

Example: [garlic reefer variant 1](../artifacts/kie-synthesis-production/curated-v7-pilot24/previews/doc_0951955dd9effb488ee65cefb3def600ec16288134934a9b0c8e0fd33601d373/variant-1.txt)
varies cartons, shipment mass and volume while retaining **−3°C** and **10 CBM/hour
ventilation**. Ventilation flow is not scaled as though it were shipment volume.

The six-container steel source retains its membership-only placement shape and
distinguishes **48,111 sheets** from its **package count**.
[Rendered example](../artifacts/kie-synthesis-production/curated-v7-pilot24/previews/doc_3dc8551dbaaba3ed066f05e5255c17cbffe941f787203b20c74b1c6843e7b85a/variant-1.txt).

### D. Lexical corrections and reviewer reliability

**18 of 72 samples, across seven families, received recorded lexical amendments.**
These include an over-detailed city-only address, copied `00000` postal placeholder,
duplicate terminal punctuation, uncertain chemical synonyms, valve SET versus
ASSEMBLY wording, a town/county pairing, and linked buyer/depot identity. This
is not a claim that first-call model output was already flawless.

Broad whole-document model reviews were slow and sometimes returned placeholder
explanations or missed dependencies. The runtime now uses a focused lexical
review with field ownership context. Of 24 scoped review calls, 21 produced
valid outputs and three failed; those three were manually reviewed. All final
72 variants received manual changed-region/semantic review, not just the model
flagged cases.

Some model findings were wrong: one inferred a fixed county absent from OCR;
another treated allowed synthetic company-name variation as failed transcription.
One proposed changing MOOROOKA 4105 to 4107, contradicted by
[Australia Post's postcode listing](https://auspost.com.au/postcode/moorooka).
Those findings were rejected, not blindly applied. No extra review loop was run
merely to obtain model agreement.

## 4. Validation and measured processing cost

**147 tests passed** in the targeted renderer/equipment/generation/curated suite.
Latest full run: 9.14 seconds. Ruff checks and formatting checks passed.
The final real CLI probe also exposed a YAML-to-strict-provider serialization
boundary (lists/float prices versus tuples/Decimal prices). This was fixed by
validating at the provider's JSON wire boundary, without weakening strict
validation. The existing publication/API-failure test now initializes the actual
runner from serialized configuration instead of bypassing its constructor.

All 24 sources replay current OCR and labels exactly. All 72 final records pass
exact independent edit replay, current frozen target validation, scenario/ID
checks, declared unchanged-field preservation and source line-count checks.

| Deliberately corrupted candidate | Rejected |
|---|---:|
| Unrelated input modification | 72 |
| Unprinted target value | 72 |
| Stale input hash | 72 |
| Identifier substituted for host-owned quantity | 66 |
| Duplicate country introduced into coherent lexical proposal | 69 |
| HS label swapped without matching input/scenario | 36 |
| Container omitted from target | 66 |
| **Total** | **453** |

These probes prove the named rejection properties. Exact replay does not, by
itself, prove that a semantic source binding is correct; source review and final
lexical adjudication supply that separate evidence. Publication is deliberately
snapshot-bound so that changing a reviewed record invalidates its approval.

Three-trial CPU medians for all 72 records, using the same renderer:

- Identity source replay: **0.131 s**.
- Generated replay: **0.143 s**, approximately **0.16 ms extra per record**.
- Full candidate validation: **0.220 s**, approximately **3.06 ms per record**.
- Instrumented positive/453-negative pass: about **8 s**, peak traced Python
  allocation about **0.66 MiB**. This is not whole-process RSS or GPU memory.

There is no honest old-V5 versus new-V7 throughput comparison because the old
end-to-end target path does not support these current labels. Local validation
is inexpensive; provider latency dominated generation/review. An observed
22-source batch run took about 19.1 minutes, not including earlier probes,
source adjudication, subsequent review or implementation time.

### API spend, including failed attempts

Across **85 recorded requests**, provider-confirmed charges total **$0.33211723**.
Some disconnected/timed-out calls lack final usage. Including recorded timeout
reservations and a conservative $0.05 allowance for each of three early compiler
timeouts gives **approximately $0.55844**. This is a budgeting allowance, not a
provider invoice; exact final billing requires reconciliation. Two explicit
404 routing rejections are counted as unbilled, not successful work.

- Generation batches plus the explicit malformed-bundle correction: **$0.07904**,
  approximately **$0.00110 per final sample**.
- All pilot calls, including exploratory and review work: approximately
  **$0.00461 confirmed / $0.00776 including the timeout allowance per sample**.
- The recorded generation rate extrapolates to roughly **$11 per 10k samples for
  lexical generation alone**. It excludes semantic review, source admission,
  corrections and human time. It is not an approved campaign estimate.

The current runtime reserves unknown timeout cost before issuing requests,
keeps failed receipts, uses no automatic retries and cannot publish failed or
unreviewed results. The three early unreserved timeouts are retained in history
and explicitly allowed for in the final audit rather than represented as free.

## 5. Position synthesis: mapped, deliberately not emitted

All 72 rendered texts retain their source line counts. Source line ordinals and
page markers can therefore provide a deterministic correspondence to the R16
alignment files. That is useful groundwork, **not proof of new physical word
coordinates**: changed words differ in width and can occupy different visual
centroids even when the line count stays constant.

The pilot publishes `joinedRawText`, not fabricated `positionedText`. A sensible
next spatial experiment is to preserve region/row anchors, mark them as synthetic
layout priors, and validate row/column ownership and long-value fit. Missing or
ambiguous source anchors should stay empty. Do not rematch generated names to
the original Paddle-recognized words, or silently call copied anchors measured
coordinates of a newly rendered document. A fuller geometric solution would
need explicit region geometry and reflow/fit behavior.

## 6. Reproduction and safe continuation

Offline validation and publication use saved receipts; no provider calls:

```bash
.venv/bin/python scripts/synthesis/validate_curated_pilot.py \
  --approve-snapshot 7b9070c44b2ab75c6299dacb3fab7362bbbc4efc865e9bfd54a15b80a36561a9
.venv/bin/python -m document_ocr.synthesis.curated \
  --config configs/synthesis/mpci_bl_curated_v7_pilot24.yaml --stage publish
.venv/bin/python scripts/synthesis/summarize_curated_pilot.py
TMPDIR=/tmp .venv/bin/pytest -q tests/test_synthesis_curated.py \
  tests/test_synthesis_raw_text_template.py tests/test_synthesis_iso_equipment.py \
  tests/test_synthesis_generation.py
```

`--stage generate` is the paid generation entry point. Use a new explicit run
configuration/output for a new campaign; do not overwrite this adjudicated pilot
or assume its approval applies to fresh model answers. The standalone rebaser
documents this pilot's source preparation, not an automatically certified
full-catalog compiler.

Updated next step: first integrate genuine scenario sampling and correct reflow,
then repeat the bounded pilot with explicit variability checks as specified in
Section 8. Only after that should the remaining **66-source shortlist** candidates
be admitted through current-source ownership and numeric checks. Reuse sound
contracts; do not re-extract every generated label. Do not restore all 1,507
historical templates merely to increase volume.

The pilot establishes a working current-target generation/rendering/review/
publication path. It does **not** establish a model-quality gain: that requires
a controlled training comparison with unchanged held-out evaluation.

## 7. Preserved input identities

Dataset: `data/curated/mpci-bl-real-v7-reviewed-r16-paddle-positions-660`.

- Training SHA-256: `ca15c382bd1a36e72db978a0acb34f9dec64e8ea6c7e00639e62bccc98058305`
- Validation SHA-256: `b8c0d4bddd4b3a452f901f3fc5e08e54d580a82768eddd5df33c97ab2c85da5a`
- Frozen task constraints SHA-256: `db5a5f2a99c4f5144c901216fae579bc48ab0df7a95736520530b3df839655f6`

These match the pre-pilot authorities. Historical generation/review errors remain
in their receipts for audit. The publication receipt records the earlier
mechanical release; the current readiness assessment is the superseding audit
at the top of this report and in Section 8.

## 8. Follow-up audit: why variability and rendering fell short

### 8.1 Diagnosis and scope of this investigation

The user identified insufficient country/goods variation and an awkward notify
address in the inspection gallery. This follow-up traced the real code path,
compared every published target to its current source, inspected the underlying
lexical values, reproduced the renderer, queried the historical spending ledger
read-only, and summed current request usage receipts. No new scripts were added,
no provider calls were made, and no source/generated records were edited.

The central finding is not that the model ignored instructions. **The new
instructions explicitly ask for same-country addresses and synonyms of the same
goods.** That restricted mutation branch bypasses the old route, registry goods,
equipment and coupled shipment samplers. It cannot deliver the variability of
the old campaign merely by increasing its sample count or changing the prompt.

### 8.2 Exact current input-to-output flow

```text
24 chosen current training OCR/target pairs
  -> historical span/numeric hints + manually reviewed rebinding declarations
  -> current source contract: variable regions + existing target expressions
  -> GLM: three name/address/product-wording bundles, tied to original facts
  -> host: new container/seal/B/L identifiers and source-relative quantities
  -> word-count-based redistribution over original physical line counts
  -> byte-span replacement and current-target reconstruction
  -> replay/schema/hash/arithmetic checks + lexical/manual review
  -> explicit snapshot approval -> published JSONL
```

The actual responsibilities are:

| Stage | Receives | Produces | Missing responsibility |
|---|---|---|---|
| Source selection | Explicit list of 24 training IDs | Three variants per ID | No campaign-level country/commodity/rare-value distribution planner |
| Rebinding | Current OCR/labels; old byte/numeric hints; source declarations | Owned regions and scalar label expressions | Most semantic fields are declared source-fixed rather than integrated with samplers |
| Lexical generation | Full original OCR as text; original lexical values; fixed context; required anchors | New names/addresses and compatible synonyms | No sampled country/locality/commodity identity supplied to the model |
| Host scenario | Source values, seed, variant index | Identifiers and representable downscaled counts/mass/volume | No new route, HS, UN, thermal scenario or equipment-category draw |
| Renderer | New single-line value and old multiline region | Same number of physical lines | No punctuation attachment, postal-component or width-aware line breaking |
| Target construction | Original target plus declared variable expressions | Current V7 target with affected scalars replaced | No independent source of new structured facts beyond those variables |
| Validation/review | Contract, values, rendered text/target and reviewer outputs | Structural replay and manual approval | No required semantic novelty, distribution coverage or reliable visual-layout gate |

Relevant implementation points:

- [Source-fixed policy](../scripts/synthesis/rebase_curated_v7.py:349) explicitly
  freezes countries, routes, dates, contacts, customs facts, HS/UN identities,
  equipment counts/categories, thermal settings and package categories.
- [Generation prompt](../src/document_ocr/synthesis/curated.py:670) requests
  `same country` and a variation/synonym within the `EXACT` product/HS/UN family.
- [Host scenario](../src/document_ocr/synthesis/curated.py:591) changes only counts,
  measurements, container identifiers and contracted generic identifiers.
- [Rendering helper](../src/document_ocr/synthesis/template_compiler/descendant.py:779)
  splits candidate text on whitespace and allocates words using source line weights.

The host begins with all source variable values. Counts move along an integer
GCD lattice; measurements independently select feasible 60–99% source multipliers
at the source's printed precision. These are variant-indexed downscales, not a
fresh joint shipment sample. Source count ratios, container-row count, identity
anchors, units and many physical facts remain fixed. Container serials change
but their original owner/category prefix is retained.

### 8.3 Measured changes across all 72 published records

The following is an exact scalar-path comparison with each record's own source.
Counts are label occurrences, including repeated party roles, not distinct facts.

| Field family | Present in documents | Label occurrences | Changed occurrences |
|---|---:|---:|---:|
| Party country | 72 | 213 | **0** |
| Route scalars: names/countries/etc. | 72 | 315 | **0** |
| HS codes | 36 | 54 | **0** |
| UN numbers | 6 | 6 | **0** |
| Equipment size/type categories | 66 | 318 | **0** |
| Package type | 72 | 72 | **0** |
| Issue/on-board dates | 60 | 96 | **0** |
| Party contact fields | 48 | 189 | **0** |
| Party names | 72 | 228 | 222 |
| Address lines | 72 | 222 | 222 |
| Goods descriptions | 72 | 72 | 66 |

There are **zero registry-selected new commodity identities** in this path.
The 66 changed descriptions measure string differences, not 66 newly sampled
goods. Six descriptions are exactly unchanged; most others are synonyms,
reordering or added adjectives. Median normalized character similarity to the
source is approximately **0.837**. Party-name similarity has median **0.782**;
this is a descriptive string measure, not a correctness score or an identity
classifier. Common legal suffixes and industry words naturally contribute.

The 24 sources already span several shipper countries, which can make the whole
pilot look geographically varied. **Each descendant nevertheless retains its
own source country.** No new country is introduced by sampling. Consignee country
surfaces are `EGYPT` in 66 records, `ET` in three and `EGYPT(EG)` in three—the
same source spellings. `ET` must not be blindly treated as a new sampled country;
its source address explicitly says SHEIKH ZAYED, GIZA. Country identity and
template-specific spelling/code conventions need separate treatment upstream.

Addresses may change a street or city within the source country through free
model wording, but no controlled locality registry draw or locality-coverage
receipt exists. An address string changing is not proof of new geographic
coverage. The code has no configured country, city, HS or UN diversity targets.

### 8.4 First and second source examples

**First source, `2763b891`:**

- Source shipper: `MIDEX MOHAMED ELBADRY`; variant 1: `MAHFOUZ MOHAMED ABDELKADER`.
  The name changes, but no policy requires a particular identity-distance or
  name distribution. Its source country POLAND is deliberately fixed.
- Source postal locality: `85-605 BYDGOSZCZ POLAND`; variant 1:
  `87-800 WŁOCŁAWEK POLAND`. This is within-country rewriting, not country sampling.
- Source product: `USED UNPACKED VEHICLE (S) KOMATSU TRACK EXCAVATOR`.
  Variants replace TRACK with CRAWLER, USED with SECOND-HAND, or UNPACKED with
  UNCRATED. The Komatsu identity, chassis `K115885`, one-unit quantity, 10,800 kg
  and 59.579 m³ remain fixed. This is not a newly sampled vehicle shipment.
- Notify name/address change, but `Mohammed Ahmed Dosoky` and `+201224852979`
  remain unchanged because contacts are outside mutable regions. This does not
  itself prove a label error; it demonstrates missing contact/entity variation.

**Second source, `33546e11`:**

- `ICE CREAM PRODUCTS` becomes FROZEN / EDIBLE / ASSORTED ICE CREAM PRODUCTS.
  There is no sampled alternative commodity, recipe, package identity or cold-chain
  scenario. The new adjectives do not constitute the expected goods variation.
- Shipper names do change substantially—for example, `EMENDATORI & VAYRA 1905
  S.R.L.` becomes `GELATERIE RIUNITE EMILIA S.R.L.`—but the sampled postal
  variants all retain FERRARA, ITALY. Similar names are not universal; absent
  controlled geographic/commodity selection is universal in this pilot.

### 8.5 Exact rendering defect and its measured scope

Original owned source region:

```text
39 Banni El-Abbas, Bab Sharqi ,
Alexandria , Egypt
```

Model-generated single-line value, before rendering:

```text
22 SULTAN HUSSEIN ST., ANTOUKHY , ALEXANDRIA , EGYPT
```

Actual deterministic rendering:

```text
22 SULTAN HUSSEIN ST., ANTOUKHY , ALEXANDRIA
, EGYPT
```

The model retained source-like spaces around some commas. The **new comma-led
line is introduced by the renderer**, which treats a whitespace-separated comma
as a word and redistributes words to preserve source line counts. It does not
keep punctuation attached to its preceding component or preserve postal line
roles. No words are lost, so its current content-preservation check passes.

A complete screen of multiline mutable postal/name/product regions found this
new leading-punctuation shape in **six occurrences, six records, two families**:
all three variants of `2763b891` and all three of `049ad12d`. The latter changes
the source split after `DREAM MALL ,` into a generated next line beginning
`, FIRST/SECOND/THIRD FLOOR ...`. There were zero corresponding comma-led source
lines in those regions. This is a confirmed renderer-induced defect, not simply
faithful preservation of an awkward source line break.

That screen measures this particular shape, not every possible layout problem.
It needs to become part of a more complete punctuation/component-aware rendering
contract, not a hardcoded exception for these two documents. Preserving source
line count is useful for later position work but is not sufficient layout quality
and should not override coherent text when positions are explicitly deferred.

### 8.6 Why the validation and reviews did not establish the intended result

Two offline probes were performed without saving modified candidates:

1. The published first-source variant with `\n, EGYPT` **passes** the current
   candidate validator. It preserves whitespace-normalized lexical content,
   expected bytes, labels, hashes and line count.
2. Restoring every name/postal/product variable to its original source value,
   then rebuilding the candidate, also **passes** that validator. For this first
   source the entire OCR and entire target equal the original, because its other
   mutable quantities remain one and its physical values are frozen. Snapshot
   approval would still need renewal; this probe isolates the absence of a
   candidate-level novelty requirement, not a bypass of the publication hash.

The existing 453 negative tests remain valid tests of their named properties,
but they do not establish natural line layout, new countries, new commodities
or adequate sampling distributions. Rejecting mismatched text/labels is different
from rejecting consistently rendered but low-diversity or badly wrapped content.

The scoped lexical reviewer sees single-line proposed values, not their final
multiline rendering. It cannot detect a line-break defect introduced afterward.
Its fixed-context instructions also endorse unchanged geography/classification.
The final manual review missed the leading-comma artifact. The previous broad
approval therefore overclaimed what had been checked successfully.

### 8.7 What the older synthesis flow did that this path omits

The older implementation still contains these distinct responsibilities:

| Existing component | Relevant functionality | Used by this pilot? |
|---|---|---|
| [route_plan.py](../src/document_ocr/synthesis/template_compiler/route_plan.py) / route projection | Train-fitted and registry-assisted origin/destination/port/locality scenarios; party-owned geography and dependent surfaces | No |
| [cargo_scenarios.py](../src/document_ocr/synthesis/template_compiler/cargo_scenarios.py) | Registry goods/HS/DG identities; compatible equipment/packages; quantitative and capacity constraints | No |
| [complete_targets.py](../src/document_ocr/synthesis/template_compiler/complete_targets.py) | Structured proposal and wider deterministic fact generation | No; only lower-level ID generators reused |
| [lexical_facts.py](../src/document_ocr/synthesis/template_compiler/lexical_facts.py) | Host-owned commodity/reference/temperature text and constrained lexical requests | No |
| [request_batches.py](../src/document_ocr/synthesis/template_compiler/request_batches.py) | Shared-context batching, per-case geography, attribution and field-specific requests | No; pilot makes a three-variant call per source |
| Byte renderer / numeric surface helpers / ID checks | Mechanical rendering and reproducibility | Yes |

In the old complete pipeline, route projection first changes the structured
proposal; the cargo sampler then selects actual registry identities and physical
facts. Host lexical preparation fills facts already known deterministically, and
only the remaining wording is sent to an agent. The pilot substitutes a smaller
source-relative `scenario_values` function for that chain.

The old sampler is coupled to historical `cargoGroups`, `cargoPackages`, container
and address conventions. It cannot safely be reactivated by changing a YAML path
to the V7 dataset. Reuse its sampling/registry logic through a current-schema
scenario interface; do not reintroduce old public labels or address projection.
Train-fitted priors also need the current split, not old training membership.

### 8.8 Cost comparison: measured, with like-for-like limitations

The recovered campaign's SQLite ledger was queried in read-only mode:
**2,989 calls, settled accounting of $6.92543453, zero uncertain or unsettled
calls**, for 10,000 accepted documents. Its configuration used
`gpt-5.6-luna`, low reasoning, shared-context batch size 16 and concurrency 16.
That historical total excludes earlier template compilation/recovery and later
dataset-repair work; it is the ledger's recorded token-cost accounting, not a
new check of the provider invoice.

Current receipts, including unsuccessful calls with returned usage:

| Pilot stage | Requests | Provider-confirmed cost |
|---|---:|---:|
| Three-variant generation batches | 24 | $0.07779404 |
| Explicit correction of one malformed bundle | 1 | $0.00124260 |
| Scoped lexical reviews | 24 | $0.08963832 |
| Earlier broad batch reviews | 23 | $0.13007282 |
| Compiler/contract probes, single-sample probes and other reviews | 13 | $0.03336945 |
| **Total** | **85** | **$0.33211723** |

The previously reported approximately $0.55844 includes allowances for unknown
timeout costs; it is not confirmed spend. Those timeout allowances must not be
confused with completed review outputs or added twice.

| Cost basis | Observed 72-sample cost | Linear equivalent per 10k |
|---|---:|---:|
| Lexical generation + its correction only | $0.07904 | **$10.98** |
| Generation + scoped lexical review, confirmed | $0.16867 | **$23.43** |
| Same, including the scoped review timeout reservation | $0.18126 | **$25.17** |
| Entire pilot, all confirmed exploratory calls | $0.33212 | $46.13 |
| Entire pilot including timeout allowances | $0.55844 | $77.56 |

The last two rows **are not reasonable steady-state forecasts**: they include
setup, abandoned broad reviews and experiments that should not repeat for each
future batch. However, even the current generation-plus-one-review design is
above the historical approximately $8/10k objective. Generation alone is about
1.6× the old settled campaign rate, not ten times it; the review design accounts
for much of the larger end-to-end difference.

The token receipts expose a concrete source of overhead despite `low` reasoning:

- Generation batches: 49,583 input tokens; 145,683 output tokens, of which
  **131,147 (90.0%) are reasoning**, leaving 14,536 visible output tokens.
- Scoped review: 22,297 input tokens; 174,839 output tokens, of which
  **165,797 (94.8%) are reasoning**, leaving 9,042 visible output tokens.
- Earlier broad review: 219,536 reasoning tokens within 226,432 output tokens.

These are actual model/provider receipts, not an assumption that low reasoning
must be cheap. The pilot pays the model to decide/rephrase facts while receiving
the full source, then pays again to reassess its lexical choices. The old flow
made more decisions on the host and batched compatible requests up to 16. The
old price rates alone do not explain the difference: its configured output price
was higher, yet its recorded total was lower. Token volume, work assigned and
review cadence matter. Native versus prompted structured output may differ too,
but no controlled test here establishes it as a causal cost explanation.

### 8.9 Recommended implementation sequence, before another pilot

This is the proposed next change, **not implemented during this investigation**.
Do not merely broaden the current prompt while leaving dependent source facts
immutable, and do not scale to more sources before demonstrating genuine variation.

1. **Make the sampled shipment—not the original label—the fact authority.**
   Reuse the existing route/geography and cargo/equipment support behind a
   schema-neutral internal scenario with current V7 target construction. Record
   sampled country/locality IDs, goods/HS/UN identities, equipment, packages,
   quantities/settings and the source capability permitting them. Fit any priors
   only on current training documents and exclude the current validation split.

2. **Expand source contracts to own every dependent occurrence.**
   A country change can affect addresses, country codes, phone prefixes, tax/
   customs wording and named geographic fragments elsewhere. A commodity change
   can affect descriptions, HS/UN, product references, brands/models, handling,
   temperatures, packages, mass and volume. Represent canonical identity
   separately from each source's display form. For truly constrained layouts,
   sample within an explicit compatible domain; distinguish that restriction from
   unnecessarily freezing the original identity. Reuse existing ownership maps
   and known auxiliary facts rather than asking an agent to rediscover them per
   descendant. Replace generic name-field descriptions that mention commodity
   classification with precise role-specific constraints.

3. **Give the model the selected facts and request only coherent wording.**
   One party bundle should include the new country/locality and all owned postal
   components. One goods request should receive the selected registry identity
   and required attributes, not an instruction to synonymize the old description.
   Keep plain text/context and described native output fields. Use compatible
   shared-context batches and deterministic formatting for already known facts.
   New party names should not merely be numbered or lightly altered source names;
   measure identity reuse without mistaking common legal suffixes for duplicates.

4. **Correct generic reflow before emitting another batch.**
   Keep punctuation attached to the appropriate token/component; honor genuine
   separate postal-region slots; enforce reasonable line length/fit rather than
   treating standalone punctuation as a word. Preserve protected bytes and label
   order. Test short/long text, standalone punctuation, names, product qualifiers
   and split postal regions. Since positions are deferred, exact old line count
   should not force unnatural wrapping. Review the actual rendered text, not just
   the pre-render lexical bundle.

5. **Make variability and cost explicit acceptance criteria.**
   On the same small source set, deliberately exercise a changed compatible
   country/route, a changed goods identity and supported HS/UN changes—not only
   new IDs. Every invariant must be justified by capability, not a blanket rule.
   Compare planned versus observed country/locality/commodity distributions,
   conditional per-template change rates, repeated identities, source-copy rates,
   rendered coherence and new-target dependencies. A plain description-string
   change does not count as a new commodity. Add coherent bad-wrap and unchanged-
   source negative cases alongside the current mismatch tests.

6. **Measure a steady-state batch before approving a large campaign.**
   Separate template/setup cost from generation and review. Measure returned
   reasoning tokens, cache reuse, retries and accepted-record cost for the actual
   proposed batch size. Test a cheaper workload/call design before committing to
   it; do not assume that choosing a cheap model or setting reasoning to low
   guarantees the budget. Reuse source-level semantic certification, but retain
   per-descendant semantic checks where factual choices remain free. Never
   remove those checks solely to hit a price target without demonstrating that
   host-owned facts and rendering checks cover the displaced responsibility.

The usable pieces are the current-target boundary, source ownership work,
deterministic dependencies, exact byte editing and provenance. The missing core
is the real scenario-sampling integration; the confirmed defect is general text
reflow; the cost problem is excessive model/review work per accepted sample.
Those are concrete engineering targets. Increasing the existing variant count
would amplify its restricted distribution, not solve them.

## 9. Full-scenario restart (2026-10-06, implemented and published)

The new acceptance criterion is a full sampling-and-rendering pilot, not lexical
mutation. The original 72 outputs and receipts remain historical evidence, not
training-ready examples. Real train/validation data are not being edited.

### Reuse boundary

The old campaign has usable pinned HS, commercial-phrase, country, port,
locality, DG and vessel registries. Its orchestration cannot simply be restored:
`shipment_scenarios._patch` accepts schema 3/5 and separate address/city fields;
the cargo driver expects old cargoGroups/cargoPackages/allocation groups and
compiler sidecars. The current 600 training records, rather than the historical
1,761-record fit set, must supply any empirical sampling support. Validation
records cannot enter that fit. Current template ownership and historical exact
spans can be reused after checking them against current OCR and labels.

The new boundary is therefore: registry-backed, current-target shipment scenario
→ concise joint party/product wording → deterministic current-target rendering
→ rendered-text and scenario validation. Scenario receipts must record commodity
identity, route identity, equipment/package semantics, quantities, thermal/DG
dependencies and the source restrictions. Holding a template's printed row
topology is appropriate; freezing every commodity and country is not.

Commercial phrases cover all 5,612 HS6 entries in the pinned registry. They can
ground meaningful commercial descriptions without asking a model to interpret
tariff boilerplate independently on every sample. Generated national tariff
suffixes are not authoritative registry entries: the pilot will use actual HS6
identities rather than random suffixes. Several printed codes are allowed for
one accounting group; they do not imply several separately quantified goods.

Addresses will be generated jointly per party from sampled geography and the
source's component granularity. Punctuation is free to change. Approximate line
span is a layout constraint, not a requirement to move a bare comma to a new
line. Repeated parties share the same generated values; tax/contact fields have
separate ownership and must remain coherent with relocation.

### Reasoning and cost finding

The live OpenRouter model metadata checked on this pass lists GLM-5.3-Flash
reasoning as mandatory, with `low`, `high`, and `max` supported and `max` the
default. The configured request explicitly sends `low`. A large reasoning
share is therefore not, by itself, evidence that a prompt was ambiguous, and
disabling reasoning is not a supported remedy for this model. We will instead
measure the shorter, fact-conditioned workload. The live Wafer endpoint now
lists input/cache/output prices of $0.10/$0.095/$0.50 per million tokens; earlier
receipt costs remain historical measured costs, not today's pricing forecast.

Sources: [OpenRouter reasoning controls](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens),
[model metadata](https://openrouter.ai/api/v1/models),
[GLM endpoint metadata](https://openrouter.ai/api/v1/models/z-ai/glm-5.3-flash/endpoints).

### Required acceptance checks

- Source/target identity and exact protected-byte replay.
- Every changed label and every dependent printed repetition accounted for.
- Sampled goods, HS, packaging, equipment and thermal/DG data jointly compatible.
- Quantities, row weights, net/gross totals, volumes and placements agree;
  membership-only placements remain unquantified.
- New addresses contain sampled geography once in the appropriate owned region,
  with no retained source-country or company fragments and no contact leakage.
- Checks run on final rendered text, not solely pre-render model strings.
- Explicit commodity/geographic/category change measurements and source-copy
  rejection; mutation tests exercise plausible-looking stale dependencies.
- Cost separated into reusable setup, generation, review/rework and uncertain
  requests. No 10k readiness claim based only on mechanically valid JSON.

### Destination variation: acceptance clarification

The user explicitly requires varied **origin and destination** countries, as in
the older campaign. An initial offline probe pinned Egypt because the selected
sources contain Egyptian customs declarations; that restriction was rejected
and removed before paid generation. Those earlier probe counts are not the new
pilot's diversity result. The destination is sampled independently from the
supported maritime-country pool, excluding the origin country.

Egypt-specific customs captions will be converted into neutral import/export
reference wording, and their owned country/name/code repetitions will follow
the sampled parties. This preserves extraction distractors and layout without
claiming that an Egyptian ACID scheme is valid in every destination. It does
not attempt to simulate every country's legal customs submission requirements.
Independent carrier/legal text and genuinely separate commercial actors are
not blindly changed by global country-string replacement.

### Implemented full-scenario flow and discovered failure modes

The new entry point is `python -m document_ocr.synthesis.curated_campaign`.
Its `generate`, `render`, `review`, `validate`, and `publish` stages share a
hash-bound campaign configuration. Publication is independent of the 600/60
real training/validation dataset; no existing training records are rewritten.

1. **Current-source ownership.** Historical byte spans are replayed against the
   current source OCR. Current V7 targets, not historical labels, own the new
   targets. Explicit ownership declarations consolidate fragmented names,
   complete postal regions, product regions and repeated auxiliary facts.
2. **Joint scenario.** Train-only observations supply compatible commodity,
   packaging and physical-load bundles. Registry-backed countries, ports and
   localities vary on both ends. The scenario sampler preserves the source's
   accounting topology, including multi-container quantities versus
   membership-only placements. A list of HS codes remains one accounting group.
3. **Deterministic shipment facts.** Container/checksum IDs, seals, voyage/B/L
   references, dates, country-appropriate fictional contacts, package counts,
   row/total masses and volumes are generated together. Private printed facts
   remain private; absence of a current target does not authorize stale print.
4. **Language generation.** One native structured PydanticAI request groups
   three variants of one source. The output schema owns exact shipment and
   field coverage. Plain-text inputs describe the sampled geography and
   commodity identity plus source structure examples. Names are new fictional
   identities; postal wording follows sampled geography with approximate
   component granularity; product wording is separate from shipment totals.
5. **Rendering and semantic review.** Disjoint exact spans render the chosen
   facts and words. Final OCR, current targets, commodity identities and party
   localities are reviewed together. Independent deterministic replay proves
   what was edited, not whether every generated word is semantically correct.
6. **Publication gate.** Every expected variant must replay from source,
   configuration and wording authority. Each review must hash-match the final
   candidates. Concrete findings require repair followed by fresh review, or
   an explicit evidence-backed adjudication rejecting the finding. Missing,
   stale, unresolved and duplicate records cannot be silently published.

The probes exposed real defects before publication:

| Defect | Diagnosis and correction |
|---|---|
| Product text invented drums/pallets while host sampled bags/flexibags | Product-only output requirements; host owns packaging and numeric totals. Final rendered review checks this semantic boundary. |
| Product text copied host load totals into `description` | Remove shipment totals from language context and state the product-only boundary explicitly in both the field description and plain-text request. |
| Three heavy road tractors inherited a truck-spares mass-per-package | Exclude parts-only donors from whole-vehicle support; fixed enumerated vehicles use an explicit source whole-unit profile. |
| Frozen-cut load extrapolated to whole carcasses through HS4 expansion | Retain exact supported frozen HS6/load identity rather than treating a tariff heading as physical equivalence. |
| Auxiliary printed origin changed while nested goods-origin label stayed old | Update the actual V7 `goodsItemDetails[].origin` leaf and reject private recipes that contradict public labels. |
| Egypt-specific legal clauses survived destination variation | Replace their exact owned sentences with neutral import/customs wording; preserve unrelated carrier/third-party geography. |
| A country could vanish when its sampled label happened to equal the old label | Validate party-owned country presence independently of whether the country value changed. |
| Free-form output keys drifted despite valid structured JSON | Required native-schema properties own exact field keys; host assigns sample hashes, not the model. |
| Postal correction copied a duplicated country from its own example | Correction now explicitly labels prior values as invalid, rather than calling them structure examples. A dedicated postal request replaces only failed owned regions, and rechecks the complete assembled address before reporting success. |
| Address requirements contradicted complete-address ownership | Removed the instruction to include a country only if the old fragment showed one. The explicit complete-address assembly, including host-owned country text and source omissions, controls placement. |
| Plausible commercial text crossed HS boundaries | Supply the authoritative heading/subheading path, neighboring classifications and national child examples. A plasticised resin cannot stand in for an unmixed PVC category; papaya cannot stand in for a residual fruit category that excludes it. |
| Generated vehicle brands/models contradicted sampled mass or whole-unit counts | Request unbranded product wording with the supplied distinguishing attributes; model names, serials and shipment quantities are not lexical choices. Their separately owned facts remain host-controlled. |
| An existing party email was supported only by a goods-area continuation | In this synthetic template, explicitly print the sampled email inside each owning party contact block. Keep repeated auxiliary occurrences synchronized. Real source labels and OCR are unchanged. |
| Template-only ISO code, DG fee class, package heading and second contact stayed old | Bind their exact regions to the scenario or shared contact identities; neutralize the DG fee caption. Final review checks these dependencies rather than trusting target JSON alone. |
| Cached wording could survive a changed system instruction | The wording identity now includes system instructions, plain-text input and the native output schema. Changed contracts invalidate saved wording. |

These are development findings, not silently accepted training examples. The
older mechanically passing candidate receipts are not publication approvals.

### Provider observations during the restart

Provider choice changes both availability and measured token behavior. Wafer's
listed input price changed during this work; its earlier price is historical.
Two DeepInfra test requests completed at $0.000271075 and $0.000301225, with
33 and 16 reported reasoning tokens respectively. This is much less reasoning
than the earlier Wafer calls despite the same explicit `low` setting. It is
not a controlled attribution to one cause: endpoint and request changed.
Subsequent DeepInfra requests returned explicit upstream 429 overloads, even
at concurrency two; a Decart probe likewise returned 429. Rejected requests
were recorded at zero cost. Routing changes are explicit in configuration;
there is no silent provider fallback. Final cost and throughput must therefore
be measured from completed accepted-generation and review receipts, not
extrapolated from the cheapest two successful calls.

## 10. Final full-scenario results and scaling boundary

### Published result and what actually varies

The release command completed with `published: true`, `expected: 72`, `valid: 72`,
`sources: 24`, no failures and no unresolved review findings. A subsequent
read-only `validate` invocation reproduced the same dataset and manifest hashes.
The final dataset SHA-256 is
`2574fe51caddc5289341b9c450a98beaeb421c742fee7c65b06ef6a1192ed34a`.

| Dimension | Observed final pilot |
|---|---:|
| Source templates / variants | 24 / 72 |
| Origin countries | 60 |
| Destination countries | 59 |
| Origin/destination country pairs | 71 |
| Egypt destinations | 1 of 72 |
| Sampled HS6 commodity identities / publicly printed HS6 | 55 / 32 |
| Package categories | 14 |
| Equipment size/type pairs | 4 |

Across families, 24/24 vary both origin and destination, 17 vary the sampled HS
set, 19 vary descriptions, 20 vary quantities, 15 vary package categories and
nine vary equipment pairs. There are 159 generated containers, six DG samples,
12 thermal samples and eight with ventilation. Descriptions differ from the
source in 68/72 records; the four unchanged descriptions are three fresh-garlic
and one fresh-apple cases with compatible empirical bundles. Private sampled
HS identities are not injected as public labels where the source does not
print an HS field.

The source `33546e11` illustrates meaningful product variation within the same
document/accounting shape:

| Variant | Route countries | Goods / HS6 | Cartons | Gross kg |
|---|---|---|---:|---:|
| 1 | Maldives → Belgium | Electrical switches / 853650; washing-machine parts / 845090 | 1,173 | 11,353.5 |
| 2 | DR Congo → Italy | Footwear uppers / 640610; plastic drain-filter buckets / 392690 | 443 | 8,631.5 |
| 3 | Ecuador → Marshall Islands | Physiological monitoring apparatus / 901819; ultrasound apparatus / 901812 | 506 | 6,914.0 |

Multiple HS codes still belong to the single source accounting group; the
sampler does not invent independent goods rows solely from multiple codes.
DG chemistry also varies: the four-container source `7d44dcc4` generates
guanidine nitrate, benzonitrile and piperazine, with their registry-linked
HS/UN/class/packing-group facts and repeated declarations synchronized.

Not every template permits every dimension to vary. The four whole-unit
profiles use registry-controlled commercial descriptions, not freely generated
body styles, models or exact engine sizes. A one-vehicle profile uses one
reviewed training donor with explicit HS870323, 1,996kg, 15.582m³ and 40GP;
this pilot does not claim a wide empirical vehicle-load library. Refrigerated
and frozen profiles retain supported commodity/form/temperature/ventilation
bundles. These constraints are deliberate quality controls, not silently
frozen countries or a claim that all fields changed in every sample.

### Final corrections and prevention mechanisms

- **Countries and customs:** removed the Egypt destination restriction; exact
  Egyptian customs captions/clauses now become neutral import/export-reference
  wording, with linked country/name/code values following the sampled parties.
  This is training-text synthesis, not simulation of every national filing law.
- **Party ownership:** complete postal fragments are generated jointly; fixed
  country fragments, repeated parties and separately printed declarations are
  distinguished. Missing/duplicate locality checks stop publication and trigger
  a bounded postal-only rewrite. Invented `ATTN:` captions inside identity
  regions are rejected. Contacts that previously lived only in goods-area
  continuations are explicitly printed inside their synthetic party blocks.
- **Package ownership:** source `3dc8551d` had obsolete bindings that confused
  product sheets with the outer package count. Those bindings were retired;
  all four printed package nouns now follow the actual package category.
  The renderer rejects a category change supported only by an overlapping
  product-description region or a stale printed noun.
- **Product meaning:** corrected four HS-boundary mistakes, the milk-constituent
  spelling, ambiguous bulk wording alongside bags, an unnecessary postal-code
  caption, and unsupported contact captions. Product meaning is checked against
  the registry hierarchy rather than merely checking that the HS number prints.
- **Whole vehicles:** removing brands from a prompt was insufficient; the model
  still invented precise engine/body specifications. These tightly coupled
  profiles now get their product phrase deterministically from the selected
  registry identity. The agent writes only their names and addresses. A unit
  test proves the product field is absent from that agent's output contract.
- **Source omissions:** source `88bab156` explicitly prints issue/on-board dates
  absent from its current labels. Hash-pinned, source-evidenced auxiliary recipes
  add these two fields to its synthetic blueprint and shift them coherently.
  The real 660 labels are unchanged; backporting that source-label correction
  is separate from this synthesis publication.
- **Native review structure:** the model no longer copies long sample hashes or
  emits a finding against `all three samples`. Required native-schema `s0/s1/s2`
  objects own coverage; the host assigns identities. Positive observations are
  not findings. Review receipts bind both candidate hashes and the active
  prompt/schema contract, so stale instructions cannot authorize publication.
- **Persistence:** inventory output now uses JSON-native lists; the earlier
  tuple/list mismatch could not pass a fresh-process replay. Exact equality
  remains enforced, with a JSON-roundtrip regression test.

There are 13 retained manual wording corrections across 12 final records,
including conservative clarifications rather than only proven defects.
Their exact before/after reasons remain in the wording receipts. Superseded
paid attempts remain in the request ledger. Final semantic approvals refer to
the repaired candidate hashes, not the original model outputs.

### Validation evidence and measured runtime

**98 targeted tests passed in 9.81 seconds; Ruff passed.** This includes sampling,
source ownership, physical quantities, identity/checksum generation, postal
repair, native output coverage, rendering, persistence and publication.

The independent first-eight, middle-eight and last-eight reviews cover all 72
current candidates, including complete OCR/target reading and contextual
comparison of variants. They report 24/24 accepted in each cohort. The final
paid rendered reviews cover all 72 current candidates with zero findings.
All source/variant pairs are visible in the gallery; accepted records were not
selected by silently dropping failed variants.

Negative controls reject self-consistent forged OCR/labels/hashes, changed
proof/validation metadata, missing/stale reviews and changed review contracts.
The actual current candidate passes replay; the altered candidates fail. The
commodity probe correctly separates all eight selected before/after pass/flag
controls, but only three of four negative controls explicitly identify the HS
mechanism: the milk example was flagged for spelling instead. This limitation
is recorded in `audit/commodity-boundary-control.json`; model review alone is
not presented as a semantic correctness oracle.

Measured on this workspace:

- Scenario plus physical preparation: 72 plans in 2.48 seconds after catalog
  initialization, including probe writes; zero failures.
- Full render/validation: about 2.98 seconds for 72 after initialization.
  CLI including imports/registry loading: 15.27 seconds, peak RSS 443,216KiB
  (about 433MiB), no swapping.
- Final full review at concurrency eight: 30.85 seconds including initialization,
  peak RSS 442,840KiB. Subsequent four-family refresh reused unchanged reviews.
- Publication with fresh authority replay: 15.18 seconds including initialization,
  peak RSS 441,612KiB. The later read-only validation reproduced the publication.

The added semantic review has a real cost versus generation alone; it is not
hidden as a free performance improvement. Deterministic sampling/rendering is
not the expensive stage. Concurrency and provider routing remain configurable.

### Cost: development versus retained production path

All API attempts for this full-scenario restart cost **$0.24554292**: 243 recorded
attempts, 191 billed and 52 rejected. No unknown/reserved charge remains.
This includes experiments, superseded generations, corrections and reviews.

The exact retained path, linked back to request receipts, costs:

| Stage | Paid requests retained | USD for 72 |
|---|---:|---:|
| Language generation | 24 | 0.02356580 |
| Postal-only corrections | 8 | 0.00148190 |
| Final rendered review | 24 | 0.03577205 |
| Total | 56 | **0.06081975** |

That is **$0.00084472 per sample**, or approximately **$8.45 per 10,000** at the
observed billing rate and this source mix. It excludes manual adjudication,
template engineering and future retry/repair overhead, so it is not a quote
for unattended delivery of 10,000 accepted samples. The earlier $6.93 campaign
also excluded template recovery and later repair work. Provider-reported bills
are lower than the configured price ceiling; the linked cost audit retains
both rather than silently assuming this discounted rate persists.
At the configured conservative token prices, the same retained workload
projects to **$12.67/10,000**, rather than the observed-billing $8.45.

[Exact final cost and variability receipt](../artifacts/kie-synthesis-production/curated-v7-full-pilot24/audit/final-cost-and-variability.json)
contains per-document call links, every attempt's cost and candidate hashes.
Independent final review receipts:
[first eight families](../artifacts/kie-synthesis-production/curated-v7-full-pilot24/audit/final-review-first8.json),
[middle eight](../artifacts/kie-synthesis-production/curated-v7-full-pilot24/audit/final-review-middle8.json),
[last eight](../artifacts/kie-synthesis-production/curated-v7-full-pilot24/audit/final-review-last8.json).
The [publication falsification probe](../artifacts/kie-synthesis-production/curated-v7-full-pilot24/audit/publication-negative-controls.json)
records accepted originals and rejected mutations.

Final generation reported 156 reasoning tokens out of 12,550 output tokens
(1.24%). Earlier high reasoning-token shares were not proof that instructions
alone were the cause; routing, request shape and output contract changed too.

### What is ready, and how to proceed

The selected 24-source path is implemented and the inspected 72-sample pilot is
published. It has actual country, route, product, packaging and numeric
variation, current V7 labels, and a replayable release gate. It is suitable for
reviewing the synthesis design and deciding the next controlled expansion.

Next, use a larger bounded campaign on these contracts to measure the residual
review/correction rate and commodity coverage, while adding further **audited
source families** to improve layout diversity. New sources need exact baseline
replay and ownership of every mutable public/private dependent fact; they do
not inherit approval merely because an old template compiled. Retain the same
review gate and cost ledger. No large campaign or training run was launched.

Positions remain deferred. The source spans and edit receipts can support a
separate position-enrichment experiment, but generated word wrapping is not
automatically authentic page geometry. No fabricated coordinates are emitted.
The published input representation is therefore **plain OCR**, not a drop-in
position-enriched counterpart of the current real-data training inputs.

This pilot does not certify deliverable postal addresses, manufacturer-issued
VINs, real shipping transactions, material densities or regulatory compliance.
It validates the stated extraction/synthesis contract for the inspected set.
The untouched real-data hashes remain:

- train: `ca15c382bd1a36e72db978a0acb34f9dec64e8ea6c7e00639e62bccc98058305`
- validation: `b8c0d4bddd4b3a452f901f3fc5e08e54d580a82768eddd5df33c97ab2c85da5a`

Reproduction entry point, using
`configs/synthesis/mpci_bl_curated_v7_full_pilot24.yaml`:

```bash
.venv/bin/python -m document_ocr.synthesis.curated_campaign validate \
  --config configs/synthesis/mpci_bl_curated_v7_full_pilot24.yaml
```

`generate`, `correct-postal`, `render`, `review`, and `publish` are the respective
execution stages. Validation/publication make no paid calls. A changed wording
contract invalidates generation caches; changed candidates or review contracts
invalidate review approval. Published conflicting files are not overwritten.

## 11. Email/website and positional extension — 2026-10-07

This supersedes the earlier contact-placeholder behavior and the statement that
positions are deferred. The earlier publications remain intact. No real labels,
real OCR, training configuration, training run or new large synthesis campaign
was changed/launched.

### Outputs to inspect

- [Source OCR followed by all 72 revised variants](../artifacts/kie-synthesis-production/curated-v7-full-pilot24-contacts-v2/samples.md)
- [All 72 position-enriched variants](../artifacts/kie-synthesis-production/curated-v7-full-pilot24-contacts-v2/positions/samples.md)
- [Plain revised dataset](../artifacts/kie-synthesis-production/curated-v7-full-pilot24-contacts-v2/dataset.jsonl)
- [Positioned dataset](../artifacts/kie-synthesis-production/curated-v7-full-pilot24-contacts-v2/positions/dataset.jsonl)
- [Exact contact delta and manual review](../artifacts/kie-synthesis-production/curated-v7-full-pilot24-contacts-v2/audit/contact-delta-review.json)
- [Final measurements, coverage and costs](../artifacts/kie-synthesis-production/curated-v7-full-pilot24-contacts-v2/audit/final-contact-position-audit.json)
- Configuration: `configs/synthesis/mpci_bl_curated_v7_full_pilot24_contacts.yaml`.

### Contact diagnosis and actual change

The previous host generator produced unrelated `company-<hex>.example` domains
for email and website *before* the LLM generated the company name. The resulting
labels matched the text, but the contacts were uniformly artificial and could
give one company two unrelated domains. This was not an OCR extraction problem.

The new small call takes the **generated party name, country when labelled, and
original email/website examples** as plain text. It returns only contact strings
through a native Pydantic schema. It does not re-generate parties, addresses,
goods, routes, quantities, phones, faxes or any other shipment content. The
prompt and field descriptions distinguish corporate/free-mail style and bare
`www.` versus scheme-bearing websites. Repeated appearances of a party share
the same generated values.

The original example now renders:

```text
NANPING HUAXIN TEXTILE FIBRE WORKS CO. LTD.
WEB WWW.HUAXINFIBRE.COM
MAIL: INFO@HUAXINFIBRE.COM
```

Other final examples include `www.merilaveoasutus.ee`,
`customer.service@albarra-vet-bh.com`, `ronaldgreaves.bb@yahoo.com`, and
`LOGISTICS@LUNSARMINERAL.COM`. The free-mail examples are retained as free-mail
style rather than forcing every business to have a corporate domain. Some
parties are named people rather than companies; their fictional contacts may
use personal mailbox names on an invented business domain. We do not claim
that domains are unregistered, geographically exclusive, deliverable or safe
to contact. No DNS lookup, email delivery or website request was made.

| Scope/check | Result |
|---|---:|
| Source families with these fields | 8 |
| Changed synthetic documents | 24 of 72 |
| Distinct generated party/company identities | 42 |
| Updated email target occurrences | 45 |
| Updated website target occurrences | 6 |
| Documents without either field | 48, byte-identical candidates |
| Unrelated label or text changes | 0 |
| Remaining hash-placeholder domains | 0 |
| Shared email/website domain pairs | 3/3 coherent |

For every document, the previous candidate is verified against its original
publication manifest. The target-leaf diff permits only the email/website
paths; source-offset edit comparisons permit only those values within their
already-owned regions. Masking old/new contact values makes every differing
rendered region identical. Every revised contact is present in the rendered
text, and the normal schema/render/scenario checks still pass.

The 48 unchanged candidates retain their prior review. For the eight affected
families, the review explicitly composes the **prior published full-document
approval + exact non-contact invariance + manual inspection of all new contact
values**. This is recorded as `prior_published_full_review_plus_manual_contact_delta`;
it is not presented as a fresh full-document LLM review. This bounded repair
did not pay to review or regenerate unchanged shipments.

### Contact probe failures and costs, including discarded attempts

The first probe produced plausible contacts, but a broader style pass showed
that some `www.` examples gained a scheme and free-mail examples became
corporate domains. A concise field-description clarification fixed the style.
One returned hostname contained a space; the native output schema now includes
the generic no-whitespace string constraint, and semantic validation also rejects
malformed contacts. One subsequent provider response was malformed JSON despite
the native-output request. It was rejected, recorded and recovered with one
**explicit** retry. Failed requests never authorize text or labels.

No automatic semantic rewrite/fallback or infinite repair loop was added. The
ordinary stage reports errors. `--retry-invalid-output` permits one new request
for a previously rejected native output and preserves every prior billed attempt.
Already accepted contact receipts are reused without another paid request.

| Accounting | Observed USD |
|---|---:|
| All 25 contact experiment requests, including discarded/failed outputs | 0.003705275 |
| Eight retained final-generation requests | 0.000996525 |
| Final-schema requests including its one failed response and retry | 0.001555575 |
| Retained incremental contact cost per 10k documents at this pilot's mix | 0.1384 |
| Retained incremental contact cost per 10k documents if all have contacts at the affected-cohort density | 0.4152 |

These are **contact-stage-only** estimates, not total synthesis cost or a fixed
quote. They use observed provider billing, three variants per source request,
and the pilot's contact density. Retained requests used 2,762 input tokens and
1,033 output tokens: 895 visible and 138 reasoning tokens (13.36% of output).
The revision's ledger includes copied historical calls as provenance; the table
above isolates new contact spending from that earlier work.

### Position method and why it preserves useful layout

Use the real source's normalized `[0,1000]` coordinates as **layout anchors**.
Do not rematch newly generated words against Paddle's old words; do not copy
coordinates by absolute output line number. The renderer already records exact
UTF-8 source byte intervals, old/new text, region keys and field ownership.

1. Verify that the alignment belongs to the exact source OCR and agrees with
   the existing positioned source input, including its checksum.
2. Replay every exact edit and carry source-line ownership through it.
3. Unchanged lines retain exactly the source coordinate.
4. Changed regions retain their page and source anchor region. If a region's
   line count changes, distribute its new lines along its original line-anchor
   sequence. Two source address lines expanded to three therefore keep the
   first and last anchors and interpolate one between them.
5. An original one-line address expanded to several lines repeats that block
   anchor; it does not invent a line pitch or push the next field downward.
6. If a needed source coordinate is missing, emit ` ||` without coordinates.
7. Page markers and blank lines remain unchanged. Strip the suffixes and the
   result must be byte-identical to the new plain OCR; targets are unchanged.

For the shipper example:

```text
Shipper/Yükleten || 98,72
NANPING HUAXIN TEXTILE FIBRE WORKS CO. LTD. || 165,85
DONGYOU INDUSTRIAL PARK, BUILDING 7 || 190,94
XIFENG ROAD DISTRICT || 166,98
NANPING, CHINA || 142,102
PHONE: +86 131 2345 4284 FAX: +86 131 2345 7268 || 191,111
WEB WWW.HUAXINFIBRE.COM || 140,120
MAIL: INFO@HUAXINFIBRE.COM || 144,129
```

The source address occupied two lines; the generated address occupies three.
The middle address coordinate is interpolated. The phone, website and email
keep their original anchors, despite being one physical OCR line further down.
The nearby country-of-origin and B/L-number headings retain their right-hand
positions (`559,72` and `829,72`), preserving the form's multi-column structure.

### Position scope, falsification and measurements

| Measurement | Result |
|---|---:|
| Synthetic documents processed | 72/72 |
| Nonblank content lines | 7,162 |
| Known coordinates | 6,037 (84.29%) |
| Unchanged lines with exact known source coordinates | 3,385 |
| Changed-region lines with inherited/interpolated coordinates | 2,652 |
| Unknown-coordinate lines, explicitly left unknown | 1,125 |
| Edits changing the number of lines | 129 |
| Known unchanged lines that naïve absolute-line copying would misplace or lose | 1,777 |
| Cross-page transfers or non-suffix text changes | 0 |

The first implementation deliberately rejected structural edits; the pilot
exposed three legitimate replacements of `HDPE\nBAGS` with `\nPALLETS`. These
contain a blank line because the old material qualifier was removed. Blank
replacement lines are now preserved and consume a physical line without being
given coordinates; page-marker edits still fail. All 72 then completed.

Tests cover expansion, contraction, deletion, blank replacement lines, UTF-8
offsets, repeated words on another page, unknown coordinates, stale text and
hashes, duplicate alignment records, altered coordinates, overlapping edits,
and introduced page markers. Checks reject corrupt inputs instead of quietly
mapping them. On all real pilot outputs, each unchanged known line was compared
back to its exact original line and every contributing anchor checked against
the output page. The receipt for every generated line records its source lines,
edit identities, method, page and coordinate.

Measured position-transfer core: **0.0865 seconds for 72 records**, median of
five runs with already loaded source data; about **1.20 ms/document**. Peak
incremental Python allocations under `tracemalloc` were **889,081 bytes** for
that core pass, excluding already-loaded inputs and framework imports. Full
publication with candidate replay and disk I/O took **3.56 seconds** after
initialization. Position enrichment makes no API calls.

The first contact integration redundantly planned sibling variants during
render and took 5.66 seconds for 72. Profiling identified that work; per-sample
contact receipts removed it. Rendering then took **3.07 seconds**, versus the
previous recorded 2.93 seconds (about 2 ms extra per record, with normal timing
noise). Contact receipts remain bound to both the per-sample request and batch.

Targeted validation: **87 tests passed**, lint passed. A real cached
`generate → contacts → render` probe on three variants reproduced the same
text/targets in 0.315 seconds with paid calls disabled. Plain publication and
the positional extension both replayed successfully. No one-time script was
left in the repository.

### Interpretation and next action

This is a working spatial-input construction method for the pilot, with
explicit provenance and preservation checks. It supplies real layout structure
without pretending that an unseen synthetic PDF was measured. It does **not**
prove model generalization or an F1 improvement: that requires the later
controlled training comparison. No random jitter or artificial independent
per-line noise was added. Before a larger campaign, expand audited source-layout
coverage and keep the real validation sources excluded; more textual variants
from these same 24 layouts are not equivalent to more independent layouts.

Revised plain dataset SHA-256:
`1e4883d49fcfca09169a2ace6190f5c5d07863650c898ffe067070a32f32de4a`.
Positioned dataset SHA-256:
`b8211a7ee912657ef3e6f5ffd21a342569e256d5b02b0c55a7da91a87add4214`.

## 12. Coordinate synthesis experiments — 2026-10-07

### Decision and scope

**Recommend coherent page-level augmentation of source-owned anchors for the
next positional synthesis option. Do not enable unrestricted font-based local
reflow.** The experiments below separate accurate reconstruction of individual
line centres from preservation of useful document-layout relationships.
These are different requirements and need different checks.

This was an offline investigation over all **600 current training sources**
and the **72 generated documents / 24 source families**. The 60 validation
documents were not used to select parameters, estimate fonts, or inspect layout
distributions. Their hashes, the training dataset hashes, all real OCR/labels,
and training configurations remain unchanged. No model training, OCR rerun,
LLM call or paid service was used. Incremental API cost: **USD 0**.

Experiment code, inventories and exported candidates are together in
[`docs/analysis/synthetic-positions-probe-20261007/`](analysis/synthetic-positions-probe-20261007/).
That directory is intentionally gitignored like the other analysis outputs.
The production positional entry point still uses `source_edit_anchors_v1`;
this report does not imply the proposed augmentation has been wired into it.

### A. Formatting correction, including already published samples

The auxiliary recipe for source `1464f450` used a single-quoted YAML string
containing literal `\n`. YAML does not turn those into newlines. The generated
port-expenses clause therefore occupied one line instead of three.

The recipe now uses a literal multiline scalar. Contract validation rejects
escaped newline/carriage-return/tab sequences in rendered text, prefixes and
suffixes instead of silently rendering them. Regression tests exercise both
rejection and preservation of real line breaks.

The complete current 72-record publication was replayed from its existing
scenario/wording/contact receipts. Only three records changed, and only the
two separators in their source-only clause changed. All 72 target objects and
all unrelated rendered bytes are identical. Existing full-document reviews
were retained with an explicit, hash-bound inspected newline-only delta for
the affected family; this was not described as a fresh LLM review. Plain OCR,
candidate receipts, source/render gallery, positional OCR/receipts and both
publication manifests were regenerated and independently replayed.

The entire previous publication remains at
`artifacts/kie-synthesis-production/curated-v7-full-pilot24-contacts-v2-before-newline-repair-20261007/`.
The validated staging duplicate was removed after publication; no source or
unique generation receipt was removed. The atomic-directory-swap attempt was
rejected by Windows/DrvFS, so the final publication uses backed-up, staged,
per-file atomic replacements and commits manifests last. Receipt verification
also recovered a reporting-only finalization exception; all persisted outputs
were then rechecked, not assumed successful from process status.

Current plain dataset SHA-256:
`d0ce6e73fdfeaaacdda49f25323e96a6d8ce47bd66704372421e2d94374aaeee`.
The ordinary campaign `validate` command reports **72 expected / 72 valid**,
24 sources, no failures. Position coverage is now **6,043 / 7,168 lines**:
3,385 unchanged known anchors, 2,658 edited-region anchors, 1,125 unknowns.
Earlier counts/hashes in section 11 describe the retained pre-fix publication.

Details: [formatting receipt](analysis/synthetic-positions-probe-20261007/formatting-repair.json).

### B. Real-data withholding test: are text-aware centres better?

We selected consecutive OCR-line triples within one source paragraph and page,
with individually matched Paddle regions and plausibly aligned outer lines.
The middle line's coordinate was withheld from the predictor. Only its text
and the two surrounding measured lines were used to predict its centre.
The hidden box was then used as the scoring reference, not a fitting target.

This yielded **10,489 line tests from 595 real training documents**. Compare:

1. Interpolation between the surrounding source centres.
2. A character-count width estimator calibrated on surrounding source widths.
3. DejaVu Sans glyph advances calibrated on those same surrounding widths,
   with the left edge and line spacing inherited from the source.

Errors below are divided by the neighbouring source text height, making them
comparable across different page raster sizes. They are geometric errors,
**not extraction F1, label accuracy, or measured synthetic-font accuracy**.

| Horizontal centre error | Median | 90th percentile | 95th percentile |
|---|---:|---:|---:|
| Anchor interpolation | 2.054 heights | 6.073 | 7.776 |
| Character-count reconstruction | 0.200 | 1.333 | 2.257 |
| Calibrated glyph-width reconstruction | **0.177** | **0.980** | **1.736** |

The glyph estimator improves horizontal error on **9,417 / 10,489 (89.8%)**
cases. Vertical interpolation is already much better in these triples:
median 0.025, 95th percentile 0.403 line-heights. This supports using source
geometry and text width rather than blindly blending unrelated old widths.

But **1,029 / 10,489** glyph predictions still miss horizontally by more than
one text height. Inspecting the worst cases explains why blanket application
would be unsafe: an OCR paragraph is not necessarily one spatial column. For
example, source `17440e5e` alternates container rows at x≈84 with `CY/CY` and
other wording at x≈1143. Text-sequential neighbours can therefore bracket a
line that actually belongs to the other column. Mixed font sizes and alignment
styles are further reasons not to infer complete layout from word length.
These deliberately retained counterexamples falsify the universal-reflow claim;
they are not removed from the headline metric to improve the result.

Evidence: [all withheld anchors](analysis/synthetic-positions-probe-20261007/withheld-real-anchors.json),
[summary](analysis/synthetic-positions-probe-20261007/results.json).

### C. Reflow on actual synthetic replacements: does it fit?

We probed all **4,086 changed render regions**. A local geometry proposal needs
complete physical-line ownership, known simple source boxes, source-supported
left alignment and measured spacing. Singleton expansion requires spacing
from nearby compatible source boxes, not an invented line pitch. New boxes
must fit the page and avoid other Paddle text regions, including regions that
have no GLM-line match. Simultaneous replacement proposals are checked against
one another as well as against old text. The font estimate is calibrated on
that region's measured source widths.

| Outcome | Render regions |
|---|---:|
| Passed these geometric prerequisites | 584 |
| Partial-line ownership; full-line reconstruction would be needed | 3,169 |
| Required source anchor missing | 153 |
| Collision with another source text region | 123 |
| Proposed box outside the page | 12 |
| No measured line pitch for expansion | 9 |
| Non-simple/multi-region source geometry | 12 |
| Irregular source spacing | 21 |
| Text removed, no replacement position needed | 3 |

The 584 geometry candidates cover 734 lines in 71 documents. These are **not
734 certified glyph positions**: font estimates are approximate, and Paddle
text boxes alone do not describe all form rules, cell boundaries or logos.
The diagnostic output is explicitly marked `DIAGNOSTIC_ONLY_NOT_TRAINING_APPROVED`.

Expansion is particularly restrictive. Of **103 expanding regions**, only
**four** pass the local reconstruction prerequisites; 22 collide, 27 have
partial-line ownership, 34 lack required anchors, nine lack a measured pitch,
six have non-simple regions and one has irregular spacing. This is a limitation
of this physical-reflow proposal, not a claim that the 99 other samples are
invalid or that their existing coarse source anchors cannot be used.

Visual inspection of all five exported examples confirmed the distinction:

- [3→4 address lines that fit inside the source shipper cell](analysis/synthetic-positions-probe-20261007/geometry-review-01.png).
- [4→5 address lines that fit inside the source shipper cell](analysis/synthetic-positions-probe-20261007/geometry-review-02.png).
- [1→3 address lines that would overwrite email and phone positions](analysis/synthetic-positions-probe-20261007/geometry-review-03.png).
- [1→2 party-name lines that would overlap the address](analysis/synthetic-positions-probe-20261007/geometry-review-04.png).
- [1→2 address lines that would overlap the contact row](analysis/synthetic-positions-probe-20261007/geometry-review-05.png).

These are inspected source rasters with original and proposed boxes overlaid;
they are not generated document images. The full candidate/rejection inventory
is [here](analysis/synthetic-positions-probe-20261007/reflow-proposals.json).

### D. Coherent augmentation and its quantization failure mode

Use one positive uniform scale and one translation for the **whole page**:

```text
x' = 500 + s × (x − 500) + dx
y' = 500 + s × (y − 500) + dy
```

The same transformation covers captions, party values, container rows and all
other known anchors. We constrain the transformed envelope using **all Paddle
text boxes**, not only successfully matched GLM lines. No independent line
jitter, guessed missing coordinates, rotation, mirroring or OCR reordering is
introduced. Source markers, blank lines, text, label values and unknown-anchor
membership remain unchanged.

Probe ranges are 0.95–1.05 for scale and ±20 grid units (2% of a page dimension)
for translation, further restricted by the actual page clearance. These are
deliberately modest experimental limits, not optimized hyperparameters. TILT
provides precedent for coherent geometric augmentation, but uses a different
architecture and cannot establish a gain for our serialized-coordinate model:
[primary paper, section 4](https://arxiv.org/pdf/2102.09550).

The first experiment already preserved page bounds and axis order, and all
well-separated nearest neighbours. Nevertheless, rounding back to integer
coordinates changed **2,409 / 259,115** uniquely nearest relationships in real
page variants and **426 / 29,585** in synthetic variants. These were close/tied
configurations, not broad layout corruption, but they disproved a stronger
claim that ordinary affine augmentation automatically preserves every nearest
relationship on a finite grid.

The refined acceptance gate therefore checks the actual integer output:

- Page bounds and one coherent transformation per page.
- All existing horizontal/vertical order and equal-coordinate alignments;
  no new axis ties caused by compression/rounding.
- The complete nearest-neighbour set of **every known point**, including
  originally tied neighbours; no changed nearest neighbour or broken tie.

We try at most 32 geometrically eligible scaling proposals. If none passes,
use an explicitly recorded **integer-translation-only mode**. This is a
first-class, exact-distance-preserving augmentation, not a silent guessed
coordinate fallback. Translations are selected only within measured clearance.
Every mode is checked against the same final contract. The lower-level geometry
sampler also has a finite 64-proposal limit; exhausting it explicitly selects
the translation-only mode. It does not clamp off-page content or drop the
document. A dense, page-filling boundary case has no admissible nonidentity
transform on this integer grid: it retains its geometry with a recorded reason
rather than corrupting positions in order to claim augmentation.

### E. Refined results and independent persisted-output verification

All 600 real training sources and 72 synthetic records were tested under five
deterministic seeds: **6,380 page variants** (1,156 real pages + 120 synthetic
pages, each ×5). There were no page/document omissions.

| Refined result | Count |
|---|---:|
| Uniform scale + translation variants accepted | 6,367 |
| Explicit integer-translation-only variants | 13 |
| Known coordinate placements checked | 290,875 |
| Accepted nearest-neighbour or tie changes | **0** |
| Accepted new axis ties / ordering changes | **0** |
| Accepted off-page positions | **0** |
| Deterministic same-seed page replays | 1,276 |
| Documents tested for independence from target-field metadata | 72 |

The label-blindness test replaces every `targetPaths` hint in the edit proof
with deliberately wrong metadata and still obtains identical positioned text.
Coordinates derive from source geometry and render ownership, not gold values.

Five **separate 72-record experimental input variants** were exported. They
contain the same generated OCR and targets, with different accepted page
transforms. They are not automatically appended to training or counted as 360
independent shipments:
[`coherent-page-variant-0.jsonl`](analysis/synthetic-positions-probe-20261007/coherent-page-variant-0.jsonl)
through `coherent-page-variant-4.jsonl` in the same directory.

An independent verifier reads those saved files, parses their suffixes and
uses scalar integer distances rather than the generator's NumPy implementation.
Across 360 record-variants it verifies:

- **30,215 known coordinates** and their correct page transforms.
- **5,625 preserved unknown-coordinate occurrences**.
- **930,420 pairwise horizontal/vertical order and alignment checks**.
- **30,215 exact nearest-neighbour-set checks**, including ties.
- Identical source text, labels, page markers and physical line counts.

There were **zero failures**. Six deliberately corrupted transform cases
(independent line jitter, wrong page transform, swapped ownership, off-page
geometry, invalid scale and quantization collapse) were rejected. An additional
25 actual synthetic-page variants from the first weaker experiment that changed nearest/tied
relationships were all rejected by the stronger checker. This tests rejection
behaviour, not merely agreement on successful generated outputs.

Additional boundary controls passed for a page-filling 1,001-point dense grid,
a singleton at the page corner, repeated identical anchors, and a two-column
layout with equal-distance neighbours. NaN and off-page source positions were
rejected. The dense-grid case retains identical integer coordinates even when
a near-identity scale is admissible; it is not counted as new layout signal.
A separate forced proposal-exhaustion control reaches the exact translation-only
mode without a document drop or an infinite retry.

Two source-page raster comparisons were also inspected after applying the same
transformation to the complete image: [multicolumn cargo form](analysis/synthetic-positions-probe-20261007/coherent-page-review-1.png),
[party and equipment form](analysis/synthetic-positions-probe-20261007/coherent-page-review-2.png).
They demonstrate a realizable whole-document geometric change, including form
lines and captions. No image was used as a training input and OCR was not rerun.

Receipts: [strict results](analysis/synthetic-positions-probe-20261007/strict-results.json),
[per-page transforms](analysis/synthetic-positions-probe-20261007/strict-augmentation-pages.json),
[independent verification](analysis/synthetic-positions-probe-20261007/independent-export-verification.json).

### F. Runtime, limits and recommended next step

The initial full probe completed in about 5.8 seconds after local inputs were
available, including 10,489 withholding tests and 6,380 augmentation variants.
The stronger processing/validation/export pass took about 8.6 seconds, excluding
the final two raster illustrations. Mean synthesis time for a synthetic page
was approximately **0.82 ms**; at this pilot's 120 pages / 72 documents this is
about **1.37 ms/document** of incremental geometric work. The actual test path
includes rejecting bad proposals, not only successful transforms.

The original anchor-transfer pass for 72 records took 0.208 seconds in this
probe; running local reconstruction checks as well took 0.669 seconds. These
are offline experiment costs, not training/GPU throughput measurements. Peak
process RSS was about 683–692 MiB while holding all real-source/columnar OCR
data. A separately traced single-page core used about 67 KiB incremental Python
memory; this excludes loaded inputs and is not a whole-process memory claim.
The independent persisted-file verifier took about 1.5 seconds.

Final targeted suite: **82 passed in 9.13 seconds**, covering auxiliary
contracts, positional transfer, templates, publication and campaign replay.
Ruff and Python compilation passed for the changed production/test files and
all retained experiment scripts. Validating all 164 auxiliary recipes took
0.079 ms before versus 0.175 ms with the escaped-control guard (median of five
1,000-iteration repetitions): **+0.096 ms per whole-contract validation**.
That negligible startup/contract-check cost does not add calls or blocking I/O
to generation. No production positional hot path was changed.

**What this establishes:** the recommended page augmentation genuinely changes
absolute positions while preserving tested source-layout relationships and
text/target ownership. The integer-translation mode preserves every pairwise
vector exactly; accepted scaled modes additionally preserve all measured axis
orders, equal-axis alignments and nearest-neighbour sets after rounding. All
source points move together, which is fundamentally different from unstructured
coordinate noise.

**What it does not establish:** new layout families, repaired original Paddle
alignments, accurate glyph centres for synthetic words, or improved model F1.
In particular, this augmentation preserves the existing coarse anchor signal;
it does not turn a single-source-line expansion into measured new line boxes.
The local-reflow experiment showed why forcing that interpretation is unsafe.

Given the desired goal is useful layout conditioning rather than exact PDF
typesetting, promote the coherent page transform as the next configurable
positional option, retaining current source-owned anchors and explicit unknowns.
Keep local font reflow experimental until a template has an independently
established spatial region and sufficient room; do not quietly apply it merely
because text looks left-aligned or an interpolation returned numbers.

The subsequent model test should compare **identical synthetic shipments and
labels**, same real dataset, optimizer, steps and seed: inherited coordinates
versus the approved coherent augmentation. Keep real validation inputs unchanged.
Compare whole-task metrics, relation/placement metrics, party-boundary errors,
parse validity and throughput. That isolates whether the preserved-and-varied
layout signal actually benefits learning; this experiment alone cannot answer
that training question.

Reproduction (all offline):

```bash
.venv/bin/python docs/analysis/synthetic-positions-probe-20261007/probe.py
.venv/bin/python docs/analysis/synthetic-positions-probe-20261007/refine_augmentation.py
.venv/bin/python docs/analysis/synthetic-positions-probe-20261007/verify_export.py
```

## 13. Production positional augmentation — 2026-10-07

### Delivered scope

Implemented the approved geometry approach in the actual `positions` campaign
stage and published **72 positioned synthetic samples**, using the existing
24-source × three-variant shipments. Wording, shipment facts and labels were
not regenerated. No paid calls, GPU processing or training were required.

- [Rendered samples with coordinates](../artifacts/kie-synthesis-production/curated-v7-full-pilot24-contacts-v2/positions-augmented-v1/samples.md)
- [Dataset JSONL](../artifacts/kie-synthesis-production/curated-v7-full-pilot24-contacts-v2/positions-augmented-v1/dataset.jsonl)
- [Publication manifest](../artifacts/kie-synthesis-production/curated-v7-full-pilot24-contacts-v2/positions-augmented-v1/manifest.json)
- [Independent validation and stress results](analysis/synthetic-positions-probe-20261007/production-validation.json)
- [Before/after benchmark](analysis/synthetic-positions-probe-20261007/production-benchmark.json)

The old plain and anchor-only publications, and all 600 real training / 60
validation records, remain unchanged. The production dataset hash is
`1c84e59d4bf50ec195f8a13e137446704d43f3858121a29d01e5bd560ccdeb49`.

### Contract and integration

1. `curated_positions.py` replays each published candidate and its exact source
   edits, then inherits anchors as before. It loads the source dataset's own
   hash-pinned Paddle run rather than accepting a second unrelated geometry path.
   Source regions, counts, page frame, PDF identity, enclosing boxes and integer
   centroids must match the saved alignment. This verifies provenance, not the
   semantic correctness of the original OCR matching.
2. `curated_layout.py` applies **one positive uniform scale plus translation to
   the whole page**. Coordinates are rounded once onto the existing 0–1000 grid.
   No labels or field identities enter sampling, and there is no per-line jitter.
3. Every proposal checks page bounds, including all recognized source boxes;
   consistency with a single affine map to within half a grid unit; left/right
   and above/below order; equal-axis alignment; and complete nearest-neighbour
   sets, including tied neighbours and coincident anchors.
4. The search has a hard bound. When no scale proposal qualifies, exact integer
   translation preserves all relative vectors and is recorded as
   `integer_translation_only`. `scale_attempts: 0` deliberately selects that
   mode. Unknown pages record `no_known_anchors`. A page filling the grid can be
   immovable; its zero-change receipt is not misrepresented as new geometry.
5. Per-sample receipts retain inherited and final coordinates, source lines,
   edit identity, transform parameters, mode, rejected-proposal reasons and
   policy. The manifest binds receipts, dataset, gallery, source publication and
   geometry hashes. A conflicting rerun refuses overwrites; the manifest is
   written last. Identical reruns reproduce the saved output exactly.

The active configuration is extended in place:

```yaml
positions:
  output_subdirectory: positions-augmented-v1
  seed: 20261007
  scale_min: 0.95
  scale_max: 1.05
  max_translation: 20
  scale_attempts: 32
```

Scale and translation bounds reproduce the tested mild page perturbation.
Unlike the prototype's nested search, production uses at most **32 total
proposals**, limiting work on crowded pages. The same final geometric contract
applies; more pages can select exact translation rather than scaling. The
five-seed test found 30 such pages versus 13 in the earlier prototype, with no
relationship loss. This is an explicit variation/runtime trade-off, not a
weakened acceptance check.

NumPy is now an explicit dependency (already installed and locked at 2.5.2);
the lock refresh changed no dependency versions. No new service or API is needed.

### Saved-output validation

The independent checker parses the **saved positional suffixes** and computes
pairwise relationships with scalar Python arithmetic, separately from the
NumPy production validator. All 72 source/candidate replays succeeded.

| Check | Result |
|---|---:|
| Published documents / pages | 72 / 120 |
| Scaled-and-translated / integer-translation pages | 118 / 2 |
| Known coordinates checked | 6,043 |
| Coordinates actually changed | 6,041 |
| Unknown coordinates preserved as ` ||` | 1,125 |
| Source boxes checked against page bounds | 11,892 |
| Pairwise ordering/alignment checks | 186,084 |
| Complete nearest-neighbour sets checked | 6,043 |
| Unexpected OCR, label, page or unknown-position changes | **0** |
| Geometric contract failures | **0** |

Median anchor displacement is **16.97 grid units**, 95th percentile 30.61;
these are Euclidean displacements, so they can exceed the per-axis translation
limit after including scale. Two anchors round back to their original positions;
this does not indicate that their entire pages were unchanged.

The production implementation was also exercised on all **600 real training
sources plus the 72 synthetic samples**, under five seeds: **6,380 page
variants and 290,875 point placements**, with no failed final contracts and
1,276 exact seed replays. This stress run did not save changes to the real data.

Negative controls reject per-line jitter, off-page transforms, rounding-induced
column collapse, rounding-induced neighbour-tie changes, invalid scales,
nonfinite/fractional source coordinates, malformed boxes and stale geometry.
Tests also cover Unicode edit spans, line expansion/contraction, repeated text
on separate pages, absent anchors, dense page-filling grids, explicit proposal
exhaustion, filtered parquet with/without row-group statistics, publication
replay and conflicting-policy overwrite prevention.

**150 targeted synthesis tests passed in 12.77 seconds.** Ruff passed. Two
source-layout diagrams were visually inspected: [party/form layout](analysis/synthetic-positions-probe-20261007/production-layout-1.png)
and [multicolumn cargo layout](analysis/synthetic-positions-probe-20261007/production-layout-2.png).
Grey boxes are source OCR regions; blue points are edited-region anchors.
The diagrams deliberately do not pretend that synthetic text has been typeset.

### Runtime, memory and cost

The complete 72-record position stage, including source replay and disk checks,
took **3.63–3.75 seconds**, versus **3.47–3.65 seconds** for anchor-only output.
Median difference: **+0.114 seconds per whole pilot**, approximately **1.6 ms
per document**. This is a small measured feature overhead, not a throughput gain.
The geometry core averaged **0.33 ms per synthetic page**, versus approximately
0.82 ms in the earlier stricter prototype. These are CPU measurements, not
training throughput estimates.

Profiling identified avoidable source-loader startup cost. Streaming selected
parquet row groups, with exact PDF filtering and bounded batches, removed the
dataset/Pandas import path and avoided materializing unrelated rows. Serial
decoding of these small groups reduced the isolated reader peak from about
139 to 90 MiB with less than 10 ms difference. Whole campaign benchmark peak
RSS was **483 MiB**, versus **430 MiB** for anchor-only processing: the extra
source geometry and reader buffers are explicit. The all-source audit peaked
at **381 MiB**. A 1,001-point dense-page test used about **1.30 MiB** traced
temporary memory; pairwise checks are chunked instead of storing full distance
tensors. No unbounded retry loops or API work were added.

**Additional API cost: $0.** Existing wording/contact receipts were reused.

### What is established, and the remaining model question

The implementation is complete for **coherent augmentation of inherited layout
anchors**. The saved samples satisfy the stated geometric contract and retain
the original source-layout signal while varying absolute positions. Source
hashes and exact text/label preservation were verified, not assumed.

This does not certify original Paddle alignment as perfect, synthesize new form
families, or locate the actual glyph centres of longer generated wording.
Single-line expansions still share the inherited coarse anchor; unknown lines
still lack coordinates. Local font-width reflow remains disabled because the
earlier collision/ownership experiment did not support applying it generally.

Whether this variation improves extraction F1 still requires the planned
controlled training comparison. No training config or validation inputs were
changed in this pass.

Reproduce the production output (cached generation only):

```bash
.venv/bin/python -m document_ocr.synthesis.curated_campaign positions \
  --config configs/synthesis/mpci_bl_curated_v7_full_pilot24_contacts.yaml \
  --project-root .
```

Recheck the delivered files and the five-seed all-source stress test:

```bash
.venv/bin/python docs/analysis/synthetic-positions-probe-20261007/validate_production.py
```

## 14. Casing, long/split goods wording, and equipment alias audit — 2026-10-07

This is a read-only audit of the current contacts-enabled 24-source/72-sample
pilot and all 660 current real records. No generation requests, training, label
repairs or pipeline changes were made. Machine-readable measurements and exact
source-region inventories are in
[`casing-goods-alias-audit-20261007.json`](../artifacts/kie-synthesis-production/curated-v7-full-pilot24-contacts-v2/audit/casing-goods-alias-audit-20261007.json).

### Target casing versus rendered casing

`labeling_agents/target_normalization.py` defines an explicit uppercase field
set: party names, address lines, countries and contact names; route/place names
and countries; vessels; goods descriptions, handling instructions and origins;
free-text equipment/package descriptions. It leaves identifiers, emails, URLs,
category tokens and units alone. `curated_campaign.assemble_lexical_target`
invokes this normalizer after constructing the target from generated regions.

All **660 real targets and 72 synthetic targets** already conform: applying the
normalizer produces zero changes. This is enforced in code, **not configurable
in the campaign YAML**. The reduced target still omits fields excluded by the
dataset policy; listing a field in the general normalizer does not add it.
An additional three-sample assembly probe uppercased versus lowercased the same
cached wording inputs; both paths produced identical uppercase targets in all
three cases. This used variant 1, the first valid campaign variant.

Rendering is a separate path, but there is no independent casing augmentation
policy. The wording prompt says **“Use uppercase human-readable text.”** The
renderer wraps changed wording without converting its case, and retains
unchanged source text verbatim. The actual generated-region values contain:

| Region kind | Uppercase | Mixed casing |
| --- | ---: | ---: |
| Party/person identity | 176 | 10 |
| Postal component | 168 | 27 |
| Product wording | 78 | 3 |

These are region values, not document counts or physical line counts. Mixed
case therefore exists, but is not a controlled distribution. For example,
`syn_full_v7_1acd31d73ee82dd0afd4d193` prints `HASSelt KOERSTRAAT 88, WAREHOUSE 2`:
this is incidental casing, not a deliberate source-style augmentation policy.

Recommended next change: keep target normalization separate from a seeded,
configurable **render-only** casing policy, applied consistently to appropriate
fields/blocks. Protect identifiers, contact endpoints and case-sensitive
technical notation. Set distribution weights after measuring source styles;
do not mistake uncontrolled generator variation for an implemented policy.
This need not require additional LLM calls.

### What the goods generator receives

`curated_campaign.wording_request` supplies sampled HS6 identities and their
classification hierarchy, national child examples, route context, a source
text-region example and a specific requirement for each output string. Thermal
cargo also receives the observed cold-chain commodity/form/settings context.
The agent does **not** receive the full source OCR/PDF in this generation call.
It is asked for concrete commercial wording retaining supplied distinguishing
qualifiers, not a list of tariff alternatives; unbranded products; no unsupplied
manufacturer/model/specification claims; and no shipment counts, package totals,
weights, volumes or tariff captions, which have separate host-owned rendering.
Products and party wording are generated together under a schema fixing the
requested keys. Whole-unit vehicle profiles instead use host-provided product
phrases and skip product LLM generation.

Explicit generated line breaks are retained for rendering. Otherwise source
region layout guides wrapping. Label assembly collapses whitespace and applies
the declared target expression and uppercase normalization. No general product
complexity, item-count or description-length distribution is sampled yet.

| Description length in characters | Real 660 | Synthetic 72 |
| --- | ---: | ---: |
| Median | 33 | 72 |
| 90th percentile | 101 | 110 |
| Maximum | 1,207 | 391 |
| Above 500 | 4 | 0 |

The 24 selected **source** descriptions max out at 194 characters. Thus the
pilot does not exercise the genuinely long real descriptions, despite accepting
multiline strings mechanically.

Confirmed topology examples:

- `doc_01d86535…`: four product regions (`BRUSH`, `SPONGE`, `BODYCARE TOOLS`,
  `PLASTIC TAKE NAILS`) are generated separately and joined by the explicit
  expression `{brush}; {sponge}; {tools}; {nails}` into **one goods item**. The
  longest synthetic description (391 characters) comes from this structure.
- `doc_3dc8551d…`: the laminated-steel description repeats four times. One
  variable renders consistently at all four occurrences; repetition is not
  four independent generated identities.
- `doc_88bab156…`: source product wording appears twice, with package counts,
  `CEREAL SEED QUALITY`, HS/customs information between the occurrences. Its
  reviewed contract replaces both identity regions with the same complete new
  description, neutralizes `CROP:` to `GOODS:` and deletes the old seed-quality
  qualifier when resampling a different commodity. This prevents stale wording
  but is **not** preservation of a general multi-fragment continuation layout.
- Training source `doc_345a2a0b…` has a **1,160-character** laboratory-supplies
  description extending across three pages, separated by container rows,
  document metadata and boilerplate. It is not in the pilot.
- Training source `doc_ecb17253…` has a **788-character** used-compressor/dryer
  description with models, pressures and serials. It is also outside the pilot.

The distinction matters: untouched text outside explicit owned regions stays
in the template; generic generation does not rediscover all product qualifiers
there. A new family needs a complete product/auxiliary ownership inventory.
The current wording planner accepts one region covering all identities, or
one region per identity. A direct probe with **two distinct fragments for one
identity** raises `product fragments need a complete commodity ownership
contract`. It does not silently synthesize an arbitrary split description.

Recommended next extension: represent ordered description fragments separately
from HS/product identity count, with repeated occurrences sharing a fragment
and an explicit complete target assembly. Render fragments around separately
owned customs/quantity/DG/reference regions. Supply coherent product attributes
for technical descriptions rather than asking for arbitrary extra prose. Pilot
long contiguous, cross-page and interleaved descriptions using **training-only**
sources. Validate no stale product tails, omitted fragments, duplicated target
content or leakage of non-product captions into descriptions. No assertion that
all long/split families already work is warranted by the current pilot.

### Equipment categories and printed aliases are different stages

Semantic size/type pairs come from compatible training-observed physical
bundles, filtered by family, required measurements, thermal settings and source
capabilities (`curated_scenarios.ScenarioCatalog._donors`). The 72 samples have
159 containers: 71 standard-height 20-foot general-purpose, 69 high-cube 40-foot
general-purpose, 7 standard-height 40-foot general-purpose and 12 high-cube
40-foot refrigerated. These are container counts, not document counts.

Printed spelling is then selected deterministically from the **source spelling
and sampled semantic pair**, chiefly by
`template_compiler.descendant._receipt_equipment_surface`. There is no random
alias pool or alias-weight configuration in this campaign:

1. Preserve a recognized source alias when its semantic pair is unchanged.
2. Preserve ISO representation and valid detail/spacing/O-for-zero style when
   switching the pair through `container_semantics.iso_equipment_surface`.
3. Preserve partial length/height declarations as partial declarations.
4. For other compact source styles, use a compatible preferred output code;
   otherwise emit explicit canonical equipment wording.

Read-only probes of the actual helper:

| Source surface | New size/type | Output |
| --- | --- | --- |
| `20DV` | 20-foot standard general-purpose | `20DV` |
| `20DV` | 40-foot high-cube general-purpose | `40HC` |
| `40HQ` | 40-foot high-cube general-purpose | `40HQ` |
| `40HR` | 40-foot high-cube refrigerated | `40HR` |
| `22G1` | 40-foot high-cube general-purpose | `45G1` |
| `22G1` | 40-foot high-cube refrigerated | `45R0` |
| `22GO` | 40-foot high-cube general-purpose | `45GO` |
| `GP20` | 40-foot high-cube general-purpose | `40HC` |

The recognized vocabulary is the reviewed grammar/tables in
`synthesis/container_semantics.py`, not a single randomized list. Its carrier
rules extend beyond the basic compact-code table. `equipment_registry.py`
provides exact ISO/BIC categories, not all carrier spellings. The older
`_equipment_surface_candidates` function is not this campaign's alias sampler.

Printed pilot examples include `20GP`, `GP20`, `40HQ`, `40HC`, `40'HC`, `HC40`,
`40 DRY 9'6`, `1X40HR`, `RH40`, `40 REEF 9'6`, `22G1`, `42G1` and `45G1`.
Their variation is inherited across source templates, not an independently
sampled distribution. A bare `40'` remains length-only, and `40HC` can remain a
height-only receipt even for a reefer. Those helper outputs alone do not prove
that a complete type label is grounded: complete supporting declarations must
be checked in the whole document when onboarding another source.

If independent alias augmentation is desired, add an explicit reviewed output
alias registry with source/carrier eligibility and reproducible sampling. Reuse
the choice across repeated declarations; check each complete alias round-trips
to the intended pair and separately validate partial declarations. Recognition
of an alias is not evidence that all aliases are presently generated.

## 15. Configurable casing and targeted complex-goods experiment — 2026-10-07

### Implemented: independent target and input casing

`curated_casing.py` now defines the policy. The campaign's `casing.target` is
`uppercase` or `preserve`; `casing.render_styles` chooses among `preserve`,
`uppercase`, and `title`. Multiple distinct styles are sampled uniformly with a
dedicated deterministic stream. The chosen style is shared across eligible
owned text in one document, including repeated parties and separately printed
localities/countries. Source headings/unowned bytes, technical goods wording,
equipment codes, units and endpoints are not title-cased. Alphanumeric tokens
inside postal/name regions are preserved too.

Target assembly, source-span rendering, independent replay, final validation
and semantic-review instruction hashing all use the configured policy. No
additional LLM request is involved. Existing real labels are unchanged. The
published pilot config explicitly retains `preserve` rendering for replay;
new runs can select `[uppercase, title]` in their own output directory.
Company-contact requests use a canonical company/country identity independently
of presentation casing, so a casing-only change does not split repeated parties
or invalidate the generated contact identity. Email/URL values are not recased.

Read-only runtime validation rendered all 72 cached shipments in four modes:

| Mode | Validated records | Text changed from published pilot | Target changes |
| --- | ---: | ---: | ---: |
| Preserve | 72 | 0 | 0 |
| Uppercase eligible text | 72 | 21 | 0 |
| Title-case eligible text | 72 | 72 | 0 |
| Seeded uppercase/title mixture | 72 | 49 | 0 |

The mixture selected 38 title-case and 34 uppercase documents. Across all
modes, whole text compared equal after uppercasing; exact-edit replay passed;
non-casing target data were unchanged. Tests additionally protect source
headings, product specifications, units, alphanumeric identifiers and endpoints,
and verify order-independent style selection and policy-sensitive review hashes.

Three repetitions per 72-record render: before the change the median was
**2.888 seconds**; mixed casing after the change was **2.906 seconds** (about
**0.24 ms/document** extra, within the small run-to-run variation). Isolated
process peak RSS was 442,508 versus 443,496 KiB (about **0.97 MiB** higher).
**383 targeted tests passed in 14.60 seconds**; changed production/test files
passed Ruff. Diagnostic casing samples and measurements:
[samples](analysis/synthesis-goods-casing-probe-20261007/casing-samples.jsonl),
[validation](analysis/synthesis-goods-casing-probe-20261007/casing-validation.json).
The alternative `target: preserve` branch was also exercised on all 72 records:
24 retained mixed-case target text, and all 72 became exactly the published
targets when the agreed uppercase normalization was reapplied. Contact receipts
remained valid without new API work.

### Clarification and correction: bare equipment lengths

The user's agreed annotation convention is already in the source resolver:
bare `20'`/`40'` means standard-height/general-purpose unless other explicit
equipment/thermal evidence says otherwise. It is a target convention, not a
claim about what bare dimensional notation physically proves.

The receipt renderer had a narrower inconsistency: when changing equipment,
it could keep bare length wording even for a newly selected high-cube or reefer
pair. It now retains bare wording only when resolving that wording returns the
new pair; otherwise it prints a distinguishing alias/canonical description.
Thus unchanged standard GP stays `40'`, while a new high-cube GP prints `40HC`,
and a high-cube reefer prints explicit compatible refrigerated wording. Alias
style remains source-derived; no alias-weight sampler was added. Round-trip
tests cover both bare lengths and standard/high-cube/refrigerated resampling.
The current 72 published records replay unchanged in preserve mode.

### Goods experiment scope and inputs

Three **training** sources were selected manually for goods-only diagnostics:

| Source | Product OCR supplied | Current target length |
| --- | --- | ---: |
| `doc_345a2a0b…` laboratory supplies | Three regions across three pages; 11, 43 and 1 lines | 1,160 chars |
| `doc_ecb17253…` compressors/dryers | Heading plus 17 serial-numbered entries, 18 lines | 788 chars |
| `doc_85b6388a…` telecom equipment | Parts/specifications/batch wording, 19 lines | 394 chars |

Exact sources, selected registry codes, input prompts, API receipts and outputs
are under [the probe directory](analysis/synthesis-goods-casing-probe-20261007/).
HS choices in this experiment were **manually selected, registry-backed goods
briefs**, not a full route/load/equipment resampling campaign. No training
dataset, source contract, published pilot or production goods prompt was edited.

The baseline used the real `wording_request`, native PydanticAI output schema,
validation/unpacking and GLM-5.3-Flash/Fireworks request runner. The long source
regions were explicitly supplied as the example. An additional laboratory
control retained the builder's existing first-occurrence-only behavior. Its
example had 247 characters versus 1,162 with all product regions supplied.
This distinguishes lack of full source context from failures despite it.

### Observed results, including failures

**Current instructions, low reasoning (four calls):**

- Full laboratory context produced 772 characters/26 lines of product wording.
  It did not invent shipment totals in this run, but introduced borosilicate
  glass under the selected residual `701790` classification, whereas the
  supplied registry lists low-expansion glass under another HS6. The wording
  and technical qualifiers therefore still require semantic review.
- First-region-only laboratory context produced explicit HS captions, package
  counts and gross/net totals despite the instruction excluding them. This is
  not a successful output just because native JSON parses.
- Compressor context produced 679 characters/6 lines, naming <=120 cm hoods
  and biological safety cabinets. The supplied registry explicitly lists those
  under `841460` and `841470`, not selected `841480`. It also reduced the long
  enumerated inventory to five entries. The host did not supply a new per-unit
  inventory count in this baseline request.
- Telecom context produced 626 characters/29 lines, including connectors and
  power-related components beyond the requested dedicated-parts scope. It is
  a review failure/candidate, not an accepted synthetic training example.

**Same inputs, high reasoning (three calls):** the compressor and telecom
outputs still had scope problems. The laboratory call expanded until its
16,000-output-token budget was exhausted and yielded truncated invalid JSON;
PydanticAI rejected it. Higher reasoning did not resolve the underlying brief
and scope problems in this small experiment. The failed request remains in
the cost ledger; it was not retried or treated as a valid label.

**Focused goods-only prompt and narrower positive product briefs (three calls,
low reasoning):** these used new fictional product wording as the explicit
task, approximate source detail, and one output string per separated block.
No word-level spans, bounding boxes, rationale or evidence outputs were asked
for. The positive brief was manually narrowed for diagnosis, so improvement
cannot be attributed to prompt shortening alone.

- Telecom produced a new **520-character** description of dedicated housings,
  backplanes, faceplates and mounting parts. This is the clearest improvement.
  It returned one line; the existing source-style wrapper rendered it into the
  owned region without changing other input bytes or labels.
- Laboratory produced **1,851 characters** across all three requested blocks.
  The existing template compiler/renderer successfully inserted the three
  blocks and assembled one complete description target, leaving intervening
  container/customs/boilerplate text and all other labels unchanged. However,
  block lengths became **20/22/22 lines**, versus **11/43/1** in the source;
  the short continuation grew disproportionately. Some technical wording
  also needs review (for example, `WEEPING BLADE TYPE`). This proves mechanical
  fragment insertion, not semantic or geometric acceptance of that sample.
- Compressors reproduced the complete source, including all original serials,
  verbatim. The count stayed at 17 but it is **rejected as synthesis**: preserving
  a list's shape is not permission to copy its identities unchanged.

The source-region renderer was exercised directly on the focused laboratory
and telecom results: **three and one owned edits respectively**, no unrelated
label changes. See [render checks](analysis/synthesis-goods-casing-probe-20261007/focused-render-checks.json).
These outputs remain diagnostic, not published training samples.

### Cost and interpretation

Ten calls cost **$0.012551875 total** (about **1.26 US cents**), including the
truncated high-reasoning request. The four baseline low calls cost $0.001548;
the three high calls $0.0095221; the three focused low calls $0.001481775. These
are provider-reported ledger charges, not a scaled whole-dataset estimate.

The evidence supports a small, general interface rather than format-heavy
instructions: **one concrete, in-scope product brief + complete source product
text + one string per genuinely separate description block**. The existing
renderer can perform those block edits and target assembly already. The
current wording-request builder still needs an explicit distinction between
separate continuation blocks and repeated occurrences before this becomes a
general pipeline feature; its present identity-count shortcut is insufficient.

Important pre-scale requirements are semantic rather than punctuation rules:

1. Resolve residual/broad HS categories to a concrete permissible product family
   before writing a long list. The current commercial phrase can repeat the
   broad heading's excluded products, competing with the more precise registry
   context. A list of all heading alternatives is not a sampled product plan.
2. Supply genuine continuation regions together; request proportionate content
   per region, not an unrelated new assortment for every fragment. Exact line
   wrapping can remain the renderer's job.
3. Preserve known inventory constraints where explicitly established. Do not
   generally equate package count with the number of product names or serials.
4. Reject stale copied identities and unsupported shipment totals; semantically
   review product/HS compatibility before publishing. Native JSON alone does
   not verify those properties.

This pass completes the requested probe and casing implementation. It does
**not** promote the unsuccessful long-goods prompt experiments into production
or claim that a broader synthesis campaign has been validated.

## 16. Approved goods-generation and geometry implementation — 2026-10-07

The follow-up has now implemented the compact-brief/full-description approach,
joint generation of separated description fragments, natural line boundaries,
and measured-space coordinate placement. The user's clarified objective supersedes
the strict tariff-classification criticisms in section 15: plausible related product
assortments are permitted; printed HS values, accounting and DG/thermal facts still
must be consistent.

**24 full shipments / eight sources** were generated, reviewed, corrected, published
and coordinate-enriched. **Nine additional long-goods probes / three sources** exercise
the same production builder and renderer without claiming full-shipment ownership for
those additional families. No current real training labels or inputs were modified.

See the [complete implementation and validation report](kie-synthesis-goods-layout-pilot-2026-10-07.md)
and [source/variation gallery](analysis/synthesis-goods-layout-pilot-20261007/SOURCE_AND_VARIATIONS.md).
This report includes problems found, scoped corrections, the 226-test result, cost
ledger ($0.06443112 including all experimental attempts), coordinate coverage,
mutation checks and measured runtime/memory.

The previous single-anchor expansion defect is resolved. Locally expanded lines
receive distinct measured-space positions when possible; otherwise coordinates
are explicitly unknown and text remains complete. All 15 long-goods regions were
positionable in the final probes. Across all 33 outputs, 3,980/4,889 content lines
have coordinates. Original source gaps account for 769 blanks; 140 further lines
are explicitly unpositioned because of insufficient/overlapping space. This is
layout conditioning, not measured synthetic glyph geometry or proof of an F1 gain.
