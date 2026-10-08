# Current-schema synthesis: 100-source / 200-record expansion

## Scope and release contract

The requested deliverable is 100 distinct usable training-source templates, with
two newly sampled complete documents per template. Each document must include
current-schema targets, rendered text, source-anchored synthesized coordinates,
and source/variation provenance. This is a separate synthetic dataset, not an
edit to the 600 real training or 60 validation records.

The campaign configuration is
`configs/synthesis/mpci_bl_curated_v7_expansion100.yaml`; its working artifacts
are in `artifacts/kie-synthesis-production/curated-v7-expansion100-v1`.
The initial inventory and subsequent explicit eligibility replacements are
preserved there. A draft contract, a successful model call, or a rendered sample
is not itself a published release. Publication requires the complete configured
scope, exact current-authority replay, resolved semantic review, and the separate
coordinate audit.

## Initial selection and admission findings

The starting selection contained the 26 previously exercised templates, all 48
remaining sources from the earlier low-complexity shortlist, and 26 additional
train-only sources. Screening uses current OCR/labels and the historical template
ownership, not old catalog acceptance as a substitute for current validation.

The inventory exposed several concrete differences between historical templates
and current labels:

- Addresses and represented-party names can span remote continuations; historical
  city/address splits cannot simply be copied to the new complete-address target.
- Product phrases inside company names are not goods-description occurrences.
- Historical allocation row indices can differ from current container ordering.
- Original-B/L counts and container counts expressed in words are not cargo
  package totals.
- Mixed package rows, missing container rows and paired metric/imperial printouts
  require their own accounting contracts rather than flattening or equal-number
  substitution.
- Discharge and onward delivery can lie in different countries; party geography
  must follow its actual route node rather than always the discharge endpoint.
- Source-only customs references, party aliases, repeated numeric rows and
  country-specific clauses must change coherently with the sampled shipment.

Thirteen initially selected candidates were explicitly replaced during eligibility
screening. Reasons and replacements are recorded in
`artifacts/synthesis-templates/mpci-bl-v7-reviewed/selection-history.yaml` and the
campaign's `audit/selection-revision.json`. Their original evidence is retained;
none of their real labels or OCR was altered. The active scope remains 100.

The final two replacements are `8c6da3dd → d69c24cb` and
`bee16be0 → 04f208b1`. The first requires a separate source-label equipment
reconciliation; the second has conflicting cargo/row weights without a printed
tare or VGM explanation, confirmed in the PDF. Four already-generated candidates
from those excluded sources are preserved in `excluded-candidates/`, outside the
published candidate set. A plausible interpretation is not used to certify them.

## Validation layers

1. **Source identity and ownership:** current OCR and target hashes; exact source
   quotes; non-overlapping owned spans; baseline label replay; explicit ownership
   for every mutable public or dependent private fact.
2. **Sampling:** train-only registry/support fitting; source-compatible cargo
   family, packaging, equipment, thermal/DG settings and route topology; coherent
   totals and placements.
3. **Rendering:** every changed label has an owned rendered surface; repeated
   facts agree; exact edit replay; no unrelated source changes; uppercase target
   policy and independently configured rendered casing.
4. **Semantic review:** complete rendered text and target, with the sampled
   shipment context. Findings are corrected and re-reviewed, or explicitly
   adjudicated against quoted current text. No blanket approval of held results.
5. **Positions:** source-anchor provenance, text preservation, coordinate bounds,
   valid page assignment, spacing/order and collision checks; explicit unknown
   coordinates where fitting would be unsupported.
6. **Release:** exactly 100 source IDs and 200 distinct records; every review and
   proof bound to current hashes; no real/validation mutation; reproducible
   publication and coordinate enrichment.

## Admission probe that was rejected

A small GLM source-fragment-selection probe made six calls costing $0.0026302.
It returned extra punctuation/count context and invalid repeated occurrence
selectors. All six proposals were held; none was used to authorize a template.
The temporary agent-admission script was removed. Exact historical ownership,
current source inspection and deterministic baseline replay are used for the
admission declarations instead. The original paid-call receipts remain in the
campaign cost ledger.

## Status

The expansion is complete: **100 sources, 200 records, two variants each**,
with published plain and position-enriched inputs, current-schema labels,
zero unresolved content findings and a passing independent geometry audit.

Publication files:

- `artifacts/kie-synthesis-production/curated-v7-expansion100-v1/dataset.jsonl`
- `artifacts/kie-synthesis-production/curated-v7-expansion100-v1/samples.md`
- `artifacts/kie-synthesis-production/curated-v7-expansion100-v1/manifest.json`
- `artifacts/kie-synthesis-production/curated-v7-expansion100-v1/positions-v2/dataset.jsonl`
- `artifacts/kie-synthesis-production/curated-v7-expansion100-v1/positions-v2/samples.md`
- `artifacts/kie-synthesis-production/curated-v7-expansion100-v1/positions-v2/manifest.json`

The plain dataset SHA-256 is
`5268bc8367e06a04e94c2fcae6320f8801a9bb6f0456b10884f09caf0b276dae`.
The position-enriched dataset SHA-256 is
`c35131d73070903cc38bc7c2f9865e772a0eb622ccb4fcf56780353b568d5fae`.
The plain publication manifest SHA-256 is
`4a9b3c3b608a78634aa7d15d8691eab3a1cda8ecec8f3c305cf4b62fd7f8e146`
after the path-only template relocation documented below. The previous manifest
(`9982fb3dd0f7ce3b7fe8bc9b664eb1a6363bc1d1f3f952460b6b4754eed3f826`)
is preserved in `audit/template-relocation/`; dataset bytes are unchanged.

## Repairs established by the expanded admission pass

The earlier 26-source pilot did not exercise every source representation found
in the additional 74 sources. The following changes address observed inputs,
not speculative format support:

- **Numeric source preparation:** current container identity, not historical row
  position, determines an allocation owner. Cargo package totals are distinguished
  from original-B/L counts and container receipts, including numbers in words.
  Unit-only regions do not become numeric measurements. Grouped numeric notation
  and explicit unit conversions require a baseline proof.
- **Private measurements:** source-only gross/net/volume and tare values use
  explicit equipment owners and declared sums. Ambiguous punctuation requires a
  reviewed, source-hash-bound interpretation; equipment capacity is not used to
  guess the original number.
- **Route topology:** origins, ports, onward destinations and independent party
  localities retain their distinct roles. An explicitly represented company's
  name can appear inside a principal's name without forcing both postal addresses
  to the same country.
- **Route-owned handling:** three sources contain demurrage or bonded-warehouse
  instructions naming a port/locality. Their labels and printed clauses now
  depend on the same sampled route values. The contract pins the original clause
  and its owned source region; it cannot modify arbitrary descriptive text.
- **Postal/contact ownership:** reviewed contracts consolidate split emails and
  phones in their owning party blocks, consume duplicated address fragments,
  and remove obsolete continuation markers only from synthetic rendering.
  Contact-only notify parties do not require inventing a company name.
- **Contact validation:** the native output schema and host validation reject
  invalid mailbox dot patterns. Previously accepted contact outputs are migrated
  only after exact request comparison and validation; original call receipts are
  retained. One rejected double-dot email is separately adjudicated with a
  correction receipt.
- **Auxiliary consistency:** source-only references, repeated company aliases,
  country-specific captions and independent correspondence offices are explicitly
  inventoried. UN/LOCODE text comes from a verified UN/LOCODE registry entry,
  not from a generic locality identifier. Public target fields are not invented
  merely because source-only text exists.
- **Generation boundary:** deterministic source, numeric, route and auxiliary
  preflight runs before paid wording. Broken contracts cannot consume a lexical
  generation call. Product wording excludes host-controlled accounting,
  freight instructions and annotation-policy language.
- **Equipment wording:** a source `40HC` surface did not state the full sampled
  equipment category after a change from general-purpose to refrigerated cargo.
  Curated rendering now resolves its proposed wording through the equipment
  registry and prints the complete sampled size/type pair when the inherited
  alias is insufficient. For example, refrigerated high cube is printed as
  `40' HIGH CUBE REFRIGERATED`; an unchanged general-purpose `40HC` remains valid.
  The registry check is tested across general-purpose, refrigerated and open-top
  categories, including whole-container receipt wording.

These are synthesis contracts. None of these changes rewrites real source OCR
or labels. In particular, removal of an obsolete synthetic continuation is not
evidence that its original real-document text was an OCR error.

## Expanded rendered-content audit: concrete acceptance checks

Baseline replay is necessary but cannot expose every dependency: the old text
and old value may agree while a newly sampled value reveals an unowned repeat.
The expansion therefore includes complete rendered-text reading in addition to
schema checks and exact-edit replay. Findings are grouped by causal ownership,
not repaired through arbitrary whole-document substitutions.

- `276b6f10`: newly sampled private per-container volumes exposed an unchanged
  source shipment total of `132.000`. A private total needs the same accounting
  owner as its component rows even when volume is omitted from the public target.
- `c1b20a71`: a source `NCM` commodity code survived beside a newly sampled HS
  code. This is a stale identifier, not an objection to a neighboring product
  description. All printed aliases of that commodity identifier must change.
- `c5b964bc` / `ca74e1e4`: source-only exporter declarations must follow the
  explicitly named exporter party. Its country need not be the loading port's
  country; route-origin macros are not a substitute for party ownership.
- `926e04e2`: the historical template had bound footer `INTE267634V` to the
  B/L number, although the current target uses the main document-number field
  `SSPHNYC9047110`. The footer is an independent source-only reference; it must
  not become an inconsistent second rendering of the B/L number.
- `a7ede7d8`: the voyage identifier differs by a trailing character on one
  repeated page. Synthetic repeats use the complete sampled voyage value,
  rather than propagate conflicting source surfaces.
- Some newly generated product wording introduced shipment totals or a country
  of origin inside the description. Those facts are controlled by the host;
  lexical correction preserves the commercial product wording while removing
  unauthorized accounting/origin declarations. Validation now rejects these
  explicit captions before publication.
- Three descriptions passed structural validation but implied physically
  unsuitable whole products for their sampled loads. The corrected text uses
  appropriately sized parts/components; the sampled counts, weights and
  equipment remain unchanged. Exact HS-to-product classification is explicitly
  outside this experiment's acceptance criteria, as requested by the user.

Every manual lexical correction retains its prior value and reason in an audit
receipt. Corrected candidates are rendered again; final review and publication
bind to their new hashes. A stale approval cannot approve changed text.

The closing lexical audit additionally removed unsupported spool/cone fill
weights, generated product-unit counts, and geographic origin declarations with
no goods-origin target. Product specifications such as fish size grades, fabric
grams per square metre and machine capacities are retained. A redundant,
arithmetically correct `25 KG PER BAG` generated phrase was also removed to keep
packing statements under host control. Explicitly source-owned goods-origin
wording remains where the same value has its proper target; the validator does
not blanket-ban every occurrence of the word `ORIGIN`.

## Final selection and variability

The selected 100 families comprise 88 ambient, four vehicle, four chilled, two
frozen, one dangerous-chemical and one dangerous-vehicle source. Transshipment
is an independent route trait, not an additional mutually exclusive cargo class.

For the planned 200 variants, the final sampling inventory records:

- 116 origin countries and 120 destination countries;
- 188 distinct HS6 identities and 15 package categories;
- 396 party localities and 122 training-only physical-support donors;
- both endpoint countries change between variants in all 100 families;
- commodity codes change in 91 families and quantities in 92.

This is not a requirement to randomize every field in every template: printed
footprint, cargo safety and source-owned dependencies constrain some choices.
Exact HS-to-product tariff classification and fictional postal deliverability
are not acceptance requirements; printed HS values must still match labels, and
party geography and physically plausible cargo wording must remain coherent.

## Stress testing and runtime boundaries

All 200 planned physical variants pass. A separate 10,000-scenario sampling probe
passed every draw. Running full physical preparation on 10,000 future variants
accepted 9,985 and rejected 15 for printed precision, positive row allocation,
net-versus-gross or post-rounding capacity constraints. There were no remaining
source-baseline contradictions in the active selection. Those 15 are explicitly
inadmissible future draws, not accepted malformed training records, and are
rejected before any paid wording request. Large campaigns must account for such
holds rather than assuming every seed will yield an output.

The numeric-only before/after comparison retained identical plans on 78 prior
samples: median processing time 0.040734 → 0.043154 seconds per 78, about 31
microseconds additional validation per document. Traced peak allocation was
83,301 → 120,927 bytes. This is negligible compared with model generation;
generation/review latency is not represented by that microbenchmark. A complete
cached 200-record render took approximately 18.5 seconds on this workspace.

A separate equipment-formatting microbenchmark over 426 calls measured median
0.247 → 0.317 milliseconds, with the same 1,430-byte traced peak. This measures
the added category-resolution check, not generation throughput. The sampler
stress reports retain their exact code/configuration snapshots; the final
200-record publication replay, rather than an older snapshot hash, establishes
the final candidates' agreement with the current contracts.

## Final positional validation

The independent checker passed all **200 documents, 370 pages and 20,269 content
lines**. It checks exact plain-text and target preservation, byte-edit ownership,
source/Paddle provenance, page assignment, normalized bounds, expansion spacing,
coherent page transforms, relative order and nearest-neighbour relationships.
It found **zero remaining coordinate collision groups or validation failures**.
All **12 deliberately corrupted negative controls** were rejected. The checker
does not call the production geometry validator to decide whether geometry passes.

Coordinate coverage is explicit:

| Outcome | Content lines |
| --- | ---: |
| Known coordinates | 15,594 (76.93%) |
| Missing measured source anchors, inherited unchanged or through an edit | 3,230 |
| Expansion cannot fit the measured available space | 1,125 |
| Unowned source text intersects the proposed expansion | 300 |
| Separately reflowed owners compete for one measured source line | 20 |
| Total | 20,269 |

All 4,675 unknown-coordinate lines retain their full text and the explicit ` ||`
suffix. These are not guessed coordinates or omitted words. Of the 3,230 missing
source-anchor lines, 1,330 are unchanged source text and 1,900 are edited text.
The two pages without any known coordinates are the two variants of
`719e3e96`, page 3: that page contains only `fair trade` and
`Schriftkauf Handel + Logistik GmbH`, neither aligned in the measured source.

The final audit discovered and repaired two general transfer defects:

1. An exact owned text region may include an empty paragraph separator. Blank
   lines now supply no coordinate anchor; actual page crossings remain rejected.
2. A caption or package count sharing a source line with a reflowed value must
   move with that line. Averaging its stale point with the moved text compressed
   spacing and created collisions. Same-source-line peers now follow the reflow;
   independent source-line fragments remain independent. Competing reflow owners
   produce explicit unknown coordinates, not overlapping invented positions.

The caption correction changes 22 anchor points across 18 records, with only
one additional known-to-unknown line. Real source inputs, all plain synthetic
text and all targets are unchanged. The prior positional candidate and the
failed audit are retained under `audit/positions-before-geometry-fix/` and
`audit/geometry-independent-before-caption-fix.json`.

Position enrichment took **20.09 seconds** for 200 records; the final metadata
replay took **19.58 seconds**. The final independent validation took about
**2.51 seconds**, with **369 MiB peak RSS**, and no paid requests.
The source-transfer microbenchmark measured **1.341 → 1.390 milliseconds** on
an unaffected path (about 0.01 seconds additional work across 200 records).
The additional provenance and line-composition checks have negligible campaign
cost. Coordinates remain approximate source-derived layout signals, not measured
glyph boxes for a newly typeset synthetic PDF.

Evidence: `audit/geometry-independent.json`,
`audit/geometry-independent.md`, `audit/geometry-independent-gallery.png`, and
`audit/geometry-blank-separator-fix.md`. Removing the appended position suffixes
independently recovers every plain input exactly; the position-enriched JSONL
also retains every original plain-record field and target without alteration.

## Tests and release checks

- **370 targeted curated-synthesis/equipment tests passed** in 16.38 seconds.
- The changed legacy date-format regression also passed separately.
- **63 targeted position/spatial tests passed**, including corruption probes,
  same-line captions, numeric/package peers, competing reflows and page boundaries.
- Ruff and `git diff --check` passed.
- After correcting a stale introductory sentence in the plain-text gallery,
  all 17 publication tests passed again. Re-publication retained the identical
  dataset and review/source entries; only the gallery hash changed. Earlier
  metadata is retained under `audit/gallery-wording-before/`.
- Final plain publication replay: **200 valid / 200 expected, 100 sources,
  zero failures**; exact content-review coverage and hashes verified.

This is not a claim that the full repository test suite is green. Four older
raw-text-pipeline test failures reproduce with the original descendant module
as well as the edited module; their differential, fixture limitations and exact
commands are recorded in `audit/numeric-legacy-test-baseline.md`. The modified
date-rendering path and the complete curated publication path pass their tests.

Detailed numeric evidence and PDF adjudications are in
`audit/NUMERIC-DEPENDENCY-CLOSURE.md`; diversity and future-draw outcomes are in
`audit/sampling-final.json` and `audit/physical-stress10000.json`.

## Content closure and cost

All 200 complete rendered documents and targets received content review. The
three disjoint closing inventories cover 126 + 34 + 40 records. The combined
`audit/content-review-coverage.json` verifies exact 200-record coverage without
overlap, current candidate file hashes, OCR hashes and target hashes against the
published manifest. Changed text was re-read; a previous pass was not reused
without matching its recorded content.

The final model review covered all 100 two-variant batches without request
errors. Eight remaining findings were explicitly rejected with exact quotations
and individual explanations, stored under `adjudications/`:

- Two arithmetic/row-ownership false alarms: one reviewer counted extra rows;
  another assumed the one printed container row applied equally to both units.
- Two outer-pallet objections: the approved labels retain the inner box level,
  while outer packing remains legitimate source-owned rendering metadata.
- Three exact HS-classification objections, outside the explicitly agreed
  neighboring-product policy. The actual HS strings still match the labels.
- One postcode-deliverability objection, outside the fictional-address policy.

Actual text/label and ownership defects were corrected and re-rendered before
the final review. Rejected findings are not a blanket override or permission to
publish unknown errors. Final publication replay validates every candidate
against its source contract, sampled shipment, wording and contact receipts.

Two sources have explicit synthesis-only completion of date labels from audited
historical date bindings. For the final replacement `d69c24cb`, the existing
normalized historical values and printed date format establish the recipe;
the sampled issue/on-board dates are added alongside the exact printed values.
Its real labels are not changed. The completion probe proves unchanged OCR,
only the two intended target additions, schema validity and exact replay.

**Total recorded API spend: USD 0.29044148** for this campaign, including
generation, contacts, postal corrections, rejected attempts/probes and repeat
reviews. That is about USD 0.00145 per published sample. It does **not** price
engineering time or manual content adjudication and is not a claim that a new
unreviewed 10,000-document campaign is already certified.

The published set has 370 pages: 72 one-page, 90 two-page, 34 three-page and four
four-page documents. Six samples have a labeled transshipment port. Four have
no labeled container (retained non-container cargo profiles); the rest range
from one to thirteen containers.

The original real-data files remain unchanged:

- train: `ca15c382bd1a36e72db978a0acb34f9dec64e8ea6c7e00639e62bccc98058305`
- validation: `b8c0d4bddd4b3a452f901f3fc5e08e54d580a82768eddd5df33c97ab2c85da5a`

No training job was started and no synthetic record was merged into the real
training or validation split.

An independent release check confirms that all 100 source IDs belong to the
600-record real training split, none belongs to the 60-record validation split,
and none of the 200 synthetic input hashes matches a validation input. This is
an exact identity/content check, not a claim that carrier layouts are unique
across the real splits.

## Follow-up: rendered-content audit, coordinate coverage and reusable catalog

Subsequent coordinate experiments are documented in the
[source-conditioned reflow investigation](kie-synthesis-position-reflow-investigation-2026-10-07.md).
They substantially improve goods/address coverage in a separate experimental
output; the published dataset and the coverage figures below remain unchanged.

This follow-up rechecked the published outputs, attributed every positioned and
unpositioned line, replayed an earlier positional pilot with the current placement
rules, and relocated the admitted templates. No model calls were made; additional
API cost is **USD 0**. No generated text or extraction label was changed.

### What has actually been reviewed

The previous full-content reviews cover **all 200 rendered documents and targets**,
not just schema validity or snippets. Their three disjoint inventories contain
126, 34 and 40 records. This follow-up verified the report-file hashes and every
candidate-file, OCR and target hash against the current artifacts, with exact
coverage and no overlaps. Consequently those reviews apply to these exact
published samples. This is a verification of prior full readings, supplemented
by targeted inspection here, not a claim of another fresh manual reading of all
200 documents in this follow-up.

The content checks cover sampled geography/party ownership, product wording,
repeated identifiers, equipment, counts, mass, volume, thermal/DG information,
and goods/container placements. Exact rendering replay separately checks the
permitted edits and agreement between rendered facts and targets. Acceptance is
under the agreed synthetic-training policies: fictional addresses are not
certified deliverable, and neighboring product wording is not rejected solely
for imperfect tariff classification.

**A material limitation remains in positional usefulness, even when text and
labels agree:** long generated descriptions often have no usable coordinates.
Some also expand very short source descriptions substantially. This should not
be hidden behind the aggregate coverage or the passing geometry checks.

### Why 76.9% is lower than the earlier approximately 85%

| Comparison | Known / content lines | Coverage |
| --- | ---: | ---: |
| Earlier contacts pilot, 24 families / 72 samples, original placement policy | 6,043 / 7,168 | 84.31% |
| **Same earlier texts**, replayed with current placement policy | 5,855 / 7,168 | 81.68% |
| Subsequent registry pilot, 24 families / 72 samples | 5,767 / 7,385 | 78.09% |
| Current variants from those same 24 families, 48 samples | 3,834 / 4,957 | 77.35% |
| Current variants from the other 76 families, 152 samples | 11,760 / 15,312 | 76.80% |
| Current complete release, 100 families / 200 samples | 15,594 / 20,269 | 76.94% |

The controlled replay isolates **188 lines, or 2.62 percentage points**, lost
under stricter current placement rules on unchanged old text. Earlier geometry
acceptance is not interchangeable with today's measured-space, ownership and
collision restrictions. The registry pilot had already fallen to 78.09% before
this expansion. Adding the other 76 families lowers the current overall figure
by only about 0.41 percentage points relative to the current 24-family subgroup.
Different generated wording, seeds and numbers of variants mean the remaining
historical differences are not a controlled estimate of one causal effect.
The coherent page scale/translation augmentation does not remove coordinates:
unknown positions are determined before that transformation.

### Are there more lines in the sources?

No, in aggregate the synthetic documents are longer:

- The 100 unique real sources have **9,531** content lines, **8,113** positioned
  (**85.12%**).
- Repeating each source twice gives a matched baseline of **19,062** lines,
  **16,226** positioned and **2,836** unpositioned.
- The 200 synthetic documents contain **20,269** lines: **1,207 more (+6.33%)**.
- **175** descendants are longer than their own source, **17** shorter and
  **8** the same length.

Across renderer-owned spans, goods wording grows from **344 source lines to
1,288 generated lines**; address wording from **1,620 to 1,927**. These are
owned-span counts, not a claim that every source physical line has exactly one
owner. Inline composition produces 1,925 final address-containing lines.
Other edits, contractions and compositions account for the remainder of the
net physical-line change.

### Which fields lose coordinates?

Categories are taken from exact renderer `targetPaths` and edit provenance,
not a keyword classifier. Mixed lines contribute to their actual field family;
goods and address rows below do not overlap. Unchanged source text is left as
its own category rather than guessed to be a particular semantic field.

| Final line group | Lines | Known | Unknown | Known coverage |
| --- | ---: | ---: | ---: | ---: |
| Contains goods description | 1,288 | 98 | 1,190 | **7.61%** |
| Contains party address | 1,925 | 827 | 1,098 | **42.96%** |
| Unchanged source text | 10,723 | 9,393 | 1,330 | 87.60% |
| Other edited fields | 6,333 | 5,276 | 1,057 | 83.31% |
| **All** | **20,269** | **15,594** | **4,675** | **76.94%** |

**160 of 200 documents have no positioned goods-description line.** Forty have
at least one; 35 have all their goods-description lines positioned. Addresses
have at least one position in 173 documents and none in 27; 25 have positions
on every address-containing line.

| Reason for unknown position | All lines | Goods-description lines | Address-containing lines |
| --- | ---: | ---: | ---: |
| Missing measured source anchor | 3,230 | 209 | 679 |
| Insufficient measured local space | 1,125 | 822 | 285 |
| Unowned source text intersects proposed expansion | 300 | 159 | 121 |
| Competing independently reflowed owners | 20 | 0 | 13 |
| **Total** | **4,675** | **1,190** | **1,098** |

Missing source anchors include 1,330 unchanged lines and 1,900 edited lines.
A missing anchor within a changed block can prevent positioning the whole
replacement, including lines whose individual original anchor was known.
Unknowns therefore are not simply the 1,207 extra generated lines.

Goods and addresses account for **48.94% of all unknown lines**, but **96.89%
of the 1,445 geometry/space-driven unknown lines**. The remaining inherited gaps
include original captions, boilerplate, footers and other edited facts. The
two wholly unanchored pages are the already-documented `719e3e96` footer pages.

### Concrete cases and family concentration

One pronounced example is source `049ad12d`: its source description is simply
`TYRE`, while variant `syn_full_v7_c60cc00208e25208208d74ce` contains **14 lines**
of motor-vehicle radiator descriptions/specifications/accessories. All 14
receive explicit unknown positions because they cannot fit the source's local
space. The words are retained and the label matches, but this is not an example
of successful goods-region positional conditioning. See the source/variation
pair in `samples.md` (source around line 3111; generated goods around line 3318).

Every family has some unknown positions; the ten largest contributors account
for **1,148 / 4,675 (24.56%)**, so this is not one or two bad templates:

| Source prefix | Unknown / all synthetic lines | Known coverage | Main explanation |
| --- | ---: | ---: | --- |
| `b5c11ec9` | 178 / 298 | 40.27% | Missing source anchors throughout affected edits |
| `3dc8551d` | 159 / 690 | 76.96% | Missing source anchors |
| `86ed2b9e` | 155 / 343 | 54.81% | 132 missing-anchor, 23 local-space lines |
| `a53a1a9e` | 111 / 271 | 59.04% | See exact per-line reason inventory |
| `35136b22` | 108 / 340 | 68.24% | See exact per-line reason inventory |

The findings support separate next steps, not weakening geometry validation:

1. Keep semantic complexity appropriate to the source description when generating
   wording; a one-word source need not become a large specification list. This
   is a source-complexity instruction, not an arbitrary character or line cap.
2. For legitimately long descriptions, investigate reflow of a larger **owned**
   cargo region with its dependent rows. Merely interpolating more tightly or
   moving unrelated fields would not resolve the measured-space problem.
3. Recover genuinely missing source alignments separately, without borrowing
   another field's anchors or treating an inferred location as measured.
4. Report coordinate coverage by field as well as globally, and retain explicit
   unknowns when a supported placement is unavailable.

These are recommendations for a subsequent positional-quality pass, not changes
silently applied to the current reviewed text or coordinates. The geometry audit
establishes plausibility/provenance of emitted coordinates; it does not establish
that most goods descriptions currently receive useful positional signal or that
training performance will improve.

Reproducible evidence, under the campaign's `audit/` directory:

- `coverage_followup.py`: all-record attribution and historical controlled replay.
- `coverage-followup.json`: source/output totals, historical comparisons, every
  family, field grouping, growth and concrete expansion examples.
- `coverage-lines.csv`: all 20,269 final content lines, their owners, positions
  and exact missing-position reasons.

### Reusable template catalog and relocation validation

The admitted template assets now live in
**`artifacts/synthesis-templates/mpci-bl-v7-reviewed/`**. The parent directory is
reserved for reusable template catalogs, separate from generation experiments.
Each of the 100 `cases/doc_<id>/` directories contains:

- `contract.json`: current source/target rebinding contract;
- `template.json`: compiled source layout and historical binding hints;
- `source.txt`, `source-positioned.txt`, `target.json`: exact current source views;
- `alignment.json`: measured source-line/region correspondence.

The catalog also contains `ownership.yaml`, `auxiliary.yaml`,
`selection-history.yaml`, a usage README and `manifest.json`. The manifest hashes
all **600 case files**, inventories capabilities and records shared dependencies.
Both contract and compiled-template files were moved byte-for-byte. The shared
declarations retain exactly the 100 selected entries, unchanged; their original
unfiltered versions and all 13 excluded candidates remain in the campaign's
audit/staging locations.

The existing campaign config now reads its four template/declaration paths from
this catalog. Preparation's admission/replacement functions honor the configured
paths rather than writing admitted assets back into the campaign directory.
Capabilities remain explicit in the campaign and indexed in the catalog manifest.
Registries, real-source data/support and Paddle geometry remain shared external
dependencies with recorded paths/hashes, not redundant per-template copies.
Generated samples, paid wording caches and review receipts remain campaign-local.
This is a reusable catalog, not a claim that it runs without the application and
its documented shared dependencies.

Every one of the **200 complete candidate objects** was recomputed before and
after relocation and compared exactly: text, labels, sampled facts, edit proofs
and cached-generation bindings are unchanged. Replay took **23.87 → 23.64 seconds**;
peak RSS was **432.8 → 434.4 MiB**. This single pair shows no material regression,
not a statistically established speedup. Plain/positioned datasets and both real
splits retain their original hashes. Manifest metadata is refreshed solely for
the new configuration paths, with prior manifests retained under
`audit/template-relocation/`.

The `.gitignore` exceptions preserve versioning of the relocated declarations,
catalog inventory and READMEs. Source OCR/labels, geometry, generated outputs and
other runtime artifacts remain ignored. **424 targeted curated-synthesis and
equipment tests passed in 17.15 seconds** after the path changes.

The final independent geometry recheck again passed all 200 documents / 370
pages / 20,269 lines, with zero collision groups or failures and all 12 negative
controls rejected (2.51 seconds, 347.7 MiB peak RSS). The normal publisher's
overwrite guard was retained: old manifests were preserved before publishing
the new path metadata. Exact comparison confirms only `configSha256` changed
in the plain manifest and only its corresponding `sourceManifestSha256`
reference changed in the position manifest. Final checks are recorded in
`audit/template-relocation/final-validation.json`.

## Next expansion: investigation for 200 templates (2026-10-08)

This section is a plan and measured offline admission probe, not a new published
campaign. The current 100 admitted templates and 500 synthetic records remain
unchanged. No paid generation or training was launched during this investigation.
The next objective is 100 additional templates, two audited descendants each,
then another 800 records: 1,500 synthetic plus 600 real training records, with
the existing 60 validation records unchanged. The 200 pilot records count toward
the requested additional 1,000; they are not an extra uncounted campaign.

### Permitted source pool and selection

The user explicitly permits sources outside current training, provided they are
not validation sources. Current reviewed labels are an efficiency advantage,
not an eligibility requirement. Keep the existing filtered B/L/sea-waybill,
at-most-five-page pool and previous explicit curation exclusions. Before final
admission, compare document identity, source hashes and duplicate-shipment
evidence against validation; the historical layout-proxy screen is an additional
conservative screen, not proof of independent shipments.

The read-only inventories are saved beside the probes in
`docs/analysis/synthesis-expansion200-plan-20261008/`:

| Pool/screen | Sources | Meaning |
| --- | ---: | --- |
| Unused current reviewed training sources | 500 | Current labels and line-position alignments exist |
| Newer historical template, exact OCR, no validation-proxy overlap or known prior rejection | 155 | Preferred rebase candidates |
| Same, with the current sampler's positive typed single-package shape | 148 | Mechanical source-shape eligibility |
| Older historical templates passing equivalent source and shape screens | 61 | Additional candidates, not excluded just for catalog age |
| Combined mechanically eligible reviewed-source shapes | 209 | Not yet admitted templates |
| Filtered sources outside the current 660 | 1,258 | Historical labels, saved raw OCR and Paddle outputs exist |
| External sources with newer historical templates and no validation-proxy overlap | 559 | Useful expanded search pool; current-label review still needed |

Of the 559 external candidates, 548 historical template source texts match saved
OCR exactly; 519 have one historical cargo group. Historical labels suggest 62
negotiable, 45 thermal, 11 DG and eight transshipment sources across 134 carriers.
These are prioritization hints: all labels use the old experimental schema and
must be reviewed against current policies before admission. Paddle detections
already exist, but these sources need raw-OCR-to-Paddle line alignment. The 145
previously reviewed real sources outside the current 660 were all filtered out;
they are not a clean ready-to-use reserve.

Among the 209 reviewed-source shapes, 29 distinct sources cover 16 negotiable,
five thermal, three DG and six transshipment cases (categories overlap).
Start admission review with this priority group; 22 already passed the two-draw
scenario probe, while seven need capability/dependency decisions. Fill the
remaining places using simpler contracts, new carriers/layouts, varied container
counts, notify references and long/split goods text. Use a reproducible seeded
tie-break among equivalent candidates, and record replacements explicitly.
External sources are useful where they add diversity absent from that group;
they must not bypass source-label review simply to meet the count of 100.

The previous one-off preparation script requires HS and gross weight for its
extra shortlist. Reusing that restriction would leave only 41 extras. This is
not a runtime requirement: the scenario sampler can sample a private registry
goods identity while preserving absence of unprinted public HS/mass fields.

### Actual rebase and sampling probe

`probe_rebase.py` exercised the production draft, compiler, ownership builder and
registry sampler on 223 candidates: 155 newer and 68 older historical templates,
before excluding unsupported package shapes. It used no new ownership or
auxiliary declarations and no language-model calls.

- 218 produced drafts and passed exact source-baseline compilation.
- 157 passed the ownership builder without added declarations.
- 66 also had no unresolved draft binding diagnostics.
- 196 sampled two new shipments successfully, including 93 sources without a
  public HS value and 41 without a public gross weight. Public field presence
  remained unchanged and private registry goods identities were populated.
- The intersection of clean baseline/ownership/no-missing-bindings and two
  successful scenario draws is **56 sources**.
- Runtime: **9.23 seconds**, peak process RSS **433.44 MiB**, API cost **$0**.

Draft diagnostics include split lexical surfaces, historical numeric values
different from current labels, and measures needing explicit row-sum ownership.
Sixty-one sources encountered overlapping owned regions; these require exact
source-span reconciliation, not permissive rendering. Three require an explicit
multi-package equation and two have unproven historical allocation identity.

Twenty-seven sources failed the minimal scenario probe: 14 lacked one positive
typed package row; six need explicit transshipment topology; two need the
goods-origin identifier representation; three need freight-payment endpoint
ownership; one lacks a compatible vehicle donor; and one has incomplete exact
allocation totals. The topology/origin/freight decisions have existing declaration
interfaces. This does not establish that every source is solvable without code
changes. The unsupported package/donor/allocation cases should be reviewed or
replaced explicitly, not silently coerced into supported shapes.

These probes did not render new wording or certify auxiliary closure, complete
source semantics or synthesized geometry. Their purpose is to measure reusable
work and identify the concrete admission decisions before paying for generation.

### Implementation and admission plan

1. Freeze source and target identities, validation exclusions and selection.
   For selected external sources, produce reviewed current-policy source labels
   and line alignments first. Do not add them to the real 600 training split.
2. Reuse historical exact source ownership through the existing current-schema
   rebaser. Apply reviewed ownership, numeric, auxiliary and capability declarations
   once per source. Check split addresses/descriptions, repeated occurrences,
   allocation identity, customs/locality text and container aliases explicitly.
3. Require baseline replay and complete mutable-fact coverage, then run seeded
   scenario/preflight checks for all declared modes before any paid wording.
   Include origin/destination changes, quantities, packaging, equipment, DG/thermal,
   transshipment and negotiable/reference-party inheritance as applicable.
4. Generate two complete descendants per newly admitted candidate using the
   existing registry sampling, lexical/contact generation, casing and joint
   coordinate-reflow flow. Keep all attempts and cost receipts campaign-local.
5. Review all 200 rendered texts and targets, grouped two-per-source against the
   source. Validate supporting text and omitted facts as well as schema; reconcile
   findings and repeat checks only on affected results. No stale review hash may
   authorize a changed result. Produce the source/variation inspection Markdown.
6. Independently audit coordinates: text preservation, page assignment, bounds,
   sequence/spacing, collisions, role ownership and source-anchor provenance.
   Exercise existing negative controls and plot difficult/multi-page examples.
   Compare coordinate coverage by generated goods/address lines, not only overall
   coverage. Explicit unknown coordinates are preferable to invented geometry;
   any new concentrated gaps require inspection before admission.
7. Publish exactly 100 new usable templates into the common catalog, retaining
   original 100 assets. Revalidate replay of the existing 500 if shared code or
   declarations change. Then synthesize the remaining 800 and apply the same
   campaign checks before assembling the expanded training dataset.

The preparation script is a campaign-specific tool with paths and count for the
old 100-source job. Do not rerun it unchanged. A parameterized admission entrypoint
should reuse its compiler/declaration machinery and accept selected source IDs,
source authority, destination and campaign paths explicitly; there is no reason
to duplicate the synthesis engine or recompile all historical templates by LLM.

Current runtime couples source authority and sampler support to the configured
dataset's `train.jsonl`. External sources therefore require an explicit synthesis
source-pool configuration, kept separate from the published real training set.
One clean option is a dedicated pool containing current 600 plus reviewed selected
external sources and the unchanged validation exclusions; its sampler support
would intentionally use that allowed pool. If support statistics must remain
600-only, separate source authority from fitting support instead. This choice
must be explicit before using external sources, not an implicit label bypass.

### Scaling and cost

A balanced count allocation is seven descendants per old template and eight
per new template: 700 + 800 = 1,500. Given five existing descendants per old
template and two new pilot descendants per new template, the remaining 800 are
200 old-template and 600 new-template draws. Sampling quotas may adjust this
balance; negotiability should be governed by template selection/quota, not by
changing the instruction independently of the template.

The previous 200-record admission campaign's all-attempt API ledger was
$0.29044148. Simple extrapolation is about $1.45 per 1,000 at that observed mix;
this is a planning reference, not a quote, and excludes engineering time and
any new external-source annotation. New work must retain generation/review/retry
cost receipts rather than reporting successful calls alone.
