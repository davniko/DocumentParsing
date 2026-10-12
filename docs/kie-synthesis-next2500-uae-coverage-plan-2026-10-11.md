# Next 2,500 UAE-endpoint synthetic samples: proposed coverage and validation

Date: 2026-10-11. **Planning only: no campaign launched, no dataset replaced.**
This follows the [latest training analysis](mpci-training-results-and-next-experiments.md#15-uae-expansion-run-j-5000-synthetic--600-real)
and [new production sampling integration](analysis/sampling-production-integration-20261011/REPORT.md).

## 1. Objective and fixed experiment boundaries

Generate **2,500** accepted improved synthetic documents to **replace the
current newer UAE 2,500-sample cohort**, as explicitly clarified by the user.
Preserve the inherited 2,500 synthetic documents and retain 600 real training
documents and 60 validation: **5,000 synthetic + 600 real = 5,600 train**.
The earlier additive 7,500/8,100 proposal is superseded; this is a coverage
replacement experiment at fixed dataset size, not another scale increase.
Do not count validation or the 30 integration inspection samples as additions.
Archive the replaced cohort and record the exact removal/join manifest.
Keep rank48, the same prompt-free
positioned input, compact targets, optimizer/LR/effective batch and10epochs
for the next coverage comparison. Synthetic-only and rank64 are separate
later experiments, not bundled into this coverage test.

Every new terminal/import locality is **United Arab Emirates**: discharge,
delivery and final/onward terminal endpoints must agree with the sampled
route contract. Origins, transshipment nodes and genuinely independent
office/payment locations remain variable. UAE does not mean every country
token anywhere in the document must sayUAE. Real and inherited synthetic
data are not retroactively moved toUAE.

Use approved policies unchanged: complete main description and justified
continuations; no leading package/STC declarations or separate Marks/customs
contamination; declared package unit and OCR-owned allocations; OCR-only
party ownership; uppercase target human text, variable rendered casing;
nullable required negotiability and notify reference; approved ambiguous
numeric-date fallback; source-supported route roles and positional geometry.

## 2. Evidence driving the distribution

J's all-document field F1/accuracy is0.9017/0.8650. On the same58parseable
documents, relation F1 improves0.9417→0.9623 and description F1
0.6379→0.6897. Size F1 slips0.8750→0.8641; declared package F1
0.8750→0.8571. Addresses are nearly flat, and shipment mass/volume ownership
remains weak. Thus add category contrasts while protecting successful
relation/list/route/metadata coverage.

The [current dataset inventory](kie-current-dataset-coverage-audit-2026-10-11.md)
finds **zero synthetic20HC**, only69total45HCcontainers across four families,
40HCdominance, missing synthetic support for several explicit declared units,
and inherited source-style alias concentration. The new sampler addresses
these controllable variables; it does not create absent document topology.

Avoid deriving new goods wording or exact labels from validation errors.
Validation identifies general failure mechanisms; training source selection
must exclude validation IDs, OCR/PDF duplicates and known shipment duplicates.

## 3. Recommended mutually exclusive sampling cohorts

|Cohort|Accepted documents|Purpose|
|---|---:|---|
|New registry-form/alias sampling|**1,000**|Independent physical forms, rare declared units, GP size/alias contrasts across varied compatible source layouts|
|Existing broad-domain/specialized paths|**1,500**|Broad HS registry coverage, thermal/DG, nested packaging, uncertain/incomplete equipment, no-container and existing documentary topology|
|Total|**2,500**|All new end/import nodesUAE|

The new catalog deliberately has ten forms across only sixHS4headings:
`1101`, `1507`, `3901`, `6912`, `7208`, `8501`. Applying it to all2,500
would improve package variability while unnecessarily narrowing commodity
coverage. The1,500 broad/specialized cohort prevents that. Compatibility
comes from reviewed commodity form and declared unit, not arbitrary donor
appearance or a blanket package-versus-container whitelist.

### 3.1 The1,000 registry-form documents

Target100documents for each reviewed profile:

|Profile|Declared unit|Reason|
|---|---|---|
|`paper_bag_powder`|Paper bag|Absent synthetic category|
|`multiwall_bag_powder`|Multiwall paper bag|Absent synthetic category|
|`jumbo_bag_granules`|Jumbo bag|Absent synthetic category; correct canonical enum|
|`ibc_liquid_oil`|Intermediate bulk container|Explicit IBC rather than generic package|
|`plastic_drum_liquid_oil`|Plastic drum|Material-specific category contrast|
|`crate_motor`|Crate|Explicit crate coverage|
|`skid_motor`|Skid|Skid versus pallet/crate contrast|
|`coil_steel`|Coil|Declared coil versus generic units|
|`wooden_case_ceramic`|Wooden case|Material-specific case coverage|
|`carton_ceramic`|Carton|Common-category control across the same commodity domain|

Profile names above match the current load-profile registry; freeze that
registry's hash when writing launch configs. Spread
each category over **at least ten compatible layouts**, rather than making a
category synonymous with one carrier/source. Across2,500, retain broad
ordinary package/carton/pallet/bag examples from the existing cohort.

Suggested equipment weights for the1,000 new-mode documents:

|GP pair|Weight|Approximate document draws|
|---|---:|---:|
|20standard|15%|150|
|20HC|25%|250|
|40standard|25%|250|
|40HC|15%|150|
|45HC|20%|200|

These are **document-level homogeneous equipment draws**, not percentages
of container objects; sources with many containers contribute more objects.
Current new-mode synthesis preserves the source container count and assigns
a compatible homogeneous pair. Mixed-container changes remain deferred.

For sources printing tare,20HC is not available until a proper tare authority
exists. Use a separate exact eligible-equipment domain for those sources;
do not invent tare, suppress its source line or silently switch equipment
after drawing. Achieve the overall20HCquota with compatible non-tare layouts.

There are31GPregistry aliases:8/5/7/5/6 respectively across the five pairs.
Target **at least ten realized documents per alias**, with multiple layouts
where feasible. Uniform alias weights are sufficient; do not copy one source
alias everywhere. Counts are based on accepted owned occurrences, not regex
hits in tariff/address/product text. Complete specialized aliases, including
40HCreefer, remain in compatible specialized cohorts, not forced intoGP.

### 3.2 Configuration is cohort-specific

The production configuration has campaign-wide alias policies. A single
equipment-registry override rejects incomplete, no-container and mixed
equipment sources; this is an explicit contract, not a permissive fallback.
Prepare separate configurations as needed for:

1. New-mode homogeneousGP with complete owned equipment aliases.
2. Complete specialized thermal/DG/other equipment with verified alias owners.
3. Existing incomplete/mixed/no-container or source-specific contracts without
   an unsupported equipment-alias override.

The1,500 total may therefore be split across several launch configs; that
does not change the mutually exclusive1,000+1,500 budget. Preserve one
shared manifest with unique sample IDs, authority pins and realized coverage.

## 4. Preserve prior overlapping coverage strata

Targets below apply to the **whole 2,500 replacement samples** and overlap. They are
not additional exclusive quotas to sum on top of1,000+1,500. They follow
the previous accepted UAE campaign; exact joint feasibility must be checked
before freezing the launch. If a floor cannot be met, report the actual
feasible count rather than inventing ownership or quietly lowering it.

|Trait|Proposed floor/approximate target|
|---|---:|
|Multi-container|985|
|At least8containers|180|
|Membership-only placements, no printed quantity|175|
|Existing incomplete-equipment contracts|129|
|Thermal|180|
|Dangerous goods|100|
|Negotiable consignee wording|200|
|Notify-party reference wording|675|
|Approved split-product passages|300|
|Repeated product presentations|324|
|Split postal presentation|432|
|Same-line identity/postal context|341|
|Volume rows, no explicit shipment total|250|
|Volume rows with shipment total|208|
|Gross rows, no explicit shipment total|299|
|Gross rows with shipment total|361|
|Genuine source-owned transshipment roles|Approximately50|

Keep real no-container/VIN distractors and sole-containerSTCquantity cases
under their existing reviewed contracts. Prefer source diversity for each
trait and cross-category variation within a layout. Do not teach one package
type as a proxy for negotiability, carrier, UAEport or container size.

No new blanket numeric/HSproduct-semantic requirement is introduced. The
model's training objective is correct extraction/relations, not HS-product
classification or physical-capacity inference. Host-controlled public
quantity/mass facts must nevertheless be rendered and labeled consistently;
an explicit new net total cannot silently appear without its target.

## 5. Actual admission probes and limitations

Two actual production planning attempts per shape-eligible source followed
by source preflight were exercised offline, with candidate search bounded24.
The run took27.70seconds and made no API requests. Sources stop at the first
exception, so do not interpret the eligibility count as578successful calls.
At least466plans succeeded across the233admitted sources.

|Stage|Current-real authority|Other admitted authority|Total|
|---|---:|---:|---:|
|Shape-eligible|272|17|289|
|Passed both plans and preflight|218|15|**233**|
|Rejected new-mode admission|54|2|**56**|

|Explicit new-mode limitation|Sources|
|---|---:|
|Existing package restriction has no compatible new profile|14|
|20HCdraw has no source-printed tare authority|19|
|Adjacent package-count owners make alias grammatical number unresolved|7|
|Complete alias occurrence disagrees with standalone source pair|6|
|Mixed/unreviewed equipment occurrences|5|
|Numeric/auxiliary ownership or unsupported equipment-count framing|5|

These are **not new proof that existing accepted descendants are malformed**.
They show why not to override every source with the new mode. Compatible
sources are enough for a varied cohort; other sources can stay in their
already-supported modes. Two successful probes do not guarantee every future
draw: enumerate and preflight every actual planned candidate.

The full source IDs, captured rejects and authority hashes are in
[next-cohort-admission.json](analysis/training-latest-20261011/next-cohort-admission.json).
Passing IDs were reconstructed from the original eligible pool minus captured
rejects and checked against the probe counts; the receipt explicitly records
that serialization method rather than pretending a second probe was run.

Known topology ceilings remain: no certified true cross-page product
continuation, partly counted synthetic allocations, known-type/missing-size
mutable source slot, or roster above14containers created by these weights.
An approved split product passage is not automatically a page-to-page
continuation. Do not advertise coverage that source contracts cannot express.

## 6. Execution and acceptance gates for the later authorized campaign

### Before any paid generation

1. Freeze a fresh seed, exact source/ownership/registry/config/schema hashes.
2. Enumerate all2,500 intended plans and preflight every one. Unsupported
   plans are explicit rejects with an alternative eligible plan selected
   before wording, never hidden changes after generation.
3. Produce a realized joint table: category×equipment×alias×source layout,
   plus all overlapping documentary strata above. Positive weight does not
   guarantee a finite count. Exact quotas require audited draw selection,
   not claiming that expected proportions were achieved.
4. Check all final/import endpointsAE; origins/intermediate roles coherent;
   validation/source/PDF/OCR/known-shipment overlap zero.
5. Check declared units, numeric owners, count/weight/volume and public-field
   availability; no invented category/length/tare/quantity to meet a quota.

### Generation and publication

Run five reviewable waves of500, with the intended200new-mode/300existing-mode
mix per wave if joint source constraints allow. Within a wave, use separate
cohort configs as required, resumable receipts and the established concurrency.
This is a proposed execution layout, not an already prepared launch file.

For every final plain-text/target pair:

- Production render replay and frozen source-binding ownership checks.
- Complete-pair model review using the rendered OCR text and full target,
  not only a suspicious substring. Adjudicate each finding; stale reviews
  cannot approve a subsequently changed candidate.
- Targeted checks for description boundaries/qualifiers/continuations,
  declared package level, per-container row quantities, net declarations,
  identity/postal/contact boundaries, negotiability/notify and date policy.
- Anonymization of actual mutable source identities/identifiers; harmless
  unchanged flavor/header formatting is not a reason to distort targets.
- Independent position audit: remove suffixes to reproduce plain text;
  targets unchanged; coordinates within page range, measured source-anchored
  layout constraints, line ordering/row associations and plausible reflow;
  unmatched coordinates remain explicit rather than losing text.
- Deliberate corruptions through the validation path: altered quantity/unit,
  unknown enum, missing/repeated container, wrong role ID, declaration/Marks
  leakage, stale approval and invalid/ownership-changing coordinates must
  not be published as accepted.

Publish only adjudicated valid samples; keep rejects/reasons separate and
replace rejected candidate slots to reach2,500accepted, not2,500attempted.
After waves, open100random pairs from50different sources, two descendants
per source, spanning new and existing cohorts and avoiding the integration30
when possible; add targeted inspection of every review finding. Record source
OCR, rendered OCR, target and geometry decisions in an inventory. This manual
audit complements, not replaces, complete-pair review of all2,500.

Before joining: exact2,500accepted uniqueIDs; required realized strata;
zero validation overlap; no unresolved finding in the published cohort;
post-publication hashes and target/input parity. Preserve the inherited 2,500;
archive and exclude the superseded newer UAE 2,500; create a new combined
dataset directory containing exactly 5,000 synthetic plus 600 real training
records, and the unchanged 60-document validation split, with explicit manifests. Audit max
source/target tokens against the launch limits before configuring training.

One separately documented inherited real-label correction remains before that
join: training source`8c6da3`, container`CXRU1576038`, explicitly prints
refrigerated equipment and−18°C but lacks `typeCategory`. The current5,600-row
training file was checked and still has that omission. Add the supported
refrigerated type without inventing length, then repin the new combined
dataset. It has no current synthetic descendants and is not one of the new
registry-mode admission failures. This planning pass does not edit the
completed run's hash-pinned training data.

## 7. Cost and model evaluation

Observed all-attempt costs, not new price quotes:

- Previous UAE campaign: **$2.49696160 /2,500**.
- Recent new-sampler integration: **$0.03585315 /30**, including retries and
  review/refinement work; linear projection approximately**$2.99 /2,500**.

Use a planning allowance of**$3–4**, with a proposed explicit total ceiling
of**$5** and per-call/all-attempt receipts. Rejected requests, retries and
reviews belong in total accounting. Actual cost depends on the selected
source lengths and adjudication needs; these observations do not guarantee a
quote. Freeze and verify provider/pricing configuration before launch.

For the eventual training comparison, retain all-document full-target F1,
accuracy, precision/recall, relation/category facts and exact graphs, JSON/
schema validity and retained checkpoints. Also compare the fixed I/J58matched
cohort or a newly named exact intersection, category/alias supports and
description/address errors. Do not hide syntactic failures or compare two
different parseable memberships as if they were the same documents.

The next hypothesis is testable: richer declared-unit and equipment contrasts
should improve size/package accuracy without sacrificing relation/metadata
strengths. UAE-domain concentration remains a separate distribution shift
from the real validation destinations; retain origin/layout diversity and
measure results rather than assuming moreUAEexamples improve every domain.
