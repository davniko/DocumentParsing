# Second 100-template synthesis admission

## Scope

Add 100 new source templates to the existing 100-template catalog, with two
complete synthetic descendants each. Preserve the existing catalog assets and
published real/synthetic datasets. Sources must exclude validation and previous
curation rejects. Current policy includes uppercase targets, OCR-grounded
nullable negotiability, explicit notify `sameAs`, current goods/package/placement
semantics, source-compatible registry sampling, and joint positional reflow.

Working campaign: `artifacts/kie-synthesis-production/curated-v7-expansion200-v1`.
Selection, source identities, reviewed declarations, failed attempts, costs and
validation receipts are retained there. Staging is separate from the shared
catalog; admission follows complete content and geometry review.

## Status

**Complete: 100 additional sources admitted; shared catalog now contains 200.**
The new campaign contains 200 published synthetic descendants, two per source,
with current labels, complete rendered text, positional receipts and reviews.
All final content findings are resolved. No training was started. No larger
synthesis campaign was launched by this admission pass. The subsequent
[1,000-fresh-sample plan](kie-synthesis-expansion1000-2026-10-08.md) keeps these
200 admission samples separate and combines the existing training500 with
1,000 new descendants to reach the planned 1,500-synthetic experiment.

## Required validation

1. Source/target identity, validation exclusions and exact baseline replay.
2. Complete ownership of sampled public and dependent private facts; no overlap.
3. Sampling/physical/route preflight before paid generation.
4. Two complete renderings per source, exact replay and current schema checks.
5. Complete content review with source comparison and adjudicated findings.
6. Independent geometry checks, negative controls and representative plots.
7. Shared catalog admission and old-campaign replay if shared code changes.

## Deliverables and exact scope

- [Shared 200-source catalog](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/README.md)
  and its [manifest](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/manifest.json).
- [Source OCR + both rendered variants](../artifacts/kie-synthesis-production/curated-v7-expansion200-v1/samples.md).
- [Positioned samples](../artifacts/kie-synthesis-production/curated-v7-expansion200-v1/positions-reflow-v1/samples.md)
  and [41 diagnostic plots covering 20 documents](../artifacts/kie-synthesis-production/curated-v7-expansion200-v1/positions-reflow-v1/audit/GALLERY.md).
- Plain `dataset.jsonl` and `positions-reflow-v1/dataset.jsonl` in the campaign
  directory. These are separate from the existing 500-synthetic training copy.
- [Campaign configuration](../configs/synthesis/mpci_bl_curated_v7_expansion200.yaml).
  It retains the audited staging paths to preserve its publication fingerprint.
  All admitted contracts, templates, source/label/alignment snapshots and shared
  declarations have also been copied into the common catalog. Full replay through
  that catalog, rather than staging, was verified.

The existing 660 real records, previous 500 published synthetic records and
original 100 per-source catalog assets were not edited. Shared YAML inventories
were extended without changing their old source entries. Catalog-wide manifest
and documentation were updated deliberately. Large per-source snapshots remain
local ignored assets under the repository's existing data policy; transfer them
with the catalog when moving synthesis to another machine.

## Selection and diversity

The preparation inventory examined 209 unused current-training sources. Of these,
174 passed exact ownership preparation and 130 had complete current target-path
coverage. Sampling constraints, dependent printed facts and semantic review were
then applied before choosing the final 100. These counts are successive screening
results, not a claim that the other 109 sources are defective or unrecoverable.

Although external non-validation sources were permitted, the reviewed training
pool supplied enough candidates for this increment. No additional real labels
needed to be invented. Final checks found **zero overlap with validation by
document ID, exact OCR or B/L number**, and zero overlap with the original catalog.
The final replacement source was included in these checks.

| Trait | New source templates | New descendants |
|---|---:|---:|
| Documents | 100 | 200 |
| One / two / three pages | 27 / 58 / 15 | 54 / 116 / 30 |
| One container | 78 | 156 |
| Multiple containers | 22 | 44 |
| Negotiable | 11 | 22 |
| Non-negotiable | 86 | 172 |
| OCR-unavailable negotiability, explicit null | 3 | 6 |
| Notify entries explicitly `sameAs: consignee` | 29 | 58 |
| Independently printed notify entries, `sameAs: null` | 62 | 124 |
| Explicit transshipment-port targets | 3 | 6 |
| Final-destination targets | 7 | 14 |

There are 96 ambient, one frozen, one chemical-DG and two vehicle templates in
this increment. The complete 200-source pool now has **184 ambient, six vehicle,
four chilled, three frozen, two chemical-DG and one vehicle-DG templates**.
It contains 12 negotiable families, compared with one in the previous 100.

Registry sampling produced 103 distinct **printed** HS-code values and 18 package
categories in the 200 descendants. Shipper and consignee geography covers 115
and 121 country codes respectively; Egypt was not fixed as the destination.
Per-source field presence and topology remain intentional: missing equipment
categories or unavailable consignee instructions are not filled merely to obtain
a desired distribution. Container counts range from one through six.

Machine-readable detail: [final summary](../artifacts/kie-synthesis-production/curated-v7-expansion200-v1/audit/final-summary.json).

## Rebasing and dependency closure

Current reviewed real OCR and reduced-V7 targets were the authority. Historical
bindings and numeric contracts supplied reusable source structure, not old gold
labels. Exact occurrences, current target paths, lexical regions and dependent
source-only text were reconciled through explicit per-source declarations.
The renderer continues to verify byte ownership and replay every edit; no global
text replacement or source-ID branch was added to the production compiler.

Representative repairs:

| Issue | Resolution and example |
|---|---|
| Split postal country ownership | `1e9d32e5` contains CHINA inside a named industrial site and again in the terminal postal component. Explicit `postal_country_owners` now assigns the country role only to the reviewed terminal fragment. Generated addresses need the country once, not in every fragment. |
| Repeated/continued parties | `fabe32eb` repeats notify/delivery identity and postal details. Shared values are retained; the synthetic delivery address is consolidated inside its own block and its obsolete remote partial address removed. The real OCR is unchanged. |
| Separate contact names | `291f67b3`, `51d5f830` and `5431122b` print two contacts separately. Existing split-path rendering emits the generated contact names once in each appropriate position, with both retained in the target. |
| Source-only physical measures | `7f8fafe7` and `9ad49fcc` have printed gross weights not present as complete public measures. They now participate privately in capacity sampling without creating unsupported public labels. |
| Per-container and aggregate tares | `a286601b` has five printed 3,700-kg tares and a six-container total of 22,270 kg. The existing aggregate resolver derives the one unprinted component as 3,770 kg; it does not assume six identical tares. Package nouns are bound in all six rows. |
| Package levels | `058d92bb` has 20 outer packages and a separately counted inner level. Its synthetic outer caption is explicit; inner quantities are positive multiples of 20. `e9120c73` and `e9275395` retain six/five private outer pallets with compatible inner categories and count multiples. Targets still contain the agreed inner level only. |
| Stale count-dependent marks | Original carton/box serial ranges in `7f8fafe7` and `e9275395` were removed from synthetic text because they encoded the old counts. Independent lot/reference information remains; no real source text was deleted. |
| Repeated package nouns | All relevant rows in `a286601b`, `e745f5ba` and `fabe32eb` follow the sampled type. `5c5e51a9` uses a generic total-package caption rather than retaining TOTAL PALLET when cartons are sampled. |
| Dependent geography | Port footnotes in `9bd26a2b`, exporter-country lines, origin marks, office captions and routing clauses now follow their reviewed owner. `5c5e51a9` no longer retains ALEXANDRIA/EGYPT in sampled destination clauses. Its contradictory, unlabeled source delivery field is omitted from synthetic descendants. |
| Country-specific source registrations | Source tax/registration values receive fictional replacements and relevant captions are generalized. These are extraction examples, not a claim of valid national registration numbers. |
| Dates and identifiers | Explicit date profiles cover split worded dates; seal/vessel/voyage aliases and split B/L identifiers follow their complete current values. Date profiles are checked against the real source date. |
| Multiple HS values | Nine families needed a matching two/three-identity sampling capability for their printed code structure. Multiple HS codes remain allowed; they are not treated as proof of multiple goods. |

One candidate, `015c845e`, was replaced by `a286601b`: PL STONE / PLSTONE aliases
share a postal block, but its identity-sharing relation needs explicit
adjudication before independently sampling party localities. Other inspected
reserve candidates retain their blockers in the inventory, including unresolved
carton ranges and remote/compound address ownership. None is silently admitted.

Reusable assets are the contracts, shared ownership/auxiliary declarations,
capabilities and source geometry. Pilot-specific wording corrections are in the
campaign receipts, not hardcoded as template-specific Python branches.

## Content review and adjudication

All 200 final candidates passed schema validation, owned-text/label grounding,
instruction inheritance, physical arithmetic/capacity checks and exact replay.
Each source's two complete rendered documents and target values then received a
scoped GLM-5.3-Flash review. Reviews bind the exact current candidate hashes;
changed samples were reviewed again. Failed or rate-limited calls did not count
as completed review.

Main-agent inspection covered the concrete findings and related source clauses,
not merely their model verdicts. Corrected lexical defects included duplicated
tokens, contradictory sterile/non-sterile wording, invented packaging capacities,
omitted locality spelling and contact-domain inconsistency. A goods/package
cross-check also found implausibly cartoned complete harvesters and invented
cable drum capacities. Those trial descriptions were corrected without changing
the host-controlled quantities, equipment or printed HS codes. Complete
descriptions, not first fragments, remain the generation context.

The final receipts contain seven rejected false-positive findings and **zero
unresolved findings**. The rejected findings concern:

- Two outer-versus-inner package-count flags, resolved using the explicit
  hierarchy and exact divisibility rather than flattening two levels together.
- Three exact HS/product-taxonomy flags. The agreed objective permits neighboring
  product detail; it does not train a tariff classifier. This does not relax
  consistency of printed quantities, equipment or DG/thermal facts.
- One title-case OCR versus uppercase-label flag, which is intentional policy.
- One unchanged independent shipping-bill reference, outside the reduced target.

Every rejection records its reason, exact rendered evidence, finding hash and
candidate hashes. Review acceptance therefore cannot silently carry over to a
different sample. See [final adjudications](../artifacts/kie-synthesis-production/curated-v7-expansion200-v1/audit/final-adjudications.json).

**Interpretation of admission:** these 100 source contracts and their two trial
descendants have passed the defined content and geometry gates. It is not a claim
that every future stochastic wording draw is automatically correct. New runs
retain rendering validation, review and adjudication before publication. The
trial corrections show why that gate remains necessary; admission is not a
license to publish unreviewed future output.

## Positions: independent validation and remaining gaps

The production source-conditioned reflow and coherent page transform were used
without changing their algorithm or fitting new per-template geometry rules.
The existing source-derived calibration remains pinned. A separate no-reflow
anchor-transfer run supplies a like-for-like baseline.

| Content lines | Anchor-transfer baseline | Reflow + augmentation | Coverage after |
|---|---:|---:|---:|
| All 20,840 | 16,609 | 17,621 | 84.6% |
| 1,159 goods-description lines | 77 | 802 | 69.2% |
| 1,858 address lines | 972 | 1,250 | 67.3% |
| 17,823 other lines | 15,560 | 15,569 | 87.4% |

There are 376 synthetic pages: 186 have accepted reflow and 190 have no eligible
reflow group. **No proposed page layout was rejected and no previously known
coordinate was lost.** 374 pages received a coherent scale/translation; two had
no known anchors. Across the published inputs, 17,595 known points differ from
their unaugmented anchors.

The remaining 3,219 unknown lines comprise 357 goods, 608 address and 2,254 other
lines. They retain the explicit ` ||` form. The receipts identify 106 groups with
incomplete source anchors and 95 whose source envelopes contain unrelated text.
Those are geometry limitations, not held sample-content reviews. Filling them
with guessed positions would undermine the layout signal. This broader set has
more source-anchor limitations than some earlier diagnostic subsets; it must not
be described as fully positioned.

Independent checks cover all samples: original text and labels unchanged,
receipt/input hashes, complete page/line inventories, line order, layout
clearance, normalized coordinate bounds, coherent transforms and preservation
of known anchors. Four negative probes deliberately introduce text mutation,
an off-page coordinate (with recomputed input/hash), a duplicate line and an
inconsistent anchor. All four are rejected. Twenty documents/41 pages were
plotted using high-gain, low-coverage and hash-selected cases. Three representative
panels were also visually inspected, including dense multi-container layout,
compact party columns and an attachment page.

See [independent probes](../artifacts/kie-synthesis-production/curated-v7-expansion200-v1/audit/geometry-independent-probes.json)
and [complete geometry statistics](../artifacts/kie-synthesis-production/curated-v7-expansion200-v1/positions-reflow-v1/audit/validation.json).
These are plausible source-conditioned centers, not measurements of newly
rendered synthetic PDF glyphs.

## Code changes, regression checks and efficiency

`curated_ownership.py` now supports the explicit postal-country region declaration.
It rejects unknown fields/regions, duplicate or empty owner lists, another
party's address, missing source evidence and deleted country owners. Without a
declaration, existing behavior is unchanged. Tests cover the split-address case,
missing sampled country and malformed declarations.

Expanding the catalog exposed repeated parsing of the entire ownership YAML for
each blueprint. The loader now caches at most two exact **byte contents** (active
and staged catalogs). It still reads current bytes on every load, so a changed
file cannot reuse a stale cache entry. Selected declarations are deep-copied;
one blueprint cannot mutate cached state for another. This is a bounded parsing
optimization, not a cache of accepted sample verdicts. Tests explicitly cover
file changes and mutation isolation.

Two older test fixtures lacked the now-required nullable negotiability field;
they were updated to `null`, without weakening the real schema. Final relevant
suite: **332 passed in 16.26 seconds**; Ruff and whitespace checks also pass.

Additional evidence:

- **1,000/1,000** offline physical/route/ownership preflight draws, ten extra variants
  per selected source; 13.66 seconds, 434.3 MiB peak RSS. This tests sampling and
  host constraints, not ungenerated LLM wording.
- Final rendering: 200/200 in 14.83 seconds before the catalog-loading optimization.
- Production position generation: 18.69 seconds for 200 samples. Independent
  geometry audit plus 41 plots: 4.24 seconds; baseline/probe process peak 482.7 MiB.
- All **700 published candidates** replay exactly through the expanded shared
  catalog: old 200, old 300, new 200. Text, targets, facts and edit proofs match.
- The isolated 200-record replay dropped from **14.50 to 4.65 seconds** after
  ownership parsing was cached: about 68% less elapsed time / 3.12× throughput.
  CPU time fell from 13.33 to 3.49 seconds; peak RSS was effectively flat at
  **435.1 versus 434.6 MiB**. Separate-process measurements distinguish actual
  per-run memory from accumulated objects in a multi-run audit process.

No changes were made to model training, evaluation metrics, real labels, original
OCR, existing synthetic samples or the positional synthesis algorithm.
Admission backs up the previous shared files and commits the new catalog manifest
last. See [admission receipt](../artifacts/kie-synthesis-production/curated-v7-expansion200-v1/audit/catalog-admission.json)
and [registered replay](../artifacts/kie-synthesis-production/curated-v7-expansion200-v1/audit/registered-replay-final.json).

## Cost and next use

Provider-reported API spend, **including retries, superseded trials, corrections
and re-reviews: $0.216433335**. Breakdown:

| Work | Cost USD |
|---|---:|
| Initial/revised wording | 0.080256 |
| Wording-validation repair | 0.004125 |
| Postal corrections | 0.005570 |
| Company contacts and contact repairs | 0.003238 |
| Complete rendered review and re-review | 0.123244 |

That is approximately **$0.00108 per final descendant** for this pilot's API
usage. It excludes engineering/manual-review time and is not a guarantee of the
cost of an arbitrary future batch. No geometry or numeric preflight incurred API
charges. All attempt receipts remain in `calls/`.

The template expansion is finished. The next synthesis campaign can sample from
the admitted 200, using the existing configurable family and negotiability
distributions, new seed/output and the same review gates. These 200 new samples
can count toward the planned 1,000-sample increment, subject to the chosen final
mix; generating another 800 would then produce 1,500 synthetic samples in total.
That additional generation and rebuilding the training mixture remain a separate
run, not something silently launched by this template-admission task. The later
request selected **1,000 fresh samples**, keeping these admission200 separate;
see the [prepared campaign](kie-synthesis-expansion1000-2026-10-08.md).

## Evidence directory

The campaign's `audit/` contains the 209-source preparation inventory, source
decisions, numeric/route/private dependency reviews, exact wording edit receipts,
final selection, validation-leak checks, pre/post hashes, stress results,
adjudications, independent geometry probes, costs and replay benchmarks.
One-time repair/admission scripts are isolated under
`docs/analysis/synthesis-expansion200-plan-20261008/`; production functionality is
in the existing synthesis modules rather than a second experimental pipeline.
