# DG template expansion: twenty additions

**Completed:** twenty new DG templates admitted, with three certified test
descendants each. The registry now contains **383 templates, including 27 DG
templates**. All sixty final inspection samples pass; none remain held.

This report records the source selection, rebase decisions, production synthesis
tests, full-content reviews and geometry checks for the DG expansion requested
on 14 October 2026. The final admission receipt is the authority for completion;
the working audit records live in [the audit directory](analysis/dg-template-expansion20-20261014/).

## Scope and source selection

The selected cohort contains seventeen unused sources outside the current
660-document real dataset, plus three previously uncompiled sources from its
600-document training split. The three substitutions were explicitly approved
by the user. No validation source is admitted, and no existing training dataset
is modified by this template-expansion task.

The wider filtered pool already has PaddleOCR results. Its additional work is
current-policy source annotation, exact GLM-to-Paddle alignment, ownership
rebasing and validation—not another PaddleOCR campaign. The inventory found
38 DG candidate sources among the wider pool. Candidate status is not automatic
approval: missing OCR occurrences, compound goods/HS ownership and unsupported
package totals must first be resolved.

Three initially selected cases were replaced rather than introducing a new
compound-goods model or inventing absent package counts. A further substitution
avoided assigning four indistinguishable retained container-quantity rows to
five containers by guesswork. The complete candidate evidence, selection
history and exclusion reasons remain in [the selection inventory](analysis/dg-template-expansion20-20261014/selection/proposed-selection.json).

The compilation authority contains 617 real training sources and 60 preserved
validation records: seventeen new rows plus three replacements of existing
compilation rows. It is not a new model-training dataset. The previously used
training labels for those three sources remain unchanged; the refinements to
their compilation annotations are explicitly documented below.

## Observed DG presentations

Each descendant retains its source's **printed field set**. A registry value
that is not printed by that source does not become an extraction target merely
because the sampler knows it.

| New source presentation | Templates | Descendants |
|---|---:|---:|
| UN only | 1 | 3 |
| UN and primary class | 4 | 12 |
| UN, class and packing group | 9 | 27 |
| UN, class, packing group and flash point | 4 | 12 |
| UN, class, packing group, subsidiary hazard and flash point | 2 | 6 |
| Total | 20 | 60 |

Consequently the additions supply fifteen packing-group layouts, six flash-point
layouts and two subsidiary-hazard layouts. These supplement the prior seven DG
templates; they do not inject every optional DG field into every document.

After admission, the complete DG repertoire is:

| Printed presentation | Before | Added | Now |
|---|---:|---:|---:|
| UN only | 0 | 1 | 1 |
| UN and primary class | 4 | 4 | 8 |
| UN, class and packing group | 1 | 9 | 10 |
| UN, class, packing group and subsidiary hazard | 1 | 0 | 1 |
| UN, class, packing group and flash point | 1 | 4 | 5 |
| UN, class, packing group, subsidiary hazard and flash point | 0 | 2 | 2 |
| Total | 7 | 20 | 27 |

Thus **18** admitted DG templates print packing groups, **7** print flash points
and **3** print subsidiary hazards. The unchanged UN/class-only presentations
remain useful negative supervision for absent optional fields.

UN, proper shipping name, class, packing group and subsidiary hazards are tied
to one selected [Hazardous Materials Table](https://www.ecfr.gov/current/title-49/subtitle-B/chapter-I/subchapter-C/part-172/subpart-B/section-172.101)
entry. Flash-point samples additionally require a matched record in the saved
NIOSH registry, not a generated temperature for an unrelated formulation.
The registry is at `artifacts/sampling-registries/niosh-flashpoints-20261014/`.
Source-specific decimal precision and Celsius/Fahrenheit presentation are
rendered with recorded conversion/rounding; only printed values are labelled.

## Rebase and production-path changes

### Source truth, ownership and privacy

- Reconstruct annotations from complete retained OCR under the current policy,
  rather than accepting historical labels or recovering PDF-only scalars.
  PDFs resolve column/layout ownership only.
- Use the complete goods-description block and genuine continuations; keep
  Marks, documentary identifiers and leading transport/packing declarations
  outside it. Retain embedded product specifications and capacities.
- Select the declared shipment package level. Nested drums inside a skid or
  supporting pallets around drums are not extra package rows.
- Preserve explicit sole-container allocations and multi-container membership.
  Do not synthesize public per-container counts for source rows that OCR omitted.
- Bind repeated identities, private contacts, tax/customs identifiers and
  commercial references across all pages. Retain public carrier identities and
  public emergency-service contact information.
- Omit exact source-only CAS/EmS/environmental declarations when the production
  registry does not supply their truth for the newly sampled chemical. This is
  explicit span ownership, not broad deletion by keyword.

### Generic functionality exercised by the new sources

- **Packing-group presentation:** numeric `1/2/3` and Roman `I/II/III` are
  rendered from the same category, preserving each owned occurrence's style.
- **Source-bound load accounting:** reviewed complete private quantity vectors
  can check capacity without adding OCR-missing allocation values to labels.
  Certificates pin the source, target, layout evidence and equipment context.
- **Free-text package presentations:** `PARCEL` and `SETS` remain source-backed
  free-text package labels rather than being forced into an invented category.
- **Incomplete equipment evidence:** missing size/type information remains
  missing publicly; source-bound private accounting does not fabricate fields.
- **Electric vehicles:** the battery-electric passenger-vehicle source samples
  the bounded UN3171/HS870380 domain. Existing UN3166 fuel-vehicle draws remain
  unchanged. This is not an unrestricted vehicle/DG substitution.
- **Source-only manufacturer:** a separately printed manufacturer receives its
  own generated identity/postal information without becoming a new extraction
  party role.
- **Repeated regulatory names:** explicit product-only surface aliases use the
  selected proper shipping name, not a complete packed-description passage.

### Current-training donor annotation differences

The approved three training-source donors are `2e8c2a1f`, `48325f63` and
`630cd159`. Their new compilation annotations are reviewed independently of
the training dataset. In particular:

- `2e8c2a1f`: repeated GlobeLink text belongs to delivery, not a blank notify
  box. The goods description excludes its leading three-drum declaration;
  formatted HS `2915.90.1800` is retained. An unsampled marine-status statement
  is not carried into arbitrary chemical descendants.
- `48325f63`: retain the formatted source HS presentation and the explicitly
  printed 17.6-tonne net weight. A personal mailbox in the goods-area continuation
  is anonymized there but is not assigned to the direct consignee block.
- `630cd159`: apply the current complete-description and printed-DG-field
  rules instead of treating the historical annotation as authoritative.

## Source coordinate evidence

The final twenty-source alignment audit covers 48 pages and 2,752 content lines.
2,320 lines have measured positions: **84.30% source coverage**. It identifies
149 repeated-text groups, uses 124 repeated-context matches and 384 spatial
group matches, and detects no reused measured-region claims.

Eight alignment corruptions per source were tested: 160 rejected controls.
The source audit and its 75 relevant tests passed. Missing geometry remains
explicit; no unresolved repeated text is assigned an arbitrary occurrence.
See [source-coordinate validation](analysis/dg-template-expansion20-20261014/spatial/final/validation-summary.json).

The synthetic geometry uses these measured source anchors and the production
reflow path. Its final coverage, all-sample validation and inspection plots are
recorded in the completion section below.

## Thermal repertoire reviewed separately

The existing registry has **thirteen temperature-bearing thermal templates**:
nine frozen and four chilled. They contain sixteen temperature-bearing
containers; all source equipment presentations are 40-foot high-cube reefers,
and all source temperature units are Celsius. Two templates print ventilation.
Six additional templates print reefer equipment but do not establish a thermal
goods/temperature profile.

This inventory does not restrict descendants to thirteen source commodity
names. The validated sampler registry has 82 frozen and 73 chilled HS6 identities.
The read-only probe exercised 260 ordinary draws and recovered 76 distinct
frozen and 47 distinct chilled codes. Forty constrained produce draws also
verified the observed apple and garlic profiles, including their printed
ventilation settings.

All nineteen related source cases' 114 asset pins matched; all thirteen thermal
source replays and position-text preservation checks passed. The current 5,000
synthetic pool has 399 temperature-bearing documents, comprising 259 frozen and
140 chilled documents. The full [thermal inventory](analysis/dg-template-expansion20-20261014/thermal/inventory.json)
contains source-level links, units, temperatures and remaining coverage gaps.

The next thermal breadth opportunities are more ventilation layouts,
Fahrenheit presentation, 20/45-foot thermal equipment presentations and any
new reviewed commodity profiles. No unsupported pharmaceutical temperature
profile was introduced, and no thermal templates were added in this DG task.

## Completion receipts

### Full-content review and corrections

All twenty complete source OCR texts and their current annotations were reviewed.
All sixty complete descendant renders and labels were individually read and
adjudicated, not just inspected through summary counters. Twenty model-review
calls additionally received each source and its three complete descendants.
After the final private-mailbox repair, the affected three descendants were
reread and received a fresh model review; the other nineteen unchanged review
requests reused their exact cached results.

The final [manual review inventory](analysis/dg-template-expansion20-20261014/manual-review.json)
contains sixty passing decisions, with source, plain input, positioned input,
target, candidate and geometry-receipt hashes. The independent
[validation receipt](analysis/dg-template-expansion20-20261014/validation.json)
also passes all sixty documents. There are **no unresolved holds in this final
inspection set**.

The audit caught and resolved these concrete issues before publication:

| Finding | Corrected layer and observed result |
|---|---|
| A complete postal address was being requested independently for several components of one address; correction context omitted fixed postal wording. | Generation now describes the jointly assembled postal region, its fixed components and explicit country owner. Ten descendants required sixteen field repairs; unrelated generated values stayed unchanged. |
| Some generated values copied instruction captions or a source company. | Narrow name/postal guards reject instruction leakage and unchanged source identity. The guards do not broadly reject ordinary words such as “region” in a genuine address. |
| A spaced container identifier was not bound to its sampled identifier. | The C706 template now owns its explicit composite surface. There is no global whitespace-matching fallback. |
| A country repeated under Marks retained its old value; a source-only limited-quantity declaration could contradict the selected chemical. | A915's repeated country shares the sampled consignee-country owner; the unsampled declaration is explicitly omitted. |
| A named hazard label remained fixed while its numeric class changed. | The unsupported source-only named label is explicitly omitted; sampled numeric declarations remain consistent. |
| Four earlier product-name occurrences remained fixed in a five-container document. | B854 binds all four to the existing product-only surface. They do not acquire the complete packed description, and Marks text does not leak into the description target. Three actual production before/after pairs, two builder reproductions and eight rejected corruption controls validate the change. |
| A private source-only email's recipient changed but its old company domain remained. | Source-only mailbox ownership now requires an explicit private-company/public-provider/public-carrier domain policy. All three AB descendants have newly generated private-company domains. Three existing public-carrier policies preserve their established behavior. |
| Two source adjudication notes incorrectly carried another case's equipment remark. | Audit metadata corrected; source text, labels and equipment values unchanged. |

The frozen [source revision inventory](analysis/dg-template-expansion20-20261014/source-render-revisions.json)
and [postal findings](analysis/dg-template-expansion20-20261014/postal-instruction-findings.md)
retain the exact before/after evidence. The source-only mailbox search covered
all 383 old-plus-new sources; the explicit privacy-policy migration changes
only three old ownership metadata leaves, not their six compiled assets or any
training sample.

Six model-review findings on one source were rejected after checking the full
source: one skid contains two steel drums, and the drums' embedded 200-litre
capacities are not competing shipment package totals. Leading `SKID STC` remains
outside the product description. The
[adjudication receipt](analysis/dg-template-expansion20-20261014/review-adjudication.json)
records these rejected claims; good labels were not altered to satisfy them.

A separate fresh compact-instruction probe produced five passing and four
held outputs out of nine. Those diagnostic outputs were **not published**.
The holds are retained transparently: one output violated its explicit country
omission contract and three copied source wording. This demonstrates rejection
of those failures, not universal first-call generation success. The approved
sixty outputs all pass the final instructions' schemas and semantic guards;
their explicit instruction-revalidation receipts preserve the previous request
hashes and confirm that rebinding did not change any values.

### Synthetic geometry

The final sixty documents contain 144 pages and 8,125 content lines. Of these,
**6,758 have coordinates: 83.18%**. Reflow improves the anchor-only count from
6,628 to 6,758, with **130 gained positions and zero lost anchors**.

| Line group | Positioned / total | Coverage |
|---|---:|---:|
| Goods-description lines | 66 / 75 | 88.00% |
| Generated address lines | 465 / 761 | 61.10% |
| Other content | 6,227 / 7,289 | 85.43% |

Sixty-five pages receive accepted reflow; the other seventy-nine have no
eligible reflow. No page receives a rejected partial geometry edit. Unknown
positions retain an explicit ` ||`; text and targets are preserved. Geometry
is source-conditioned layout signal, not a claim that these points are exact
glyph centroids of a separately rendered synthetic PDF.

All sixty samples pass ancestry, page bounds, coherent transform/reflow,
text preservation and label preservation checks. The final independent check
rejects 160 deliberate corruptions: eighty text/geometry mutations and eighty
label/render mutations. A second geometric audit rejects its own eighty
controls. These are separate checks of some repeated failure classes, not
240 distinct real-world defect types.

The [geometry gallery](analysis/dg-template-expansion20-20261014/geometry/GALLERY.md)
contains twenty-five plotted pages across ten selected sources. The montage
and a full-resolution page were visually inspected in addition to all-sample
numerical checks. The [geometry receipt](analysis/dg-template-expansion20-20261014/geometry/review.json)
records every sample, plot and preserved publication hash.

### Tests and measured processing impact

The final post-admission root run passes **982 tests across 22 targeted modules
in 32.63 seconds**. The preceding pre-admission run also passed all 982 tests
in 32.87 seconds. Ruff passes for the changed components and audit helpers.
See [test receipt](analysis/dg-template-expansion20-20261014/final-tests.json).
The previously blocked auxiliary-repair fixture now uses its matching immutable
source/label epoch; archived values are revalidated under current guards before
temporary receipt rebinding. Original caches and stale-request rejection guards
remain unchanged.

Measured relevant changes are small and bounded:

- Postal validation: 17.68 to 18.46 microseconds per valid address, an added
  0.78 microseconds; measured peak allocations slightly decrease.
- Final prompt-plus-schema characters across sixty generation requests:
  569,622 to 556,210, **2.35% smaller** than the original instruction format.
- Source-only mailbox preparation: 130.329 to 129.817 microseconds across the
  four applicable owners; peak allocations change by 47 bytes.
- Product-only alias work: 2.290 to 2.339 microseconds per invocation.
- Generic scenario output is bit-identical in the unchanged control cohort;
  median 0.1498 to 0.1495 milliseconds. Fuel-vehicle controls are also
  bit-identical; the measured change is about 0.5%.
- Complete-allocation control outputs are bit-identical over one hundred draws:
  18.525 to 19.053 milliseconds per hundred draws, about 5.3 microseconds
  additional work per draw. Peak traced allocations are unchanged at 1,970,931
  bytes, and the repeated timing ranges overlap.
- Packing-group style parsing: 18.419 to 16.841 microseconds, an 8.57% reduction.

These measurements describe the CPU paths and instruction footprint. They are
not a GPU training benchmark. No training was launched and the current training
dataset's five pinned files remain unchanged.

### Recorded API usage and cost

The final [accounting receipt](analysis/dg-template-expansion20-20261014/final-accounting.json)
reconciles every persisted provider response with the published manifest.
Its scope is this final 20-template/60-descendant campaign and its recorded
correction/probe calls. It excludes Codex session usage and any earlier staging
probes outside the campaign ledger.

| Stage | Calls | Input tokens | Output tokens | Recorded USD |
|---|---:|---:|---:|---:|
| Lexical generation | 20 | 75,492 | 12,244 | 0.01683140 |
| Contacts | 11 | 3,819 | 6,318 | 0.00368577 |
| Initial postal corrections | 8 | 9,699 | 3,980 | 0.00344485 |
| Audited postal recovery | 8 | 12,596 | 2,280 | 0.00301404 |
| Fresh compact-instruction probes | 3 | 14,353 | 2,399 | 0.00335245 |
| Complete-render reviews | 21 | 222,652 | 11,083 | 0.03750314 |
| Total | 71 | 338,611 | 38,304 | **0.06783165** |

Reasoning accounts for 16,581 of the 38,304 output tokens; it is not an
additional token category to sum again. Total recorded input-plus-output tokens
are 376,915. All-attempt costs include rejected corrections and experimental
probes, not just final accepted values. The semantic dispositions are 52 fully
accepted calls, eleven partially accepted batches, six rejected calls and two
superseded calls. These are call/batch dispositions, not document acceptance
counts; sixty final documents are approved.

Two in-flight requests from an interrupted early probe have no persisted
provider response or response ID. Their billing is **unknown**, not zero, and
is excluded from the recorded total. The displayed cost is therefore the
audited recorded amount, not an assertion of an exhaustive provider invoice.

### Registry publication and preservation

The [admission receipt](analysis/dg-template-expansion20-20261014/admission.json)
records successful promotion from **363 to 383** registered cases. It installs
the twenty new six-asset case directories, ownership and auxiliary declarations,
617-plus-60 source authority, and a reusable
[production validation config](../configs/synthesis/mpci_bl_curated_v7_dg_expansion20_validation.yaml).

The transactional publisher commits the manifest last, after fresh source
readiness, sixty preflights, sixty accepted-candidate replays and 1,089
pre-existing-template preflights. All old six-asset hashes are preserved.
Sixty-three old per-case source-authority metadata pointers were reconciled;
eight pointed to older target annotations even though their actual compiled
case assets already matched the current authority. Only their pointers change;
no old source text, target or coordinate asset is rewritten.

Seventy-four admission-helper tests pass, including rollback, stale evidence,
unauthorized metadata changes and independent plain/positioned input corruption.
The final dry-run takes 57.87 seconds and reports a 678.66 MiB process peak,
including application imports. It issues no model requests.

After the actual promotion, a separate
[post-admission regression](analysis/dg-template-expansion20-20261014/regression-after-admission.json)
passes all **363 old templates / 1,089 scenario plans** in 25.94 seconds,
with a 678.66 MiB process peak and zero paid calls. An independent live-registry
check also confirms all **383 registered cases' asset pins**, and recounts the
27 DG templates and their printed-field presentations shown above.

The sixty validated descendants remain in their own inspection publication;
they were **not silently appended to training**. Inspection materials:

- [Original sources and all rendered descendants](../artifacts/kie-synthesis-production/curated-v7-dg-expansion20-validation-v1/samples.md).
- [Position-enriched descendants](../artifacts/kie-synthesis-production/curated-v7-dg-expansion20-validation-v1/positions-reflow-v1/samples.md).
- [Plain decoder targets](../artifacts/kie-synthesis-production/curated-v7-dg-expansion20-validation-v1/dataset.jsonl).
- [Geometry plots](analysis/dg-template-expansion20-20261014/geometry/GALLERY.md).

Decoder targets are the extraction object itself, with no `schemaVersion` or
`documentPatch` wrapper. Internal source contracts retain their annotation
envelope; this does not change the training-output contract.
