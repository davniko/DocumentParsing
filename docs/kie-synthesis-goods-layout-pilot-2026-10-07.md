# Goods-description synthesis and bounded coordinate placement

## Result and inspection files

The implementation and pilot are complete: **24 full synthetic shipments from eight
reviewed sources**, plus **nine goods-only variations from three long-description
sources**. The full 24 were rendered, reviewed, corrected, reviewed again, published,
and position-enriched. The nine focused probes exercise the same wording builder,
native PydanticAI output, target assembly, renderer and coordinate functions.

- [All 11 source OCRs followed by their three variations](analysis/synthesis-goods-layout-pilot-20261007/SOURCE_AND_VARIATIONS.md).
- [Geometry and preservation measurements](analysis/synthesis-goods-layout-pilot-20261007/validation.json).
- Geometry plots: [laboratory](analysis/synthesis-goods-layout-pilot-20261007/geometry-laboratory.png),
  [compressors](analysis/synthesis-goods-layout-pilot-20261007/geometry-compressors.png),
  [telecom](analysis/synthesis-goods-layout-pilot-20261007/geometry-telecom.png).
- [Production pilot configuration](../configs/synthesis/mpci_bl_curated_v7_goods_layout_pilot24.yaml).
- Published plain dataset: `artifacts/kie-synthesis-production/curated-v7-goods-layout-pilot24-v1/dataset.jsonl`.
- Published coordinate dataset: `artifacts/kie-synthesis-production/curated-v7-goods-layout-pilot24-v1/positions-v2/dataset.jsonl`.

The gallery links each variation's coordinate-bearing text, labels and geometry
receipt. Original training/validation JSONLs remain unchanged; neither these probes
nor the pilot have been added to training automatically.

The long compressor/laboratory/telecom sources have reviewed goods spans, not complete
new whole-shipment ownership contracts. Those nine outputs are deliberately marked
**goods-only**, not passed off as another nine fully resampled shipments. All their
non-description labels and non-owned input bytes remain at the source baseline. The
compressor probe baseline projects its description to the actual OCR's list punctuation;
both the original and probe-baseline targets are saved. No real labels were edited.

## Policy and implementation

The objective is realistic, variable extraction examples, not teaching tariff
classification. These findings supersede the earlier probe's strict HS/product-scope
rejections. Multiple related products, fictional brands/models/serials and technical
specifications are permitted within a single accounting goods group. Printed HS codes
must still equal the sampled labels. DG, thermal facts and accounting remain coherent.

1. The host samples commodity identities, routes, localities, equipment, packages,
   counts and measures through the existing scenario sampler.
2. `wording_request` groups product variables by their actual description-target
   expressions. Three continuation fragments are not interpreted as three HS codes.
3. The model receives the **complete set of owned source product fragments**, a compact
   commodity brief and explicit assembly, plus host-owned physical facts as compatibility
   context. Broad sibling/national tariff trees are no longer supplied to wording.
4. Native structured output requires one string per requested region, not spans,
   rationale, hashes or new accounting labels. Repeated occurrences reuse the same value.
5. Natural paragraphs and list boundaries are retained. There are **no character or
   line-count quotas**. Product specifications can include numbers; shipment totals,
   packing/fill quantities and transport settings belong to separately controlled facts.
6. Deterministic assembly produces the description label from the exact generated
   fragments, normalized under the configurable target casing policy. Customs headings,
   totals and unrelated text between fragments are not absorbed into the description.
7. Rendering uses byte-owned edits, followed by final-text semantic review and explicit
   correction. Publication verifies current requests, targets, receipts and review hashes.

The production builder also supports a `goods_only` request for focused experiments.
This uses the same builder/renderer, not a different simplified prompt masquerading
as the production path. The full 24 used complete party/contact generation as well.

### Diversity observed in the full pilot

- 19 origin countries/territories; 22 destinations. Egypt is not pinned as destination.
- 10 package categories, including cartons, bags, plastic bags, bins, rolls, pallets,
  cases, boxes, generic packages and intermediate bulk containers.
- 66 containers: 18 twenty-foot standard GP, 17 forty-foot standard GP, 28 forty-foot
  high-cube GP, and three forty-foot high-cube refrigerated.
- Ambient, DG and frozen families; repeated goods, four-fragment descriptions,
  repeated multi-page shipment tables and separate customs/handling text.

These are the combinations supported and sampled by this pilot, not a claim that
every registry equipment/package combination was tested.

### Longer goods results

| Source | Source lines by owned region | Three variants' rendered lines | Result |
| --- | --- | --- | --- |
| Compressors | 18 | 14; 14; 13 | Distinct equipment lists with fictional models/serials and technical qualifiers |
| Laboratory | 11 / 43 / 1 | 5/17/3; 6/28/2; 7/28/3 | One description assembled across three pages; intervening material preserved |
| Telecom | 19 | 14; 15; 15 | Distinct component lists, model references and batch wording |

All 15 generated long-goods regions obtained usable positions. These counts are
observations, not imposed limits. The small continuation can grow without repeating
one coordinate or forcing the content into the original line count.

## Coordinate algorithm and what its validation means

Positions are **coarse layout anchors**, not estimates of exact synthetic glyph boxes.
We do not render a synthetic PDF or assert the new strings' physical font widths.

- Verify source/Paddle hashes and edit provenance before using geometry.
- Retain known source anchors when the line count is unchanged.
- For a contracted single line, use the original region centre.
- For changed multi-line spans, use measured source line height, vertical order and
  same-column neighboring regions. Interpolate when spacing is sufficient.
- When more local space is needed, reflow within the owned span plus at most half
  the adjacent gap. Neighboring changed regions cannot each claim the entire gap.
- Missing anchors, intersecting unowned text or insufficient space produce **explicit
  unknown coordinates**. Text is never deleted or squeezed into duplicated points.
- Finally apply the existing coherent per-page scale/translation. All known lines
  share that transform. Integer bounds, axis order/alignment and complete nearest-neighbor
  sets, including ties, must remain valid. No independent per-line jitter is used.

The production pilot published all 24 records / 48 pages; all 48 pages used a valid
scale-and-translation transform and all 2,692 known points changed. Labels and plain
OCR remain unchanged by this stage.

### Coverage across all 33 outputs

| Outcome | Content lines |
| --- | ---: |
| Known coordinates | 3,980 |
| Unknown because source position was unavailable | 769 |
| Unknown because local space was insufficient | 110 |
| Unknown because unowned source text intersects the region | 30 |
| Total | 4,889 |

Coverage is **81.4%**. The 140 geometry-driven abstentions are **2.9% of output lines**;
they are mostly outside the focused long-goods regions. This is an intentional loss
of positional coverage, not missing text or labels. The 24 full shipments alone have
2,692/3,454 positioned lines; the nine focused probes have 1,288/1,435.

### Falsification and preservation checks

- Exact source-to-output byte replay for all 33 documents; no unrelated edits.
- Coordinate suffix removal recovers the complete original rendered text.
- Page ownership, integer bounds, spacing, envelopes and page transforms checked.
- Five independent augmentation seeds on all 33 outputs: **165 document replays**.
- Injected collapsed coordinates rejected in **15/15 applicable documents**.
- The previous laboratory failure expanding **one line to 22** now abstains for
  insufficient space rather than placing 22 lines at one point. Its text survives.
- Unit tests inject corrupted hashes, shifted ownership, invalid geometry, copied
  descriptions, missing schema fields, repeated descriptions and invented totals/captions.
- Real pixel normalization is exercised: a floating-point operation-order mismatch
  previously made source-owned boxes look foreign; the fixed path retains their identity.
- Plots inspected for the three long-description sources. Gray rectangles show measured
  source layout (transformed on the synthetic side), not newly measured synthetic boxes.

## Problems found and addressed during the pilot

The first outputs were **not** accepted solely because their JSON parsed.

1. Some generated descriptions invented packing methods or unit fill masses at odds
   with sampled packages. Physical context and field ownership were clarified. Goods
   were regenerated without regenerating the accepted companies/contacts. One residual
   PVC packing claim was removed with an exact correction receipt.
2. Frozen descriptions introduced unsupported cut/processing details and malformed
   wording. Three were corrected to retain the actual donor commodity/form while allowing
   fictional brand/product-line variation. Final complete-text reviews accepted them.
3. A laboratory response contained drafting commentary; it was superseded, not published.
   Another returned three identical variants. The general batch validator now rejects
   duplicate complete generated descriptions, including when cached/new results are mixed.
   One bounded corrective call produced the distinct variants shown in the gallery.
4. A short source word was mistakenly usable as a character-width proxy during the
   experiment, creating excessive wrapping. That approach was removed. Natural generated
   line boundaries now determine goods layout, and geometry decides positionability.
5. A measured-box normalization mismatch was fixed; uncertain boxes are not silently
   rounded into a claim of ownership.
6. Final shipment review found one generated Brazilian state inconsistent with its sampled
   town. The state/state-road claim was removed from that party's generated address, keeping
   the sampled locality/country and other postal components. This is not a postal-deliverability
   audit. Earlier explicit postal corrections remain in their receipts.
7. The frozen source had a separate `FREIGHT PAYABLE IN NEW YORK` auxiliary span left
   fixed while the main payment-place field was sampled. Its **existing template binding**
   now shares the payment-place target and retains the caption. Future descendants update
   both occurrences consistently; bank/third-party names elsewhere stay source context.

Five final manual wording/postal corrections are recorded in
[manual-corrections.json](analysis/synthesis-goods-layout-pilot-20261007/manual-corrections.json).
The larger goods-only refresh preserved all non-goods values exactly and retained
pre-edit receipts. Final full-pilot review: **24/24 covered, zero outstanding findings**.
These were reviewed/corrected results, not a 100% first-pass generation success claim.

## Tests, cost, performance and side effects

The relevant synthesis/equipment suite passed **226 tests**. Ruff passed for the
changed production modules and their tests. All final publication and coordinate
stages were exercised through their actual CLI, not just helper functions.

API ledger total for this experiment, including superseded outputs, corrective calls
and re-reviews: **$0.06443112** (about **6.44 US cents**).

| Stage family | Cost USD |
| --- | ---: |
| Full wording, including earlier attempts | 0.01561890 |
| Company contacts + one syntax correction | 0.00130185 |
| Postal corrections | 0.00287215 |
| Goods-only refresh of full shipments | 0.00420020 |
| Complete rendered reviews/re-reviews | 0.03067848 |
| Long-goods probes and one duplicate correction | 0.00975954 |

This is a development/pilot ledger, not a reliable 10k-document production-price
projection. Cached replays incur no new API charge. No higher-reasoning model was needed.

Measured coordinate transfer median: about **1.49 ms before → 1.75 ms after** on the
33 real-source variations; total measured transfer about **53.6 → 65.7 ms**. The measured
space checks add about 0.26 ms/document in this probe. This is an explicit, small CPU
cost for stronger geometry validation, not a throughput-speedup claim. Full 24-record
rendering took **1.24 s**; final production positioning including geometry/provenance
checks and publication took **1.59 s**. The complete offline 33-document audit with
five-seed stress tests and plots took **10.66 s**, peak RSS **251.2 MiB**, zero swaps.
Timings vary slightly per replay; exact transfer observations are in `validation.json`.

Existing published pilots were not overwritten. Current real train/validation inputs
and labels were not edited. Broader source contracts and historical training configs
were not promoted to readiness. The source-specific freight correction is declared
template data, not a special-case code branch. The scientific plausibility of every
invented commercial specification and real-world postal deliverability are not certified.

## Recommended next use

The implemented route is suitable for the reviewed pilot families: compact briefs,
complete goods context, joint fragment generation, deterministic facts/labels, bounded
layout placement, final semantic review and receipt-bound publication. Keep the same
gates when expanding; do not infer whole-shipment readiness for a new family from a
successful goods-only probe. The next source expansion should compile/review its
complete ownership first. Position coverage can be measured separately from text
quality; unknown coordinates are an explicit feature of the input contract.

This pilot verifies usable geometry and consistent text/label assembly. Whether the
additional layout diversity improves extraction F1 requires the later training experiment;
it is not established by geometric checks alone.
