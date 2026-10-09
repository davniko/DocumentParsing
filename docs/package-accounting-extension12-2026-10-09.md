# Packaging and description validation: 12 additional templates

Date: 2026-10-09. Follow-up to the [eight-template repair pilot](package-accounting-repair-2026-10-09.md).

## Scope and selection

Twelve additional templates, two freshly generated variants each: 24 new validation samples. None overlaps the preceding eight-template pilot. This is deliberately risk-stratified coverage, not a prevalence estimate from random sampling. The pilot remains separate from training. A subsequent [footer-only repair](external-reference-suffix-repair-2026-10-09.md) updated seven existing synthetic training inputs and two pilot inputs; all labels and coordinates remain unchanged.

Configuration: [pilot.yaml](analysis/package-accounting-extension12-20261009/pilot.yaml). All selected sources belong to the active reviewed 200-template catalog. Offline source-readiness and all 24 scenario preflights passed before generation (5.90 seconds, peak RSS 439.77 MiB, no paid requests).

| Source prefix | Why selected / expected invariant |
|---|---|
| `06a7c8b2` | Keep adjudicated product prefix `1012322163DXH DEGREE 1`; exclude preceding table headers and following HS/customs/LC fields; 25 pallets belong to one container. |
| `9d4a6b90` | Full induction-liner model/specification list; exclude preceding `10 PALLET OF`; repeated pallet declarations must agree. Attachment party continuation is not additional goods text. |
| `1f166783` | Preserve embedded `(5 GALLON)` product capacity and part-number wording; exclude separate STC, HS and purchase-order lines. One crate supports one sole-container allocation. |
| `1b796092` | Awkward OCR product continuation around three container identifiers; 143 declared pallets with membership-only placements. No per-container quantities may be invented. |
| `7f8fafe7` | Negotiable yarn shipment with two containers; 860 cartons total but no printed allocation quantities. Marks-column carton ranges must not leak into product description or authorize calculated splits. |
| `3dc8551d` | Six containers and repeated pages; embedded sheet quantity is product wording, distinct from 99 declared packages. Membership-only placements remain quantity-free. |
| `30ad41fb` | Vehicle transshipment, product serial wording and handling instructions. Repeated `1 UNIT` is one declared package, not additive repeats. |
| `2763b891` | Unpacked vehicle with no container: no equipment or placement invented; Marks-column chassis wording remains outside description. |
| `21d3ced7` | Two-container dangerous goods with equal allocations, dedicated UN/class fields and notify same-as instruction. Repeated proper shipping name appears only once in description. |
| `1464f450` | Chilled apples, wooden boxes, temperature and ventilation; product wording must stay distinct from thermal instructions and package counts. |
| `719e3e96` | Flattened Marks / `3 PALLETS S.T.C.` / pastry-bag product block; bags are the commodity, not the declared package level. Preserve product-adjacent non-stackable qualifier. |
| `9bd26a2b` | Complete pigment product list, excluded exporter identifier and unequal 600/1,000-bag container allocations. |

The source boundaries were checked against the current OCR/targets before generation. Generated identities, product models and capacities may vary; the invariant is correct ownership and preservation of the generated product wording, not copying the literal source product values.

## Validation plan

1. Use the production sampler, wording/contact generation, rendering and semantic-review stages with the current policy and exact pinned registries.
2. Check all rendered samples against their labels, emphasizing description boundaries, declared package types, repeated totals and printed-vs-absent allocation quantities. Review the complete text as well as numeric inventories.
3. Resolve findings with explicit evidence; any changed candidate must be replayed and reviewed again. Keep failed attempts and charges in the ledger.
4. Publish only the complete reviewed cohort; synthesize positions through the current source-anchor/reflow/augmentation implementation.
5. Check text/target preservation, coordinate bounds, source-envelope ownership, ordering and coherent transforms. Falsify with deliberately corrupted records; inspect plotted layouts.
6. Verify the live training data and prior pilot remain unchanged. Report costs, failures/corrections, geometry coverage and exact artifact links.

## Completed result

**24/24 new samples completed generation, rendering, semantic review, replay validation and position publication.** Together with the preceding pilot this gives **40 samples from 20 distinct templates**. The expanded test found no unresolved description-target, declared-package-level or allocation mismatch in its final samples. It did require the bounded corrections described below; this was not a flawless first generation attempt.

Every sample's complete rendered text and target was supplied to the semantic reviewer, not just a numeric inventory or excerpt. Final review receipts bind the exact candidate hashes and contain zero findings. The root and an additional code-agent review supplied another inspection layer; [adjudication notes](analysis/package-accounting-extension12-20261009/semantic-adjudication.md) specify its coverage and the **untargeted auxiliary-footer limitation** it found. That limitation was subsequently repaired as described below. The two changed candidates have explicit manual exact-delta review receipts preserving the original full-text model reviews; no new model review is claimed.

### Packaging and description evidence

The cohort contains **42 containers and 42 placements**: **20 quantified allocations** and **22 membership-only allocations**. All quantified, completely declared splits sum to their respective shipment package totals. All membership-only placements remain quantity-free. Two vehicle documents correctly contain no containers or placements. Four documents remain negotiable; explicit notify-party same-as behavior is inherited correctly.

| Source | Concrete generated result / observation |
|---|---|
| `06a7c8b2` | Nine packages or 24 pallets, each assigned to the sole container. New complete product lists exclude the preceding flattened headers and following customs/reference fields. |
| `9d4a6b90` | 29 bundles or 496 cartons; repeated declarations agree. The original leading `10 PALLET OF` accounting remains outside the description. |
| `1b796092` | 7,817 packages or 39 packages across three identified containers. Both owned product fragments are retained, but no per-container count is invented. |
| `7f8fafe7` | 942 cartons or two wooden cases across two containers; membership-only relations stay quantity-free despite the tempting shipment total. |
| `3dc8551d` | 2,916 cartons or 248 bundles, repeated consistently across four pages; six container memberships per document remain unquantified. Product specifications do not replace the declared accounting unit. |
| `30ad41fb` | Eight packages or five packages, with matching sole-container allocations; complete long machinery descriptions retain specifications. Repeated totals are not added together. |
| `21d3ced7` | DG allocations of 1,820 + 1,820 = 3,640 cartons and 86 + 85 = 171 cartons. Registry UN/class fields and printed proper shipping names remain aligned. |
| `1464f450` | 905 cartons of chilled fish or 1,695 cartons of chilled duck products. Reefer equipment, 0 °C settings and printed 0 CBM/h ventilation agree. Embedded product packing specifications remain in description; separate handling settings retain their own fields. |
| `719e3e96` | Eight or 13 declared pallets; the generic `PACKAGES` footer remains consistent with the pallet count. The commodity's bag/spool wording does not become an extra package category. |
| `9bd26a2b` | 23 pallets split 9 + 14, or three packages split 1 + 2. Complete generated product lists exclude the exporter-ID block and separate totals. |

Description lengths range from 23 to 1,857 characters. This includes short DG names, long enumerated equipment lists, product specifications, split OCR passages and repeated pages. This confirms the tested boundary/accounting behavior; it is not an engineering certification of every fictional product specification or a universal guarantee for unseen templates.

### Problems caught and resolved

1. **Generated shipment-accounting text inside a product passage** (`3dc8551d`): the existing generation guard rejected it, and the bounded wording-repair call resolved it before rendering.
2. **Generated equipment masses exceeded the host shipment gross weight** (`30ad41fb`): the guard blocked publication. Its first automatic correction still failed. A scoped product-only correction made the first variant valid (23,610 kg of item masses versus 26,286 kg shipment gross). The second still exceeded its 16,429 kg gross, so explicit manual adjudication removed five optional, newly invented item-mass assertions from the *synthetic wording proposal*. Product models, serials, dimensions, capacities and the host-controlled shipment facts remain. Both input and label are generated from the corrected wording. This is not retrospective deletion of printed weight information from a real description label. [Exact receipts](analysis/package-accounting-extension12-20261009/generation-hold-resolution.json), [five-assertion decision](analysis/package-accounting-extension12-20261009/manual-product-mass-adjudication.json).
3. **Postal locality/repetition holds**: six variants across four sources received scoped address correction. One resulting address still repeated `MEXICO` through an optional `CIUDAD DE MEXICO` phrase; that optional generated city phrase was removed by explicit adjudication, retaining the sampled locality/country and street/district. The final postal checks pass. No real address input was rewritten.
4. **Tyre wording ambiguity** (`1b796092`, variant 1): the reviewer flagged `TUBELESS` alongside an inner-tube continuation. The synthetic proposal now says `TUBE TYPE`; all other product text is preserved. The renderer updated both input and label, and a fresh review of the changed candidate passed. [Before-edit records and decision](analysis/package-accounting-extension12-20261009/semantic-adjudication.md).

These events demonstrate rejection and recovery, not a guarantee that generation cannot propose an error. No new document-specific rule was added to production code. The original failed-call ledger and generation summary remain intact rather than being rewritten to imply first-pass success.

### Auxiliary issue: resolved in a subsequent narrow repair

Both `9bd26a2b` variants originally retained the source footer `CN-02-91320282050282384L` under `ADDITIONAL EXTERNAL REFERENCES`, although the mutable shipper/exporter ID was resampled. The historical composite-reference binding prevented the nested shorter identifier from independently replacing this suffix. This was an incomplete-resampling issue in an **excluded target field**, not a package, allocation or description error. Following the user's clarification that the prefix may remain literal flavor text, the live template now binds only the suffix to the shared exporter ID. Both pilot inputs and all seven affected current synthetic training inputs are repaired. Labels, coordinates, the prefix and unrelated text are unchanged. See the [repair report and exact receipts](external-reference-suffix-repair-2026-10-09.md).

Some unrelated carrier/registration boilerplate also remains source-fixed by design. The pass does not claim to regenerate every printed identifier. Generated email/website pairs were absent in these twelve families, so this extension adds no new live contact-domain coverage; the existing independent-domain regression tests still pass.

## Coordinates and geometric validation

All **3,249 nonempty lines across 24 documents / 54 pages** were checked independently against the serialized geometry receipts. Plain OCR and labels are exactly preserved in every positioned record.

| Line cohort | Lines | Anchor-only positioned | Final positioned | Lost positions |
|---|---:|---:|---:|---:|
| Goods-description lines | 192 | 14 (7.3%) | 143 (74.5%) | 0 |
| Address lines | 283 | 121 (42.8%) | 159 (56.2%) | 0 |
| Other lines | 2,774 | 2,432 (87.7%) | 2,432 (87.7%) | 0 |
| **All** | **3,249** | **2,567 (79.0%)** | **2,734 (84.1%)** | **0** |

Reflow adds **167** supported positions, including 129 goods lines. Twenty-seven page reflows pass; 26 pages have no eligible region. One proposed reflow, the expanded eight-machine description, is rejected because the resulting median glyph height would fall below the calibrated density floor (6.677 versus 6.815 normalized units). That page retains its known anchor positions and explicit ` ||` for unknown lines. No text is dropped and no unsafe partial reflow is committed. The five-machine variant fits and receives ordered product coordinates.

Other missing-coordinate regions are explicitly held for incomplete source anchors (18 groups) or foreign text inside the ownership envelope (six groups). There are 515 unknown lines in total; their existence is not disguised as geometric coverage. Two pages have no known source anchors. All remaining pages receive coherent scale/translation where supported.

The independent check verifies line identity, serialized position consistency, page bounds, generated-line ordering, source-envelope ownership and coherent transforms. It rejects four deliberate corruptions: changed text/line count, off-page coordinates, duplicate receipt lines and an inconsistent source anchor.

The gallery contains **all 24 documents / 54 plotted pages**. The root visually opened nine page plots covering the rejected and accepted long-machinery variants, repeated six-container goods, split descriptions, chilled goods, DG, and unequal container splits. They show the intended source-conditioned ordering and layout separation. These are plausible synthetic envelopes, not measured bounding boxes from synthetic PDFs.

## Tests, performance and preservation

- **593 tests passed in 32.15 seconds**: curated campaign/sampling/templates/contacts/publication/positions/packaging/descriptions, direct labeling, and the exact dataset-repair regression suite.
- **94/94 integrity corruptions rejected**: 24 unprinted description edits, 24 wrong package totals, 24 unauthorized OCR edits with recomputed hashes, and 22 wrong or invented allocation quantities. These prove tamper/replay enforcement; semantic correctness is additionally checked by source policy and rendered-text review, not inferred from replay alone.
- **4/4 geometry corruptions rejected** independently.
- Final replay/publication-precondition inventory: 7.03–7.08 seconds across two checks, peak RSS 820.74 MiB. Geometry baseline/probes: 6.65 seconds, peak RSS 477.98 MiB. Position synthesis: 1.70 seconds. Full 54-page plot/coverage audit: 1.96 seconds. Final rendering: 0.94 seconds for 24 samples. These are measured operation runtimes after Python imports, not a before/after speedup claim; no production hot path was changed in this extension.
- `git diff --check` passes.
- At initial pilot completion, the 2,100-train / 60-validation dataset files and the preceding 16-sample pilot matched their pre-test hashes. The subsequent footer repair changes seven existing synthetic training inputs, with its own exact before/after receipt; it leaves validation and the preceding pilot untouched. No samples were merged into training, no training configuration changed, and no training started.

[Exact integrity/count/hash receipt](analysis/package-accounting-extension12-20261009/validation.json), [geometry negative controls](../artifacts/kie-synthesis-production/package-accounting-extension12-20261009/audit/geometry-independent-probes.json), [full geometry validation](../artifacts/kie-synthesis-production/package-accounting-extension12-20261009/positions-reflow-v1/audit/validation.json).

## Recorded API cost

All 32 calls, including failed proposals, repairs and the changed-candidate rereview, total **$0.02676085** (about **$0.001115 per final sample**). This is the external API ledger, excluding human/Codex work and local compute. Manual adjudication time means it should not be presented as an unattended bulk-production estimate.

| Stage | Calls | Cost, USD |
|---|---:|---:|
| Initial wording | 12 | 0.00899990 |
| Guard-triggered wording repairs | 2 | 0.00236945 |
| Scoped product correction | 1 | 0.00112820 |
| Scoped postal corrections | 4 | 0.00081700 |
| Complete rendered-text review, including rereview | 13 | 0.01344630 |
| **Total** | **32** | **0.02676085** |

## Inspection artifacts and conclusion

- [Original OCR followed by both rendered variants for each source](../artifacts/kie-synthesis-production/package-accounting-extension12-20261009/samples.md).
- [All position-enriched samples](../artifacts/kie-synthesis-production/package-accounting-extension12-20261009/positions-reflow-v1/samples.md).
- [Geometry gallery](../artifacts/kie-synthesis-production/package-accounting-extension12-20261009/positions-reflow-v1/audit/GALLERY.md).
- [Plain dataset publication manifest](../artifacts/kie-synthesis-production/package-accounting-extension12-20261009/manifest.json).

The extended test supports the repaired goods-description boundaries and declared-package/allocation policy on twelve additional, deliberately difficult families. The safeguards correctly reject unsupported shipment facts and geometrically unsafe expansion; scoped review/adjudication remains necessary. The disclosed composite-reference footer has also been repaired without changing any targets. These artifacts are a completed validation cohort, kept separate from production training data.
