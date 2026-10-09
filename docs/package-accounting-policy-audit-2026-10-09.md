# Package accounting audit: declared shipment units, not an inner/outer default

Date: 2026-10-09. Status: **audit and proposed repair only**. No dataset, template, production prompt or implementation was changed by this pass. The preceding description repairs remain intact. No model inference, generation or training was launched.

Subsequent implementation: the approved proposals and sole-container follow-up are now applied. See the [repair, validation and fresh-synthesis report](package-accounting-repair-2026-10-09.md). The counts below remain the original audit's proposal inventory, not the later publication total.

## 1. Outcome

The concern is substantiated. The current extraction schema and cargo reviewers explicitly prefer **innermost shipment packages**. That makes some labels follow the contents of the shipping units, while the document's package column and container rows count the units actually declared for carriage. Synthesis then faithfully varies the selected inner count, preserving the wrong accounting choice.

The recommended policy is:

> Extract the document's declared shipment package count and type. Keep container allocations in that same accounting unit. Treat contained packing and product capacities as separate facts, not replacement shipment counts. Do not assume that either the innermost or outermost physical layer is always the declared unit.

Concrete, reviewed correction proposals:

| Cohort | Audited inventory | Identified proposed label changes | Share |
|---|---:|---:|---:|
| Real training | 600 | 32 | 5.33% |
| Real validation | 60 | 5 | 8.33% |
| Synthetic training | 1,500 | 29 | 1.93% |
| Total | 2,160 | **66** | 3.06% |
| Active template catalog | 200 source families | **4** need accounting-contract changes | 2.00% |

The 29 descendants comprise **five from the original 500** and **24 from the next 1,000**. This is a bounded repair set, not a recommendation to regenerate the whole dataset.

The 37 real proposals change 31 package rows to pallets, four to generic packages representing mixed handling units, one to a skid, and one to cartons. The target count remains unknown in one pallet case because OCR omitted the count. Proposed container-quantity changes are:

| Cohort | Replace existing count | Add previously omitted supported count | Remove unsupported-level count |
|---|---:|---:|---:|
| Real training | 27 | 5 | 1 |
| Validation | 2 | 9 | 0 |
| Synthetic | 29 | 0 | 0 |

These counts concern quantity leaves, not documents. Container identities and memberships are unchanged.

There are also genuine source contradictions and incomplete OCR. They are listed separately below and in the casebook. They must not be silently “fixed” through arithmetic or by importing PDF-only values.

## 2. Evidence, coverage and limitations

Active dataset: [real600 + synthetic1500](../data/curated/mpci-bl-real600-synthetic1500-v7-positions-v1/). Training has 2,100 rows; validation has 60. Synthetic.jsonl is a mirror of the synthetic training cohort, not another 1,500 independent examples.

Active catalog: [200 reviewed templates](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/).

The investigation combined:

1. A full inventory of package rows and container allocations across all 2,160 records.
2. Comparison with the preceding frozen audit, verifying that package facts had not changed during the description-only repair.
3. Counted-package discovery, a separate uncounted/reversed/capacity-interrupted word scan, missing-total checks, mixed-row checks and total/placement checks.
4. Review of the initial 83 real multiple-counted-word candidates; all have explicit change/retain/conflict dispositions. Additional scans found six proposed corrections outside that queue: train159,195,256,484 and validation47,49.
5. Targeted inspection of source OCR, the 149 synthetic counted-word contexts, and family closure through all 29 descendants of the four affected active sources. Broader lexical scans also checked uncounted packing and product/package-name ambiguity.
6. PDF layout inspection for the difficult accounting-column cases. Sixteen new PDF page previews were inspected, plus three existing validation previews. PDF establishes ownership/layout, not permission to add absent OCR values.
7. Inspection of the real execution path: Pydantic field descriptions, cargo mapping/review, curated source contracts, sampled quantities/categories, allocation apportionment, physical scaling, and auxiliary rendered facts.
8. An in-memory proposed repair and schema/non-interference checks. No proposed target was published.

Artifacts:

- [Full inventory and OCR occurrences](analysis/package-accounting-policy-audit-20261009/inventory.json)
- [Every proposed real change, retained control, OCR and PDF link](analysis/package-accounting-policy-audit-20261009/casebook.md)
- [Exact before/after package and allocation proposals, including all descendants](analysis/package-accounting-policy-audit-20261009/proposed-changes.json)
- [Affected source families and all descendant IDs](analysis/package-accounting-policy-audit-20261009/affected-families.json)
- [Source limitations and conflict dispositions](analysis/package-accounting-policy-audit-20261009/source-limitations.json)
- [Read-only inventory script](analysis/package-accounting-policy-audit-20261009/audit.py), [adjudication/dry-run script](analysis/package-accounting-policy-audit-20261009/adjudicate.py), [regression probes](analysis/package-accounting-policy-audit-20261009/test_audit.py)

**What the scope means:** 66 is the exact enumerated proposal set identified and checked here. It is not an assertion that every other field in all 2,160 documents has been manually recertified, or a mathematical guarantee that lexical discovery finds every possible semantic ambiguity. The full scans are coverage/discovery; the reviewed accounting decisions authorize the proposals. These are deliberately separate claims.

## 3. What the target form and external specification establish

The local [MPCI field catalog](../artifacts/mpci-ai-schema/field-catalog.md) separates:

- `goodsItemDetails[].numberAndTypeOfPackages[]`: quantity and package type.
- `goodsItemDetails[].splitGoodsPlacement[]`: container foreign key and optional package quantity.
- Goods description, weight/volume and dangerous-goods fields.

The placement has **no package-row foreign key or explicit packaging-level identifier**. Therefore a scalar placement count must have a coherent interpretation relative to the selected goods package accounting. A total in bags and an allocation in pallets is not self-describing in this target.

Several package rows are representable, but this alone does not encode a nested graph. `5 pallets + 300 bags` would double-count nested levels if treated as additive shipment packages. Conversely, `12 IBCs + 8 pallets` can legitimately represent 20 separate shipping units.

NAIC's public business specification describes package type/quantity as the packages carrying the goods, and container-goods placement as the relationship to containers. The reviewed business-data table does **not** prescribe our existing universal innermost preference. This is not a claim that the entire MPCI implementation guide has no additional constraints: the complete message implementation profile was not available in this audit. [NAIC business specification, table10, printed page28](https://naic.icp.gov.ae/portal/assets/mpci-guidelines/UAE%20MPCI%20Business%20Specification%20Document%20V1.0.pdf).

CUSCAR explicitly distinguishes goods package particulars (`GID`), packaging level (`PAC`) and container placement (`SGP`). That supports distinguishing levels; it does not justify flattening all levels into one additive total. [UNECE CUSCAR D.96A, segment group13](https://service.unece.org/trade/untdid/d96a/trmd/cuscar_d.htm).

Accordingly, **declared shipment accounting** is the recommended extraction convention for this simplified target, grounded in the source and compatible with the inspected form. It should not be misrepresented as an externally mandated “always outermost” customs rule.

## 4. Taxonomy and concrete cases

### A. Inner contents replaced an explicit shipment declaration

Validation028 declares 100 pallets and five container rows of20 pallets. Product detail mentions4,000 paper bags. Current gold keeps4,000 bags and omits all five allocation quantities. Proposed:100 pallets and20 per container. [OCR](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_8625ddda8e039b329e706ec7bf9cd50cd059c7e0874c42ea0bf3358716d5e653/ocr.txt) · [PDF](../data/corpora/paddle-bl-swb-filtered-max5/files/5339e00e8da119e19dfb735baeeabdb1e89d06915fc52114315dda1a515b6d97.pdf).

Validation049 similarly has three20-pallet rows,60 pallets total and2,400 PE bags as contents. Current gold keeps the bags and drops the printed container counts. Proposed:60 pallets,20/20/20. [OCR](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_68b8d417e52c765166df5d27ad53807992af5d07ec337b63f4d3d7b84f16af6f/ocr.txt) · [PDF](../data/corpora/paddle-bl-swb-filtered-max5/files/573716198d7ddd1c03a12a311b0eaaf01ab3aaaf6e5cc3e6415ef562de7746ef.pdf).

Train363 is the same problem even though all old arithmetic balances:167 cartons=128+39; declared21 pallets=16+5. The correct accounting unit must be established before arithmetic can validate it.

Train057 declares ONE(1)PALLET ONLY. Its product block has8CTNS=5062PCS. The package column's1PALLET was omitted in OCR, but the total survived, so the proposed one-pallet target is OCR-grounded. It is not derived from the PDF alone.

### B. An uncounted inner word erased a fully printed outer declaration

Validation047 has `1 Container Said to Contain 1400 CARTONS`, repeats the carton total and provides a1,400-carton container row. Product text says bags packed in cartons. Current target is just `PACKAGE_BAG`, without total or allocation. Proposed:1,400 cartons and1,400 in the container. [OCR](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_684f442c38b6f4943745b7bc51f7db2cc92ee329f8801202d40359b3a192f6e2/ocr.txt) · [PDF](../data/corpora/paddle-bl-swb-filtered-max5/files/042218d78ede4b4823e49b95e739cd9f8e276b118257d740b59667925cdad73b.pdf).

Train195 likewise labels uncounted steel drums although210 pallets are declared. Its printed zero-pallet container rows are contradictory source data: use210 pallets at goods level, preserve membership, and leave individual quantities unknown. Do not divide210 by7 or treat the zeros as reliable empty-container assertions.

This is why searching only for two adjacent `number + package noun` phrases is insufficient.

### C. Mixed loose and palletized units: neither inner-total nor pallet-total alone

Four training cases have an explicit generic shipment count over a mixture:

| Train row | Printed hierarchy | Current target | Proposed target |
|---|---|---|---|
|180|19 pallets containing728 cartons +3 loose cartons =22 packages|731 cartons|22 packages|
|323|16 pallets containing606 cartons +5 loose cartons =21 packages|611 cartons|21 packages|
|354|16 pallets containing610 cartons +4 loose cartons =20 packages|614 cartons|20 packages|
|564|16 pallets containing620 cartons +16 loose cartons =32 packages|636 cartons|32 packages|

All are single-container allocations at that explicit generic count. Do not relabel the aggregate as all pallets. Preserve the explicit aggregate as `PACKAGE_PACKAGE`; separately listed component types can remain contextual/audit facts rather than manufacturing nested target rows.

### D. Genuine same-level mixtures should remain mixtures

Eight real training records currently have multiple package target rows. Several are clearly legitimate additive groups:

- Train426:12 IBCs +8 pallets, with20 shipment units in its container.
- Train482:1 unpacked unit +2 pallets.
- Train138:192 IBCs +104 packages.
- Train142:2,383 cartons +633 pieces.
- Train210/389:packages and rolls accounted as distinct same-level portions.

These are not multiple goods merely because their packaging differs, and they are not automatically hierarchy errors. Do not collapse genuine disjoint groups or add nested groups.

### E. Bags/drums/pieces can legitimately be the declared unit

Controls falsify a blanket “use outer pallets” correction:

- Train021:five drums are the declaration; two pallets describe supporting packing.
- Train089:the PDF's number/kind entry begins188CARTONS IN5PALLETS. The cartons remain the declared unit; five pallets also appear in Marks.
- Train176:120DRUMS ONLY is explicit;120DRUMS=10PLTS is secondary packing.
- Train488:the attachment places273/1354/1361/651/695 in `No.of.Pkgs`, while its descriptions enumerate underlying carton/pallet packing. Their sum4,334 pieces is the declared accounting. A generic prohibition on “pieces” would be wrong.
- Train048/307/446/511 and validation055:container rows explicitly count bags or drums although palletization is mentioned.

The distinction is **the role of a count in this document**, not the package noun or the numeric magnitude.

### F. Incomplete OCR and real source contradictions are separate problems

Six documents have fully numbered retained allocations whose sum differs from the goods total: train016,201,297,325,345 and validation012. Inspection supports incomplete retained ownership/identifiers, not a command to redistribute counts. The total can be valid while the available allocation list is partial.

Examples:

- Train325:39 valid990-bag container rows against40 declared containers. The remaining OCR identifier is malformed (`MSKU93333590`). Do not invent its canonical correction.
- Train159:70 pallets total; only one of two container IDs is retained in OCR. Preserve its35-pallet portion, not70.
- Train484:PDF has1 pallet, but OCR only has `PALLET SLAC / 40 BAGS`. Proposed package type is pallet, quantity absent. Neither “singular noun implies1” nor importing the PDF's1 is acceptable.
- Train391:143 bales in product wording versus144 in the declaration and48+48+48. Current missing total reflects a real conflict.
- Train271:carrier declaration says457 fibreboard boxes; product packing says410 boxes plus47 drums on19 pallets. A hierarchy switch alone cannot resolve the contradictory type. Existing precise composition is defensible but should carry an explicit source-conflict review outcome.
- Train588:primary `CONTAINER SLAC1000x20KG BAGS`, then20 heat-treated pallets and a combined weight/pallet total. Recommend retaining1,000 bags under declaration precedence, with an explicit interpretation note. It is a useful borderline fixture, not a reason to impose a universal outer rule.

All16 specific limitation/boundary records are linked in the casebook; this list overlaps proposals/controls and must not be added to the66.

### G. Discovery false positives and an adjacent synthesis wording risk

The initial scans find75+8 real and149 synthetic multiple-counted-word candidates. They are not error counts. False positives include:

- Freight `20.00 / Unit`, `Qty/Pkg` captions, address lots and PO boxes.
- HS/UN/product identifiers touching a package word over a line break.
- A package noun used as the commodity itself: bags, drums, pallet crates.
- Generic `packages` plus its specific type at the same count.
- Capacity/unit/piece counts internal to products, not shipment totals.

The broader word scan intentionally has even more noise:397 real-train,42 validation and1,137 synthetic hits. Its purpose is to catch omissions by the narrow scanner, not to approve or reject labels.

One adjacent synthetic wording example deserves a separate guard: `syn_full_v7_f3a79bbac236cdda53916fd8` declares1,718 BOXES but generated prose adds `TOTAL:1718 CARTONS`. These can be near-synonyms in prose, but are separate registry tokens. It is not a proven inner/outer hierarchy error and is not included in66. Generated description wording should not independently rename the host-selected shipment unit or append its own accounting total. Review that footer rather than launch a broad new regeneration pass.

## 5. Root cause through the pipeline

### Extraction and review

- [PackagesV7 and PlacementV7](../src/document_ocr/label_schemas/bill_of_lading_v7.py): docstrings/field descriptions explicitly say inner level; the goods package field reinforces it.
- [CargoCount.level](../src/document_ocr/labeling_agents/direct_cargo.py): `target` means innermost, `outer` means containing packaging, and `product_capacity` covers retail/set contents. It lacks a clear contained-shipment-context role when the declared target is the outer unit.
- [Cargo mapper](../prompts/labeling_agents/direct_cargo_mapper.md) and [relations reviewer](../prompts/labeling_agents/direct_relations_reviewer.md): both direct the model toward the innermost shipment level. A later caution about unquantified words does not remove this competing preference.

Thus this is not primarily random agent failure. Agents can follow the written policy and produce the undesired accounting choice consistently. Another agent using the same policy is not a remedy.

### Schema validation

An actual probe against `BillOfLadingExtractionV7Label` accepts the inconsistent combination “4,000 paper bags total + five allocations of20” from validation028. JSON typing and relational foreign-key validity do not carry the missing unit semantics.

A sum check is also inadequate: train363's old inner counts balance perfectly. And forcing every sum to match would corrupt the legitimate partial cases above.

### Templates and synthesis

Four active source contracts project inner counts into the public package/placement targets while rendering outer declarations as fixed auxiliary text:

| Source row / family | Public level now | Separate rendered accounting | Descendants |
|---|---|---|---:|
|018 / `e9275395`|54 source boxes, sampled inner categories/counts|5 pallets|7|
|095 / `058d92bb`|600 source boxes, sampled carton counts|20 OUTER PACKAGES|7|
|236 / `e9120c73`|240 source sacks, sampled carton counts|6 pallets|7|
|319 / `551657f3`|146 source boxes, sampled box counts|4 pallets|8|

For example, source095's descendants print20 OUTER PACKAGES while sampled inner cartons vary520/920/1460/580/360/480/660. Under the proposed policy, the public count should be20, not those content counts. The OCR itself is not contradictory; it contains a hierarchy. The target projection selects the wrong level.

The old counted-word scanner missed this family because `OUTER` interrupts count/noun adjacency. Family closure and auxiliary-contract inspection recovered all seven.

The active [curated scenario sampler](../src/document_ocr/synthesis/curated_scenarios.py) currently:

- Accepts one positive typed package row for a cargo observation.
- Samples a quantity/category, and apportions that public quantity over fully counted source allocations.
- Rejects mixed known/unknown allocation topology rather than inventing a fallback.
- Scales quantities and physical measures using donor/source per-package relationships.

The [physical renderer](../src/document_ocr/synthesis/curated_physical.py) also uses public/source/donor package quantities as the scaling basis. **Changing only the public count from600 boxes to20 packages without rebasing its physical/load basis could silently change mass/volume scaling.** Current labels can be corrected from existing OCR without that problem; future generation needs the contract change, not only a JSON edit.

The older package-compatibility prompt also prefers the directly containing package. It must be aligned if that route is reused, but it should not be confused with the entire active curated path.

## 6. Recommended operational policy

Use these rules as one coherent policy, not a chain of special cases:

1. **Identify package accounting scope.** Separate shipment package declarations, container portions, contained packing, product capacity, equipment count and unrelated numbers.
2. **Select the declared shipment unit.** Prefer the identified package/count column and explicit container package rows. A shipment declaration/total can supply missing facts. Use product packing to clarify that unit only when ownership is supported; do not replace it merely because deeper packing exists.
3. **Keep quantity and type paired.** Do not combine a pallet count with a bag category. A generic aggregate can remain generic where it includes heterogeneous units or its subtype is not fully established.
4. **Keep additive and nested structures distinct.** Sum only disjoint units at the same accounting scope. Do not sum pallets with their contained cartons, or a total with its portions.
5. **Use the same unit for allocations.** A printed bag-to-pallet ratio does not automatically authorize conversion of container counts. Only use a count at the selected level with established container ownership. Otherwise retain membership and omit its count.
6. **Preserve incomplete inputs honestly.** No forced balancing, equal division, serial-range counting, mass/capacity division, container-size-as-count or PDF-only value recovery. A known type can survive with an unknown count.
7. **Keep the description policy separate.** Product wording such as “walnuts packed in15kg bags” remains in the bounded description when it belongs there, even if the shipment target is five pallets. Do not reinsert detached declarations removed by the description pass, and do not strip embedded wording to make the numeric targets look simpler.
8. **Flag conflicting declarations.** Return a short issue category, competing printed facts, and a recommendation for review. Repetition of a value or a mathematically convenient sum is not authority to choose it.

This avoids an expanded public target graph. Accounting-role evidence belongs in review metadata and source contracts, not necessarily the model's training output.

## 7. Repair plan after policy approval

### A. Existing real and synthetic labels

Apply the enumerated37 real and29 descendant proposals with an exact before-state/hash check. Write receipts. Propagate the real changes to active mirrors and source targets; reconcile the synthetic mirror with combined training rows. Preserve historical frozen run targets and metrics.

No raw OCR edits, coordinate synthesis, party regeneration or goods-description regeneration is required for these66 proposals. Every proposed container membership is already present. The ambiguity/source-conflict list remains explicit, not silently coerced into the repair set.

### B. Four active templates

Rebind the public package count/type and placement quantity to the declared accounting occurrences. Retain inner content counts as private context where the template prints them. One source fix should cover all descendants; avoid29 sample-ID branches.

For future synthesis, make the shipment accounting unit and contained packing separate sampled/rendered facts in those contracts. Keep consistent totals, per-container counts and any printed hierarchy. Rebase the physical per-unit/donor relationship at the same time. Existing OCR's fixed outer counts may be retained for the label-only repair, but future variability must be deliberate rather than an accidental frozen source fragment.

Recompile the four templates and run at least two contrasting descendants per family. One should exercise a different sampled package/product style where allowed. Audit every count occurrence and its target ownership, including the source-only auxiliary occurrences. Re-run geometry only if rendering changed; a label-only correction leaves coordinates unchanged.

### C. Extraction/review contracts

Replace the inner-default wording consistently in `PackagesV7`, `PlacementV7`, the goods field, `CargoCount`, mapper and reviewer. Define target as the declared accounting level; retain non-target contained/transport packing as contextual evidence. Do not merely append another caution after the conflicting inner-level instruction.

The public field names and shape need not change. Review metadata should make uncertainty explicit and distinguish source contradiction, unavailable quantity, unavailable ownership and conflicting level selection.

### D. Validation gates

- Schema and identifier checks.
- Reviewed accounting-level ownership: each target count/type and each allocation refers to the same selected shipment level, or the allocation count is absent.
- Whole-cohort before/after field masks: only package rows/placement quantities may change; descriptions, addresses, identifiers, OCR and coordinates must not.
- Arithmetic with a completeness flag: require equality for a complete owned partition; permit documented partial coverage; reject sums across nested levels.
- Source-to-template-to-descendant checks: map public and private counts separately; verify repeated rendered copies, printed hierarchy and physical scaling.
- Negative fixtures: preserve declared inner-unit controls, additive mixed units, omitted OCR counts and partial allocations; reject “fixes” that erase these distinctions.
- Snapshot/freeze the benchmark and separately replay old predictions against revised gold. Report both target versions; gold corrections are not newly learned model improvements.

## 8. Implications for the previous relation error analysis

The [per-prediction review](analysis/package-accounting-policy-audit-20261009/prior-gold-absent-allocation-review.json) records container IDs, predicted counts and dispositions. As in the original23-count inventory, it excludes two separate predictions that use vehicle identifiers as container IDs.

The previous23 gold-absent container quantities across seven valid-container documents split more informatively after this audit:

- **Nine are supported under the proposed accounting policy:** five20-pallet predictions in validation028, three20-pallet predictions in validation049, and the1,400-carton prediction in validation047.
- **Two646 predictions** in validation009 remain ownership-sensitive because OCR lost a third identifier and column/position information. They are not confirmed fabricated numbers, nor approved allocations from string presence alone.
- **Twelve others** in validation008,038,042 are not licensed by this packaging-level correction. They still need restraint or input/ownership handling; changing inner/outer policy is not a universal explanation for relation errors.

The model is not uniformly right on the affected documents:

- Validation047 already predicts the coherent1,400-carton package and allocation, matching the proposed correction.
- Validation028 predicts the pallet allocations but keeps4,000 bags and an invalid duplicated category token. This remains a mixed-level model error even though its individual allocation counts are grounded.
- Validation049 predicts60 and20/20/20 but an invalid `PACKAGE_BAG_STEEL` type. The quantity recovery is good; the category is not.
- Validation046/052 follow the existing inner carton/bag targets and would need to learn the revised pallet convention.

Therefore correcting labels should remove contradictory supervision and improve measurement. It does **not** prove a particular F1 increase before training, and it does not replace coverage work on categories or allocation restraint.

## 9. Probe results, resource use and preservation

- In-memory proposal run: **66/66 validate against the current strict JSON schema**.
- **Zero unrelated target-field changes** in the proposal diff.
- **Zero container identity/membership changes**.
- **15 regression probes passed** in0.72s, including the deliberate demonstration that today's schema accepts a semantic unit mismatch.
- Both balanced-but-wrong-policy counts and legitimate non-balancing partial inputs are covered; this falsifies arithmetic-only validation.
- Full inventory:9.95s on this workspace. Proposal/schema/preservation run: approximately3.5s. These are audit timings, not a production throughput benchmark.
- Reported process peak RSS for the inventory:820.74MiB; this is not a new training/synthesis memory requirement. The audit uses streaming file hashing and streams active JSONL records, while materializing its evidence inventory.
- **1,210 active dataset/catalog file hashes unchanged** before/after the audit. The scripts only publish analysis artifacts under this report's subdirectory.
- **Paid API calls:0. Cost:$0.** Production performance and behavior are unchanged because this was an audit, not an implementation pass.

Bottom line: there is a real, localized policy problem, and its primary correction is **declared package accounting with unit-consistent allocations**. The existing inputs support the enumerated corrections; wholesale regeneration is unnecessary. Future synthesis does require four carefully rebased accounting contracts and aligned extraction/review instructions, rather than another global inner/outer heuristic.
