# Declared-package accounting repair and synthesis validation

Date: 2026-10-09. This implements the [approved accounting audit](package-accounting-policy-audit-2026-10-09.md), including the sole-container STC clarification. Dataset publication is complete; final pilot results are recorded below.

## 1. Applied policy

Use the document's **declared shipment accounting unit** for goods package totals and container allocation quantities. Neither innermost contents nor outermost handling units are the default. Independent same-level package groups can remain separate; nested quantities are not additive package rows.

`1x40'HC container S.T.C. 38 PACKAGE(S)` supports an allocation of 38 when the actual sole container is identified. A separate container table is unnecessary. One surviving identifier in OCR that declares multiple containers does **not** license copying the full shipment quantity onto that identifier.

Examples:

| Printed situation | Target |
|---|---|
| 100 pallets; five container rows of 20; 4,000 bags inside | 100 pallets; allocation 20 to each identified container |
| 19 pallets containing 728 cartons, plus 3 loose cartons; 22 packages declared | 22 generic packages, not 22 pallets or 731 cartons |
| 120 drums declared; packed on 10 pallets | 120 drums; pallets remain supporting packing |
| `1 Container Said to Contain 1400 CARTONS` with one container row | 1,400 cartons total and 1,400 allocated |
| 70 pallets in two containers; only one ID and its 35-pallet row survive OCR | 70 total; 35 allocated to that ID, not 70 |
| `PALLET SLAC / 40 BAGS`, pallet numeral absent from OCR | Pallet type, unknown pallet quantity; do not substitute 40 |

The [central label policy](kie-real-baseline-label-policy-2026-10-05.md) records this convention and preserves the earlier main-product-passage description policy. Embedded packing qualifiers stay in genuine product wording; standalone accounting, loading declarations and separately owned Marks stay outside the description. This is our annotation contract, not an assertion that customs rules universally require outer packages.

## 2. Current dataset: exact publication scope

Dataset size is unchanged: **600 real + 1,500 synthetic training documents; 60 real validation documents**.

| Cohort | Labels corrected | Inputs corrected |
|---|---:|---:|
| Real train | 33 | 0 |
| Real validation | 5 | 0 |
| Synthetic train | 29 packaging + 1 description | 1 separate document |
| Total unique label repairs | **68: 67 packaging + 1 description** | **1 additional input-only repair** |

Of the 29 synthetic packaging label repairs, five are in the original 500 and 24 in the additional 1,000. All are the complete existing descendant closure of the four affected active template families. Including the separate description and input-only fixes, this pass changes **69 unique documents**.

The initial audit supplied 66 exact proposals. The sole-container follow-up recovered one further OCR-present count: train 301's detached `3`, spatially in the cargo-total footer, belongs to its sole container. The target has quantity 3 with no package type. PDF-only `PLASTIC PALLETS` and measurements were **not** imported. Both dataset publisher and root reviewer inspected its PDF layout.

The separate synthetic correction changes `TOTAL: 1718 CARTONS.` to `TOTAL: 1718 BOXES.` in `syn_full_v7_f3a79bbac236cdda53916fd8`, matching its existing declared and labeled boxes. This is an exact input-only correction in both raw and positioned mirrors. It preserves every coordinate and all labels. New generation already rejects standalone accounting introduced by product wording; the motivating footer has an explicit regression fixture.

The final retrospective description screen checked all 1,500 synthetic targets. Of four candidate counted-package lines, one is an independently removable terminal declaration: `6 PACKAGES TOTAL` in `syn_full_v7_56d2ccb71ad7f7c7c9e2b170`. Only that footer was removed from its description label; its OCR, product wording, package totals, allocations and coordinates remain unchanged. The other three introduce attached product lists (`10 PACKAGES OF HEAVY EARTHMOVING WEAR PARTS:`, `535 CARTONS CONTAINING:`, `558 CTNS CONTAINING:`) and were explicitly retained after contextual review. This distinction avoids turning a conservative generation guard into a destructive retrospective text filter. [Four decisions](analysis/package-accounting-repair-20261009/description-accounting-followup/decisions.json), [exact publication receipt](analysis/package-accounting-repair-20261009/description-accounting-followup/receipt.json).

The publication updated the mixed dataset, r14/r16 real mirrors, standalone labels and current manifest hashes. Historical training runs and campaign artifacts were not rewritten. Full pre-edit backup:

`data/curated/backups/package-accounting-20261009/`

The [detailed dataset report](analysis/package-accounting-repair-20261009/REPORT.md), [exact proposals](analysis/package-accounting-policy-audit-20261009/proposed-changes.json), [supplemental count recovery](analysis/package-accounting-repair-20261009/single-container-followup/proposal.json) and [final composite validator receipt](analysis/package-accounting-repair-20261009/validation-final.json) provide all before/after identities and proofs.

### Sole-container validation

All 660 real records were screened; 408 have one labeled container:

- 401 have matching package/allocation counts, including the repaired explicit-STC and detached-footer controls.
- Four have partial OCR of a multi-container shipment. Their printed smaller allocations remain intact.
- Three lack the shipment-count numeral in OCR; counts remain absent.
- **Zero known-positive-total/missing-allocation candidates remain** in this structural screen.

This combines full machine inventory with focused source/PDF adjudication of the exceptions; it is not a claim that all 408 documents received a fresh full-document manual review. [Inventory](analysis/package-accounting-repair-20261009/single-container-audit.json).

## 3. Template and generation corrections

Four of the 200 admitted templates required accounting rebasing:

| Source prefix / real row | Declared public unit | Private contained packing | Exact sampling constraint |
|---|---|---|---|
| `e9275395` / 18 | 5 pallets | 54 boxes | Declared quantity multiple of 5 |
| `058d92bb` / 95 | 20 packages | 600 boxes | Any positive whole declared quantity |
| `e9120c73` / 236 | 6 pallets | 240 sacks (`SACOS`) | Any positive whole declared quantity |
| `551657f3` / 319 | 4 pallets | 146 boxes | Declared quantity multiple of 2 |

Changes in the live reusable catalog:

1. Rebind public package/count/category owners and allocation quantities to the declared level.
2. Bind all repeated declared numerals and written-out numbers, rather than retaining stale source totals.
3. Retire the superseded fixed composite owners.
4. Keep contained counts private, with an explicit source-certified ratio. Integer arithmetic uses exact rational numbers; impossible fractional counts fail rather than round.
5. Validate source count occurrences and the adjacent printed contained unit. Supply those same sampled private facts to both wording generation and final-text review.
6. Restrict commodity sampling when a retained inner packing noun requires it. The `SACOS` family uses its measured source physical bundle and six registry starch/inulin identities, rather than unrelated machine loads. Global ambient chapter 11 is explicitly enabled in the future config; the exact family domain is 1108. This is configuration, not a document-ID branch in code.

Original source OCR stays unchanged. All four rebases passed exact identity replay. Existing source-template alignments remain the source of geometry. Catalog file hashes, target hashes, capabilities and source-dataset dependencies were refreshed; the four source certificates and backups are in [catalog receipt](analysis/package-accounting-repair-20261009/catalog-stage-receipt.json).

### Findings from fresh synthesis, not just unit tests

The first pilot exposed missing **private-fact context**: reviewers could see generated contained box/sack counts in OCR but not in their supplied facts. It also exposed a real commodity/packing contradiction: a sack-retaining source drew tube-rolling machinery. These were corrected at the generation/review contract and sampling-capability layers.

A proposed polymer-only domain had no complete eligible physical donor and was rejected by preflight. The final choice uses the source's proven sack/pallet bundle with registry starch variants. The ambient chapter whitelist also needed an explicit chapter 11 entry. No fallback donor, invented load or fabricated rounded packing count was introduced.

Postal checks continued to hold missing/repeated geography for the existing bounded postal-correction stage. Those holds do not authorize silently dropping address content.

Root inspection also caught a false-negative semantic review: two electronic-goods variants had a new standalone `146 BOXES PER SHIPMENT UNIT COUNT...` line inside their product passages. The existing generation guard covered leading accounting and count-only lines, but not counted-package lines with trailing prose after an earlier product line. Its line-anchoring was corrected and both variants were put through the bounded production correction. This is a generation-time rejection, **not** a regex that strips numbers from real labels. Embedded genuine product specifications remain protected by regression tests.

### Email/website side note

As explicitly clarified by the user, email and website domains need not agree. The generation prompt, final reviewer context and validator now allow independent corporate/free-mail domains. Source-style website syntax, syntactic validity, exact ownership and placeholder checks remain. The extra contact-correction retry added during this pilot was removed because its motivating shared-domain requirement was unnecessary.

Existing dataset contacts were not changed or regenerated. The eight pilot samples with contact receipts were revalidated offline against the relaxed policy with **zero changed contact values** and explicit before/after metadata receipts. One tiny correction request had already happened before the user's clarification and remains honestly included in the cost ledger. [Receipt](analysis/package-accounting-repair-20261009/contact-policy-relaxation.json).

## 4. Production components

- `bill_of_lading_v7.py`: package totals and allocation field descriptions now define declared accounting and sole-container support explicitly.
- `direct_cargo.py` and cargo/relation prompts: count roles are `target`, `packing_context`, `product_capacity`; only declared target counts participate in proposed arithmetic. The old ambiguous `outer` review role is rejected rather than silently reinterpreted.
- `curated_packaging.py`: exact contained-count scaling, source-unit verification, sampling divisibility check and private generation/review context.
- `curated_ownership.py`: the reviewed private dependency is wired into actual rendering and blueprint validation.
- `curated_campaign.py` / `curated_wording.py`: sampled contained facts reach generation and semantic review; configuration is checked before paid generation.
- Compiler and critic prompts plus package-compatibility instructions: same declared-accounting convention, repeated-count ownership, contained dependencies and commodity compatibility.
- Future task contract: add registry-backed `PACKAGE_SKID` (MPCI code SI). The current frozen whitelist had omitted this valid category; an actual training startup check caught it. No target/schema invention was needed.

Historical configurations are left unchanged. The **unrun** corrected training config is:

[`t5gemma2_270m_lora.mpci_bl_real600_synthetic1500_descblocks_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml`](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic1500_descblocks_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml).

The future synthesis config is [description_blocks_next](../configs/synthesis/mpci_bl_curated_v7_description_blocks_next.yaml). Its 200 templates remain available; no bulk campaign was launched.

## 5. Validation evidence

- Complete dataset comparison against the immutable backup: 2,160 unique documents, 3,480 mirrored JSONL rows, 1,320 standalone labels; 86 changed files and 3,904 byte-unchanged files. **Zero unrelated label changes, coordinate changes or container-membership changes.** The one authorized description-footer correction is explicitly included in the expected diff.
- Seven deliberate publication mutations were rejected. Targeted regressions cover partial-OCR total copying, wrong counts, PDF-only type leakage, stale source text, unknown counts and unrelated edits.
- Actual CPU training startup inspection passes 2,100 train + 60 validation, including the constrained vocabulary. No GPU model load or training was started.
- All 1,200 catalog case files plus shared files and source-dataset hashes verified. All 200 sources passed two new scenario preflights (**400**); the four repaired families passed another 100 draws each (**400**). Exact integer contained-count relationships hold throughout.
- Family stress-test quantity diversity: 73 distinct counts for the generic-package family; 2 for the four-pallet source bundle; 4 for the sack/pallet bundle; 8 for the five-pallet-ratio family. Narrower counts reflect observed physical bundles and integrality, not a silent fixed fallback.

The completed fresh-synthesis, semantic-review and geometry results are in section 7.

### Measured runtime and memory

- Dataset strict-schema validation, seven alternating repetitions over all 2,160 before/after targets: median 0.20720 s → 0.20774 s (+0.55 ms total, 0.26%). Packaging repairs grow targets by 272 bytes total; the separate later description-footer removal is not part of that benchmark.
- Final full dataset/mirror/byte audit, including the description follow-up: 23.13 s, peak RSS 293.37 MiB.
- Final all-template startup/preflight/stress verification: 18.26 s, peak RSS 820.74 MiB; the 400-plan sweep itself 4.56 s. These are CPU checks, no paid calls, and include the final live training-data hash.
- Added private-packing context measured on 16 actual pilot requests, seven alternating repetitions: 0.02239 s without vs 0.02387 s with context, **0.092 ms per request**. Maximum traced request allocation 117,269 bytes. This isolates the new context overhead, not a claim about historical end-to-end throughput.

Receipts: [dataset benchmark](analysis/package-accounting-repair-20261009/benchmark.json), [context benchmark](analysis/package-accounting-repair-20261009/private-context-benchmark.json), [runtime validation](analysis/package-accounting-repair-20261009/runtime-final.json), [startup vocabulary fix](analysis/package-accounting-repair-20261009/future-constraints/receipt.json).

## 6. Source limitations are preserved, not repaired by guessing

The audit's 16 source-conflict/partial-OCR/boundary records retain their documented dispositions. They overlap repair/control cohorts and are not 16 additional unapplied corrections. None is among the 200 active synthesis templates. Missing OCR numerals, malformed missing IDs and conflicting source declarations do not become invented allocations or enforced equal splits.

The scope of this repair is packaging accounting and its integration with the already-repaired descriptions. It does not claim that every unrelated extraction field has been reannotated, or that a finite pilot proves the absence of every future semantic mistake. The publication gates remain strict: every future candidate must pass replay, semantic review/adjudication and geometry validation.

## 7. Completed synthesis pilot and final approval

The final [pilot configuration](analysis/package-accounting-repair-20261009/pilot-v2.yaml) generated **16 samples from eight templates**, two variants per template. It covers all four accounting rebases plus a long-description family, dangerous goods with four containers, refrigerated goods, and a five-container allocation case. These are actual generation/render/review/publication calls, not mocked LLM outputs or sampler-only tests.

Examples observed in the final rendered output:

- 10 declared pallets with 108 contained boxes, and 20 pallets with 216 boxes: the 54/5 relationship remains exact in all rendered occurrences.
- 2 declared packages with 60 boxes, and 10 packages with 300 boxes: public allocations use packages, not contained boxes.
- 6 pallets with 240 sacks of wheat starch, and 4 pallets with 160 sacks of potato starch: commodity form and retained packing agree.
- 4 pallets with 146 boxes of electronic components: independent packing declarations remain outside the product-description target.
- Four-container dangerous-goods cases preserve registry facts and allocation arithmetic (272 boxes split 68 each; 6,770 drums split 1,693 + 1,693 + 1,692 + 1,692).
- Frozen goods retain reefer equipment and the sampled −18 °C settings; the five-container control retains its complete allocation sum.

All 16 final candidates pass publication and full rendered-text/label semantic review with **zero unresolved findings**. The root follow-up did identify and correct the two electronic-description accounting lines discussed in section 3; those corrected candidates were rereviewed. One malformed model response was rejected before rendering and retried explicitly, with its original response and charge retained. Passing the first reviewer was not treated as sufficient evidence on its own.

Review material:

- [Original source OCR followed by all rendered variants](../artifacts/kie-synthesis-production/package-accounting-validation-20261009-v2/samples.md).
- [All final positioned variants](../artifacts/kie-synthesis-production/package-accounting-validation-20261009-v2/positions-reflow-v1/samples.md).
- [Final plain-text dataset and manifest](../artifacts/kie-synthesis-production/package-accounting-validation-20261009-v2/manifest.json).
- [Geometry gallery: eight documents, 16 plotted pages](../artifacts/kie-synthesis-production/package-accounting-validation-20261009-v2/positions-reflow-v1/audit/GALLERY.md).

### Coordinate validation

All **1,938 nonempty lines** across 16 samples / 38 pages were checked. The final coordinate pass preserves the plain text and targets exactly:

| Line cohort | Lines | Anchor-only positions | Final positions | Lost positions |
|---|---:|---:|---:|---:|
| Goods | 97 | 8 | 62 | 0 |
| Addresses | 205 | 97 | 125 | 0 |
| Other | 1,636 | 1,388 | 1,388 | 0 |
| All | **1,938** | **1,493 (77.0%)** | **1,575 (81.3%)** | **0** |

Fourteen pages use accepted joint reflow; 24 have no eligible reflow region. No proposed page reflow was rejected. Remaining unknown coordinates retain explicit empty position markers: missing source anchors and foreign text inside an ownership envelope are not filled by guessing. Goods coverage is 63.9% and address coverage 61.0% in this deliberately difficult pilot; this is not a claim that every line has a position.

The independent geometry probe rejected **four of four deliberately corrupted inputs**: changed text/line count, off-page coordinates, duplicate lines, and an inconsistent source anchor. The root also opened and visually inspected the four-panel plots for the electronic-components, pallet/box and sack/pallet families. These show ordered product/address points confined to their owned layout regions, with the coherent page transform applied afterward. The envelopes represent plausible layout regions, not measured glyph positions from nonexistent synthetic PDFs.

[Machine geometry audit](../artifacts/kie-synthesis-production/package-accounting-validation-20261009-v2/positions-reflow-v1/audit/validation.json), [independent negative controls](../artifacts/kie-synthesis-production/package-accounting-validation-20261009-v2/audit/geometry-independent-probes.json). Position production took 1.26 s; the gallery/coverage audit took 0.84 s; the separate negative-control probe took 6.22 s with peak RSS 485.41 MiB. No paid calls were needed for geometry.

### Tests, cost and release state

Final relevant suite: **593 passed in 32.28 s**. This includes the 13 exact-publication regression tests, nested-accounting/integrality cases, sole-container controls, description-boundary tests, and independent email/website-domain cases. Ruff passes for the changed packaging, contacts, orchestration, schema and direct-cargo components and their targeted tests.

Total paid model cost across both pilot iterations, all reviews/corrections and the malformed-response attempt: **$0.04398655** (about 4.4 US cents). The contact-policy relaxation and existing-value revalidation incurred no additional paid requests. [All-attempt cost ledger](analysis/package-accounting-repair-20261009/cost-summary.json).

The current dataset repairs, all four template rebases, extraction/compiler/generation policy alignment, and the requested small synthesis validation are complete. No identified issue in this pass remains awaiting repair. The 200-template catalog is ready for a larger run using the same strict preflight, generation, review and geometry gates. The 16 validation samples are kept separately and **have not been added to the training dataset**. No training or larger synthesis run was launched.

Final mixed training SHA-256: `e2ad4444fdf87d2d3932337c252b7b426d8c8d1d1661070e5215b04df0ac8847`.

Final validation SHA-256: `9f3b4c36fe5d6310b3fb79a6d380a06d24d2fe8583acf80748d83861b2fe94da`.

Final 16-sample plain pilot SHA-256: `eda10ee7d44a8abceca382b8bb40a9250cbc0a02bce4c856bf6875151be83a74`.

Final positioned pilot SHA-256: `82fa9aec2f793237744dda001f731a45b85082a36c3df83fb51abada7b0bb5f1`.

## Corrected-data training comparison prepared (2026-10-09)

The existing, unrun `descblocks` rank-48 configuration linked above has been refreshed
after the [final footer repair](external-reference-suffix-repair-2026-10-09.md). It
includes all main-product-passage description and declared packaging/allocation
corrections, plus the seven synthetic input-only footer corrections. Its name is
retained to avoid creating another duplicate experiment recipe.

Current pinned files:

- Train: `0340136994089effdd153cc76bc052dd95179c60b2868711a036d717989da642`
  — 600 real + 1,500 synthetic records.
- Validation: `9f3b4c36fe5d6310b3fb79a6d380a06d24d2fe8583acf80748d83861b2fe94da`
  — the same 60 real documents, with repaired labels.

Comparison with the completed rank-48 run's saved configuration confirms unchanged
model/tokenizer revisions, rank 48, alpha 32, EVA initialization, learning rate
0.0001, ScheduleFree AdamW, 33 optimizer-owned warmup steps (5% of 660 updates),
microbatch 1, accumulation 32, seeds, compact targets, input-only positional prompt,
10 epochs, and all other training/evaluation settings. Evaluation and checkpoint
saving remain every 165 updates (epochs 2.5, 5, 7.5 and 10); all scheduled checkpoints
are retained and the best field-F1 checkpoint is loaded/exported. The new run/adapter
identity keeps previous outputs intact. Dataset pins, the updated schema/category
contract and policy tags are the intended differences. No model checkpoint is resumed.

Validation performed through the freshly rebuilt **same training Docker image**:

- Full schema/hash/record inspection: 2,160 records pass; host inspection takes
  1.51 seconds (1,433 records/second).
- Actual tokenizer/cache preparation: 40.24 seconds; **zero truncated inputs and
  zero excluded targets**. Train maxima: 8,066 input / 3,139 target tokens. Validation
  maxima: 5,698 input / 1,348 target tokens. Existing limits (19,200 / 5,500) and
  validation generation budget (3,072) remain unchanged.
- 32 targeted configuration/optimizer tests pass on the host. One Torch-dependent
  test is skipped there because Torch is container-only; the image lacks pytest,
  so its equivalent CPU-only optimizer probe is run directly in that image.
- A new comparison regression test rejects unintended hyperparameter changes.
  Ruff and `git diff --check` pass.

Only configuration, its regression test and this documentation change in this
preparation step. No dataset edits, GPU model loading or training were performed.
No hot path changed, so these preparation timings are not a training-speed benchmark.
Tokenization cache files were generated for the corrected dataset.

When comparing results, rescore earlier saved predictions against these repaired
validation labels as well as reporting historical metrics: label corrections make
a raw comparison of the two runs' originally logged scores non-identical in target.

```bash
docker compose --profile training run --rm --build kie-trainer train \
  --config configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic1500_descblocks_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml \
  --project-root /workspace
```
