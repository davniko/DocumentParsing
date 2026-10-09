# MPCI extraction: run results and next experiments

Updated: 2026-10-09. This is the consolidated results ledger for the current run and the preceding six runs, followed by an evidence-based plan for the next experiment. Keep this document as the entry point rather than reconstructing the experiment history from artifact directories.

**Current conclusion:** the rank48 / 5% warmup run is the strongest recent run overall: its best scheduled evaluation reaches **field F1 0.8858, accuracy 0.8401, relation F1 0.9120**. The next intervention should target demonstrated training-data coverage and evidence-ownership gaps, not another broad hyperparameter sweep. Categories and descriptions have different failure mechanisms; neither is adequately described as “the model needs more examples.”

This investigation performed local artifact inspection, CPU metric replay, source-text review and deterministic sampler probes. It did **not** modify datasets, training/synthesis code, configs or checkpoints; run inference/training; or make paid model calls. Proposed interventions below are not already implemented.

## 1. Reading the metrics correctly

- The target is **both** field F1 and field accuracy above 0.90 for “good,” and approximately 0.95 for “great.” A relation F1 above 0.90 does not establish that goal for the complete extraction.
- Field precision/recall/F1 count aligned scalar values, with list-order-invariant matching. **Field accuracy is correct values divided by the union of occupied aligned gold/prediction paths**, not whole-document accuracy and not token accuracy. An incorrect value at one path contributes a precision and recall error but occupies one union path.
- Relation and category metrics compare projected facts. Their accuracy is set overlap/Jaccard. An aligned placement-field score is not necessarily identical to the relation-set score because matching and deduplication differ.
- Invalid JSON gets no recovered-field credit. Parseable schema-invalid JSON can receive partial field credit. Explicit nulls in the target count under the current contract.
- Whole descriptions/addresses are individual string fields. A missing word and a completely wrong description both lose that field's exact-match credit. The error taxonomy below distinguishes their practical severity without silently changing the primary metric.
- **Native scheduled evaluation** means the values logged during that run, using its contemporary labels/settings. **Common-gold replay** means CPU rescoring of saved generated text against the current 60-document validation target, case-insensitively. It is not new model inference.
- Best scheduled evaluation and the separate final prediction pass are different observations. Their scores can differ slightly even when the exported weights are correct. Do not mix a best-epoch aggregate with an epoch10 error inventory.

### Important historical corrections

1. The earlier “compact JSON” experiment did **not** isolate pretty versus compact targets: the previous run's actual cached decoder targets were compact too. The change included casing/equipment labels, evaluation settings and prompt/schema changes.
2. R14 to R16 changed address-punctuation labels as well as adding positional input. That transition is not a pure positional ablation.
3. Removing the explicit prompt was a cleaner comparison: the R16 positional data/settings were retained. Each training run started from the base model/EVA setup; these are not successive fine-tuning stages of the preceding adapter.
4. Older runs did not all use the current explicit-null instruction policy. Before the 1,500-synthetic run, 38 `sameAs` nulls and two negotiability nulls added 40 target facts across 36 validation documents. This particularly changes historical schema-validity scores when evaluated against today's schema.
5. The previous rank32 1,500-synthetic run selected epoch7.5 but exported epoch10. The rank48 run exports its selected epoch10 checkpoint correctly. Granular comparisons below use the **available saved epoch10 outputs**, not the earlier run's best epoch7.5 predictions.

## 2. Experiment registry: seven completed runs

All use T5Gemma2-270M, LoRA/EVA and ScheduleFree AdamW. Alpha is 32 throughout this ledger. Each has 60 validation documents. The configuration links are the launch files; the corresponding run directory contains the authoritative `resolved-config.json` and artifact manifest.

|ID|Run / configuration|Training set|Principal change|Epochs / updates|Internal warmup|
|---|---|---:|---|---:|---:|
|A|[Real600, R12, 20 epochs](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real660_reduced_v7_e20_eva_a32_r32_local_schedulefree_v1.yaml)|600 real|Longer real-only baseline; raw OCR plus prompt; historical labels and case-sensitive scoring|20 / 380|5 updates|
|B|[Real600, R14](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real660_r14_reduced_v7_e10_compact_eva_a32_r32_local_schedulefree_v1.yaml)|600 real|Equipment/casing fixes; case-insensitive evaluation; generation limit 2,048→3,072; still raw OCR plus prompt|10 / 190|5|
|C|[Real600, R16 positions](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real660_r16_positions_v7_e10_compact_eva_a32_r32_local_schedulefree_v1.yaml)|600 real|Positional input, still prompted; updated address-punctuation targets|10 / 190|5|
|D|[Real600, input-only positions](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real660_r16_positions_inputonly_v7_e10_compact_eva_a32_r32_local_schedulefree_v1.yaml)|600 real|Remove explicit prompt, retain positional OCR|10 / 190|5|
|E|[Real600 + synthetic500](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic500_positions_inputonly_v7_e10_compact_eva_a32_r32_local_schedulefree_v1.yaml)|1,100|Add 500 synthetic records from 100 templates; four evaluations|10 / 350|5|
|F|[Real600 + synthetic1500, rank32](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic1500_positions_inputonly_v7_e10_compact_eva_a32_r32_local_schedulefree_v1.yaml)|2,100|Expand to 200 templates / 1,500 synthetic records; instruction policy and targeted sample repairs|10 / 660|5|
|G|[Same dataset, rank48](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic1500_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml)|2,100|Rank32→48; warmup5→33; keep all checkpoints; correct best-adapter export|10 / 660|33, or 5%|

F and G have identical training/validation file hashes, input/target formatting, seeds, LR, batch/accumulation, model/tokenizer revisions and total updates. Their effective rsLoRA scale nevertheless changes from 32/√32 = 5.657 to 32/√48 = 4.619; EVA also adapts to the new rank. Therefore G is **rank48 plus longer warmup**, not an isolated rank test.

### Best scheduled evaluation in each run

The relation/category values below are at the **field-F1-selected checkpoint**, not independently selected peaks.

|ID|Best epoch / step|Field F1|Field accuracy|Relation F1|Category F1|JSON valid|Schema valid|
|---|---:|---:|---:|---:|---:|---:|---:|
|A|20 / 380|0.7961|0.7189|0.8198|0.8000|55/60|47/60|
|B|5 / 95|0.8285|0.7639|0.8814|0.8259|56/60|43/60|
|C|5 / 95|0.8252|0.7465|0.8700|0.7788|54/60|44/60|
|D|10 / 190|0.8317|0.7604|0.8578|0.7829|55/60|46/60|
|E|10 / 350|0.8421|0.7687|0.8925|0.8268|53/60|44/60|
|F|7.503 / 495|0.8756|0.8262|0.8758|0.8473|58/60|50/60|
|G|10 / 660|**0.8858**|**0.8401**|**0.9120**|0.8597|58/60|48/60|

These are useful historical results, not seven controlled experiments on an unchanged scoring contract. In particular A is case-sensitive, and B/C best checkpoints differ from their exported final predictions.

### Complete scheduled progression

All values in this table are native logged metrics. Epochs are rounded; integer update steps are exact.

|Run|Epoch / step|Eval loss|Field F1|Field accuracy|Relation F1|Category F1|Valid JSON|
|---|---:|---:|---:|---:|---:|---:|---:|
|A|5 / 95|0.07129|0.7831|0.6929|0.8247|0.7607|53|
|A|10 / 190|0.08360|0.7880|0.7012|0.8683|0.8356|51|
|A|15 / 285|0.10827|0.7947|0.7153|0.8753|0.8401|52|
|A|20 / 380|0.12015|0.7961|0.7189|0.8198|0.8000|55|
|B|5 / 95|0.08353|0.8285|0.7639|0.8814|0.8259|56|
|B|10 / 190|0.09830|0.7903|0.6960|0.7539|0.7598|51|
|C|5 / 95|0.07663|0.8252|0.7465|0.8700|0.7788|54|
|C|10 / 190|0.09201|0.8135|0.7289|0.8295|0.7510|53|
|D|5 / 95|0.07734|0.8112|0.7262|0.8232|0.8242|56|
|D|10 / 190|0.09610|0.8317|0.7604|0.8578|0.7829|55|
|E|2.524 / 88|0.06931|0.7799|0.6783|0.7928|0.6798|52|
|E|5.029 / 176|0.07404|0.8393|0.7685|0.8606|0.8108|55|
|E|7.553 / 264|0.08400|0.8229|0.7352|0.8678|0.7962|51|
|E|10 / 350|0.09542|0.8421|0.7687|0.8925|0.8268|53|
|F|2.503 / 165|0.07227|0.8180|0.7305|0.7880|0.8085|54|
|F|5 / 330|0.07371|0.8697|0.8122|0.8957|0.8755|56|
|F|7.503 / 495|0.09122|0.8756|0.8262|0.8758|0.8473|58|
|F|10 / 660|0.09385|0.8535|0.7866|0.8687|0.8429|55|
|G|2.503 / 165|0.07073|0.8523|0.7885|0.8438|0.8007|59|
|G|5 / 330|0.07407|0.8506|0.7849|0.8605|0.8302|56|
|G|7.503 / 495|0.08269|0.8234|0.7374|0.7990|0.7904|53|
|G|10 / 660|0.09223|0.8858|0.8401|0.9120|0.8597|58|

More epochs are not uniformly better. G improves late while F peaks earlier, and both lose ground at some intermediate evaluations. G's eval loss rises while its final extraction scores improve. Select checkpoints on extraction metrics, not minimum teacher-forced loss. Retain predictions as well as weights at each evaluation: aggregate logs alone cannot localize the intermediate regressions.

## 3. Common-current-gold comparison of saved predictions

Same 60 document IDs; complete current target, **including addresses and goods descriptions**; case-insensitive scoring. This is not the older 30k comparison restricted to unchanged fields. A–E were not all trained to today's contract, so some historical penalties are policy/format differences rather than evidence of poorer semantic recognition.

|ID|Field precision|Field recall|Field F1|Field accuracy|Fact F1|Relation F1 / accuracy|Category F1 / accuracy|JSON valid|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|A¹|0.8422|0.7319|0.7832|0.6997|0.7803|0.8139 / 0.6862|0.7859 / 0.6472|54|
|B|0.8487|0.6863|0.7589|0.6643|0.7593|0.7539 / 0.6050|0.7598 / 0.6126|51|
|C|0.8688|0.7432|0.8011|0.7100|0.7987|0.8271 / 0.7051|0.7305 / 0.5754|53|
|D|0.8609|0.8083|0.8338|0.7653|0.8310|0.8631 / 0.7592|0.7897 / 0.6524|56|
|E|0.8864|0.7903|0.8356|0.7581|0.8303|0.8925 / 0.8059|0.8268 / 0.7048|53|
|F|0.8854|0.8344|0.8591|0.7965|0.8575|0.8718 / 0.7727|0.8668 / 0.7649|56|
|G|**0.8922**|**0.8776**|**0.8849**|**0.8390**|**0.8823**|**0.9120 / 0.8382**|0.8597 / 0.7540|58|

¹ A did not retain native prediction text in its run directory. This row uses the previously saved checkpoint380 inference replay, whose adapter hash was reverified against the run's export. It is explicitly a later replay, not recovery of the original scheduled evaluation.

Historical schema validity against the current schema is deliberately not used as the headline trend: newly required instruction fields invalidate many otherwise readable older outputs. G has 47 schema-valid outputs in this saved pass, compared with 48 at its scheduled epoch10 evaluation. F has 45 in its saved pass. Exported G weights exactly match checkpoint660; the small scheduled/final replay difference is not another confirmed export failure.

### Field groups across all seven runs

Complete-current-target **field F1**, all60 including malformed outputs. The current gold has 2,566 scalar facts. This table measures extraction, not per-field-group document pass rates.

|Group / gold facts|A|B|C|D|E|F|G|
|---|---:|---:|---:|---:|---:|---:|---:|
|Document, transport, freight / 375|0.9131|0.8576|0.9024|0.9113|0.9134|0.9264|**0.9561**|
|Locations / 317|0.9028|0.8986|0.8792|0.9071|0.9033|0.9271|**0.9470**|
|Containers / 466|0.8308|0.7886|0.8065|0.8749|0.8901|0.9089|**0.9198**|
|Aligned placements / 211|0.8139|0.7539|0.8271|0.8631|0.8904|0.8765|**0.9120**|
|Other party fields / 562|0.7837|0.7814|0.7766|0.7992|0.7776|0.8425|**0.8740**|
|Other goods fields / 398|0.7962|0.7913|0.7898|0.7937|0.8079|0.8105|**0.8385**|
|Address lines / 177|0.2083|0.1902|0.6407|0.7130|0.7027|0.7267|**0.7875**|
|Descriptions / 60|0.4561|0.4505|0.3894|0.4483|0.4248|**0.4828**|0.4237|

The early address scores are heavily penalized by changed punctuation policy; do not interpret A→C as solely learned address improvement. The cleanest current diagnosis is F versus G, where gold and input files are identical.

### F versus G: separate parsing coverage from content

|Measure|Rank32 F|Rank48 G|
|---|---:|---:|
|All60 correct scalar facts / 2,566|2,141|2,252|
|All60 wrong/extra scalar predictions|277|272|
|Same54 parseable documents: field F1|0.8873|0.8961|
|Same54: field accuracy|0.8462|0.8585|
|Same54: relation F1|0.9002|0.9113|
|Same54: category F1|0.8893|0.8709|
|Documents above both 0.90 field F1 and accuracy|20/60|26/60|
|Documents at least 0.95 on both|11/60|14/60|

Of the +111 correct scalar facts, +26 come from the same54 parseable documents; +135 from four newly parseable documents; −50 from two newly malformed documents. Approximately **77% of the net gain is changed parsing coverage**, while a smaller genuine extraction improvement remains on the matched cohort.

On G's own58 parseable documents, complete-target F1 is **0.8951**, accuracy **0.8576**. Correct syntax alone therefore does not demonstrate the 0.90/0.90 target. All included difficult fields remain in those scores.

A paired 10,000-resample document bootstrap gives F→G full-set F1 improvement interval **−0.47 to +6.12 percentage points**, accuracy **−1.01 to +10.11**. On the matched54, corresponding intervals are **−0.54 to +2.31** and **−0.81 to +3.36**. These small, repeatedly used validation samples support a promising point estimate, not a strong causal claim about rank alone.

## 4. Current strengths to preserve

On the same54 parseable documents, G scores:

- Document/transport/freight F1 **0.9683**, versus 0.9597 before.
- Locations **0.9690**, versus 0.9516.
- Containers **0.9300**, versus 0.9311: effectively flat overall, despite category-presence problems described below.
- Relation F1 **0.9113**, versus 0.9002.
- Addresses **0.7938**, versus 0.7516; exact addresses 121→129 of161.

The [ten-container SIJ0559535 document](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_5cabcd4b918a0689f0221f4feeae863190dd2795f1cbae565b781b23887971e3/ocr.txt) improves from eight recovered containers to all ten correct identifiers, seals and placements of five packages each. This is a concrete relation/list-completion gain worth retaining. Its description still has an extra PO prefix: a strong graph result need not mean every text field is correct.

Recommendation: keep the rank48/alpha32/5%-warmup, input-only positional configuration as the next baseline, retain the general real and synthetic coverage, and change **data coverage/style deliberately**. Do not simultaneously change rank, optimizer, prompt, representation and sampling, which would obscure the outcome.

## 5. Goods descriptions: audited errors and the training mismatch

The [description audit](analysis/run-ledger-and-next-experiment-20261009/descriptions/REPORT.md) reviews every current parsed description mismatch against OCR and compares the preceding rank32 prediction. [Per-document evidence](analysis/run-ledger-and-next-experiment-20261009/descriptions/inventory.json) includes the actual strings and ownership analysis.

### Error taxonomy

|Primary outcome|Documents|Meaning|
|---|---:|---|
|Exactly correct|25|Complete gold description recovered|
|Qualifier/continuation omission|11|Identity often present, but specification, modifier or continuation missing|
|Product identity / block loss|3|Core product disappears or wrong block is selected|
|Extra other-role text|7|References, marks or party wording appended to goods|
|Mixed omission and addition|6|Both missing product text and unwanted text|
|Character-copy error|2|Substantive spelling/copy corruption|
|Formatting only|3|Spacing/punctuation exact-match failure|
|Redundant repeated wording removed|1|Semantic loss less severe than the exact score suggests|
|Malformed JSON|2|No safely parsed description|
|**Total**|**60**|33 parsed mismatches, plus two malformed outputs|

Against rank32, the same54 parseable documents contain six exact→wrong regressions, two wrong→exact improvements, 21 exact→exact and 25 wrong→wrong. Description performance did not benefit from the overall gain.

Examples:

- [Filters](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_839005dd35a27c06f4a6184e24a654c8dfc6f01483253283cae8158a08aff397/ocr.txt): `LIQUID FLUID FILTER CARTRIDGES` becomes `LIQUID FLUID FILTER`. The continuation carries product identity, not disposable formatting.
- [Cheese](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_9f97c8edd5d4fd625822b0dd8287e6b32e8356fe3381970b185ffaba9677a6ca/ocr.txt): `GOUDA CHEESE 48% F.I.D.M. BLOCK 15KG` becomes only `BLOCK 15KG`.
- [Capacitor](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_1c3797a521af25c4dd20bc95238d6605c4ef404b343374bee3833c7be71541fd/ocr.txt): `CAPACITOR` becomes `CAPACITOR MANUFACTORY`; `MANUFACTORY` belongs to the shipper.
- [Bentonite](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_891555b99ca49335da7e37f1b2ccaf0d347400b5b8c4980527e2142cf6c84145/ocr.txt): `BYLON (BENTONITE)` is replaced with a corrupted consignee-address string.

### More verbose synthesis is not the missing ingredient

|Cohort|Median description words|Mean|90th percentile|Descriptions over50 words|
|---|---:|---:|---:|---:|
|Real training600|5|7.48|15|5 / 600|
|Synthetic training1500|71|73.36|111|1,185 / 1,500|
|Validation60|6|10.05|21|1 / 60|

These are word-count diagnostics, **not recommended generation length caps**. The current synthetic descriptions are already much longer than typical real ones. An instruction encouraging natural product assortments has worked, but it produces a style distribution unlike the real set. Even a short source such as `COMPRESSORS AND ACCESSORIES` can turn into a detailed multi-component product list.

There is also a stronger ownership/topology signal:

- **1,426/1,500 synthetic descriptions** occur as one contiguous OCR span after case/whitespace normalization.
- Only **10/200 template contracts** have more than one product region, representing74 synthetic descendants.
- In validation, 38 descriptions pass the contiguous-span test; G gets **24/38** exactly right. The other22 yield **1/22** exact.
- The same split was already poor in F: **27/38 versus1/22**. Thus it is not a new rank48-only weakness.
- Removing punctuation for this diagnostic moves three cases to the contiguous group, but leaves19 noncontiguous cases with only one exact extraction. Noncontiguity is still a proxy: it includes label transformations and cannot alone prove a layout cause.
- On the same54 parseable documents, a diagnostic word-multiset comparison gives description macro precision approximately **0.902** and recall **0.798** for G; prior recall was **0.839**. Word overlap is not a substitute for correct product meaning, but it supports the omission diagnosis.

### Boundary policy still needs a small, bounded adjudication

Thirteen description-inventory rows carry policy-review flags; that is **not 13 established bad labels**. Related KRONOS titanium labels differ in retention of `PAPER BAGS`; lot/batch wording is retained in one goods target from the marks area but omitted in another. Repeated/reordered product wording and inline destination qualifiers are the other bounded questions. These require source-based policy decisions before synthesizing more of the same boundary choices, not rewriting validation labels to favor predictions.

The finite task is to adjudicate the flagged boundary cases, then query training labels for those **same semantic patterns**. Preserve old gold and report old/new scores separately if any correction is approved. Do not use this as a reason to reopen every annotation or discard accepted work.

### Recommended intervention

1. Generate the same *kind* of description as the source: simple identity, identity plus specifications, or a genuinely detailed assortment. Preserve the existing ability to generate long descriptions without making every source verbose. No character quotas or exact HS-to-product classification task is needed.
2. Increase **independent source-family coverage** for interrupted, split and continued product text, using training/admitted non-validation sources. Retain customs references, marks and party text in their actual separate roles around those regions.
3. Provide the wording agent all product-owned regions together, as the pipeline already does. Audit that rendering preserves their ownership and that the label includes the complete generated product text while excluding the interleaved material.
4. Preserve and check coordinates for every changed region; do not repair ownership by inventing geometry. Reuse the existing geometrically validated expansion path, with its explicit missing-coordinate behavior.
5. Measure exact description match **and** product-token recall/contamination by source topology and description style. Keep exact full-field metrics primary.

This is not a request for another generic “long description” module: joint-fragment generation already exists. The needed change is the distribution and ownership coverage of its outputs.

## 6. Categories: three separate mechanisms

See the [category and quantity audit](analysis/run-ledger-and-next-experiment-20261009/categories/REPORT.md), [field/error inventory](analysis/run-ledger-and-next-experiment-20261009/categories/findings.json), and [deterministic sampling probe](analysis/run-ledger-and-next-experiment-20261009/categories/sampling-probe.json).

### 6.1 Invented enum strings

Five final outputs include invalid package categories: `PACKAGE_CRACK`, `PACKAGE_EGYPT`, `PACKAGE_BAG_STEEL`, `PACKAGE_BAG_PAPER_PAPER_MULTI_WALL`, and `PACKAGE_BULLET`. This refines the previous report's four: the fifth document first failed structural validation for duplicated seals, so the later frozen-vocabulary check was not reached. The new field-level scan finds both defects. This is a schema/vocabulary emission problem, distinct from selecting the wrong valid category.

A schema-constrained inference experiment is appropriate for this layer. XGrammar documents JSON-schema grammar compilation and token masking, including a Transformers example; that example uses a causal LM, so it does **not** itself validate our encoder-decoder integration. Test the actual T5Gemma2 path before relying on it. Constraints can rule out forbidden strings; they do not determine which allowed type the OCR supports. [Official XGrammar quick start](https://xgrammar.mlc.ai/docs/latest/start/quick_start.html).

### 6.2 Unsupported presence is a larger equipment issue than family confusion

Rank48 correctly emits **99/99 matched, gold-supported container type-category values** when it emits them: no wrong valid family substitution in that subset. It still has seven missing and12 extra type facts. Eleven extra type facts, paired with11 extra size facts, are guessed attributes on known container IDs.

Those **11 guessed pairs /22 extra category facts** concentrate in three documents:

- The seven-container refractory document extrapolates a printed `20ST` attribute from one row onto four other extracted IDs without equivalent row evidence.
- Two other documents have four and three extracted IDs respectively without printed equipment categories; the model supplies standard size/type anyway.

Size classification also has four wrong supported values, alongside95 correct, seven missing and12 extra. This is a mixed problem: unknown handling/ownership, plus size/alias interpretation. It is **not** evidence that the model cannot learn general-purpose versus refrigerated equipment.

The agreed “bare 40-foot wording defaults to standard” rule is unchanged. It applies when that length wording is printed; it does not authorize supplying size from no equipment wording at all.

### 6.3 Package classification is much weaker than container-family recognition

|Projected category family, all60|F precision / recall / F1|G precision / recall / F1|G correct / gold|
|---|---:|---:|---:|
|Container size|0.9038 / 0.8868 / 0.8952|0.8559 / 0.8962 / 0.8756|95 / 106|
|Container type|0.9231 / 0.9057 / 0.9143|0.8919 / 0.9340 / 0.9124|99 / 106|
|Package type|0.7321 / 0.7455 / 0.7387|0.7241 / 0.7636 / 0.7434|42 / 55|

The same54 parseable documents show package F1 falling **0.7573→0.7379**. Thus its slight all60 improvement is not a matched-cohort classification improvement. One validation document supplies the DG hazard-class and packing-group expectations; both models miss those facts, but a one-document slice cannot establish overall DG quality.

Examples include IBC wording reduced to generic packages, steel drums reduced to generic drums, plain paper bags predicted as multi-wall bags, and outer cartons selected where the current target represents uncounted inner bags. Fallback text also matters: `PLASTIC PALLET`, `RACKS` and `BALES` have policy-specific fallback targets where a more specific canonical category cannot be supported. Some predictions are physically plausible broad categories but wrong for that preservation contract. Teaching that distinction requires explicit examples; schema validity alone cannot distinguish them.

### 6.4 Several package classes are structurally absent from synthetic training

Document counts with each package category:

|Category|Real600|Synthetic1500|Validation60|
|---|---:|---:|---:|
|Paper bag|0|0|2|
|Plastic bag|1|0|1|
|Steel drum|3|0|1|
|Intermediate bulk container|7|0|1|
|Unpackaged single unit|5|0|1|

The synthetic set covers23 package types versus42 in the real training set. Zero support is not simply an unlucky random draw: a deterministic probe of the current 200-template sampling path found:

|Category|Why ordinary additional draws do not fix it|
|---|---|
|Paper bag|No train-fit donor observations, despite the vocabulary and generic renderer supporting the category|
|Plastic bag|Its only donor HS100710 is outside the configured ambient chapter pool;91 templates can accept a donor structurally but none has a nonempty compatible identity pool|
|IBC|Five usable donor observations,178 structurally accepting template pools, but no nonempty compatible identity pools under the current HS/equipment constraints|
|Steel drum|Eligible donors exist for two DG templates, but the chemical cargo path overrides packaging to DRUM/CARTON/BOX|
|Unpackaged single unit|Five donor observations, but no eligible current template|

Literal equipment-alias coverage is also sparse: `40RA`, `DC20`, and `20HO` appear in neither current real training nor synthetic OCR; `40RQ` occurs in two real training records and zero synthetic records. These are literal-code counts, not a claim that the underlying equipment semantics have no training coverage. Source/carrier context matters: the reviewed CMA CGM `40RA` mapping is documented as high-cube in the project's source-aware alias policy, so it must not be globally remapped to standard merely to match a prediction.

### Recommended intervention

- Add/enable legitimate **category + cargo-family + package + equipment** combinations at the actual sampler eligibility layer, and retain compatible selected subtypes through cargo generation. Do not merely increase a category's weight when its feasible pool is empty.
- Use registry/category knowledge and independently admitted source templates for missing classes. Do not seed synthesis with validation documents or their exact product/party values.
- Add examples where container IDs are available but attributes are not, and where only one row is typed. Keep known/unknown distinctions in rendering and targets. Continue the agreed defaults for printed shorthand; do not teach unsupported completion.
- Sample permitted source-aware aliases with category-preserving round-trip tests. Do not collapse operator-dependent codes into unconditional global aliases.
- Before spending on generation, publish feasible candidate counts for each intended stratum. A requested stratum with zero support should be a visible configuration failure, not a silent substitution.

## 7. Quantities and relations: preserve recall, improve restraint

The final relation gains are mostly recall: correct facts **187→202/211**, missing/wrong **24→9**, while wrong/extra facts improve only **31→30**. On the same54 parseable documents, exact relation graphs actually change42→41, despite higher micro F1. Better aggregate recovery and complete-graph correctness are different outcomes.

The targeted audit finds **23 placement quantities absent from gold across seven documents**. They are not all fabricated digit strings:

- Eight are printed **outer pallet counts** assigned where the target package level is the inner bags.
- Others copy a container count, a size-like number20/22, or a shipment total3700 into a per-container quantity.
- Two `646` predictions occur in OCR with two container IDs but a declaration of three containers and repeated unpositioned quantity rows. These need layout/ownership adjudication before being called confirmed semantic false positives. They are currently gold-absent predictions, not automatically proven wrong labels or wrong model behavior.

Training support helps explain what to test next:

- Real training has24 membership-only documents /94 membership edges.
- Synthetic training has51 such documents /196 edges, but they descend from only **seven families**.
- Validation has seven membership-only documents /23 edges, plus one mixed document.
- All1,500 synthetic documents have a **positive typed package row**. Current generation rejects some unknown/mixed quantity structures rather than representing them.

Therefore, enable explicit unknown/membership-only/mixed support in the synthesis contract where the source supports it; adding more examples through an always-positive path cannot teach those distinctions. Use contrastive examples with the same shipment facts but different *printed ownership evidence*: shipment total only, supported per-container split, and inner-vs-outer packaging. Every text change must have an aligned target and valid geometry; indiscriminate deletion of numbers is not acceptable augmentation.

Keep the existing inner-package policy and exact-derived-total policy. The objective is to learn which quantity belongs to which entity, not to weaken those definitions. Measure false-positive quantities separately from relation recall so new restraint does not silently lose correct placements.

## 8. Addresses, remaining structure, and error budget

Address extraction is improving, not solved. On all60, exact addresses rise125→139 of177, and F1 rises0.7267→0.7875. On the matched54 the gain is121→129/161. This analysis does not repeat a full address semantic adjudication or propose another address-reformatting migration. Preserve the current casing/delimiter policy and use the existing error inventory to select any later narrowly scoped address audit.

Ignoring commas/whitespace changes G's complete-target F1 only **0.8849→0.8864**, accuracy **0.8390→0.8405**, adding four exact scalar matches. Formatting is not the main remaining error source.

G also has two malformed JSON outputs and11 additional parseable schema-invalid outputs. The latter include invented enums, duplicate seals, invalid equipment IDs, dangling placement IDs, zero volume and an OCR-like date spelling. Syntax constraints cannot alone enforce all cross-record or semantic rules. Audit constrained inference on the real implementation before claiming a production-validity rate.

### Gold-assisted error-budget calculation — not model results

The following diagnostic replaces selected erroneous **aligned scalar fields with their gold values only in already parseable outputs**. Malformed outputs are left unchanged. It answers how much these fields could matter in this saved result, not what a proposed training change will achieve. Scenarios overlap and cannot be added.

|Hypothetical perfect fields|Full60 field F1|Full60 field accuracy|
|---|---:|---:|
|None: actual saved predictions|0.8849|0.8390|
|Descriptions only|0.8978|0.8513|
|Addresses only|0.8988|0.8536|
|Descriptions + addresses|0.9117|0.8659|
|Equipment/package/DG category leaf fields only|0.8987|0.8588|
|Package quantities only|0.8945|0.8525|
|All four field families together|0.9356|0.9001|

This explains why fixing descriptions alone is insufficient for the joint0.90/0.90 goal, despite their conspicuously low exact match. Descriptions are60 of2,566 gold scalar facts; quantities, categories, addresses and structural coverage also matter. The combined row is an algebraic upper-bound-style diagnostic on selected aligned paths, not an attainable-score promise or a simulated decoder.

## 9. Recommended next experiment and acceptance plan

### Step 1 — freeze the baseline and resolve the finite label questions

Keep G's checkpoints, prediction files, dataset hashes and current metric definition immutable. Adjudicate the specific lot/package-description boundary candidates and the two uncertain646 placements. Only if evidence establishes a gold error, fix the same policy consistently in affected training/validation records with a receipt and retain old-gold scores. Do not optimize annotation decisions against the preferred prediction.

This is a **small policy/ownership review**, not a restart of dataset annotation. Confirmed model errors can already be used to define training-data requirements; they are not blocked by every unresolved example.

### Step 2 — make a coverage manifest before another synthesis campaign

Count eligible families and feasible sampling pools for:

1. Short/simple, specification-rich and genuinely long product descriptions.
2. Contiguous, interrupted and multi-region product ownership.
3. Paper/plastic bags, steel drums, IBCs and unpackaged cargo, with their compatible goods/equipment paths.
4. Known/unknown equipment attributes and valid source-aware aliases.
5. Membership-only, quantified and mixed placements; inner/outer packaging; shipment totals versus row quantities.
6. Long container lists, thermal/DG goods, negotiability, transshipment and the other working distributions we must retain.

Use training sources or independently admitted non-validation sources; check document and shipment/family overlap. Select counts from the available independent families and desired coverage, not from the number of descendants alone. More copies of seven membership-only layouts do not equal broad layout coverage.

### Step 3 — targeted implementation and a reviewable preflight

The code-level work belongs in sampler feasibility/subtype preservation, unknown-quantity support, source-style conditioning and template admission—not in a special-case postprocessor correcting particular validation documents.

Before the main generation volume, exercise every newly enabled combination with deterministic plan/render/label tests, including the branch where evidence is absent. Audit text and labels together for material subtype, quantity ownership, complete product identity and distractor exclusion. Re-run coordinate plausibility checks on changed regions. Unchanged general synthesis remains covered by its existing tests plus a regression sample.

Do not certify a stratum merely because its JSON validates. Its acceptance condition is the demonstrated correspondence between rendered evidence and target meaning.

### Step 4 — one controlled data experiment, not a sweep

Recommended next training experiment: retain the real600 and current general coverage, and add a **targeted synthetic tranche** after the feasibility/preflight above. Keep G's rank48, alpha32, 5% warmup ratio, optimizer, input-only positional representation, tokenizer, target format and evaluation policy. If the proposed additional1,000-sample scale is retained, allocate those1,000 by audited feasible strata rather than uniformly across templates; several traits can coexist in the same sample.

For description style, include ordinary short source-shaped variants as well as complex ownership examples; do not solve the imbalance by truncating existing descriptions or discarding valid long examples. Track the resulting mix explicitly. For categories, new samples must actually contain the previously unreachable classes, not only a configuration request for them.

A larger dataset at the same epoch count also increases optimizer updates. Report both epoch-matched and approximately update-matched comparisons; do not attribute all improvement to data diversity alone. This is one practical coverage intervention with component diagnostics, not a clean estimate of each component's independent causal effect.

### Step 5 — measure the desired changes and protect the strengths

Before launch, persist this scorecard and the sample/cohort IDs:

|Layer|Success evidence|Regression check|
|Whole task|Full60 field precision/recall/F1/accuracy; documents above both targets; valid JSON/schema|No headline score restricted to easy or parseable cases|
|Descriptions|Exact field match, product-token recall, contamination inventory; by contiguous/split and short/long cohorts|Do not improve recall by appending party/reference/marks text|
|Package categories|Per-class support, correct/incorrect/missing, invalid enums|Do not replace rare valid types with generic labels to improve validity|
|Equipment|Supported attribute accuracy **and** false-positive attributes on untyped IDs|Retain correct99/99 emitted type-family behavior and valid defaults for printed shorthand|
|Quantities/relations|Relation precision/recall/F1/accuracy, exact graphs, supported quantity accuracy, unsupported quantity rate|Do not trade current high relation recall for indiscriminate omission|
|Strong fields|Locations, transport/freight, container IDs/seals, party names/addresses|Report paired regressions, including recoveries/losses caused by JSON parsing|
|Generation/positions|Audited semantic correspondence, feasible-stratum counts, geometry/coordinate coverage|No silent fallback to a different package type or unsupported coordinates|

Save **per-document predictions at every evaluation**, not only the final pass, alongside retained checkpoints. This removes the current blind spot at epochs2.5/5/7.5. Choose best by the predeclared extraction metric, and verify exported weights against that checkpoint.

Use the same-document parseable intersection only as a secondary semantic comparison. Keep all60 as primary. Use paired document bootstrap and inspect the actual improved/regressed cases; with60 repeatedly used documents, small changes are noisy and adaptive overfitting is a risk. A fresh independent holdout is valuable before a later strong generalization claim, but it need not block this focused experiment.

### What not to do next

- Do not add another rank/alpha/optimizer change before testing the demonstrated data gaps.
- Do not assume more long goods prose fixes continuation/ownership.
- Do not rely on category weights when eligibility filters or downstream overrides make a class unreachable.
- Do not change field descriptions and expect the **input-only** T5 model to see them: those descriptions influence extraction/synthesis agents and resulting data, not its current training prompt.
- Do not promise constrained decoding will repair correct-but-wrong enums, field ownership or cross-array relationships.
- Do not make address punctuation another broad migration: the measured formatting-only gain is small.

## 10. Runtime, checkpoint integrity, and evidence

Trainer runtimes including scheduled evaluation were A168.13min, B93.09, C94.73, D71.98, E134.74, F197.62 and G202.73. Different dataset sizes, input lengths, epoch counts and evaluation schedules prevent interpreting that sequence as a throughput benchmark.

The cleaner F/G comparison adds2.58% Trainer runtime. Most of the difference is evaluation time; subtracting scheduled evaluation leaves approximately1.13% extra time, still including saving and other overhead. EVA initialization separately rises172.6→411.8sec. Maximum scheduled-eval allocated GPU memory rises8.254→8.353GiB. These are recorded run observations, not an isolated kernel benchmark.

G's much lower cumulative training loss is mostly the early transient:99.84% of the eventual cumulative-loss difference is already present at update10. Final loss windows are nearly equal,0.004883 versus0.004890. Maximum logged gradient norm drops671.82→131.94, but median only2.221→2.108. Neither statistic proves rank alone caused the extraction improvement.

G retains checkpoints165/330/495/660 with optimizer state, and its final export matches checkpoint660 across504 tensors. B/C/F exported the last checkpoint even where another was selected as best. The ledger records the matching exported checkpoint paths and prediction hashes explicitly.

### Reproducible evidence

- [Seven-run metrics/config/provenance ledger](analysis/run-ledger-and-next-experiment-20261009/ledger.json).
- [Field-group metrics, all60 and per-run parseable cohorts](analysis/run-ledger-and-next-experiment-20261009/ledger-field-groups.csv).
- [Gold-assisted error-budget calculations](analysis/run-ledger-and-next-experiment-20261009/oracle-error-budget.json).
- [Ledger CPU replay](analysis/run-ledger-and-next-experiment-20261009/analyze_ledger.py).
- [Current run's full audit, plots and comparison](analysis/real600-synthetic1500-rank48-results-20261009/REPORT.md).
- [Prior1,500-synthetic run](analysis/real600-synthetic1500-results-20261008/REPORT.md).
- [Input-only / first500-synthetic analysis](analysis/real660-inputonly-and-synthetic500-20261007/REPORT.md).
- [Positional-input comparison](analysis/real660-position-results-20261006/REPORT.md).
- [R14 comparison and correction of the compact-target interpretation](analysis/real660-r14-training-results-20261006/REPORT.md).

The new replay checks identical validation IDs, current validation file hash, the historical A replay's adapter hash, best-metric agreement with Trainer state, and conservation of scalar counts between aggregate and per-field/group accounting. Its new gold-assisted calculations operate only on parsed aligned leaves and are labeled separately from model results. Targeted audits preserve their full inspected inventories and deterministic sampling probes. Production code and data remain unchanged.

Final validation: the CPU replay reproduced every available structured final metric for F/G within1e-12, took **8.30 seconds** and peaked at **125,840KiB resident memory**. Programmatic checks verified all22 visible checkpoint rows, all seven common-gold rows and56 field-group entries against the generated inventories. All77 local links across this document and the two targeted reports resolve. The ledger script passes Ruff; the two targeted audits report their own replay/assertion checks. This is an analysis-runtime measurement, not a training-performance claim.

**Bottom line:** the latest model can already recover many core fields and relations well. The next credible route to improvement is better teaching of **where to stop, where to continue, what belongs to which entity, and when an attribute is not supported**, plus reachable rare-category coverage—not simply more verbosity or a larger adapter.

## 11. Follow-up: points1–2 reviewed before choosing synthesis volumes

**Policy update:** the subsequent user clarification approves intact descriptive phrases but excludes separately owned Marks/Numbers content. The Marks-transfer and blanket origin-separation recommendations below are historical proposals, not the current policy. The [description-block audit](analysis/description-block-policy-audit-20261009/REPORT.md) supersedes those recommendations; no repair was applied from them.

The [policy and feasibility review](analysis/run-ledger-and-next-experiment-20261009/policy-and-feasibility/REPORT.md) now examines the13 flagged description cases, an additional source-order example, the uncertain646 allocation case,12 training-only witnesses and the actual sampling functions. It preserves current labels, predictions and reported scores.

Main policy recommendations, **not yet applied**:

- Copy intact product-specific packing/capacity expressions, including count×capacity, while keeping separate accounting totals out. This is an explicit proposed relaxation of the current blanket exclusion of shipment counts from descriptions, not a silent “repair.” Training currently keeps a multiplier in one of five identified count×mass examples and removes it in four.
- Include goods-owned lots, dates and specifications even in marking/continuation areas; exclude package-number ranges, package seals, routing marks and company identities that are not established product wording.
- Put standalone explicit origin declarations in goods origin consistently; preserve an attached product suffix such as-EGYPT unless a separate role is established.
- Follow OCR order and printed punctuation; deduplicate whole repeated copies, not arbitrary substrings or differently rounded dimensions.

Feasibility refinements:

- **Steel drums already have a working configuration interface.** In-memory scenario probes produced57/60 valid steel-drum draws; three were properly rejected by cube guards. A new sampler is unnecessary; complete rendering validation is still required.
- One unpackaged train-source candidate works through existing whole-unit functionality,10/10 scenario probes. It needs template admission.
- Membership-only allocations already have seven admitted families; entirely untyped equipment has six;178 admitted sources have descriptions≤10 words. These do not require a new generation subsystem.
- Mixed quantities, fallback/no-count/no-package rows require scenario **and** physical-planning changes. Current source eligibility alone cannot be relaxed safely.
- Plain paper bags have no real600 donor. A separate reviewed non-validation support-source interface is not currently exposed by campaign configuration, despite that source policy being allowed in principle.
- Carrier/source-dependent40RA and20HO aliases need context-aware surface handling, not unconditional global mappings.

The646 PDF has three container rows, but OCR omitted one identifier and retained two repeated descriptions without coordinates. Its two predicted quantities must not be treated as confirmed fabrications; it is a separate incomplete-input/ownership-sensitive evaluation case. It does not block other work.

No next-campaign volume or weights were chosen in this follow-up. Those should be based on the agreed policy and verified feasible combinations, not requested counts that the sampler cannot actually produce.

## 12. Description-block scope audit after the policy clarification

The [full audit and recurrence-prevention plan](analysis/description-block-policy-audit-20261009/REPORT.md) covers the current **600 real training, 60 real validation and 1,500 synthetic records**, plus all **200 effective synthesis families**. It is an analysis pass: labels, OCR, positions, production code, templates and recorded metrics are unchanged. No paid labeling/generation requests or training were run.

The clarified rule is **block ownership first, intact transcription second**: copy the actual goods-description wording/continuations in source order, preserving embedded packing/count/capacity and product qualifiers. Do not import product specifications from a separately owned Marks block. Keep structured package/mass/HS/relations semantics unchanged. A separately labeled accounting or customs field is not made descriptive just because it is geometrically near goods text.

### Established scope

- **Real:** 145 candidate flags across all660. Detailed review of45 documents (35 PDF-backed /45 pages, plus ten OCR-only phrase checks) establishes at least37 descriptions needing change (31 training, six validation), including one with an unresolved continuation subquestion. Three reviewed controls have correct content, four remain boundary questions, and one has a missing-description OCR problem. The remaining103 flagged sources are candidates, not established bad labels or certified passes; three confirmed phrase omissions were outside the flag set.
- **Synthetic:**52 current descendants across seven families have confirmed description-policy mismatches.45 have a label/binding-only correction direction; seven generate their product only in Marks and require a localized rendered-region decision. A separate25-family/188-descendant queue covers ordinary quantity introductions (173), a cross-column mutable region (eight), and an inherited source-only extent (seven). These188 are not188 proven wrong description labels.
- **Input exceptions:** glycerin2b88d69f and RATHIPONe9275395 have proper goods-column descriptions in PDF that are absent from OCR; Marks text was used instead. Resolve input recovery/exclusion separately rather than introduce PDF-only target values. RATHIPON is additional to the45 real adjudications above.

### Why existing checks passed

All1,500 saved generated product values survive into rendered text and target. No post-generation numeric stripping was found, and all200 effective blueprints compile. The main failure lies earlier: old extraction rules exclude accounting-like wording and import product facts from Marks; rebasing and final assembly faithfully reproduce those shortened/cross-block target expressions.

Actual compiler probes accept both an old target omitting an embedded count and an expression reversing product order while importing separate Marks. Replay proves agreement with a mapping, not completeness/correctness of that mapping. The casing normalizer is not responsible.

### Repair direction before further coverage expansion

Resolve description spans once per affected source, preserving source block/continuation order. Reuse current edit receipts to project the final rendered spans into descendant targets. Retain the working separation between model-generated wording and host-controlled facts: host-rendered packing inside a description can appear in the description label without letting the LLM invent shipment totals.

Update the schema, cargo map/review/correction wording and compiled description projection together. Test both omission and contamination: deleted multipliers, appended Marks-only codes, reversed continuations and stale host quantities must fail. Positive controls must keep goods-owned lots/seal ranges and valid interleaved-HS exclusions. Label-only changes preserve input/coordinate hashes; wrong-region cases require separately checked text/geometry edits.

Do not globally delete lots, origin wording, capacities or numbers, and do not use nearest-heading regexes as ownership proof. Do not regenerate the whole dataset: current generated wording is retained correctly and most confirmed synthetic corrections can reuse it unchanged.

Detailed evidence: [real review](analysis/description-block-policy-audit-20261009/real/REPORT.md), [synthetic family-by-family report](analysis/description-block-policy-audit-20261009/synthetic/REPORT.md), [active-code trace and falsification probes](analysis/description-block-policy-audit-20261009/pipeline/REPORT.md). Historical model scores remain against historical gold; measure any correction effect only after both models are replayed against the same versioned reference.
