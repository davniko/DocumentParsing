# MPCI extraction: run results and next experiments

Updated: 2026-10-17. This is the consolidated results ledger for **fourteen completed runs**, now including corrected/anonymized **N/R27** alongside synthetic-only **L/R22** and restored-real **M/R23**. Keep this document as the entry point rather than reconstructing the experiment history from artifact directories.

**Current conclusion:** N/R27 completes with **6,900 synthetic + 600 anonymized real**. Its selected/exported model is **checkpoint 1174, epoch 4.9984**, not epoch 10: scheduled field F1/accuracy **0.9038/0.8640**, separate saved final pass **0.9030/0.8629**. This is the highest native final field F1 in the ledger, but not a new across-the-board best or a controlled identical-target comparison. On unchanged facts in the same 57 parseable M/N documents, overall F1 is effectively tied (**0.9162→0.9158**) and accuracy slightly lower (**0.8899→0.8857**). Phones improve **94→100/111** and unchanged countries **212→221/228**; descriptions, shipment totals and equipment-category precision remain weak. Relations tie on that shared cohort; category regression survives target/parse controls. Export/resume integrity verifies. See [section 19](#19-correctedanonymized-nr27-6900-synthetic-plus-600-real) and its [source-reviewed errors](analysis/r27-training-results-20261017/SOURCE_REVIEW.md).

The latest **training-results audit** used local artifact inspection, CPU metric replay and source-text review only; no dataset, production code, config or checkpoint was changed and no inference/training was launched. The separately requested wording probes are documented in their own report. **Sections3–18 preserve historical analyses and plans**; their “current” wording describes those snapshots, not today's labels. J/K trained on wrapped targets; L/M/N use direct targets. M restores printed separators in14 validation HS strings across13 documents. Historical models are **not** scored against replacement gold. Section18 excludes the entire HS family for its historical comparison; section19 conservatively excludes only changed document/field families against each run's own frozen reference, including subsequent country/package/phone changes. N includes the later phone/container-ID presentation repairs, new goods/DG wording, country policy/presentations, surgical real anonymization and bounded package coverage changes together; it does not isolate any one repair.

## 1. Reading the metrics correctly

- The target is **both** field F1 and field accuracy above 0.90 for “good,” and approximately 0.95 for “great.” A relation F1 above 0.90 does not establish that goal for the complete extraction.
- Field precision/recall/F1 count aligned scalar values, with list-order-invariant matching. **Field accuracy is correct values divided by the union of occupied aligned gold/prediction paths**, not whole-document accuracy and not token accuracy. An incorrect value at one path contributes a precision and recall error but occupies one union path.
- Relation and category metrics compare projected facts. Their accuracy is set overlap/Jaccard. An aligned placement-field score is not necessarily identical to the relation-set score because matching and deduplication differ.
- Invalid JSON gets no recovered-field credit. Parseable schema-invalid JSON can receive partial field credit. Explicit nulls in the target count under the current contract.
- Whole descriptions/addresses are individual string fields. A missing word and a completely wrong description both lose that field's exact-match credit. The error taxonomy below distinguishes their practical severity without silently changing the primary metric.
- **Native scheduled evaluation** means the values logged during that run, using its contemporary labels/settings. Earlier **common-gold replays** below refer to their explicitly dated frozen snapshots, not a license to score historical models against subsequent policy changes. Section18 replays each run against its **own original reference** and compares only unchanged families across contracts. None of these CPU replays is new model inference.
- Best scheduled evaluation and the separate final prediction pass are different observations. Their scores can differ slightly even when the exported weights are correct. Do not mix a best-epoch aggregate with an epoch10 error inventory.

### Important historical corrections

1. The earlier “compact JSON” experiment did **not** isolate pretty versus compact targets: the previous run's actual cached decoder targets were compact too. The change included casing/equipment labels, evaluation settings and prompt/schema changes.
2. R14 to R16 changed address-punctuation labels as well as adding positional input. That transition is not a pure positional ablation.
3. Removing the explicit prompt was a cleaner comparison: the R16 positional data/settings were retained. Each training run started from the base model/EVA setup; these are not successive fine-tuning stages of the preceding adapter.
4. Older runs did not all use the current explicit-null instruction policy. Before the 1,500-synthetic run, 38 `sameAs` nulls and two negotiability nulls added 40 target facts across 36 validation documents. This particularly changes historical schema-validity scores when evaluated against today's schema.
5. The previous rank32 1,500-synthetic run selected epoch7.5 but exported epoch10. The rank48 run exports its selected epoch10 checkpoint correctly. Granular comparisons below use the **available saved epoch10 outputs**, not the earlier run's best epoch7.5 predictions.

## 2. Experiment registry: fourteen completed runs

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
|H|[Corrected description blocks and declared packaging, rank48](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic1500_descblocks_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml)|2,100|Same G settings; corrected description boundaries, package level and allocation labels/data; no added training records|10 / 660|33, or 5%|
|I|[Expanded coverage, 2,500 synthetic, rank48](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic2500_descblocks_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml)|3,100|Add1,000 synthetic records;300 templates; retain description/declared-package policies and input-only positions|10 /970|49, or5%|
|J|[UAE expansion, 5,000 synthetic, rank48](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic5000_uae_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml)|5,600|Add2,500 UAE-endpoint synthetic records;360-template registry; repaired party/date/delivery contracts; retain real600|10 /1750|88, or5%|
|K|[UAE coverage replacement R19, rank48](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic5000_uae_coverage_r19_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml)|5,600|Replace2,500 UAE descendants with registry-form/alias/mixed-equipment coverage; same360 templates and sample count; finite shared-label repairs|10 /1750|88, or5%|
|L|[Synthetic-only7,000, R22](../configs/training/production/t5gemma2_270m_lora.mpci_bl_synthetic7000_old_uae_reinforcement_r22_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml)|7,000 synthetic|Remove real600; repaired/expanded synthetic pool plus old-UAE reinforcement; direct target|10 /2190|110, or5%|
|M|[Real600 + synthetic7,000, R23](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic7000_hs_preserved_r23_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml)|7,600|Retain L's exact synthetic shard; restore real600 and printed real/validation HS formatting; same rank48 and direct target|10 /2380|119, or5%|
|N|[Anonymized real600 + synthetic6,900, R27](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic6900_packages_r27_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml)|7,500|Phone/ID formats, country/wording/privacy repairs,200 package examples added/300 Egypt synthetic archived; keep rank48/settings; resumed after reboot|10 /2350|118, or5%|

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
|H|7.503 / 495|0.8691|0.8057|**0.9287**|**0.8736**|55/60|45/60|
|I|10 /970|**0.8967**|**0.8562**|**0.9348**|**0.8939**|**59/60**|**52/60**|
|J|10 /1750|**0.9013**|**0.8643**|**0.9450**|**0.9009**|**59/60**|**51/60**|
|K|10 /1750|0.8941|0.8520|0.9336|0.8996|59/60|51/60|
|L|10 /2190|0.8375|0.7877|0.8303|0.8377|58/60|38/60|
|M|7.501 /1785|**0.8990**|**0.8581**|0.9291|**0.9276**|58/60|49/60|
|N|4.998 /1174|**0.9038**|**0.8640**|0.9364|0.9005|59/60|49/60|

These are useful historical results, not fourteen controlled experiments on an unchanged scoring contract. In particular A is case-sensitive, B/C best checkpoints differ from their exported final predictions, and H has corrected validation labels. H's saved predictions are from its correctly reloaded best495 checkpoint, not epoch10. H/I share identical training-time validation file hashes; I's saved predictions come from its correctly exported970 checkpoint. J uses later approved labels; section15 records that historical comparison. J/K have identical validation references and training hyperparameters. L/M/N add further data/target changes; sections18–19 separate own-contract scores from fair unchanged-fact comparisons. M and N's separate selected-model prediction passes differ slightly from their scheduled best evaluations despite verified correct exports.

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
|H|2.503 / 165|0.06608|0.8482|0.7753|0.8350|0.8008|57|
|H|5 / 330|0.06169|0.8617|0.7970|0.8826|0.8405|56|
|H|7.503 / 495|0.06848|0.8691|0.8057|0.9287|0.8736|55|
|H|10 / 660|0.07597|0.8607|0.8003|0.9249|0.8346|56|
|I|2.506 /243|0.06068|0.8487|0.7769|0.8756|0.8096|55|
|I|5.010 /486|0.06317|0.8735|0.8162|0.9210|0.8809|56|
|I|7.516 /729|0.06912|0.8695|0.8070|0.8927|0.7914|57|
|I|10 /970|0.07949|0.8967|0.8562|0.9348|0.8939|59|
|J|2.503 /438|0.05362|0.8882|0.8421|0.8878|0.8658|59|
|J|5.006 /876|0.06344|0.8886|0.8414|0.9044|0.8909|58|
|J|7.509 /1314|0.06427|0.8650|0.8017|0.8662|0.8588|56|
|J|10 /1750|0.07333|0.9013|0.8643|0.9450|0.9009|59|
|K|2.503 /438|0.05324|0.8878|0.8374|0.8768|0.9081|58|
|K|5.006 /876|0.06529|0.8916|0.8442|0.9202|0.8625|58|
|K|7.509 /1314|0.06534|0.8821|0.8314|0.9157|0.8731|58|
|K|10 /1750|0.08245|0.8941|0.8520|0.9336|0.8996|59|
|L|2.503 /548|0.13870|0.8135|0.7496|0.8889|0.8419|55|
|L|5.005 /1096|0.14581|0.8194|0.7604|0.8637|0.8231|57|
|L|7.507 /1644|0.16431|0.8175|0.7576|0.7981|0.8203|57|
|L|10 /2190|0.18553|0.8375|0.7877|0.8303|0.8377|58|
|M|2.501 /595|0.06124|0.8727|0.8147|0.9104|0.8797|55|
|M|5 /1190|0.06547|0.8880|0.8410|0.9425|0.9074|57|
|M|7.501 /1785|0.07045|0.8990|0.8581|0.9291|0.9276|58|
|M|10 /2380|0.07852|0.8758|0.8164|0.8986|0.8820|56|
|N|2.499 /587|0.05616|0.8951|0.8522|0.9303|0.8833|59|
|N|4.998 /1174|0.06556|0.9038|0.8640|0.9364|0.9005|59|
|N|7.495 /1761|0.07118|0.8803|0.8219|0.8873|0.8517|57|
|N|9.994 /2348|0.07542|0.8925|0.8420|0.9569|0.8983|57|
|N|10 /2350|0.07687|0.8734|0.8098|0.9052|0.8538|56|

More epochs are not uniformly better. G improves late while F peaks earlier, and both lose ground at some intermediate evaluations. G's eval loss rises while its final extraction scores improve. Select checkpoints on extraction metrics, not minimum teacher-forced loss. Retain predictions as well as weights at each evaluation: aggregate logs alone cannot localize the intermediate regressions.

## 3. Historical pre-repair common-gold comparison of saved predictions

**Frozen G-era reference:** this section uses2,566 scalar facts and211 relation facts before the description/packaging repairs. The current comparison in section13 uses2,576 scalar facts and220 relation facts. Do not compare numbers across these two tables as if the gold were unchanged.

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

## 13. Corrected description and declared-packaging run H

### 13.1 What changed and how this comparison was made

H trains on **600 real +1,500 synthetic documents**, with the same60 validation documents as G. The changes are the approved goods-description block policy and declared package-level/allocation repairs. Rank48, alpha32, rsLoRA, EVA settings, input-only positional input, compact JSON, LR0.0001, ScheduleFree AdamW,33 warmup updates, batch1×accumulation32, seed424,660 total updates, generation maximum3,072 and evaluation/checkpoint interval165 are unchanged. No newly admitted300-registry templates or extra synthetic records are included.

The [resolved-configuration and integrity audit](analysis/corrected-data-run-results-20261009/TRAINING_INTEGRITY.md) compares every flattened setting, recorded package/source hashes, export hashes and checkpoint contents. Only data/policy/contract identifiers differ among resolved settings; the recorded extraction schema module changed, while training/evaluation/runtime code hashes match. EVA sees the same candidate order but converges after187 instead of171 forwards, so its data-dependent initialization is not asserted to produce identical starting weights.

Validation gold changed in **27/60 documents,46 scalar paths**:

|Changed field|Paths|
|---|---:|
|Goods description|25|
|Declared package quantity|5|
|Declared package category|5|
|Per-container placement quantity|11|

There are now **2,576 gold scalar values,220 relation facts and269 category facts**, versus2,566/211/269 before. All old saved outputs were rescored against this same corrected complete target. This includes every address, description, numeric value and explicit null, not just unchanged easy fields. G's saved F1 changes from0.8849 against its old gold to0.8867 against corrected gold without changing its predictions: part of an apparent metric change can therefore come from annotation correction alone.

Exact pinned current validation SHA-256: `9f3b4c36fe5d6310b3fb79a6d380a06d24d2fe8583acf80748d83861b2fe94da`; training SHA: `0340136994089effdd153cc76bc052dd95179c60b2868711a036d717989da642`.

### 13.2 Training progression and best checkpoint

These are **native scheduled metrics**, against each run's contemporary gold. Epoch2.5/7.5 are rounded from2.502857/7.502857. This table supports within-run progression and a qualified same-stage comparison, not an exact common-gold intermediate comparison.

|Epoch|G field F1 / accuracy|H field F1 / accuracy|G relation F1|H relation F1|G category F1|H category F1|G/H valid JSON|
|---|---:|---:|---:|---:|---:|---:|---:|
|2.5|0.8523 /0.7885|0.8482 /0.7753|0.8438|0.8350|0.8007|0.8008|59/57|
|5|0.8506 /0.7849|0.8617 /0.7970|0.8605|0.8826|0.8302|0.8405|56/56|
|7.5|0.8234 /0.7374|**0.8691 /0.8057**|0.7990|**0.9287**|0.7904|**0.8736**|53/55|
|10|**0.8858 /0.8401**|0.8607 /0.8003|0.9120|0.9249|0.8597|0.8346|58/56|

H is ahead at epochs5 and7.5 on the native field scores, but behind G's late recovery at10. H reaches its best field F1 at7.5; continuing to10 loses **0.843 percentage points of field F1** and **3.905 points of category F1**, while relation F1 stays above0.92. Its validation loss is lowest at5, not at the best extraction checkpoint:0.06608→0.06169→0.06848→0.07597.

The final export is **checkpoint495, epoch7.5**. Its adapter SHA-256 `267b65f8ffce647fb72ee617e5413debfe0d1efa3fa80866ba7e8a170da83d1c` matches495 only, byte-for-byte. G's export matches660 only. Both runs retain checkpoints165/330/495/660 with optimizer/scheduler/RNG/trainer state. The earlier wrong-best-export defect has not recurred.

H's separate saved best-weight prediction pass scores F1 **0.869002**, slightly below its scheduled selection score0.869108; this is not evidence of exporting the wrong weights. All per-field/error comparisons below use those saved predictions. **Only final/best prediction text was retained**, so epoch5/7.5/10 per-document common-gold rescoring would require new checkpoint inference, which this audit did not run. All scheduled aggregate metrics are available and reported.

### 13.3 Eight runs against the same corrected complete gold

Saved exports, all60 documents, case-insensitive/order-invariant primary scorer. A is the hash-verified historical checkpoint380 replay; B/C/F retain their previously documented export-selection caveats. The table is a common scoring contract, not a controlled experiment isolating one factor across all eight runs.

|Run|Field precision|Recall|F1|Accuracy|Fact F1|Relation F1|Category F1|JSON /60|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|A: real600,20 epochs|0.8453|0.7318|0.7844|0.7018|0.7820|0.8107|0.7780|54|
|B: equipment/casing|0.8535|0.6875|0.7616|0.6675|0.7610|0.7673|0.7598|51|
|C: positions,prompted|0.8743|0.7450|0.8044|0.7142|0.8013|0.8382|0.7305|53|
|D: positions,input-only|0.8672|0.8109|0.8381|0.7708|0.8339|0.8818|0.7934|56|
|E: +500 synthetic|0.8903|0.7908|0.8376|0.7606|0.8319|0.8970|0.8268|53|
|F: +1,500,rank32|0.8916|0.8370|0.8634|0.8021|0.8608|0.8858|0.8668|56|
|G: +1,500,rank48|0.8958|**0.8777**|**0.8867**|**0.8424**|**0.8830**|0.9248|0.8597|58|
|H: corrected data,rank48|**0.9082**|0.8331|0.8690|0.8059|0.8632|**0.9287**|**0.8736**|55|

G still has the strongest **unconstrained all-document field scores**. H improves precision and relation/category F1, but loses field recall. It has2,146 correct scalar values versus2,261 in G, and217 wrong/extra scalar predictions versus263. Thus it is more restrained, but retrieves115 fewer correct values overall. The next subsection identifies where that loss comes from rather than treating it as a uniform semantic decline.

### 13.4 Parsing coverage versus complete-target content

|Scope / model|Precision|Recall|Field F1|Field accuracy|Relation F1|Category F1|
|---|---:|---:|---:|---:|---:|---:|
|All60,G|0.8958|0.8777|0.8867|0.8424|0.9248|0.8597|
|All60,H|0.9082|0.8331|0.8690|0.8059|0.9287|0.8736|
|Same54 both parse,G|0.8994|0.8990|0.8992|0.8640|0.9312|0.8599|
|Same54 both parse,H|0.9091|0.8995|**0.9042**|**0.8686**|**0.9506**|**0.8983**|
|H's own55 parseable|0.9082|0.8987|0.9034|0.8671|0.9461|0.8987|

The matched54 contain2,367 gold scalars, including160 addresses and54 descriptions. They are **not** the older30k comparison that excludes changed goods/addresses. G/H recover2,128/2,129 correct values; wrong/extra scalar predictions decrease238→213. The small aggregate content gain comes predominantly from better precision and targeted descriptions/packages, with other regressions offsetting much of the added recall.

Exact loss decomposition:

|Parse transition|Documents|G correct values|H correct values|Difference|
|---|---:|---:|---:|---:|
|Both parse|54|2,128|2,129|+1|
|G parses,H malformed|4|133|0|−133|
|G malformed,H parses|1|0|17|+17|
|Both malformed|1|0|0|0|
|**All60**|**60**|**2,261**|**2,146**|**−115**|

Changing parsing coverage loses a net116 correct values; matched content adds one. That explains the full-set recall decline directly. It does **not** mean the malformed documents' latent content is correct or that constrained decoding would recover exactly133 facts. Syntax/schema constraints are relevant to deployment, but their effect on these weights has not been measured here; cross-array ownership and semantic correctness are not guaranteed by JSON grammar.

More documents now reach the strongest quality tier: **20/60 have both field F1 and accuracy≥0.95, versus14/60**. But the count above both0.90 decreases28→24. Whole-document canonical exact match improves1→4. This is a more polarized outcome, not uniform improvement.

A paired10,000-resample document bootstrap gives all60 H−G field-F1 interval **−5.21 to+0.94 percentage points**, accuracy **−9.26 to+0.83**. On matched54 the intervals are **−0.86 to+1.86** and **−1.28 to+2.18**. These small repeatedly used validation data do not establish a statistically clear overall content improvement or the causal effect of any one repair. The concrete case-level description/quantity gains below are stronger targeted evidence than the small aggregate delta.

### 13.5 Where content improved and where it regressed

Matched54, production exact-value scoring; both models see the same gold and have valid JSON. Each group includes every supported scalar within that group.

|Field group|G F1|H F1|Correct values,G→H|Interpretation|
|---|---:|---:|---:|---|
|Document/transport/freight|0.9713|0.9699|339→338|Core metadata remains strong|
|Locations|0.9722|0.9439|280→269|Real copying/normalization/omission regression|
|Containers|0.9205|0.9303|411→414|Aggregate improvement; difficult lists still fail|
|Aligned placements|0.9312|0.9506|203→202|Much fewer wrong/extra quantities|
|Other party values|0.8953|0.8896|436→435|Near-flat overall, local boundary errors|
|Other goods values|0.8418|0.8564|306→310|Package and net-weight gains offset volume decline|
|Address lines|0.7913|0.7697|127→122|Notify addresses are the main deteriorating role|
|Goods descriptions|0.4815|**0.7222**|**26→39**|Largest targeted gain; no exact→wrong regressions|

**Descriptions:**13 wrong→exact improvements and zero exact→wrong regressions on matched54. Seven improvements concern changed description gold, while six concern unchanged gold, so this is not merely changing the answer key. All60 exact descriptions rise28→39, despite more malformed documents. Examples:

- Titanium dioxide now copies `TITANIUM DIOXIDE KRONOS 2064 800 PAPER BAGS (25 KG) ON 20 PALLETS`, rather than importing separate Marks `ALEXANDRIA 1-20 MADE IN NORWAY`.
- Castor oil retains `PACKING :IN 225KG NEW STEEL DRUMS PALLETIZED`, rather than appending detached batch lines.
- Paprika retains `1680X25KG.`; bentonite now emits `BYLON (BENTONITE)` rather than a consignee-address string.
- The ten-container pipe document removes its unwanted PO prefix while retaining the product description.

The [complete source-checked description/address report](analysis/corrected-data-run-results-20261009/descriptions/REPORT.md) classifies every one of H's16 parsed description errors: six missing detail/continuation, four neighboring-field leaks, two embedded-caption omissions, two copying typos, one repeated summary and one container-ID-as-description. Five malformed documents are counted separately. None of these16 becomes exact by ignoring punctuation/space. Some failures are minor caption/letter differences; others omit real product identity or entire multi-page sections. Long/split dye descriptions and qualifiers beyond intervening accounting/HS rows remain particular coverage targets.

**Declared package totals/types:** matched54 package-quantity field F1 improves **0.7850→0.8785** and package-category field F1 **0.7379→0.8350**. This is direct evidence that the declared-level corrections are learnable. It does not certify every package classification.

**Addresses:** exact strings decline127→122 of160; six improve and eleven regress. Notify-address F1 declines0.8060→0.7077, while shipper/consignee aggregate F1 remains0.7664/0.7755. Confirmed errors include a personal name entering consignee/notify addresses, a company-name suffix entering the address, a whole landmark/street line being dropped, contact text retained and copying corruption. The source review separately marks gold-policy-sensitive disagreements such as printed `PC112`, `(EG)`, repeated postcode occurrences and explicit starred continuations; these are not automatically hallucinations. No labels were changed to improve this run's score.

**Route/place wording:** loading-port-name F1 declines0.9630→0.9259; delivery-place-name0.9143→0.8485. Current errors mix truncated wording (`NEW YORK, NY→NEW YORK`), corrupted copying (`BALTIMORE→BALMORE`, `LEVERKUSEN→LEVERKUENS`), field separation (`KOPER (SLOVENIA; KOPER)` retained whole rather than city/country split), country mistakes (`UNITED ARAB EMIRATES→UNITED STATES`) and omissions. This is not primarily letter case.

**Volumes:** value F1 declines0.8889→0.7667 on31 gold values in matched54. Source checks confirm difficult aggregation/ownership examples: two40.0000CBM container rows have80.0 gold but H emits120.0; five20.000CBM rows have100.0 gold but H emits40.0;52.200+49.880CBM has102.08 gold but H emits132.2. Other cases omit the value entirely or add0.0/an item dimension as shipment volume. These are separate from package-level improvements.

**Strong individual fields, matched54:** negotiability is54/54 exact; HS-code F1 is0.9783; vessel name0.9720; on-board date0.9737; discharge-port name0.9815; freight arrangement0.9885. Temperature-setpoint values and units are each5/5 exact, improving from four exact with one extra/wrong pair. Rare DG hazard and packing-group values remain0/1; that tiny support establishes a missed example, not a reliable estimate of DG-wide performance.

### 13.6 Relations and categories: preserve the distinction between precision and recall

All60, corrected gold:

|Relation measure|G|H|
|---|---:|---:|
|Precision /recall|0.9009 /0.9500|**0.9395** /0.9182|
|F1 /Jaccard accuracy|0.9248 /0.8601|**0.9287 /0.8670**|
|Correct facts /220|209|202|
|Wrong/extra facts|23|**13**|
|Missing facts|11|18|
|Exactly correct relation graph,all60|47|46|
|Exactly correct graph,54 documents with gold relations|43|**44**|

The important split is between package-quantity ownership and container membership:

|Fact kind,all60|G correct /predicted /gold|H correct /predicted /gold|G→H F1|
|---|---:|---:|---:|
|Container package quantity|93/114/100|93/101/100|**0.8692→0.9254**|
|Goods uses container|116/118/120|109/114/120|0.9748→0.9316|

Quantity precision improves **0.8158→0.9208**, while recall stays0.93:13 wrong/extra quantity facts disappear without losing correct ones. Container membership loses seven correct facts, so it must not be described as a blanket graph-recall improvement.

On the matched54, quantity correct facts improve91→93 with wrong/extra19→7; goods-container membership falls112→109. Relation graph exactness improves43→45/54, and relation Jaccard accuracy0.8712→0.9058. Thus the precision benefit survives removing syntax failures, while difficult list/membership behavior still needs protection. The seven-container refractory document still extracts only three containers; the ten-container pipe document retains all ten. More geometry does not by itself force list completion.

Categories improve all60 F1 **0.8597→0.8736**, accuracy0.7540→0.7756 and exact category sets33→38/60. Correct category facts actually change236→235, with wrong/extra44→34: again restraint rather than a large recall gain. Package-category projected F1 improves0.7434→0.8000. On matched54 category F1 improves0.8599→0.8983; the projected package-category F1 differs slightly from the aligned field score because fact-set projection and list alignment are different contracts.

### 13.7 JSON/schema failure shape and formatting sensitivity

H has55 parseable outputs and45 schema-valid outputs. Its five syntax failures are **not generation-cap cutoffs**: all60 outputs reach EOS; maximum saved length is1,241 tokens, well below3,072. The saved strings show:

- A stray `]` after the buffalo-meat description string.
- A fused route key/value in the Istanbul document (`PORT OFLoading`).
- Premature object closure followed by extra top-level fields in two documents.
- An unmatched quote after a numeric placement quantity in the two-container furniture document.

The last case also selects seal-like values as package quantities; fixing JSON alone would not fix that semantic error. The machine-readable [structural-failure inventory](analysis/corrected-data-run-results-20261009/metrics.json) preserves exact error positions/contexts and every parsed schema failure.

H's ten additional parseable-but-schema-invalid documents include unsupported package categories, seal/serial/header text used as container identifiers, references to absent containers and a zero volume. Categories and identifier grammar can be constrained structurally; uniqueness, ownership and agreement across arrays require additional semantic checks. No automatic output repair was used in the reported metrics.

Formatting changes do not close the remaining gap:

|All60 diagnostic|G F1 /accuracy|H F1 /accuracy|
|---|---:|---:|
|Primary,ignore casing only|0.8867 /0.8424|0.8690 /0.8059|
|Ignore commas and repeated whitespace|0.8882 /0.8439|0.8710 /0.8077|
|Broader text-punctuation diagnostic|0.8910 /0.8465|0.8722 /0.8089|

The comma/whitespace diagnostic recovers only five extra H scalar matches; broader text punctuation recovers eight. It does not normalize digits, dates, identifiers or units as if they were interchangeable. On matched54, broader punctuation recovers only two more H addresses and no descriptions. H has **zero case-only field misses** against current gold; uppercase normalization is working for this evaluation. These diagnostics are not replacements for the primary scorer.

### 13.8 Optimization, runtime and confidence in the evidence

Final cumulative training loss improves0.033682→0.032908 (−2.30%). First logged pre-clipping norms are86.11/86.23, maximum131.94/130.22, median2.11/2.26 and final1.597/1.075. No logged norm is nonfinite. Later spikes are smaller in H, but it is not a uniformly lower-gradient run. These data do not support blaming an optimizer explosion for the observed errors. Logged snapshots are133/660 updates, not all steps.

Trainer runtime improves12,163.63→12,032.16s (−1.08%); subtracting scheduled evaluation leaves8,893.74→8,913.43s (+0.22%, including saving/other overhead). Peak scheduled-eval allocation changes8.353→8.392GiB. No input truncations or target-length exclusions occur; all2,100 training and60 validation rows are retained. This is essentially the same training workload cost, not a kernel benchmark.

The CPU comparison has no API cost and performs no inference. Its [replay](analysis/corrected-data-run-results-20261009/analyze.py) verifies60/60 IDs, pinned validation/contract/output hashes, A's replay adapter provenance, H's exact saved gold, all H structured final metrics within1e-12 of the run artifact, group/field-count conservation and pre/post unchanged source hashes. It independently reconstructs aligned scalar counts and applies10,000 paired bootstrap resamples. Additional real-gold probes pass: identity gives perfect scores; recursively shuffled lists give identical scores; changed human-text casing retains perfect field scores; one injected wrong scalar loses exactly one match; malformed JSON gets no recovered-document credit. Final replay took **3.02s**, with **820.74MiB peak process RSS** including imported dependencies. All eight new run-table rows and eight matched group-table rows were checked against generated evidence; all **89 local links** across the ledger and new detailed reports resolve. The independently rerun checkpoint audit produced byte-for-byte identical evidence.

The existing `tests/test_training_metrics.py` run yields **18 passed,15 failed**. All failures are stale V7 test inputs missing the now-required `negotiability` field, before the metric assertions execute; the test file was not changed in this analysis-only task. This is a test-maintenance defect, not evidence the60 valid real references fail the scorer. It is explicitly recorded rather than reporting a clean suite. The current-gold guard probes above exercise the intended behavior on today's actual schema and pass; Ruff/compilation and document-link checks also pass. Replay timing/RSS are recorded in the evidence JSON; these CPU audit figures are separate from training benchmarks.

### 13.9 What this establishes for the next synthesis discussion

1. **Keep the corrected description/package policy.** It produces observable intended behavior: intact packing qualifiers, less Marks/party contamination, more correct declared package totals/types and fewer wrong allocation quantities. Reverting to the old gold would undo that improvement.
2. **Protect strengths while expanding coverage.** Core metadata, HS codes, negotiability, thermal settings and most container/relations remain strong. The next synthesis design should retain those distributions, rather than concentrate every new sample on one troublesome field.
3. **Target the remaining mechanisms, not simply volume.** Candidate coverage strata are multi-page/split product continuations; adjacent Marks/reference/party fields; complete multi-container lists; shipment versus row-level mass/volume; declared package types and membership-only/no-container negative examples; and name/address/contact boundaries, especially notify blocks. Specific counts/weights are deliberately not chosen in this results pass.
4. **Do not hide output failures.** Keep all-document field metrics primary, and retain the fixed matched/parseable content diagnostics alongside them. Test actual constrained decoding separately when authorized; do not claim its unmeasured gains in these scores.
5. **Retain intermediate prediction text in a future evaluation workflow.** All weights are safe, but per-document outputs at intermediate evaluations would make future comparisons much more diagnostic. No training-code change was made here.

**Conclusion:** this is a successful targeted policy experiment, not a new across-the-board best model. The complete-target parseable F1 has now crossed0.90, and relation content reaches0.9506 on the matched cohort, but field accuracy remains0.8686 and all-document scores suffer from syntax failures. Description continuation, address ownership, numeric aggregation and list completion are now concrete remaining weaknesses to address in the next coverage experiment.

**Coverage-design follow-up:** the [next1,000 synthesis plan](kie-synthesis-next1000-coverage-plan-2026-10-09.md) maps the requested600/200/200 template allocation to actual supported topology, package-category and volume-aggregation coverage. CPU-only full-recipe tests demonstrate24 plastic-bag and10 steel-drum plans, identify one precisely diagnosed physical-row feasibility rejection, and distinguish absent cross-page source coverage from existing long wording. No generation or training was launched.

Evidence: [full metrics/probes/pins](analysis/corrected-data-run-results-20261009/metrics.json), [per-field metrics](analysis/corrected-data-run-results-20261009/fields.csv), [group metrics](analysis/corrected-data-run-results-20261009/groups.csv), [per-document scores/traits](analysis/corrected-data-run-results-20261009/documents.csv), [exact error strings](analysis/corrected-data-run-results-20261009/errors.csv), [old/new gold differences](analysis/corrected-data-run-results-20261009/gold-changes.json), [description/address adjudication](analysis/corrected-data-run-results-20261009/descriptions/REPORT.md), [training/export integrity](analysis/corrected-data-run-results-20261009/TRAINING_INTEGRITY.md).

## 14. Expanded-coverage run I: 2,500 synthetic +600 real

### 14.1 Actual workload, controlled settings and reference versions

The run directory, resolved configuration, preprocessing report and MLflow tags agree: **600 real +2,500 synthetic =3,100 training documents**, plus60 real validation documents. All3,100 training rows and60 validation rows were retained. There are **zero input truncations and zero target-length exclusions**. The completed run is not a3,000-synthetic campaign. Reaching5,000 synthetic from the present2,500 requires **2,500 additional samples**; an additional2,000 would produce4,500. This arithmetic does not alter the authorized60-template admission task or authorize a larger generation campaign.

Compared with H, I adds1,000 synthetic examples and expands the source registry200→300 templates. Rank48, alpha32, rsLoRA, EVA settings, compact targets, positional input-only text, LR0.0001, ScheduleFree AdamW, effective batch32, seed424,10epochs, maximum generation3,072 and model/tokenizer revisions are unchanged. Updates increase660→970; proportional warmup33→49; evaluation/checkpoint interval165→243. Every source-code hash captured by the two training environments is identical. This is primarily a **data expansion/coverage experiment**, with more optimizer updates and data-dependent EVA initialization—not an isolated test of sample count independently of training exposure.

H and I have the **same training-time validation file SHA-256**, `9f3b4c36fe5d6310b3fb79a6d380a06d24d2fe8583acf80748d83861b2fe94da`, containing2,576 scalar values,220 relation facts and269 category facts. The saved `reference_text` for both runs exactly matches that complete file for all60 IDs. Consequently the H/I frozen-gold comparison is direct; it includes every description, complete address, date, number and explicit null.

After training, the approved party/date/delivery repairs changed the validation targets in **8documents,13scalar paths**: three shipper names, two shipper address lines, two shipper countries, four issue dates, one on-board date and one delivery-place name. The scalar support remains2,576 because additions and removals balance. The OCR and positional input are unchanged in all60. Both models are also rescored against this newer gold in14.3, without pretending I trained on it.

The recorded I training-file hash is `66a3b60f5595fe2645f3555b907d7fe8504fe49fda7d6264a204df29a5796131`. That old path now has a different hash following subsequent authorized repairs. The frozen run configuration/preprocessing report are the evidence for what the loader used; today's mutable file is not asserted to reproduce that training snapshot. This does **not** affect the prediction comparison: frozen validation still matches its recorded hash and every saved reference exactly. The next run should use the published boundary-repaired dataset and its repinned launch contract, not the historical launch file for I.

### 14.2 Progression and checkpoint integrity

Native scheduled results; H/I use identical complete validation gold here. Integer steps are exact; epochs are rounded.

|Epoch|I step|H field F1 /accuracy|I field F1 /accuracy|H→I relation F1|H→I category F1|H/I JSON valid|
|---|---:|---:|---:|---:|---:|---:|
|2.5|243|0.8482 /0.7753|0.8487 /0.7769|0.8350→0.8756|0.8008→0.8096|57/55|
|5|486|0.8617 /0.7970|0.8735 /0.8162|0.8826→0.9210|0.8405→0.8809|56/56|
|7.5|729|0.8691 /0.8057|0.8695 /0.8070|0.9287→0.8927|0.8736→0.7914|55/57|
|10|970|0.8607 /0.8003|**0.8967 /0.8562**|0.9249→**0.9348**|0.8346→**0.8939**|56/59|

I's final checkpoint is its best by the declared field-F1 selector. It recovers strongly after the7.5-epoch dip. That dip is substantial in categories (**0.8809→0.7914→0.8939**), not just a rounding difference. Only scheduled aggregates, not per-document intermediate predictions, were retained. Therefore the exact category/field causes of the7.5 dip cannot be localized from saved outputs without a new inference pass; none was run here.

Validation loss rises **0.06068→0.06317→0.06912→0.07949**, while final extraction improves. This again favors checkpoint selection by extraction metrics rather than minimum teacher-forced loss. All60 final generated sequences reach EOS, with maximum1,236tokens, far below3,072.

All four I checkpoints **243,486,729,970** are retained, including optimizer, scheduler, RNG and trainer state. The final adapter's SHA-256 is `88bfa7fe527b06e79901220895796eda6128c0243f71f960c86c64e70882059c`, matching **checkpoint970 only**, byte-for-byte. It contains504F32 tensors/91,127,808tensor bytes. The export-selection fix is working; no best-checkpoint loss/export regression was found. The authoritative manifest is `complete`; the separately written status file retains an earlier `artifacts_complete_manifest_pending` message, not evidence that training itself is unfinished.

The separate final prediction pass gives F1 **0.896147**, versus scheduled0.896714; field accuracy **0.855489**, versus0.856182. These are distinct generation observations using the correctly selected weights. All detailed counts below refer to the actual saved prediction pass, never to an inferred intermediate output.

### 14.3 Complete-target results, parsing coverage and post-run gold

Case-insensitive, list-order-invariant scoring. No generated JSON was repaired and no fields were removed from these comparisons.

|Reference /scope|Model|Precision|Recall|Field F1|Field accuracy|Relation F1|Category F1|
|---|---|---:|---:|---:|---:|---:|---:|
|Frozen training gold,all60|H|0.9082|0.8331|0.8690|0.8059|0.9287|0.8736|
|Frozen training gold,all60|I|0.9030|0.8894|**0.8961**|**0.8555**|**0.9348**|**0.8939**|
|Frozen gold,same54 both parse|H|0.9081|0.8992|0.9036|0.8675|0.9456|0.8996|
|Frozen gold,same54 both parse|I|0.9127|0.9077|**0.9102**|**0.8778**|0.9412|0.8990|
|Frozen gold,I's own59 parse|I|0.9030|0.9023|0.9027|0.8675|0.9391|0.8972|
|Today's repaired gold,all60|H|0.9090|0.8339|0.8698|0.8069|0.9287|0.8736|
|Today's repaired gold,all60|I|0.9030|0.8894|0.8961|0.8555|0.9348|0.8939|
|Today's repaired gold,same54|H|0.9089|0.9000|0.9045|0.8686|0.9456|0.8996|
|Today's repaired gold,same54|I|0.9127|0.9077|0.9102|0.8778|0.9412|0.8990|

The54-document intersection here differs from the earlier G/H54-document intersection: the count is the same, the membership and gold support are not. This matched cohort has2,351 scalar facts,159 addresses and54 descriptions. It is **not** the historical30k unchanged-component mask, and does not exclude the difficult text fields.

All60 saved-output comparisons on the same frozen complete gold now give G **0.8867/0.8424**, H **0.8690/0.8059**, I **0.8961/0.8555** for F1/accuracy. I is the strongest aggregate field result in this ledger: +0.94F1/+1.31accuracy percentage points over G and +2.71/+4.96 over H. Its extraction-fact F1 improves H→I **0.8632→0.8925**. The goal of both field F1 and accuracy>0.90 has not yet been reached; even I's parseable-content accuracy remains below0.90.

Exact decomposition of H→I correct scalar recovery:

|Parse transition|Documents|H correct values|I correct values|Difference|
|---|---:|---:|---:|---:|
|Both parse|54|2,114|2,134|+20|
|H malformed,I parses|5|0|157|+157|
|H parses,I malformed|1|32|0|−32|
|Both malformed|0|0|0|0|
|**All60**|**60**|**2,146**|**2,291**|**+145**|

Thus125/145additional correct values come from changed parsing coverage, and20from matched-content improvement. It is real aggregate progress, but not evidence that every field mechanism improved. Wrong/extra scalar predictions increase217→246 over all60 because more output survives parsing; on matched54 they decrease214→204 while correct values increase20.

The number of frozen-gold documents with both F1 and accuracy>0.90 changes **24→23**; those≥0.95 change **20→15**. Strict canonical whole-document matches change **4→1**. With today's repaired gold those counts are26→24,20→16 and5→2 respectively. These measures expose local regressions despite better micro averages. Mean per-document F1/accuracy nevertheless improve **0.8247/0.7959→0.8868/0.8559** on frozen gold, so the overall improvement is not solely a large-document weighting artifact.

A paired10,000-resample document bootstrap on frozen gold gives H→I F1 delta interval **−0.10to+6.08percentage points**, accuracy **+0.02to+10.56**; matched54 intervals are **−0.57to+1.93** and **−0.51to+2.67**. These repeatedly reused60documents support measured improvement and targeted case findings, not an unrestricted causal/generalization guarantee. Today's repaired-gold interval also spans zero for all-document accuracy.

### 14.4 Where improvement survives the matched-content comparison

Same54 parseable documents, frozen complete gold:

|Field group|H F1|I F1|Correct values,H→I|Interpretation|
|---|---:|---:|---:|---|
|Document/transport/freight|0.9681|0.9679|334→332|Near-flat strong baseline; fewer extra values|
|Locations|0.9426|**0.9553**|271→278|Receipt/delivery/country recovery improves|
|Containers|0.9297|**0.9373**|410→411|Identifiers/seals improve; size categories regress|
|Other party values|0.8907|**0.9104**|432→442|Names and several country fields improve|
|Other goods values|0.8547|**0.8631**|306→312|Volumes/net weights improve; gross totals remain weak|
|Address lines|0.7810|0.7886|123→125|Small improvement, still well below desired accuracy|
|Aligned placement fields|0.9456|0.9412|200→200|Same correct facts, two more wrong/extra predictions|
|Goods descriptions|0.7037|**0.6296**|38→34|Important local regression; continuation/boundary gap remains|

Particularly strong supported fields include **negotiability54/54**, net-weight values/units **14/14**, discharge-port names **53/54**, vessel names **52/53**, freight arrangement **43/44**, receipt-country **12/12** and on-board dates **36/36**. Their F1 can differ slightly from correct/gold ratios because unnecessary predictions count too. HS codes remain strong at0.9663F1, versus0.9778; rare support must not be mistaken for broad certainty.

Useful changes: container identifiers **106→111/114correct**, seals **105→110/115**, consignee names **38→42/49**, notify names **26→29/32**, shipper names **48→50/53** and volume values **22→27/30**. All ten container identifiers in the long pipe document are now recovered, whereas earlier small-data runs stopped after three. However seven of its45HC containers become40HC, and another loses size: list completion and correct categorization are distinct skills.

Remaining/new regressions include container-size F1 **0.9029→0.8744**; gross-weight value F1 **0.7912→0.7556**; package-quantity F1 **0.8785→0.8598**; handling-instruction F1 **0.5000→0.4516**; thermal setpoint values/units **5→4correct**. Package-category F1 is unchanged0.8462 on the matched cohort, while all60 improves because more documents parse. The single supported DG hazard is now recovered; its packing group remains missed. A one-example hazard result is not aDG-wide performance estimate.

### 14.5 Description errors and numeric ownership

I has **37exact descriptions/60**, including37/59parsed outputs. H has39/60. On matched54 there are **30exact in both**, **four wrong→exact**, **eight exact→wrong** and12wrong in both. All22parsed I description errors were read alongside their source goods-region OCR and H's output; the [source-checked inventory](analysis/training-latest-20261010/SOURCE_REVIEW.md) records each case.

Examples distinguish meaningful extraction failure from small formatting loss:

- **Embedded information omitted:** `PAPRIKA POWDER 60 ASTA HTST 1680x25KG.` loses `1680x25KG.`; red-oak lumber loses `691 pieces 10258 BF`; a loader loses its serial/dimensions; pharmaceutical cellulose loses `PHARMACEUTICAL GRADE` and `MCC 102`. The labels' approved policy intentionally retains these product-block qualifiers.
- **Continuation/list incomplete:** the long dye document now copies much of page1, a substantial improvement over H's tiny fragment, but still omits page2 and part of the page1HS caption. The fungicide description again loses the product-code/packing variants after its HS row. Fabric extraction keeps only the final color line.
- **Neighboring content copied:** refractory goods acquire separate Marks/supplier identity; a parts description acquires a consignee-company suffix; pipe description reacquires its precedingPOline; soya lecithin acquires detached batch/seal lines.
- **Boundary/count mistakes:** separate leading vehicle quantities enter the description; a standalone100-pallet line is appended to titanium dioxide; aQTY.column prefix enters the resin description. Conversely `5RACKSOFSTEELCASING.` loses its goods-owned `5RACKSOF` wording.
- **Wrong repeated occurrence/copying:** the source prints `SPARE PARTS` in the main block but `SPEAR PARTS` in an auxiliary repeated item-name line; I selects the latter. Another output rewrites `Acquisition` to `procurement`; one outputs the adjacent shipment mark rather than `FINNISH WHITEWOOD`.

These are model predictions, **not a finding that the repaired dataset is inconsistent**. No labels were edited to make this evaluation score higher. They justify targeted description-boundary/continuation coverage while preserving the policy, rather than another indiscriminate label rewrite.

Relations show a similar split. All60 projected facts H→I: **202→208/220correct**, wrong/extra **13→17**, missing **18→12**. Goods-container membership improves F1 **0.9316→0.9664**, but package-quantity F1 declines **0.9254→0.8986**:93correct quantity facts in both, with predicted quantities101→107. On matched54, quantity correct facts decline92→90/96 and wrong/extra8→12. Whole relation graphs are46→47exact/all60, but44→43exact/matched54. Do not describe the relation gain as uniform quantity/graph improvement.

Concrete quantity errors include **tare weight3,700** assigned to one container as a package count instead of its declared2,377packages; six copies of22from size-like `22GO` equipment wording; equipment length40as a quantity; and quantities1/4created around pallet/container wording without an OCR-owned row allocation. The646case remains **gold-absent/ownership-sensitive**, not newly declared a proven fabrication merely because its digits appear in OCR. Some new failures use vehicle VINs as container IDs, or a vessel name as a placement ID. The full [error CSV](analysis/training-latest-20261010/errors.csv) retains every exact path/value and distinguishes absence, wrong values and extra predictions.

Gross-weight errors include choosing embedded `1680x25KG` as1680.25, a9.144-metre product dimension as mass, a single24,348row instead of243,480shipment total, or an incorrect sum. Volume recovery improves strongly but still includes131.192→144.7,102.08→144.4and189→63errors. The task contains real numeric role/aggregation challenges beyond JSON syntax.

### 14.6 Structural failures and punctuation sensitivity

I has **one syntax failure**, in the Epistar party/contact output: a phone-like `contactName` string closes with `]` instead of `}`. It is not a generation-limit cutoff. The other **seven parseable schema/contract failures** are three unsupported package enums (`PACKAGE_BALE`, `PACKAGE_RACK`, `PACKAGE_BAL`), a notify address containing an email, two vehicle/VIN placement-ID errors and one vessel-name placement-ID error. Their exact validation diagnostics are saved in[metrics.json](analysis/training-latest-20261010/metrics.json). No automatic repair inflated the reported scores.

XGrammar/schema-constrained inference is a relevant planned deployment step. It can restrict syntactic structure and supported literal categories; it does not prove correct addresses, ownership, completeness or cross-array relations. This audit neither runs constrained inference nor predicts its metric gains.

Formatting sensitivity remains small, all60 frozen gold:

|Diagnostic|H F1 /accuracy|I F1 /accuracy|
|---|---:|---:|
|Primary,ignore case only|0.8690 /0.8059|0.8961 /0.8555|
|Ignore commas/repeated whitespace|0.8710 /0.8077|0.8973 /0.8566|
|Broader text-punctuation diagnostic|0.8722 /0.8089|0.8993 /0.8585|

I has one case-only scalar miss. Ignoring commas/whitespace recovers three more values; broader punctuation recovers eight. These changes do not remove missing words/numbers, continuation omissions, wrong entities or wrong roles. Consistency improvements are useful, but punctuation is not the dominant remaining ceiling.

### 14.7 Optimization, runtime and audit validation

I's final cumulative train loss is **0.026262**, versusH0.032908,20.20%lower. Logged gradient norm maximum drops130.22→120.13; median2.261→1.759; final1.075→1.246. The final quarter's median loss is0.004568 versus0.005308. The evidence does not show a gradient explosion; isolated later gradient spikes exist but are far below the initial transient.

Trainer runtime increases **12,032.16→16,294.16s** (3.34→4.53hours,+35.42%) for47.62%more training records/46.97%more updates. Scheduled evaluation totals3,118.74→3,409.93s. Subtracting those evaluations gives8,913.43→12,884.23s, including saving/other overhead. Observed non-padding input-token throughput on that remainder is5,895→5,987tokens/s. Peak scheduled-eval allocated memory isessentiallyflat **8.392→8.396GiB**. EVA runs187→167forwards and415.64→397.10s. These are run observations with different data lengths/order, not an isolated throughput or kernel benchmark.

The [reproducible CPU audit](analysis/training-latest-20261010/analyze.py) checks all60identities, frozen/fullreference equality, current-input equality, adapter/checkpoint hashes against manifests, all four resumable checkpoints, zero truncation/exclusion and all pinned files before/after. Both H/I final native structured metrics are reproduced within1e-12 using the current scoring implementation; schema-description changes do not silently change this metric replay. An independent list-aligned scalar diagnostic reproduces the primary scorer and conserves counts through per-field/group breakdowns.

Falsification probes on **both reference versions** pass: identical output gives perfect scores; recursive list permutations do not change them; human-text case changes retain perfect field/relation matches; one wrong scalar loses exactly one correct value; one malformed document receives no recovered-field credit. Ruff passes for the audit script. The final replay took **5.46seconds with137.18MiB peak process RSS**, including imported audit/scoring dependencies. All61 local links across this ledger and the new source-review inventory resolve. **API cost:$0; no training or inference launched.**

### 14.8 Coverage implications and the declared next experiments

The next template admission/synthesis design should protect current metadata/route/name strengths while adding actual supported examples of:

1. Full main-product blocks with embedded packing/specification, adjacent separate Marks/PO/batch fields, and genuine page-to-page/product-code continuations.
2. Correct size/type contrasts—especially45HCversus40HC,20HCversus20standard, and known-type/missing-length cases—without inventing size for a source that omits it.
3. Declared package level and per-container ownership, including membership-only/no-quantity negatives, explicit sole-containerSTC allocations, and size/seal/shipment-total distractors.
4. Complete row lists and shipment-versus-row mass/volume, with both explicit totals and supported aggregation cases.
5. Remaining address/name/contact boundaries and enough thermal/DG support to estimate preservation of those rare strengths.

User's intended progression is recorded, not launched here: expand to**5,000synthetic**, put **all new import/end localities inUAE**, compare synthetic-onlytraining against the mixture, then test**rank64**. New generation should be primarily coverage-driven rather than equal descendant counts. It remains useful to separate these experiment changes if their individual effects are to be interpreted. The present task admits60new templates in20-template groups, withtwo audited descendants each; it does not yet make those pilotdescendants part of training or launch the next bulkcampaign.

Evidence: [complete metrics/progression/integrity/guards](analysis/training-latest-20261010/metrics.json), [source-checked description inventory](analysis/training-latest-20261010/SOURCE_REVIEW.md), [field metrics](analysis/training-latest-20261010/fields.csv), [group metrics](analysis/training-latest-20261010/groups.csv), [document scores/traits](analysis/training-latest-20261010/documents.csv), [errors](analysis/training-latest-20261010/errors.csv), [post-training gold changes](analysis/training-latest-20261010/gold-changes.json), [CPU replay](analysis/training-latest-20261010/analyze.py).

## 15. UAE expansion run J: 5,000 synthetic +600 real

### 15.1 Completion, data and comparison authority

The run completed2026-10-11. MLflow is `FINISHED`; the immutable artifact
manifest is complete. All **60 artifact hashes and byte counts** verify.
Training used **5,600 records**, validation60. The current train/validation
files still match their frozen launch pins:

- Train: `08e44d14b371d517b711b5ef2cde0c99cdb8e7d9a44f0839e8492762db693075`.
- Validation: `2948e9001d2429c63e9e918597d9dd10e7ebec3b26d9f4c1758a1dfe168318be`.

All rows were retained; zero source truncations, zero target exclusions.
Maximum input/target lengths are8,066/3,160training and5,698/1,348validation,
against19,200/5,500 configured limits. The3,072 inference-generation cap is a
different limit from the5,500 training-target allowance.

All checkpoints **438,876,1314,1750** remain available. Best is **1750, epoch10**.
The exported adapter equals that checkpoint byte-for-byte, SHA-256
`5b359110500a56b6c31be6fe2482f6281ceecfac56da822b8ed532fbffdd0368`.
The pre-manifest status sidecar still saying `artifacts_complete_manifest_pending`
is not a failed or incomplete training run.

Relative to I:5,000 versus2,500 synthetic examples,360 versus300 registered
templates, approved party/date/delivery repairs and refreshed launch contracts.
The added UAE campaign used358 source families from the360 registry. Both runs
retain600 real training documents and60 validation. Rank48, alpha32, EVA,
rsLoRA, LR1e-4, bf16/SDPA, effective batch1×32,10epochs, input-only positioned
text, compact JSON and3,072-token generation are unchanged. Updates970→1750,
warmup49→88 preserve approximately5%; save/eval243→438 preserve2.5-epoch spacing.

This is not a pure isolated sample-count experiment: data composition,
source coverage, label repairs, optimization exposure and EVA's sampled
initialization differ. In particular, the new independent registry-form and
equipment/package-alias integration **postdates J's data generation**. J's
gains cannot yet be attributed to that integration.

The historical I validation file is still hash-exact, but its training path
was subsequently edited. All60 G/H/I/J validation OCR and positioned strings
are identical. Saved prediction `reference_text` gives frozen historical gold;
CPU replay here scores every model against **the same complete J gold**.
H/I references differ from J in8 documents/13scalar paths, G in31/59. J's
reference matches all current targets exactly. No addresses/descriptions,
nulls or other difficult fields are masked out.

### 15.2 Scheduled progression and final saved outputs

These scheduled rows use each run's own contemporary gold; section15.3
removes that reference difference for the saved final output comparison.

|Epoch|J step|I field F1 /accuracy|J field F1 /accuracy|I→J relation F1|I→J category F1|I/J valid JSON|
|---|---:|---:|---:|---:|---:|---:|
|2.5|438|0.8487 /0.7769|**0.8882 /0.8421**|0.8756→0.8878|0.8096→0.8658|55/59|
|5|876|0.8735 /0.8162|**0.8886 /0.8414**|0.9210→0.9044|0.8809→0.8909|56/58|
|7.5|1314|0.8695 /0.8070|0.8650 /0.8017|0.8927→0.8662|0.7914→0.8588|57/56|
|10|1750|0.8967 /0.8562|**0.9013 /0.8643**|0.9348→**0.9450**|0.8939→**0.9009**|59/59|

J starts substantially ahead, dips at7.5, then reaches its best extraction
scores at10. At5/7.5 its relation score is below I, despite the stronger final
result. J eval loss **0.05362→0.06344→0.06427→0.07333** again favors selecting
on extraction quality rather than minimum teacher-forced loss. Intermediate
prediction strings are not saved, so these logs cannot identify the exact
document-level cause of the dip without new inference; none was launched.

Final reevaluation of the selected weights gives field F1/accuracy
**0.901691/0.864957**, slightly different from scheduled
**0.901336/0.864305**. Detailed inventories use the saved final prediction pass.

### 15.3 Complete-target comparison on common current gold

Case-insensitive and list-order-invariant. Syntax failures receive no recovered
credit; parseable schema-invalid outputs can receive partial field credit.

|Saved-output run|Precision|Recall|Field F1|Field accuracy|Relation F1 /accuracy|Category F1 /accuracy|JSON /schema valid|
|---|---:|---:|---:|---:|---:|---:|---:|
|G:1,500 synthetic,old policy|0.8962|0.8781|0.8871|0.8428|0.9248 /0.8601|0.8597 /0.7540|58 /47|
|H:1,500,corrected descriptions/packages|0.9090|0.8339|0.8698|0.8069|0.9287 /0.8670|0.8736 /0.7756|55 /45|
|I:2,500 synthetic|0.9030|0.8894|0.8961|0.8555|0.9348 /0.8776|0.8939 /0.8081|59 /52|
|J:5,000 synthetic|**0.9135**|**0.8901**|**0.9017**|**0.8650**|**0.9450 /0.8957**|**0.9009 /0.8197**|59 /51|

Support is2,576 scalar values,220 relation facts and269 category facts on
all60. I→J changes are **+0.55F1, +0.95accuracy, +1.01relation-F1,
+0.71category-F1 percentage points**. Overall field recall barely moves;
precision increases1.05points. Correct scalars2,291→2,293, wrong/extra
predictions246→217. The main aggregate gain is greater restraint/precision,
not a large net recall increase. Extraction-fact F1 is0.8925→0.8976.

The same58 documents parse in both I/J:

|Matched full-target measure|I|J|
|---|---:|---:|
|Field precision /recall|0.9043 /0.9076|**0.9135 /0.9121**|
|Field F1 /accuracy|0.9060 /0.8731|**0.9128 /0.8856**|
|Correct /gold scalars|2,250 /2,479|**2,261 /2,479**|
|Wrong/extra scalar predictions|238|**214**|
|Relation F1 /accuracy|0.9417 /0.8899|**0.9623 /0.9273**|
|Relation precision /recall|0.9224 /0.9619|**0.9533 /0.9714**|
|Category F1 /accuracy|0.9040 /0.8247|**0.9070 /0.8299**|

Thus improvement survives controlling for parse membership and includes full
addresses/descriptions. It is not the much higher historical30k
unchanged-components comparison. J's own59 parseable outputs score
**0.9125F1/0.8850accuracy**, relation0.9626/0.9279. The52-document intersection
where all G/H/I/J parse gives full-target F1/accuracy:
**0.9057/0.8752→0.9117/0.8810→0.9147/0.8857→0.9221/0.899827**.
That last accuracy is near0.90, not above it, and the intersection is a
conditional diagnostic, not the primary all-document score.

Parsing is59/60 in both I/J but the failed document changes: Epistar becomes
parseable, recovering32 values; the seven-container refractory document
becomes malformed, losing41 values. The58 matched documents add11 correct
values; syntax-membership change subtracts9, leaving+2overall.

Documents with both field F1 and accuracy>0.90 increase **24→32/60**;
both≥0.95 increase **16→20/60**. Strict canonical whole-document matches2→5.
Macro document F1/accuracy0.8862/0.8555→0.8931/0.8680. These distinguish local
and whole-document quality from the micro average.

The paired10,000-resample95% interval for the I→J field-F1 change is
**−1.73to+2.59points** all60, **−0.29to+1.64** matched58. Accuracy intervals
are−3.12to+4.60 and−0.17to+2.64points. Reused60-document validation shows
measured progress and concrete case-level changes, not a statistically
established generalization guarantee or isolated causal effect of scale.

### 15.4 Strengths and remaining weaknesses, matched58

|Field group|I F1|J F1|Correct values I→J|
|---|---:|---:|---:|
|Document/transport/freight|0.9714|**0.9755**|356→359|
|Locations|0.9518|**0.9693**|296→300|
|Containers|0.9385|**0.9506**|420→423|
|Other party values|0.8895|0.8860|471→470|
|Other goods values|0.8693|0.8638|336→333|
|Address lines|0.7765|0.7834|132→132|
|Aligned placement fields|0.9417|**0.9623**|202→204|
|Goods descriptions|0.6379|**0.6897**|37→40|

Strong supported fields: **negotiability58/58**, vessel names57/57,
issue dates45/45 (one extra), loading ports55/58, discharge ports57/57
(one extra), HS codes47/49, identifiers110/112 and seals111/114. Thermal
setpoint values improve5→6/6exact. That is preservation/recovery on small
support, not proof of thermal generalization. The only DG hazard is correct,
while the only packing group remains missed.

Relations genuinely improve in restraint this time. Matched package-quantity
facts93→94/98correct, predicted107→101; F1 **0.9073→0.9447**,
precision0.8692→0.9307. Membership109→110/112correct, F1
0.9732→0.9778. Exact relation graphs47→50/58; all60,47→51 (supported
documents44→48/54). All-document correct relation facts208→206 despite
the higher F1, because the syntax failure now affects a container-rich
document. Matched content is the appropriate evidence of relation learning.

Categories are not uniformly improving:

|Category,matched58|I correct /gold|J correct /gold|I→J F1|
|---|---:|---:|---:|
|Equipment size|91 /104|89 /104|0.8750→0.8641|
|Equipment type|99 /104|101 /104|0.9519→**0.9806**|
|Declared package type|49 /54|48 /54|0.8750→0.8571|

Category-exact documents44→42 on both matched/all cohorts; better category
micro F1 must not be described as uniformly better category graphs. The ten
pipe-container IDs are complete, but all ten45HCsizes become40HC. The
20HO open-top,40RQ reefer and carrier-defined CMA40RA high-cube sizes also
default to standard. The new sampling interface directly enables useful
contrasts, but rare specialized codes still need their dedicated source paths.

Addresses show **no additional correct strings** on matched58; fewer extra
addresses improve F1. Shipper/consignee exact address counts43→42/54 and
41→40/52; notify25→26/35; delivery agent19→20/23. Shipper names52→54/57,
but consignee44→43/53 and notify31→29/36. Identity/postal/contact ownership
is still a priority alongside punctuation/copy accuracy.

Goods measurements: gross-value correct count36/46 is unchanged, with one
fewer extra prediction; volume29→27/32, net15→14/15, package quantity
50→49/57. Recurrent source-checked failures select a row instead of a
shipment total, misassociate a real row count, or produce incorrect
aggregation/digit copying. Complete descriptions improve41/60overall,
with7matched improvements and4regressions, but embedded capacities,
product identifiers, adjoining declarations and page2 continuation remain
weak. All18parsed description errors were source-checked in the
[inventory](analysis/training-latest-20261011/SOURCE_REVIEW.md).

Descriptive low-score traits are not sufficient filters:6zero-container
documents average0.804accuracy;6with≥5containers average0.714, partly
because one receives0for syntax;18with2–4containers average0.912.
Only two descriptions are≥200characters and neither exceeds both0.90
thresholds. Multi-page documents average0.880accuracy versus0.850for
one-page documents, so page count alone does not explain the failures.

### 15.5 Structure and formatting diagnostics

There is one invalid JSON output. It repeats three container identities
**24/24/23times** before reaching3,072tokens. Source has seven containers.
This is a repetition loop, not evidence that a modestly larger generation
budget would solve list completion.59/60sequences reach EOS.

Eight other outputs are parseable but fail the schema/task contract:
three unsupported package enums; three placement IDs taken from VIN/vessel
text; zero volume; and placements referencing two missing container records.
Together these explain51/60schema-valid outputs. I had52/60, so schema
adherence did not improve. Frozen-enum/JSON-constrained inference can address
structural admissibility when those constraints are supplied; it does not
establish correct semantic ownership, completeness or arithmetic.

|J scoring diagnostic,all60|Field F1|Field accuracy|Correct scalars|
|---|---:|---:|---:|
|Primary,case-insensitive|0.9017|0.8650|2,293|
|Ignore commas/repeated whitespace|0.9037|0.8668|2,298|
|Broader human-text punctuation tolerance|0.9072|0.8702|2,307|

One casing-only miss, five further comma/spacing matches and fourteen
broader punctuation matches do not account for the remaining gap. The
broader diagnostic does not excuse missing words/digits or wrong roles and
does not replace the primary metric. No target was normalized to make a
prediction pass.

### 15.6 Runtime, evidence and validation limits

Train loss0.026262→**0.018074** (−31.2%). Runtime4.526→**7.373hours**
(+62.9%) for+80.6%trainrecords. Logged throughput1.903→2.110samples/s.
Scheduled eval time3,409.93→3,181.04s; excluding scheduled evaluations,
input throughput5,987→5,927tokens/s. This approximately1%reduction occurs
with changed sample-length/composition, not a kernel-code modification.
EVA167→192forwards,397.10→418.26s. Final prediction pass887.39s.
MLflow device-global GPU peak25,414.4→25,403.4MB; host peak8,167.3→7,820.7MB.
Those system-tracker measurements are not model-only allocated-memory peaks.

The local [replay](analysis/training-latest-20261011/analyze.py) reproduces
every G/H/I/J native final structured metric within1e-12. Independent
list-aligned counts conserve through fields/groups. Before/after pins verify
no input/reference/prediction changes. Real-reference falsification guards
pass on both current J and frozen I gold: identity, recursive list
permutations, human-text casing, one wrong scalar and one malformed document.
Final replay2.35s, peak process RSS987.61MiB including imported dependencies;
Ruff passes. **API cost:$0. No inference/training/synthesis launched.**

The existing three-file metric/constraint/order pytest run produced26passes
and15failures. The failed old v7 test references omit the now-required
`negotiability` field; one also omits required notify `sameAs` values. They are
rejected before scoring. They are stale fixtures,
not observed failures of this real-data replay. No production metric/schema
change or out-of-scope fixture repair was made; this test debt is explicit,
not reported as a passing suite.

### 15.7 Historical coverage plan preceding K

This subsection preserves the plan and dataset state at the J audit. Its
replacement cohort was subsequently generated and trained in K, evaluated
below. Pending repairs mentioned here are historical, not a current backlog.

The [new proposed coverage plan](kie-synthesis-next2500-uae-coverage-plan-2026-10-11.md)
recommends **1,000 new registry-form/alias samples +1,500 broad-domain and
specialized samples**, preserving previous overlapping topology/thermal/DG
strata. All new terminal/import nodes remainUAE. New-mode emphasis is20HC,
45HC, declared paper/multiwall/jumbo/IBC/drum/crate/skid/coil units and31GP
aliases. Applying its ten-profile catalog to all2,500 would unnecessarily
narrow commodity coverage.

Actual offline source admission exercised two production plans per eligible
source: **233sources passed both**,56rejected new-mode admission for explicit
package, tare, alias-owner or equipment-occurrence limitations. These are
mode restrictions, not newly proven defects in current accepted data. Separate
configs preserve valid incomplete/mixed/no-container/specialized paths.
The exact2,500draws must pass preflight and realized coverage checks before
paid wording; no implicit donor/size/tare fallback is allowed.

The user subsequently clarified that these improved 2,500 replace the newer
UAE 2,500 cohort; they are not an addition. Preserve inherited 2,500:
**5,000 synthetic + 600 real = 5,600 training**, with 60 validation unchanged.
The earlier additive 7,500/8,100 proposal is superseded. This next experiment
compares coverage at fixed dataset size, rather than another scale increase.
Keep rank48/settings for this coverage comparison; synthetic-only/rank64
remain separate future ablations. Neither generation nor training is
authorized or launched by this analysis task.

The earlier coverage audit's single real-training refrigerated-type omission
(`8c6da3`,−18°C) is still present in the current hash-pinned training file;
it does not affect validation metrics or any current synthetic descendant.
The new coverage plan explicitly schedules that supported type-only
correction before preparing/repinning the next combined dataset. No broad
label rewrite is implied, and no completed-run data was edited here.

Evidence: [metric replay and guards](analysis/training-latest-20261011/metrics.json),
[run/MLflow/export integrity](analysis/training-latest-20261011/run-integrity-evidence.json),
[field breakdown](analysis/training-latest-20261011/fields.csv),
[group breakdown](analysis/training-latest-20261011/groups.csv),
[document scores](analysis/training-latest-20261011/documents.csv),
[exact errors](analysis/training-latest-20261011/errors.csv),
[source review](analysis/training-latest-20261011/SOURCE_REVIEW.md),
[next-cohort admission evidence](analysis/training-latest-20261011/next-cohort-admission.json).

## 16. UAE coverage replacement run K

### 16.1 Completion and actual experimental change

K completed on **2026-10-13**; MLflow run
`ea7f4fade74a44a0ad1be14f91c93811` is FINISHED and the completed artifact
manifest verifies all **60 artifact hashes and byte counts**. The stale
status sidecar still says artifacts_complete_manifest_pending; the completed
manifest, exported weights and finished MLflow run establish completion.

Training is still **600 real +5,000 synthetic =5,600 records**, using the
360-template registry. K replaces J's2,500 UAE descendants with2,500 new
coverage-oriented UAE descendants; it does not add training examples.
The inherited2,500 non-UAE descendants and600 real records remain. Of3,100
shared IDs,3,077 rows are unchanged;23 targets have finite repairs
(five real,eighteen non-UAE) and three shared synthetic inputs change.
The repairs include party fields,three descriptions and one container category.

J/K trainer source fingerprints and package versions are identical. Rank48,
alpha32,rsLoRA,EVA,LR1e-4,microbatch1/accumulation32,bf16/SDPA,10epochs,
1,750updates,88warmup steps,438-step evaluation/saving,input-only positioned
OCR and compact JSON targets remain unchanged. EVA's actual sampled
initialization changes with the dataset (192→182forwards).
This is a coverage/data-composition comparison with finite label repairs,
not a pure scale,rank or kernel experiment.

The launched training/validation file hashes still verify:

- Train: `001fa2a63ac426109993547524d0cc5b5ca7e0408b7a8bdbc8a681486fdb5d49`.
- Validation: `2948e9001d2429c63e9e918597d9dd10e7ebec3b26d9f4c1758a1dfe168318be`.

All60 J/K frozen reference strings are identical. All60 validation OCR and
positioned inputs are identical across G/H/I/J/K. Current R20 direct targets
have exactly the same factual objects as K's frozen documentPatch targets.
Every field remains in the comparison, including complete addresses,
descriptions, null instructions and relations; there is no unchanged-field mask.

**K predates the R20 private-contact/private-identifier repairs and direct-target
conversion.** Its output is the historical wrapped target. The new direct-object
training config is not the run analyzed here. These scores validate factual
performance against current gold, not a trained direct-output format.

### 16.2 Training coverage actually changed

Counts below are over the complete5,600-record training files. Equipment
numbers count container rows; document traits count documents.

|Coverage|J|K|
|---|---:|---:|
|20-foot standard container rows|3,320|3,311|
|20-foot high-cube rows|0|635|
|40-foot standard rows|378|677|
|40-foot high-cube rows|7,250|5,164|
|45-foot high-cube rows|69|1,146|
|Refrigerated container rows|520|593|
|Pressurized-tank rows|3|22|
|Multiple-container documents|1,892|1,801|
|Mixed assigned equipment-pair documents|9|586|
|Documents with thermal setpoints|303|423|

The new training signal supplies the intended previously missing20HC and
much stronger45HC contrasts. Mixed equipment expands substantially, while
the overall multiple-container population is slightly smaller. These counts
describe the treatment, not proof that every individual change caused a
validation result. See the full
[coverage comparison](analysis/training-latest-20261013/coverage-comparison.json)
for equipment families and declared package categories.

### 16.3 Checkpoint progression and verified export

|Epoch / step|Eval loss|K field F1|K field accuracy|J field F1 /accuracy|K relation F1|K category F1|K JSON /schema valid|
|---|---:|---:|---:|---:|---:|---:|---:|
|2.503 /438|0.05324|0.8878|0.8374|0.8882 /0.8421|0.8768|0.9081|58 /49|
|5.006 /876|0.06529|0.8916|0.8442|0.8886 /0.8414|0.9202|0.8625|58 /48|
|7.509 /1314|0.06534|0.8821|0.8314|0.8650 /0.8017|0.9157|0.8731|58 /47|
|10 /1750|0.08245|**0.8941**|**0.8520**|0.9013 /0.8643|**0.9336**|0.8996|59 /51|

K improves slightly over J at5epochs and more clearly at7.5, but finishes
below J at10. Its7.5 dip is less severe than J's. Category F1 peaks at2.5;
the best complete field extraction remains10. Teacher-forced eval loss again
increases while final extraction improves, so minimum loss is not the
appropriate checkpoint criterion.

All four checkpoints remain available. Best/exported checkpoint is
**1750,epoch10**, and the exported adapter matches it byte-for-byte:
`1bf89e4bf5b1c7e1765ca2db46a138f4607205cfd488d05066d5a3e1a2a17e77`.
The verified export contains504tensors/91,127,808tensorbytes.

Scheduled best F1/accuracy is **0.894071/0.851977**. The separate final
prediction pass is **0.893676/0.851601**. The inventories and comparisons below
use the saved final strings, not the slightly different scheduled aggregate.

### 16.4 Complete target comparison

Primary scores use all60documents, case-insensitive and list-order-invariant.

|Saved outputs on complete common gold|I|J|K|
|---|---:|---:|---:|
|Field precision|0.9030|0.9135|0.9102|
|Field recall|0.8894|0.8901|0.8777|
|Field F1|0.8961|**0.9017**|0.8937|
|Field accuracy|0.8555|**0.8650**|0.8516|
|Extraction-fact F1|0.8925|0.8976|0.8900|
|Relation F1 /accuracy|0.9348 /0.8776|**0.9450 /0.8957**|0.9336 /0.8756|
|Category F1 /accuracy|0.8939 /0.8081|0.9009 /0.8197|0.8996 /0.8176|
|JSON /schema valid|59 /52|59 /51|59 /51|

J→K changes are **−0.80 field-F1,−1.34 field-accuracy,
−1.13 relation-F1 and−0.13 category-F1 percentage points**.
Correct scalar values fall2,293→2,261/2,576; wrong/extra predictions
increase217→223. J remains the best all-document result.

Syntax coverage is59/60 in both runs, but membership changes. The seven-container
refractory document becomes parseable in K, recovering45scalar values.
The frozen-meat document becomes malformed, losing36previously correct values.
That swap contributes **+9correct values**; the same58 jointly parseable
documents lose **41**, producing the net−32.

|Same58 jointly parseable documents|J|K|
|---|---:|---:|
|Field precision /recall|0.9164 /0.9145|0.9168 /0.8979|
|Field F1 /accuracy|**0.9154 /0.8893**|0.9073 /0.8762|
|Correct /gold scalar values|2,257 /2,468|2,216 /2,468|
|Wrong/extra scalar predictions|206|201|
|Relation precision /recall|0.9533 /0.9714|0.9744 /0.9048|
|Relation F1 /accuracy|**0.9623 /0.9273**|0.9383 /0.8837|
|Category F1 /accuracy|0.9049 /0.8264|**0.9284 /0.8664**|

The principal content tradeoff is lower recall, not a broad precision collapse.
Relations become more restrained: wrong/extra facts10→5, but missing/wrong
supported facts6→20; exact graphs50→48/58. Category gains survive the same-input,
same-parse comparison, even though the all60 aggregate is nearly flat.

K's own59parseable outputs reach **0.9022 field-F1/0.8673accuracy**.
On52documents parseable in all G/H/I/J/K, J→K is
0.9221/0.8998→0.9128/0.8844. These are complete-target diagnostics, not
the historical30k unchanged-component comparison and not the primary score.

Documents exceeding both0.90field-F1 and accuracy fall **32→25/60**;
both≥0.95 remain **20/60**. Strict whole-document matches remain **5/60**.
Macro document F1/accuracy changes0.8931/0.8680→0.8856/0.8572.

Paired10,000-resample95% intervals for the J→K change span
−2.72to+1.10field-F1 points and−4.91to+2.12accuracy points on all60.
Matched58 intervals are−1.81to+0.15 and−2.79to+0.15.
The point estimate favors J overall; the small reused validation set does
not establish that coverage replacement necessarily harms unseen documents.

### 16.5 What improves and what regresses

|Field group on matched58|J F1|K F1|Correct values J→K|
|---|---:|---:|---:|
|Document/transport/freight|0.9756|0.9741|360→357|
|Locations|0.9757|0.9427|301→288|
|Containers|0.9502|0.9477|420→408|
|Other party values|0.8923|0.8887|468→471|
|Other goods values|0.8646|0.8676|332→331|
|Address lines|0.7821|0.7798|131→131|
|Aligned placement fields|0.9623|0.9383|204→190|
|Goods descriptions|0.7069|0.6897|41→40|

**Targeted size recognition improves.** Correct size categories rise
89→96/104, F10.8641→0.9505. In the ten-container pipe document, J recovered
all identifiers but typed every45HC as40HC. K recognizes45HC correctly for
its seven recovered containers, but omits three containers and their placements.
The20HO case's height improves to20HC, while both raw outputs still assign
GENERAL_PURPOSE instead of OPEN_TOP. K also resolves40RQ to high cube;
the carrier-defined40RA remains standard-height incorrectly.

Equipment-family recovery falls101→96/104; its matched F1 remains0.9552.
The declared package category stays47/53exact, F10.8468. Greater package
training diversity has not yet translated into a measured gain on this
validation cohort. The single DG hazard remains correct and its packing group
remains missing. Matched thermal setpoints remain5/5exact; this is small support,
not a DG/thermal-wide performance claim.

**Inventory and relation recall are the clearest weaknesses.** Matched
container identifiers110→102/112, seals110→104/113 and placement identifiers
110→100/112. The bentonite output deteriorates from four to three container
records for six expected containers and misuses other IDs as seals.
Placement quantities94→90/98correct: their precision improves, but omissions
increase. These are list completion/ownership failures, not simply bad enums.

**Locations lose13correct values.** Loading-port names56→52/58 and place-of-receipt
names24→22/26. Source-checked examples include BALTIMORE→BALLEY,
ADELAIDE→ADELAIDEIDE, NEW YORK,NY→NEW YORK, ISTANBUL,MARDAS→ISTANBUL
with MARDAS moved into country, and ANTWERP placed under vesselName rather
than portOfLoading. Some are copying defects, others cross-field ownership.

**Addresses are nearly flat overall.** Shipper exact addresses improve43→46/55,
consignee40→38/52, delivery-agent19→18/22. The total is still131/167exact.
Remaining errors mix omitted street/locality/postcode content, name leakage,
duplicated words and punctuation differences. Changing casing alone does not
recover any K facts; ignoring commas/repeated whitespace recovers six.

**Descriptions remain the main low-scoring text family.** K has40/60exact
descriptions versus J41; on matched58 there are four improvements and five
regressions. All19parsed K description errors are inspected against source
OCR and the approved target. Errors include omitted product models/qualifiers,
Marks/specification contamination, detached packaging/weight inclusion,
wrong repeated-description occurrence and missing page continuation.
See the [source inventory](analysis/training-latest-20261013/SOURCE_REVIEW.md).

Net-weight values improve13→14/14exact; shipment package quantities49→50/57;
handling-instruction values12→14/21. HS codes regress47→44/49. None of these
partial gains offsets the inventory/location losses.

### 16.6 Syntax and formatting are separate from the content gap

K's only invalid JSON is the frozen-meat document: an extra closing brace after
description closes the goods item before grossWeight. It is not truncated.
**All60 K sequences reach EOS**, maximum generated length1,204tokens versus
the3,072cap. J's prior malformed refractory repetition loop is gone in K.

Eight parseable K outputs fail the schema/task contract: duplicate handling
clauses in two documents; one malformed package enum; VIN/vessel-derived
container or placement IDs in three documents; zero volume; and a placement
referencing absent containerInformation. Several of these are ownership or
cross-array/de-duplication constraints, not merely JSON syntax.

|K formatting diagnostic on all60|Field F1|Field accuracy|Correct values|
|---|---:|---:|---:|
|Primary case-insensitive|0.8937|0.8516|2,261|
|Ignore commas/repeated whitespace|0.8960|0.8539|2,267|
|Broader human-text punctuation tolerance|0.8996|0.8573|2,276|

The primary-score shortfall is not mainly a comma/casing disagreement.
Constrained inference remains a separate experiment; no hypothetical grammar
benefit is credited to these saved unconstrained outputs. It does not recover
the omitted container/placement inventory or correct location ownership.

### 16.7 Runtime and next experimental implications

Train loss improves0.018074→0.017460, while runtime rises7.373→7.577hours
(+2.8%). Logged samples/s changes2.110→2.053; scheduled evaluations total
3,181.04→3,262.39s. Excluding scheduled evaluation, input throughput is
5,927→5,783tokens/s (−2.4%) on slightly different input volume.
Trainer code and package fingerprints are unchanged; this is not evidence
of a kernel regression. Final prediction runtime improves887.39→792.54s.

Evaluation model peaks remain approximately8.40GiB allocated/13.51GiB reserved.
MLflow device-global peak is25,474.3MB and host peak8,221.8MB; these are system
tracker measurements, not model-only memory. Training/validation source/target
maxima remain8,066/3,160 and5,698/1,348; zero truncations or target exclusions.

The defensible next step is to preserve J as the best aggregate baseline and
measure further changes independently. K supports retaining the new size
contrasts, but motivates explicit inventory-completion, membership-only and
long-continuation tests alongside them. Category micro gains must not hide
losses of containers/relations or routine location facts.

R20's new direct-target R48 config is available, but a run on it would also
change privacy-normalized inputs and output format. Raising rank64 in the same
run would not isolate rank. Likewise removing600real examples changes the
training distribution. Keep separate comparisons on one frozen dataset/target
contract rather than attribute every simultaneous change to sample quality
or rank. No new run or dataset change was performed by this results audit.

The CPU replay reproduces all G/H/I/J/K final structured metrics within1e-12,
conserves independently aligned field/group counts and verifies input,
reference and prediction pins before/after. Identity, recursive list-order,
casing and injected-error guards pass. The focused metric/relation/order
suite passes **41tests**. Repeated replays take2.78–3.28s and342.57MiB peak RSS.
**Paid API cost:$0; no inference, training or synthesis launched.**

Evidence: [replay and guards](analysis/training-latest-20261013/metrics.json),
[checkpoint integrity](analysis/training-latest-20261013/run-integrity-evidence.json),
[coverage counts](analysis/training-latest-20261013/coverage-comparison.json),
[field breakdown](analysis/training-latest-20261013/fields.csv),
[group breakdown](analysis/training-latest-20261013/groups.csv),
[document scores](analysis/training-latest-20261013/documents.csv),
[exact errors](analysis/training-latest-20261013/errors.csv),
[source review](analysis/training-latest-20261013/SOURCE_REVIEW.md).

## 17. Synthetic-only composition plan after K

The [2026-10-14 cohort audit and selection plan](kie-synthetic-only-augmentation-plan-2026-10-14.md)
retains the current 5,000 repaired synthetic records, removes 600 real training
records, and prepares exact 1,000/1,500 additions from the repaired old UAE
sidecar. A further 300 Egypt-endpoint records would give 6,300/6,800 synthetic
training records. Validation remains the same 60 real documents.

The recommended 1,000 old records already recover all old UAE five-plus-container,
membership-only and transshipment examples, while covering all 358 available
source layouts. Four old VIA-composite party records are excluded from
augmentation; none occur in the current 5,000. Their exclusion does not require
rewriting or rejecting the remaining old cohort.

Removing the real set exposes specific positive-supervision gaps: all synthetic
HS codes have six digits, whereas 41/51 validation code values have 8/10/12
digits; DG packing groups, flash points, subsidiary hazards and free-text
package fallback have no current/old synthetic examples. Ten named package
categories also occur only in real training. The 300 Egypt additions should
target these formats/fields using validated source bindings, not merely change
destinations. The report includes concrete source candidates and exact member
hashes. This audit did not launch generation/training or alter published data.

## 18. Synthetic-only L/R22 and restored-real M/R23

### 18.1 Completion, actual data and selected weights

M completed2026-10-15 at13:33:33UTC. Its complete artifact manifest and read-only
MLflow `FINISHED` state agree; all60 artifact hashes/sizes verify, totaling
1,354,469,696bytes. A stale `status.json` still says
`artifacts_complete_manifest_pending`; it is not the completion authority.

M trains on **exactly L's immutable7,000-synthetic shard plus600 real**. The
synthetic-file SHA256 is
`90b0be46d6b0387dd270b7f2d0a56264ad4515f596c3e08e6df1e2478e6f8939`.
Real/validation targets restore printed HS separators; validation input text and
coordinates remain identical. All7,600 training/60 validation records are
retained: **zero truncations, target exclusions or filtered rows**.

Rank48/alpha32, rsLoRA/EVA, LR1e-4, effective batch32, bfloat16/SDPA,
ScheduleFree AdamW and10 epochs remain unchanged. Updates increase2,190→2,380;
5% warmup110→119; evaluation/save interval548→595. Dataset-dependent EVA
initialization sees141 distinct documents in L versus256 in M despite identical
settings. This is not a fixed-initialization or fixed-update ablation.

**Best: checkpoint1785, epoch7.501.** The exported504 tensors match the selected
checkpoint exactly and differ from checkpoint2380. Adapter SHA256:
`46ba5afe8000b7c3be6f1f94d091eb78e9c1fffa8919c313b0c29cb213c4ede3`.
All595/1190/1785/2380 checkpoints remain. Final saved predictions come from a
separate pass using those selected weights, not the epoch10 model.

**M predates R24 phone/R25 container-ID presentation repairs and the new
description-prompt treatments.** Do not credit those changes for its recovery.
L/M have identical captured runtime package/hardware/source fingerprints; no
runtime/kernel/config corruption was found. J/K retain older wrapper/task/metric
fingerprints rather than being silently treated as identical software contracts.

Evidence: [completion, hashes, frozen configs and runtime inventory](analysis/training-latest-20261015/runtime-inventory.json).

### 18.2 Progression and the epoch10 decline

M improves through epoch7.5 in field scores; relation F1 peaks at epoch5. Eval
loss rises0.06124→0.07852 while extraction first improves then regresses.
Section2 records all four scheduled evaluations.

|Observation|Field F1|Field accuracy|Relation F1|Category F1|JSON / schema|Exact documents|
|---|---:|---:|---:|---:|---:|---:|
|Scheduled epoch7.5 /1785|0.899014|0.858057|0.929062|0.927644|58 /49|7|
|Scheduled epoch10 /2380|0.875780|0.816441|0.898551|0.882012|56 /49|4|
|Separate final pass, selected1785 weights|0.897967|0.857197|0.933333|0.927644|58 /49|7|

The7.5→10 decline is **2.32 percentage points F1 and4.16 points accuracy**.
Recall falls0.88470→0.84433 and JSON coverage58→56; it is not just a small
precision trade. The separate selected-weight pass differs slightly from the
scheduled best despite verified correct export. Its loss is0.06708, not the
scheduled1785 loss0.07045.

Only final/best per-document prediction text is retained. Intermediate weights
and aggregate metrics exist, but intermediate generations do not. Assigning the
epoch10 decline to specific documents/fields would require new inference, which
this audit did not launch.

### 18.3 Native results: each run's original contract

Saved final predictions below use **their own original references**, retaining
their contemporary HS and schema expectations.

|Run|Field precision|Field recall|Field F1|Field accuracy|Relation F1|Category F1|JSON / schema|Exact docs|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|J:5,000 synthetic +600 real|0.913546|0.890140|0.901691|0.864957|0.944954|0.900935|59 /51|5|
|K:coverage replacement +600 real|0.910225|0.877717|0.893676|0.851601|0.933649|0.899628|59 /51|5|
|L/R22:7,000 synthetic only|0.860713|0.815606|0.837552|0.788072|0.830275|0.837736|58 /38|1|
|M/R23:7,000 synthetic +600 real|0.913288|0.883152|0.897967|0.857197|0.933333|0.927644|58 /49|7|

M versus L gains **6.04pp field F1,6.91pp accuracy,10.31pp relation F1 and8.99pp
category F1**. Equal JSON counts do not imply identical malformed-document
membership. This establishes recovery when the real-data shard is restored in
this setup, not an isolated cause attributable solely to geography, realism,
phone formats, EVA initialization or additional updates.

### 18.4 Fair cross-contract comparison: unchanged non-HS facts

All four runs share exactly60 OCR and coordinate-enriched inputs. Projecting away
the wrapper, J/K/L gold is identical; M changes **only14 HS strings across13
documents**, preserving digits. No historical prediction is scored against those
new strings. Exclude **all51 HS gold values and every HS prediction**, leaving
2,525 identical non-HS facts. Full addresses, descriptions and continuations stay.

|All60, non-HS only|Correct /2,525|Field F1|Field accuracy|
|---|---:|---:|---:|
|J|2,245|0.900341|0.863462|
|K|2,216|0.893008|0.850998|
|L/R22|2,069|0.841228|0.791205|
|M/R23|2,227|0.896538|0.855551|

Recovery over synthetic-only persists after excluding HS; it is not a separator
scoring artifact. J still has a small all-document advantage. Separating JSON
coverage from content:

|Same58 J/M parseable documents,2,442 non-HS facts|Correct|Field F1|Field accuracy|
|---|---:|---:|---:|
|J|2,222|0.910656|0.882797|
|M|2,227|0.911771|0.883730|

**Parsed content is essentially tied with J**, not a demonstrated new overall
high. M retrieves five more unchanged facts in this shared cohort. Its additional
malformed plywood document loses23 non-HS facts J recovered, accounting for the
all-document net deficit of18. On shared57 K/M parseable documents, F1 improves
0.906178→0.913916 and accuracy0.875000→0.886502. Across the55 parsed by all
four, non-HS F1 is J0.918509 /K0.910640 /L0.863319 /M0.917805: same conclusion.

M's **own58 parseable documents, complete current target including HS**, score
F1 **0.913105**, accuracy **0.885214**; relations0.950820 and categories0.934579.
This includes all addresses/descriptions, unlike the earlier30k comparison's
unchanged-label subset. It remains below the joint0.90/0.90 goal. Native
per-document F1 and accuracy both exceed0.90 in31/60 documents, versus J32,
K25 and L16; both reach0.95 in18/60 versus J20, K20 and L7. These repeatedly
used60 validation documents do not establish statistical superiority or an
independent generalization guarantee.

### 18.5 Granular strengths and remaining weaknesses

Exact aligned correct values on all60, **not field-F1 scores**. A malformed
document contributes no recovered facts.

|Field family / expected values|J|K|L/R22|M/R23|
|---|---:|---:|---:|---:|
|Container size /106|90|97|94|**102**|
|Container type /106|102|97|94|102|
|Container identifiers /120|111|109|97|108|
|Seals /122|112|111|116|110|
|Placement identifiers /120|111|107|96|109|
|Placement quantities /100|95|90|90|95|
|Declared package quantity /59|50|51|51|**54**|
|Package category /55|48|47|47|46|
|Address lines /175|133|134|133|**138**|
|Party names /182|154|153|147|154|
|Phone values /116|95|99|36|95|
|Country values /256|237|233|216|228|
|Dates /86|85|84|81|83|
|Full descriptions /60|41|40|34|39|

Category gains are primarily equipment-size/type gains, **not** uniformly better
package classification. Seals, countries and complete descriptions do not surpass
J. Phone recovery36→95 after restoring real examples is large, but does not prove
formatting alone caused it. M retains5 wrong,16 missing and7 extra aligned phone
values; two missing occur in malformed JSON. The pending phone/container rendering
repairs address known deficiencies; their training effect is not yet measured.

M independently gets **48/51 current-format HS values** correct, all48 predictions
correct, including restored separators: HS F1 **0.969697**. Only its own HS gold
is used. Three values are missing, including one in malformed JSON. All three DG
scalar facts are correct, but support is too small to claim broad DG mastery.

**Descriptions:**39 exact,19 parsed mismatches and2 malformed. Direct review of
all19 finds lost `99.7% AS PER ISRI`, missing attached packing qualifiers/models,
missed two-page dyestuff continuation, added detached totals/batches, minor copying
errors, and replacement of product wording with carrier boilerplate or a party's
address. The wording-generation probe is not itself a fix for all these extraction
failure modes. See [source-reviewed errors](analysis/training-latest-20261015/SOURCE_REVIEW.md).

### 18.6 Syntax and schema failures

- `13ea92fa`: seven-container refractory document. Generated text repeats nested
  freight/goods blocks, emits an initial container prefix only and ends inside a
  string. Evaluation records one3,072-token sequence and one missing EOS. This
  output is consistent with that event, but exact per-document token/EOS metadata
  was not retained; its capped-event attribution is not independently proven.
- `61d1d525`: short plywood document. Premature `}]` after an unsupported master
  B/L value creates extra JSON data; shipper wording also absorbs `FOR AND` from
  the on-behalf-of clause. This is **not** a long-output truncation.

Together these lose84 expected scalar values. Nine other outputs parse but fail
the bound schema: one zero volume; four using VIN/machinery/vessel-like equipment
IDs; one retaining hyphens where container-ID targets require normalization; two
invalid package enums; one placement referring to an absent container entry.
Exact errors are preserved in [schema-invalid-reasons.json](analysis/training-latest-20261015/schema-invalid-reasons.json).
Grammar constraints and semantic cross-array constraints differ; no hypothetical
constrained-decoding benefit is credited to these saved predictions.

### 18.7 Runtime, reproducibility and bounded implications

M trainer runtime35,515.931s (**9.866h**), train loss0.014253,188,372,720 input
tokens. Scheduled evals total3,353.071s; selected-weight final prediction takes
851.376s. Excluding scheduled evals, input throughput5,856.84tokens/s still
includes saving/other overhead. Evaluation peaks:8.312GiB allocated /13.488GiB
reserved. EVA initialization452.12s. Train maximum source/target8,073/3,146
tokens; validation5,698/1,334. MLflow global-device25,195.5MB and host8,562.7MB
peaks are not model-only memory.

The root reviewed malformed outputs, all19 parsed description errors and
representative source OCR. Read-only helpers verified run/export integrity and
built the CPU replay; neither changed data, production code, configs or weights.
Replays reproduce native field/category/relation metrics within1e-12, conserve
counts and verify22 protected pins. Historical native schema flags remain the
authority: today's schema would inflate J54 versus native51 and L40 versus38.
M's fully bound current replay matches every native metric.

Repeat CPU scoring takes1.69–1.85s after imports. A fresh-process timed replay
takes2.89s wall-clock and peaks at71,736KiB (**70.1MiB**) process RSS.
**65 focused tests pass** across metrics, relation/order
and all four description-probe suites. No GPU/model inference or training was
started. Training-audit API cost:$0. The separate authorized wording retry costs
$0.01586938 and is documented in its own report.

**Next discussion:** retain the recovered real-data signal and equipment coverage;
measure the already repaired phone/ID presentations separately; use the verified
best7.5 adapter; distinguish product-boundary/list/continuation extraction errors
from generated-description style/length. These are concrete targets, not proof
that more samples guarantee monotonic gains or that the synthetic pool is globally
poisoned. This audit does not silently change the next experiment.

Evidence: [own-reference replay and cohorts](analysis/training-latest-20261015/per-field-replay.json),
[implementation](analysis/training-latest-20261015/per-field-replay.py),
[fields](analysis/training-latest-20261015/per-field-fields.csv),
[disjoint groups](analysis/training-latest-20261015/per-field-groups.csv),
[diagnostic facets](analysis/training-latest-20261015/per-field-facets.csv),
[document scores](analysis/training-latest-20261015/per-field-documents.csv),
[exact errors and OCR/PDF paths](analysis/training-latest-20261015/per-field-errors.csv),
[original frozen references](analysis/training-latest-20261015/per-field-frozen-references.json).

## 19. Corrected/anonymized N/R27: 6,900 synthetic plus 600 real

### 19.1 Completion, changes and verified selected export

N completed on 2026-10-17 with **7,500 training records: 6,900 synthetic and
600 surgically anonymized real records**, and the same 60 validation documents.
Relative to M/R23, the combined treatment includes phone presentation,
container-ID rendering, country presentations/canonical country labels,
goods/DG description regeneration, privacy and consistency repairs,
200 new UAE package examples, and archival of the earlier 300 Egypt synthetic
additions. This is **not** a pure scale-up or isolated anonymization experiment.

Model/tokenizer revisions, rank 48/alpha 32, rsLoRA/EVA settings, LR 1e-4,
microbatch 1/accumulation 32, seeds, ScheduleFree AdamW, bf16/SDPA, input-only
positions, sequence limits and decoding settings agree with M. Apart from
data/contracts and run metadata, the only resolved-setting changes are
warmup 119→118 and evaluation/save interval 595→587, tracking 2,380→2,350 total
updates. All 19 captured source/lock hashes and recorded package versions match M.

**Selected/exported model: checkpoint 1174, epoch 4.9984.** All 504 export tensors
match that checkpoint byte-for-byte; the epoch-10 adapter is different. Adapter
SHA-256:

```text
5eab5620fdd9122508dd0018db7520089e7f37c0bc8e61f458de7f114264dfe5
```

The completed immutable manifest verifies all 72 artifacts and 1,661,845,609 bytes.
The status sidecar's pending-manifest marker is written just before that final
manifest by design; it does not overturn the completed state. Full verification:
[progression report](analysis/r27-training-results-20261017/progression.md) and
[exact metrics/checkpoint receipt](analysis/r27-training-results-20261017/progression-receipt.json).

### 19.2 Progression: peak at epoch 5, not 10

| Observation | Epoch / step | Eval loss | Field F1 | Field accuracy | Relation F1 | Category F1 | JSON / schema valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Scheduled | 2.4992 / 587 | 0.056158 | 0.895086 | 0.852209 | 0.930337 | 0.883303 | 59 / 51 |
| **Scheduled selected** | **4.9984 / 1174** | **0.065564** | **0.903790** | **0.863991** | 0.936364 | 0.900542 | 59 / 49 |
| Scheduled | 7.4949 / 1761 | 0.071176 | 0.880258 | 0.821871 | 0.887290 | 0.851711 | 57 / 49 |
| Scheduled interval | 9.9941 / 2348 | 0.075418 | 0.892539 | 0.842046 | **0.956916** | 0.898336 | 57 / 46 |
| Terminal evaluation | 10 / 2350 | 0.076867 | 0.873438 | 0.809791 | 0.905213 | 0.853846 | 56 / 45 |
| Separate final pass on selected weights | Weights at 4.9984 | 0.062170 | 0.903013 | 0.862927 | 0.936364 | 0.900542 | 59 / 49 |

There are **five training-time evaluations**, not four: interval 587 gives step
2348 near epoch 10, and terminal-step evaluation also runs at 2350. The separate
final selected-model pass is not a sixth training checkpoint. Epoch 5 is the
correct complete-field selection; choosing 2348 for its relation peak would be
a different objective.

Later decline includes recall loss: 0.901512 at epoch 5→0.848003 at 7.5, despite
higher precision. JSON coverage falls 59→57→56. Intermediate raw predictions
are not retained, so this audit cannot localize those declines to individual
samples.

The 2348→2350 swing is unusually large for two updates and warrants controlled
checkpoint generation replay. It exists in both event logs and Trainer history;
CPU adapter comparison confirms distinct but close weights, relative L2 change
9.45957e-5. There is **no established cause** from these artifacts: do not call
it corruption, overfitting, kernel nondeterminism or a particular malformed
document without the intermediate generations. Best export remains correct.

### 19.3 Native final outputs: context, not replacement historical gold

All rows below use their **own original full references**, not old generations
scored against R27's revised targets. Addresses/descriptions remain included.

| Saved selected outputs | Field F1 | Field accuracy | Relation F1 / accuracy | Category F1 / accuracy | JSON / schema | Exact docs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| J, old UAE | 0.901691 | 0.864957 | 0.944954 / 0.895652 | 0.900935 / 0.819728 | 59 / 51 | 5 |
| K, replacement UAE | 0.893676 | 0.851601 | 0.933649 / 0.875556 | 0.899628 / 0.817568 | 59 / 51 | 5 |
| L/R22, synthetic-only | 0.837552 | 0.788072 | 0.830275 / 0.709804 | 0.837736 / 0.720779 | 58 / 38 | 1 |
| M/R23, real restored | 0.897967 | 0.857197 | 0.933333 / 0.875000 | 0.927644 / 0.865052 | 58 / 49 | 7 |
| **N/R27, corrected/anonymized** | **0.903013** | **0.862927** | 0.936364 / 0.880342 | 0.900542 / 0.819079 | 59 / 49 | 5 |

N has the highest native saved field F1, but J retains higher native accuracy
and relations; M retains higher categories. N's native M-relative changes are
+0.50 pp field F1, +0.57 pp field accuracy, +0.30 pp relation F1 and −2.71 pp
category F1. These contextual numbers alone do not establish a clean capability
improvement.

Complete-target counts: **2,323 correct / 2,579 gold**, with 2,566 predicted scalar
values. On N's own 59 parseable documents, F1 is **0.909554**, accuracy
**0.874953**. This includes the difficult fields, not just an easy-label subset,
and remains below the joint 0.90/0.90 goal. Both thresholds are exceeded by
29/60 documents; 19/60 reach both 0.95; strict complete-target exactness is 5/60.

### 19.4 Fair comparison with M/J: changed gold and parsing controlled

Every compared run shares exactly the same 60 raw OCR and positioned inputs.
M→N changes 26 scalar paths across 16 documents: 15 canonical party-country
strings, four free-text package→category conversions (eight paths), and three
added delivery-agent phone continuations. J/K/L also differ in 14 HS strings
across 13 documents. **All 175 addresses and all 60 descriptions remain unchanged.**

The [comparison audit](analysis/r27-training-results-20261017/comparison-contract.md)
replays each generation against its own frozen reference, then removes an
entire document/field family from both prediction and gold when that family
changed. This includes predictions in that changed family, not just gold paths;
unchanged absent-on-both-sides families still count extras. No old model is
penalized for newer HS separators, canonical country labels or new package enums.

| Unchanged-fact scope | Gold | Earlier F1 / accuracy | N F1 / accuracy |
| --- | ---: | ---: | ---: |
| M/N, all 60 | 2,555 | 0.900119 / 0.860403 | 0.902922 / 0.862819 |
| M/N, same 57 parseable | 2,435 | 0.916222 / 0.889908 | 0.915781 / 0.885726 |
| J/N, all 60 | 2,541 | 0.904525 / 0.869682 | 0.902819 / 0.862849 |
| J/N, same 58 parseable | 2,445 | 0.915490 / 0.890171 | 0.915565 / 0.885478 |
| L/N, same 58 parseable | 2,452 | 0.862197 / 0.826173 | 0.913370 / 0.880329 |

On the shared 57, M/N correct facts rise 2,231→2,240, but predictions rise
2,435→2,457: slightly higher recovery with less restraint. The all-60 M/N
correct-value gain is 40, of which nine come from the shared 57 and 31 from
changed parsing coverage. An overall native high therefore does **not** establish
a broad new content high. Content F1 is effectively tied with M and J; recovery
over synthetic-only L remains clear.

Paired 10,000-draw 95% document-bootstrap intervals for M→N field F1 are
−1.70 to +2.33 pp on all 60 and −1.06 to +0.95 pp on the same 57.
J/N's same-58 interval is −0.98 to +1.02 pp. These small, repeatedly used
validation sets do not establish superiority over M/J. They are descriptive
intervals, not an independent test.

### 19.5 Granular content: specific gains, persistent regressions

The table below is on **the same 57 parseable M/N documents**, with only changed
families excluded. Gold support is identical in both columns. These are aligned
scalar-field scores, not identifier-anchored category-set metrics.

| Unchanged field family | Gold | M correct→N correct | M F1→N F1 |
| --- | ---: | ---: | ---: |
| Phones | 111 | 94→**100** | 0.870370→**0.917431** |
| Countries | 228 | 212→**221** | 0.929825→**0.960870** |
| Full addresses | 166 | 137→135 | 0.813056→0.810811 |
| Full descriptions | 57 | 39→37 | 0.684211→0.649123 |
| Container identifiers | 112 | 107→106 | 0.955357→0.959276 |
| Seals | 114 | 109→108 | 0.931624→0.927039 |
| Container size | 104 | 101→99 | 0.971154→0.947368 |
| Container type | 104 | 101→99 | 0.971154→0.947368 |
| Declared package quantity | 56 | 53→50 | 0.938053→0.877193 |
| Unchanged package category | 53 | 46→47 | 0.859813→0.870370 |
| Placement quantity | 98 | 94→94 | 0.954315→0.944724 |

Phone precision and recall both improve. This aligns with the intended format
coverage repair, but the combined treatment cannot isolate that renderer's
causal contribution. One concrete success is the tubing document's four
spaced/hyphenated ISO container IDs: all now normalize correctly, while M left
hyphens in two. That sample nevertheless invents a one-SET package from the
product wording, so format correctness is not complete semantic correctness.

All-60 country recovery is 244/256, but comparing it with M's 228/256 must account
for the 15 changed country strings. The controlled 228-value cohort above
confirms an improvement independently of those replacements. This is not an
ISO-code output experiment or an all-country generalization test.

**Relation aggregates tie exactly on the shared 57:** both have 201 correct /
213 predicted / 210 gold, F1 **0.950355**, accuracy **0.905405**. Native all-60
F1 gains partly reflect parsing membership, while exact graphs decline 50→48/60.

**Category regression survives the controls.** Excluding changed package-category
facts, shared-57 category F1 falls 0.941176→0.920455. Current all-60 size/type
are 100/106 correct in aligned scalar slots but 98/106 when anchored to container
identity. Overprediction on unassigned rows and wrong/omitted identities matter,
alongside explicit classification errors. A token can be right on an aligned
row yet not be a correct category fact for the intended container.

### 19.6 Source-reviewed failures and next targets

The [source review](analysis/r27-training-results-20261017/SOURCE_REVIEW.md)
records all 21 parseable description mismatches, the malformed output and
representative package/contact/inventory failures. Main findings:

- **Descriptions:** 38/60 exact; 21 parsed mismatches and one JSON failure. Product,
  model/grade, embedded packing and continuation omissions persist, including
  `AS PER ISRI`, `CARTRIDGES`, `MCC 102` and two-page dye wording. Other outputs
  append Marks/supplier/batch/PO/equipment material. Removing commas/whitespace
  fixes none; neither does the broader punctuation diagnostic. More verbose
  generation alone is not the demonstrated solution.
- **Shipment totals:** local allocations can be correct while totals are wrong:
  two 990-bag rows correctly assigned but total 490 rather than 1,980;
  990/891/539 correctly assigned but total 1,990 rather than 2,420; 25/27 cases
  but total 55. Another output returns local 3,065/4,015 package rows instead of
  declared 7,080.
- **Package ownership/level:** 100 declared pallets becomes 100 PAPER BAGS despite
  correct 20-per-container splits and 4,000 inner bags in the description.
  A product called `TUBING SET` becomes an invented declared one SET.
  Two BALE and one RACK categories are now correct under the new vocabulary,
  but classification coverage alone does not teach these boundary distinctions.
- **Contact ownership/restraint:** a forwarding-agent phone appears under
  delivery; a seal becomes a phone; a date becomes a shipper phone; the delivery
  agent's Italian contact/country leaks into consignee/notify. Phone-format
  gains do not solve role assignment.
- **Inventory:** repeated/omitted IDs and duplicate-container entries persist.
  Six documents with at least five containers have micro F1/accuracy
  0.858732/0.797521; six without ISO containers score 0.822300/0.751592.
  These are descriptive small cohorts, not filtering recommendations. Page count
  alone is uninformative: one-/multi-page accuracy is 0.862808/0.862990.

There is one bounded target-boundary review candidate: `4 PLT RESIN` has current
description `PLT RESIN`, while the model returns `RESIN`. The same target is in
earlier runs. It may retain the package-unit prefix inconsistently with the
declared-description policy; it is documented unchanged, not silently relabeled
to improve this score. Its one scalar cannot explain the overall pattern.

Suggested next coverage should preserve broad/phone/country strength while
targeting **role and boundary contrasts**: declared versus contained packaging,
local versus shipment totals, product SET versus shipment SET, absent fields
versus nearby distractors, complete product continuations and high-cardinality
inventories. The R28 donor-free/factored-package work should incorporate those
contrasts rather than adding only rare enum frequencies.

### 19.7 JSON, schema, resume and validation limits

The selected pass has 59/60 JSON-valid and 49/60 schema-valid outputs. Its one
malformed output has a stray `]` after a scalar contact name holding a phone,
not truncation; all 60 sequences reach EOS and maximum generation is 1,210 tokens.
It loses 37 gold facts. The other ten schema failures are five invalid equipment
identities, one duplicate container, one duplicate HS list, zero volume, a malformed
ISO date, and one ISO-container check-digit failure. No unsupported package enum
occurs. [Exact errors](analysis/r27-training-results-20261017/schema-errors.json).

Constrained decoding can enforce supplied grammar/enum/regex rules; correct
ownership, complete inventory, cross-references/checksums and totals must not be
credited as solved by hypothetical constraints in this unconstrained run.

The reboot/resume is verified: original/resumed environments are byte-identical,
all retained optimizer/scheduler counters match checkpoints, and saved RNG
archives are present. Progress after step 587 was replayed; at least 378 logged
update positions were discarded. Append-only logs contain 76 duplicated loss-log
steps, so use the retained Trainer history rather than summing both branches.

Saved `train_runtime: 26514.2806` and `train_loss: 0.00405857` describe the resumed
Trainer invocation, **not whole-run speed or an unbiased ten-epoch loss**.
First recorded train-begin→train-end spans 11.395 hours, including the reboot gap;
the retained-history logged-step-weighted cumulative loss is 0.01530077.
There is no justified training-speed comparison from the resumed summary.

CPU replay reproduces every current native field/relation/category/schema metric
within 1e-12 and all per-document validity flags. Eight checks cover identity,
list permutation, wrong scalar, malformed JSON, human-text case, native replay,
count conservation and protected pins. The historical replay keeps original native
schema flags where current schemas differ; all 29 comparison pins verify.
Ruff passes for both analysis utilities. Root replay core takes 0.189 seconds
after imports, peak process RSS 356.1 MiB; comparison/bootstraps take 1.529 seconds,
356.1 MiB. These are CPU-audit measurements, not model-memory or training-speed
benchmarks. **No GPU/model inference, training, paid calls or active-data writes.**

Evidence: [own-reference metrics/guards](analysis/r27-training-results-20261017/metrics.json),
[replay utility](analysis/r27-training-results-20261017/analyze.py),
[all fields](analysis/r27-training-results-20261017/fields.csv),
[groups](analysis/r27-training-results-20261017/groups.csv),
[document scores](analysis/r27-training-results-20261017/documents.csv),
[errors](analysis/r27-training-results-20261017/errors.csv),
[source review](analysis/r27-training-results-20261017/SOURCE_REVIEW.md),
[fair comparison](analysis/r27-training-results-20261017/comparison-contract.md)
and [comparison receipt](analysis/r27-training-results-20261017/comparison-contract.json).
