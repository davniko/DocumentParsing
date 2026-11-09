# MPCI extraction: run results and next experiments

Updated: 2026-11-08. This is the consolidated results ledger for **eighteen completed runs**, now including **R/R30 rank96**, compared with **Q/R30 rank64** on Paddle-only inputs, 9,400 synthetic +600 anonymized real and RunPod effective batch48; P/R28; the same-data **O/R27 rank64 versus N/R27 rank48** comparison; synthetic-only **L/R22** and restored-real **M/R23**. Keep this document as the entry point rather than reconstructing the experiment history from artifact directories.

**Default for subsequent experiments: LoRA rank64**, explicitly accepted by the
user after the native terminal checkpoint comparison. Historical rank48 configs
and completed-run exports are not rewritten. The contained R28 sampling follow-up
is documented in [its report](analysis/r28-donor-independent-followup-20261018/REPORT.md).

**Current result:** **R/R30 rank96 checkpoint1566, epoch7.4944, is preferred for this run**: field F1/accuracy **0.910305/0.879665**, relation F1/accuracy **0.937500/0.882353**, category F1 **0.925865**, JSON60/60, schema53/60. Selected evaluation repeats exactly and all504 export tensors match1566. Terminal2090 has field F1 **0.903769**, the same aggregate relation F1, higher category F1 **0.931092**, but two new malformed outputs. Against Q/rank64 best, common-gold field F1 improves **0.902568→0.910305**;41 of44 net additional correct fields come from two previously malformed outputs. On actual-container documents, relation F1 is **0.948936→0.949367**; new false relationships in no-container documents explain the overall relation loss. Five training targets and two validation party-boundary targets changed before this run; inputs changed in only one synthetic record, not validation. Full archival and local MLflow import are verified. See [section32](#32-rr30-rank96-completed-runpod-run) and the [detailed audit](analysis/r30-rank96-results-20261108/REPORT.md). Rank64 remains the agreed default pending a separate decision; this audit does not change configs or releases.

**Previous Paddle-only result:** **Q/R30's selected checkpoint1044, approximately epoch5, remains preferred within the rank64 run**, with field F1/accuracy **0.902204/0.865176**, relation F1/accuracy **0.948936/0.902834**, JSON58/60 and schema55/60. Its two saved evaluations agree byte-for-byte, and all504 exported adapter tensors match1044. Terminal2090 is weaker on native field/relations (0.895197/0.889344 F1), despite better descriptions and whole-document exact9/60. Relative to R28,56/60 validation targets changed and effective batch32→48; this was not a pure OCR ablation. See [section31](#31-qr30-completed-paddle-only-runpod-run) and its [full audit](analysis/r30-training-results-20261108/REPORT.md).

**Previous R28 result:** **P/R28's field-F1-selected checkpoint and terminal checkpoint are both3130, epoch10**, with field F1/accuracy **0.902930/0.862200**, relation F1/accuracy **0.928736/0.866953**, category F1 **0.930582**, JSON58/60, schema52/60 and whole-document exact8/60. The saved scheduled and final prediction passes agree exactly; all504 exported adapter tensors match3130. This is **not an unqualified improvement over R27**: category precision and validity improve, but container inventory/allocation recall regresses. Excluding handling symmetrically because its gold changed, field F1 is **0.902739→0.903639**, a small gain. **Checkpoint3128 is a meaningful alternative**, with field F10.902180, relation F10.937355 and schema53/60. Preserve both; no export is changed by this audit. See [section25](#25-pr28-completed-10000-record-rank64-run).

**Relation follow-up:** reviewed syntax recovery and exact-duplicate removal on
diagnostic copies give relation F1 **0.977679 for R27 versus 0.956720 for R28**.
The remaining nine-correct-fact deficit is concentrated in three documents.
Training targets pass the structural checks, but high-count inventory diversity
and equal-load combinations remain narrow. These are not new production scores
or a proven causal attribution. See [section26](#26-r28-cargo-relation-regression-diagnosis).

**Previous decision, retained for R27:** **O/R27 rank64 terminal checkpoint2350, epoch10, is the user-designated best overall checkpoint of that run**, with field F1/accuracy **0.8991/0.8518**, relation F1/accuracy **0.9707/0.9430**,215/220 correct relations, eight extra and five missed, and53/60 exact relation graphs. Its original evaluation was reproduced exactly, twice; the first mismatched-batch replay is superseded, not its replacement score. Against N/rank48's terminal checkpoint on identical native terminal batches, logged field F1 improves **0.8734→0.8991**, accuracy **0.8098→0.8518**, relations **0.9052→0.9707** and categories **0.8538→0.8948**. This supports a beneficial fixed-alpha rank48→64 experiment, with the source-level and shared-parseable comparisons documented below. Both train on the same6,900 synthetic +600 anonymized real samples. O's automatic field-F1 export remains **checkpoint587**, not the user-designated task-weighted best2350; no checkpoint/export has been overwritten. [Section21](#21-or27-rank64-completed-same-data-comparison) preserves the selected-export-only comparison; [section23](#23-evaluation-non-reproduction-root-cause-fixed-and-native-terminal-result-recovered) diagnoses and fixes evaluation reproducibility. The preferred-checkpoint and terminal-rank comparison is in [section24](#24-preferred-rank64-terminal-checkpoint-and-terminal-rank48-comparison).

For that **R27 rank comparison**, on the **same52 documents parseable in both terminal models**, field F1 improves **0.9178→0.9203** and relations **0.9740→0.9767**, while category F1 regresses **0.9110→0.8966**. Most of the larger all60 gain comes from recovering heavier documents, particularly a ten-container case where rank48 repeats one container64 times until truncation. That is a meaningful output improvement, but not uniform semantic dominance or a measured constrained-decoding result.

The R28 audit includes artifact inspection, CPU metric replay and source-text review; its two existing final evaluations already agree, so no new GPU inference was needed. The earlier R27 audit additionally performed recovered-order GPU checkpoint evaluations. Training datasets, historical checkpoints, launch configurations and selected exports remain unchanged by this audit. No new training was launched. **Sections3–24 preserve dated historical analyses and plans**, with superseding notes where needed; their “current” wording describes those snapshots, not today's conclusion. J/K trained on wrapped targets; L/M/N/O/P use direct targets. M restores printed separators in14 validation HS strings across13 documents. Historical models are **not** scored against replacement gold. Section18 excludes the entire HS family for its historical comparison; section19 conservatively excludes only changed document/field families against each run's own frozen reference, including subsequent country/package/phone changes. N includes the later phone/container-ID presentation repairs, new goods/DG wording, country policy/presentations, surgical real anonymization and bounded package coverage changes together; it does not isolate any one repair. N/O have identical inputs, targets and scoring contracts. O/P have identical validation inputs and non-handling gold; section25 excludes handling for the policy-controlled scalar comparison and explicitly records their different native evaluation batch plans.

## 1. Reading the metrics correctly

- The target is **both** field F1 and field accuracy above 0.90 for “good,” and approximately 0.95 for “great.” A relation F1 above 0.90 does not establish that goal for the complete extraction.
- Field precision/recall/F1 count aligned scalar values, with list-order-invariant matching. **Field accuracy is correct values divided by the union of occupied aligned gold/prediction paths**, not whole-document accuracy and not token accuracy. An incorrect value at one path contributes a precision and recall error but occupies one union path.
- Relation and category metrics compare projected facts. Their accuracy is set overlap/Jaccard. An aligned placement-field score is not necessarily identical to the relation-set score because matching and deduplication differ.
- Invalid JSON gets no recovered-field credit. Parseable schema-invalid JSON can receive partial field credit. Explicit nulls in the target count under the current contract.
- Whole descriptions/addresses are individual string fields. A missing word and a completely wrong description both lose that field's exact-match credit. The error taxonomy below distinguishes their practical severity without silently changing the primary metric.
- **Native scheduled evaluation** means the values logged during that run, using its contemporary labels/settings. Earlier **common-gold replays** below refer to their explicitly dated frozen snapshots, not a license to score historical models against subsequent policy changes. Section18 replays each run against its **own original reference** and compares only unchanged families across contracts. None of these CPU replays is new model inference.
- Best scheduled evaluation and the separate final prediction pass are different observations. Historically randomized evaluation batches caused score differences despite identical weights; section23 fixes this and establishes exact replay contracts. Do not mix a best-epoch aggregate with an epoch10 error inventory, and do not accept an unexplained mismatch as normal evaluation noise.

### Important historical corrections

1. The earlier “compact JSON” experiment did **not** isolate pretty versus compact targets: the previous run's actual cached decoder targets were compact too. The change included casing/equipment labels, evaluation settings and prompt/schema changes.
2. R14 to R16 changed address-punctuation labels as well as adding positional input. That transition is not a pure positional ablation.
3. Removing the explicit prompt was a cleaner comparison: the R16 positional data/settings were retained. Each training run started from the base model/EVA setup; these are not successive fine-tuning stages of the preceding adapter.
4. Older runs did not all use the current explicit-null instruction policy. Before the 1,500-synthetic run, 38 `sameAs` nulls and two negotiability nulls added 40 target facts across 36 validation documents. This particularly changes historical schema-validity scores when evaluated against today's schema.
5. The previous rank32 1,500-synthetic run selected epoch7.5 but exported epoch10. The rank48 run exports its selected epoch10 checkpoint correctly. Granular comparisons below use the **available saved epoch10 outputs**, not the earlier run's best epoch7.5 predictions.

## 2. Experiment registry: eighteen completed runs

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
|O|[Same R27 dataset, rank64](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic6900_packages_r27_positions_inputonly_v7_e10_compact_eva_a32_r64_local_schedulefree_v1.yaml)|7,500|Rank48→64 only scientific config change; alpha32/EVA/settings retained; resumed after power outage; verified best-checkpoint export|10 /2350|118, or5%|
|P|[R28 real600 + synthetic9,400, rank64](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic9400_r28_positions_inputonly_v7_e10_compact_eva_a32_r64_local_schedulefree_v1.yaml)|10,000|Donor-free coverage/refreshes, alias and multi-package-row synthesis, thermal/DG/party coverage and finite data/handling repairs; stable evaluation batching; retain rank64/settings|10 /3130|157, or5%|
|Q|[R30 Paddle-only real600 + synthetic9,400, rank64 RunPod](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic9400_r30_paddle_centroids_inputonly_v7_e10_compact_eva_a32_r64_runpod_schedulefree_v1.yaml)|10,000|Paddle-only centroid inputs and rebased targets/templates; equal-load refresh; effective batch48 on96GB RTX PRO6000; retain rank64/alpha32/LR/epochs|10 /2090|105, or5%|
|R|[R30 Paddle-only, rank96 RunPod](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic9400_r30_paddle_centroids_inputonly_v7_e10_compact_eva_a32_r96_runpod_schedulefree_v1.yaml)|10,000|Rank64→96; same batch48/alpha32/schedule/code; five training and two validation party-target repairs, one synthetic input repair|10 /2090|105, or5%|

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
|O|2.499 /587|0.9001|0.8549|0.9291|0.8897|58/60|52/60|
|P|10 /3130|0.9029|0.8622|0.9287|0.9306|58/60|52/60|
|Q|4.9984 /1044|0.9022|0.8652|0.9489|0.9121|58/60|55/60|
|R|7.4944 /1566|**0.9103**|**0.8797**|0.9375|0.9259|**60/60**|53/60|

**Task-weighted preferred checkpoint, distinct from automatic field-F1 selection:**
O/rank64 **epoch10 /2350**: field F1/accuracy0.899093/0.851755,
relation F1/accuracy0.970655/0.942982, category F10.894831, JSON56/60,
schema48/60. This remains the preferred R27 checkpoint by the user's stated
priority; its complete native evaluation has been reproduced exactly. R28's
selected3130 and alternative3128 are assessed separately in section25.

These are useful historical results, not eighteen controlled experiments on an unchanged scoring contract. In particular A is case-sensitive, B/C best checkpoints differ from their exported final predictions, and H has corrected validation labels. H's saved predictions are from its correctly reloaded best495 checkpoint, not epoch10. H/I share identical training-time validation file hashes; I's saved predictions come from its correctly exported970 checkpoint. J uses later approved labels; section15 records that historical comparison. J/K have identical validation references and training hyperparameters. L/M/N add further data/target changes; sections18–19 separate own-contract scores from fair unchanged-fact comparisons. N/O are an identical-data/target fixed-alpha rank comparison. M, N and O's separate selected-model prediction passes differ slightly from their scheduled best evaluations despite verified correct exports. P's scheduled terminal and final selected-model passes agree exactly under the corrected evaluation implementation. Q's selected1044 repeat also agrees exactly; its Paddle rebase changes56/60 validation targets, so section31 separately controls changed field families and parseability. R's selected1566 repeat agrees exactly; section32 reports both native scores and explicitly supplemental common-gold rescoring for the two amended party-boundary targets, without replacing any historical metric.

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
|O|2.499 /587|0.05377|0.9001|0.8549|0.9291|0.8897|58|
|O|4.998 /1174|0.06293|0.8919|0.8423|0.9336|0.8873|57|
|O|7.495 /1761|0.06648|0.8827|0.8257|0.9171|0.8550|56|
|O|9.994 /2348|0.07036|0.8859|0.8298|0.9420|0.8828|55|
|O|10 /2350|0.07197|0.8991|0.8518|0.9707|0.8948|56|
|P|2.4992 /782|0.05315|0.8933|0.8455|0.9120|0.8775|58|
|P|4.9984 /1564|0.06320|0.8805|0.8195|0.9065|0.8897|54|
|P|7.4960 /2346|0.06769|0.8768|0.8144|0.9371|0.8927|54|
|P|9.9952 /3128|0.07019|0.9022|0.8590|0.9374|0.9306|58|
|P|10 /3130|0.07007|0.9029|0.8622|0.9287|0.9306|58|
|Q|2.4992 /522|0.04630|0.8798|0.8243|0.9295|0.8783|55|
|Q|4.9984 /1044|0.05086|0.9022|0.8652|0.9489|0.9121|58|
|Q|7.4944 /1566|0.05901|0.8953|0.8476|0.9130|0.8893|57|
|Q|9.9936 /2088|0.06091|0.8953|0.8505|0.8893|0.9091|58|
|Q|10 /2090|0.06095|0.8952|0.8512|0.8893|0.9138|58|
|R|2.4992 /522|0.04981|0.8840|0.8317|0.8938|0.8702|56|
|R|4.9984 /1044|0.05182|0.9042|0.8651|0.9237|0.9073|58|
|R|7.4944 /1566|0.06184|0.9103|0.8797|0.9375|0.9259|60|
|R|9.9936 /2088|0.06646|0.9043|0.8664|0.9375|0.9311|58|
|R|10 /2090|0.06643|0.9038|0.8661|0.9375|0.9311|58|

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

## 20. Prepared R27 rank-64 comparison and R28 implementation gates

Historical preparation snapshot, 2026-10-17: at that time **not launched and not counted as a completed run**. The subsequently user-launched run is now complete and analyzed as **O in section21**. The settings, preflight and launch block below describe preparation, not a current instruction to restart or resume the completed run; the launch YAML was later updated for its power-outage continuation. Immutable run artifacts are the authority for the completed experiment.
The [rank-64 YAML](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic6900_packages_r27_positions_inputonly_v7_e10_compact_eva_a32_r64_local_schedulefree_v1.yaml)
uses the exact N/R27 data, prompt, task constraints, model/tokenizer revisions,
seeds and settings. Rank changes 48→64; alpha stays 32, dropout 0.05, EVA512,
LR 1e-4, batch 1/accumulation 32, ten epochs, 118 internal warmup updates,
587-step evaluation/save intervals, unlimited checkpoint retention and
field-F1 best-model loading/export all remain unchanged. With rsLoRA,
alpha/sqrt(rank) changes 4.6188→4.0000; this is the requested fixed-alpha rank
comparison, not a constant-scale experiment. Adapter parameter count grows by
one third at the unchanged target-module boundary; this is not a measured GPU
memory or throughput estimate.

Run ID, adapter name and rank tag are unique. The older launch YAML's explicit
reboot checkpoint and MLflow resume ID are cleared: this is a new base-model/EVA
run, not a rank-48 checkpoint continuation. No data version was created.

Validation: three focused tests prove the full resolved-config difference,
schedule/data/export invariants and identical bound prompt/schema. Ruff passes.
The installed Compose `kie-tools` image passes `validate-config`, full
`inspect-dataset` (7,560 records, 6.199 seconds) and tokenizer-only
`prepare-dataset` (16.744 seconds). There are no exclusions or source truncations;
maximum training source/target lengths are 8,077/3,145 and validation lengths
5,698/1,334, within the unchanged training limits. No model or GPU training was
loaded. Config SHA-256:
`c2a53c6c81231745ed4a80d65f03246f64e29202e8f30654f0f89fa9d59096ae`.
GPU time and memory must be measured during the user-launched experiment;
neither speed parity nor a capacity-driven quality gain is established yet.

Launch, including the documented Compose/Buildx credential-helper workaround:

```bash
mpci_docker_config="$(mktemp -d /tmp/mpci-docker-public.XXXXXX)"

DOCKER_CONFIG="$mpci_docker_config" \
docker --config "$mpci_docker_config" compose --profile training run --rm --build kie-trainer train \
  --config configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic6900_packages_r27_positions_inputonly_v7_e10_compact_eva_a32_r64_local_schedulefree_v1.yaml \
  --project-root /workspace
```

### R28: order of work and current-data disposition

The [consolidated R28 investigation](analysis/r28-packaging-decoupling-20261016/REPORT.md)
and [scheduled requirements](kie-r27-package-vocabulary-audit-2026-10-16.md#policy-and-architectural-requirements-for-r28)
remain the design authority. The section-19 results add semantic contrast
priorities; they do not certify a cause for every regression or prove that
donor-bound samples are intrinsically incorrect.

1. **Settle package semantics before changing targets.** Review the saved 405-code
   application crosswalk into package family and evidenced optional material,
   construction, compression, closure, liner/composite and dimensional features.
   Family/material alone cannot recover every code. Unknown attributes stay
   absent; generic BALE does not imply compression. SET/BALE semantic extraction
   is already supported in R27; their saved-application projection is unresolved.
   Do not emit contradictory code/type pairs. The exact R28 field set still
   needs approval, rather than treating the earlier schema sketch as final.
2. **Decouple facts from presentation.** Replace the audited 447/453 donor-bound
   defaults with an independent goods/form/package shipment ledger. Templates
   retain owned spans, repetitions, formatting, printed field presence, topology
   and anchors. Counts, masses/cube, nested ratios and equipment/tare are sampled
   coherently, not borrowed from an observed shipment. A missing binding is a
   rebase task, not evidence of commodity incompatibility or permission to fall
   back to donors.
3. **Broaden registry reachability deliberately.** Account for all 5,612 HS6
   identities and full usable DG assets. Build the reviewed offline many-to-many
   goods→form→package catalogue; no complete open HS-to-maritime-package matrix
   was verified. Record supported/held outcomes, and eligible/reachable/attempted/
   accepted coverage rather than hiding filtered codes. DG needs coherent
   identities and supported field combinations, without requiring an HS link for
   a layout that prints no HS. Preserve thermal/equipment compatibility and
   printed-only DG/thermal labels.
4. **Retain declared accounting and resolve hierarchy scope separately.** Primary
   quantities/allocations stay at the same declared unit. Distinguish contained
   levels from additive package kinds. NAIC permits up to five GID package
   composites, but level ordering, SGP unit binding and additive mixed-package
   projection still need accepted fixtures/application-builder evidence. Do not
   assume an arbitrary tree or expand the decoder target without settling this.
5. **Migrate by evidence, not wholesale regeneration.** Inventory existing samples
   into target-only migrations, targeted re-rendering and full replacement/new
   synthesis. Preserve valid text/labels and the real/validation membership;
   migrate newly chosen attributes only where printed evidence supports them.
   Keep R27 intact as the rank comparison baseline. Carry phone/container/country/
   HS formatting, anonymization and geometric owners through every changed span.
   Adjudicate `PLT RESIN` and the two BULKS/ITEMS holds before claiming those
   boundaries are settled. Do not reintroduce original private values.
6. **Use crossed pilots and semantic contrasts before scale.** Test the same facts
   in different layouts and unrelated goods/packages in the same layout; remove
   source shipment observations and prove fixed facts do not change. Cover
   generic/material-rich, nested/additive, absent-count/no-container,
   single/uniform/mixed equipment and partial DG/thermal controls. Include local
   versus shipment totals, product SET versus declared SET, description
   continuation versus Marks/PO/batch, repeated-party contacts and large
   inventories. Audit labels, raw text, repeats, allocations and geometry, then
   benchmark planning/rejection rates and memory before a larger publication.

R28 implementation and migration are **not** performed by this config handoff.
Final replacement/addition quotas should follow the current-data inventory and
pilot results, not an unsupported fixed regeneration count. A country-ISO-target
experiment remains a separate ablation; this rank-64 run retains current country
names and every existing target policy.

## 21. O/R27 rank64: completed same-data comparison

This preserves the original **automatically selected-export** audit. The current
task-weighted preferred checkpoint is terminal2350; section24 supplies its
certified terminal-versus-terminal and shared-parseable comparisons. Selected-
export rankings below are not the current overall checkpoint decision.

Audited **2026-10-18**, after completion at **00:24:59 Europe/Ljubljana**
(2026-10-17 22:24:59 UTC). Run O is the user-launched
[rank64 configuration](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic6900_packages_r27_positions_inputonly_v7_e10_compact_eva_a32_r64_local_schedulefree_v1.yaml).
It finished2350/2350 updates and ten epochs, including a verified restart from
checkpoint1174 after the power outage. All five checkpoints remain available:
587,1174,1761,2348,2350. **The field-F1-selected and exported adapter is587**, not
the last checkpoint.

The run directory is
`artifacts/kie-training/t5gemma2-270m-mpci-bl-real600-synthetic6900-packages-r27-positions-inputonly-v7-e10-compact-eva-a32-r64-local-sfadamw-v1`.
[Integrity/progression receipt](analysis/r27-rank64-results-20261018/progression-integrity.json).

### 21.1 What was actually controlled

This is a substantially cleaner comparison than N versus historical M/J/K:
N and O have identical training/validation shard hashes, all60 validation inputs
and reference strings, task constraints, prompt, token limits, evaluation config,
base-model/tokenizer revisions, captured source-module hashes and dependency
versions. Dataset cache identity is identical. Both train on **6,900 synthetic +
600 anonymized real** and evaluate the same60 held-out documents. No current gold
has been substituted into a historical model comparison.

Resolved-config differences are exactly rank48→64 and its associated adapter
name/run ID/rank tag. Alpha32, rsLoRA, EVA512, dropout0.05, LR1e-4, batch1 with
accumulation32, seeds424, internal warmup118 updates and587-step eval/save
interval remain fixed. Trainable parameters increase **22,781,952→30,375,936
(+33.33%)**, with252 target modules/504 adapter tensors unchanged. Fixed-alpha
rsLoRA changes scale32/√rank **4.6188→4.0000**. EVA's rank-dependent calibration
also changes, so this is a fixed-alpha rank experiment, not a separately isolated
constant-scale capacity test. Each run starts from the base model/EVA, not the
previous trained adapter.

### 21.2 Training progression and which result to use

|O stage|Epoch / step|Eval loss|Field F1|Field accuracy|Relation F1|Category F1|JSON / schema valid|
|---|---:|---:|---:|---:|---:|---:|---:|
|Scheduled, selected best|2.4992 /587|0.053769|0.900118|0.854893|0.929062|0.889693|58 /52|
|Scheduled|4.9984 /1174|0.062930|0.891871|0.842341|0.933638|0.887273|57 /48|
|Scheduled|7.4949 /1761|0.066482|0.882682|0.825681|0.917051|0.855046|56 /49|
|Scheduled interval|9.9941 /2348|0.070358|0.885858|0.829763|0.941995|0.882784|55 /48|
|Automatic terminal-step evaluation|10 /2350|0.071969|0.899093|0.851755|**0.970655**|0.894831|56 /48|
|Separate final pass, reloaded selected587|Weights from2.4992|**0.053186**|**0.900691**|**0.855589**|**0.929062**|**0.889693**|**58 /51**|

At the common first587-step evaluation, rank64 improves field F1 over rank48
**0.895086→0.900118**, but rank48 subsequently reaches0.903790 at1174 while
rank64 declines to0.891871. Rank64's validation loss rises throughout the
training-time evaluations after2.5 epochs; lower late training loss does not
translate into better validation field extraction. This is consistent with
poorer later generalization, but does not prove a particular overfitting,
capacity or dataset-causation explanation.

The final2350 relation score is genuinely strong: precision0.964126,
recall0.977273, **215/220 correct relations, eight extra, five missed**, with
53/60 exact relation graphs. However, it belongs to the terminal model, not the
selected/exported model. Its field F1 is0.899093 and JSON validity56/60. It is an
interesting retained relation-specialized candidate, not evidence that the
exported rank64 model improves relations. Intermediate/terminal raw prediction
texts were not retained; aggregate logs cannot support a source-level inventory
of how those215 relations were recovered. Such a checkpoint comparison would
require separately authorized inference.

Step2348→2350 changes some generation metrics sharply in both ranks despite
only two updates. The logs prove the aggregate change, not its mechanism; it is
not evidence of data corruption, wrong export or a diagnosed numerical problem.
The scheduled587 and separate selected-model pass also differ slightly—field
F1 .900118→.900691, schema52→51—so keep those stages distinct. Independently
verified adapter hashes establish correct checkpoint selection despite that
reevaluation variation.

### 21.3 Selected exports: exact same-contract comparison

These are each run's separate `checkpoints/eval_results.json`, reproduced from
its saved final `predictions/validation.jsonl`; not a mixture of scheduled peaks
and terminal predictions. Deltas below are percentage points except counts.

|Metric|N rank48, selected1174|O rank64, selected587|O−N|
|---|---:|---:|---:|
|Eval loss|0.062170|0.053186|−0.008985 absolute|
|Field precision|0.905300|0.917538|+1.224 pp|
|Field recall|0.900737|0.884451|−1.629 pp|
|Field F1|**0.903013**|0.900691|−0.232 pp|
|Field accuracy|**0.862927**|0.855589|−0.734 pp|
|Extraction-fact F1|0.899660|0.897580|−0.208 pp|
|Relation F1|0.936364|0.929062|−0.730 pp|
|Relation accuracy|0.880342|0.867521|−1.282 pp|
|Exact relation graphs|48/60|48/60|No change|
|Category F1|0.900542|0.889693|−1.085 pp|
|Category accuracy|0.819079|0.801303|−1.778 pp|
|Exact category graphs|45/60|44/60|−1 document|
|JSON valid|59/60|58/60|−1 document|
|Schema valid|49/60|51/60|+2 documents|
|Canonical exact documents|5/60|7/60|+2 documents|
|Both document field F1 and accuracy>0.90|29/60|28/60|−1 document|
|Both document field F1 and accuracy≥0.95|19/60|23/60|+4 documents|

Rank64 produces fewer scalar predictions **2566→2486**, fewer correct scalars
**2323→2281/2579 gold**, and fewer incorrect/extra predicted scalars243→205.
Higher precision and more excellent/exact documents coexist with lower aggregate
recall and a somewhat weaker tail. Across all60,25 documents improve in field
F1,26 regress and9 tie. This is a trade-off, not broad improvement or collapse.

### 21.4 Removing the parseability confound

|Same57 documents parseable in both exports|Rank48|Rank64|
|---|---:|---:|
|Field F1|0.914585|0.913264|
|Field accuracy|0.882582|0.878468|
|Correct scalar fields /2472 gold|2270|2248|
|Relation F1|0.944444|0.932715|
|Correct relations /216 gold|204|201|
|Category F1|0.904059|0.895028|
|Correct category facts /266 gold|245|243|

Rank64 recovers formerly malformed `45ed05eb`, but newly breaks `684f442c` and
`da651f2f`. Those three documents account for20 of the42 lost correct scalar
fields; the other22 are inside the shared parseable cohort. Relation loss is
also inside that cohort, not explained away by the extra malformed document.

The10,000 paired-document bootstrap interval for the field-F1 delta is
**[−0.021954,+0.018059]** over all60 and **[−0.009592,+0.007120]** over the same57.
Both include zero. These are descriptive sensitivity checks on a reused small
validation set, not independent statistical confirmation or a causal rank study.
The measured overall difference is small; calling rank64 decisively worse would
be stronger than this evidence supports, just as calling it a new winner would.

### 21.5 What improves and what remains weak

All counts in this table use the **same57 parseable** documents and unchanged
gold. These are scalar aligned fields; anchored relation/category facts are
separately reported above.

|Field group / gold values|Rank48 correct|Rank64 correct|F1 rank48→rank64|
|---|---:|---:|---:|
|Party names /172|151|156|0.8779→0.9096|
|Goods descriptions /57|36|38|0.6316→0.6667|
|Declared package quantities /56|50|52|0.8772→0.9204|
|Goods gross-weight values /46|33|36|0.7333→0.7742|
|Phone values /113|103|97|0.9279→0.9023|
|Country values /244|237|232|0.9634→0.9547|
|HS strings /49|47|45|0.9691→0.9474|
|Address lines /166|136|135|0.8168→0.8108|
|Container IDs /118|110|109|0.9524→0.9520|
|Container seal values /119|110|110|0.9129→0.9244|
|Container sizes /104|99|97|0.9296→0.9108|
|Container types /104|99|97|0.9296→0.9108|
|Package categories /56|49|48|0.8596→0.8496|
|Placement package quantities /98|93|91|0.9394→0.9192|

The source-reviewed error patterns are not interchangeable:

- **Description boundaries improve selectively.** Six formerly wrong descriptions
  become exact: lecithin without batch/top-seal extras; copper scrap retaining
  `AS PER ISRI`; complete loader dimensions; titanium dioxide retaining
  `800 PAPER BAGS (25 KG) ON 20 PALLETS`; bentonite instead of a quantity; cheese
  retaining `BLOCK 15KG`. Four exact descriptions become nonexact: fabric loses
  `WEIGHT:320GSM`, calcium chloride copies `ISPM15` as `ISM15`, spare parts adds
  `CFS-FO`, and a press-brake description loses its leading `1 -`. Persistent
  failures still include Marks/PO leakage and truncation of a long dye catalogue.
  A vehicle-condition clause is selected instead of its product block. Exact
  description38→39/60 is modest progress, not a solved description task.
- **Totals can improve while allocations degrade.** The ten-container
  `5cabcd4b` gets packages50 and gross230850 right but drops another real
  container/allocation; it emits7 inventory rows versus9 before, including one
  formerly duplicated/misidentified row. `6fa0b967` correctly recovers total7080
  but substitutes it for a local4015 allocation. `ff146232` copies the single
  shipment total48 into both container allocations. The issue is ownership and
  accounting scope, not lack of arithmetic constraints.
- **Equipment/category precision is not fixed by rank.** One copied ID changes
  `MSDU`→`MSUD`, losing both anchored relations. Another ID formerly treated as a
  seal is correctly restored. Three unsupported type/size assignments are newly
  added in a layout lacking those gold attributes. Anchored size/type recovery
  is97→97/104 each but extras increase; list-aligned size/type counts differ
  because their matching is not identity-anchored. Shared-cohort refrigerated
  typing remains6/6; open-top remains0/1; 45-foot size8→7/10.
- **Package semantics still matter.** Pallet-versus-inner-paper-bag classification
  improves on `8625ddda`; another DG switches main-declaration cartons to the
  fibreboard boxes printed in a separate chemical-detail block. A loader sample
  invents out-of-vocabulary `PACKAGE_VINCES_LATVIA_COLUMBO`. Generic BALE remains
  2/2; there is **no SET gold category** in this frozen validation snapshot, so
  added training SET coverage cannot be evaluated by SET recall here.
- **Phone loss is principally completeness/type/ownership.** Notify phones fall
  23→19/23; several contacts disappear, and one telephone goes into an email array.
  Format-diverse R27 data and correct publication hashes do not establish that
  those distinctions have been learned.
- **Country restraint improves in some roles, recall falls in others.** Four
  unsupported Italian/Netherlands/delivery-country predictions are removed, but
  printed Slovenia/Egypt/Malaysia/Singapore role countries are omitted elsewhere.
  `NEW YORK,NY` still produces a spurious country (`NYAN`→`NY`), receipt country
  becomes `PHILADELPHIA`, and terminal `MARDAS` is placed in a country field.
  This is not merely the previously discussed `ARE`/`A.R.E` alias edge case.
- **Small DG support cannot establish DG generalization.** The single UN3077
  example changes its correctly extracted class9 to `FLAMMABLE_LIQUIDS`; its
  packing group remains correct. This is a real observed regression, but only
  one hazard-class example, not evidence of a population-wide DG failure.

Comma/whitespace relaxation recovers only2 rank48 and1 rank64 fields. Removing
all punctuation/spacing in string scalars recovers12 and11 respectively:
diagnostic all60 F1 **0.907677→0.905035**. These are deliberately non-policy
diagnostics using unchanged list alignment; stripping meaningful identifier
punctuation is not a proposed target correction. Formatting does not explain
the remaining gap. [Detailed source review](analysis/r27-rank64-results-20261018/SOURCE_REVIEW.md).

### 21.6 JSON/schema, best-export and restart checks

Rank64's two malformed outputs contain invalid numeric spacing and mixed quote
delimiters; neither truncates. The other seven schema-invalid outputs comprise
four with invalid equipment identity formats, one duplicate-phone list, one
zero volume and one address containing contact information. This is a first
reported-error taxonomy, not an assertion those documents have only one defect;
the loader also contains an unsupported category token. The export recovers
five formerly schema-invalid documents and newly breaks three; six remain
invalid, sometimes for a different reason. Full errors and transitions are saved.
Constrained decoding can address supplied syntax/enum/regex rules, but correct
role assignment, complete inventory, local allocations and field completeness
are not guaranteed by those constraints. No constrained replay was run.

The final adapter's SHA-256 exactly equals checkpoint587:
`5c054ed611c48a7faf4b2ab4a770fefc27a24558471346cba5448230df97f307`.
It differs from terminal2350. Rank48's export likewise equals its selected1174.
Both verified receipts contain504F32 tensors. The run manifest is `complete`;
`status.json` retains the older `artifacts_complete_manifest_pending` marker,
but manifest, selected export and final predictions are all present. This audit
does not alter that artifact or infer incomplete training from the stale marker.

Rank64 resumed1174 after logging through1205, replaying31 logged update
positions. Original/resumed environment snapshots are byte-identical; all five
checkpoints retain optimizer/scheduler/RNG files. Saved `train_runtime17468.4506`
and `train_loss0.00183745` describe the resumed invocation only. N resumed at a
different point, so those summary fields cannot measure ten-epoch throughput or
mean loss against one another. Retained logged cumulative loss is
0.01530077→0.01493969; this is a logged-loss diagnostic, not the resumed summary.

### 21.7 Measured cost, reproducibility and next decision

Adapter file bytes increase91,201,176→121,577,328. Selected-evaluation peak
allocated GPU memory is approximately8.311→8.412GiB (+0.101GiB). EVA initialization
is400.78→750.65seconds (158→225passes under identical stopping settings).
Matched normal-update median seconds are11.914→12.361 for steps100–575,
11.871→11.592 for1250–1750, and11.978→11.724 for1850–2300. Mixed directions and
restart/external-GPU conditions do **not** support a clean overall throughput
gain or loss claim. These are observed costs, not a new performance benchmark.

The CPU audit independently reproduces all native field, extraction-fact,
relation, category, validity and exact-match metrics within1e-12. Ten embedded
guards cover identity, recursive list permutation, one wrong scalar, malformed
JSON receiving no facts, human-text case, formatting controls, native replay,
count conservation, identical input/target contract and protected hashes.
Independent root replay passes in2.058seconds after imports, peak RSS411.15MiB;
Ruff passes. No models/GPU, paid calls, dataset/config/checkpoint writes or new
training launches were performed.

**Selected-export-only decision:** N/rank48 remains the higher broad-field-F1
export. This does **not** establish it as the preferred overall checkpoint;
section22 evaluates O's terminal checkpoint as a serious task-weighted candidate.
Rank64 is not uniformly better and does not justify treating capacity as the
established cause of all remaining errors.
Continue the approved R28 semantic/coverage work, emphasizing crossed
goods/package/layout examples and negative contrasts: shipment versus local
totals, declared versus contained packaging, product continuations versus
Marks/reference/condition clauses, contact type/party ownership, printed versus
absent country components, and complete repeated inventories. Measure those
cohorts together with the broad metrics, rather than selecting isolated gains.
This analysis makes no production or dataset changes and launches no next run.

Evidence: [paired replay/guards/pins](analysis/r27-rank64-results-20261018/paired-metrics.json),
[reproducible CPU utility](analysis/r27-rank64-results-20261018/paired_audit.py),
[progression/config/export receipt](analysis/r27-rank64-results-20261018/progression-integrity.json),
[field groups](analysis/r27-rank64-results-20261018/paired-groups.csv),
[all fields](analysis/r27-rank64-results-20261018/paired-fields.csv),
[category values](analysis/r27-rank64-results-20261018/paired-category-values.csv),
[document scores](analysis/r27-rank64-results-20261018/paired-documents.csv),
[all errors](analysis/r27-rank64-results-20261018/paired-errors.csv),
[fact transitions](analysis/r27-rank64-results-20261018/paired-fact-transitions.csv),
[schema errors](analysis/r27-rank64-results-20261018/paired-schema-errors.json)
and [source-reviewed findings](analysis/r27-rank64-results-20261018/SOURCE_REVIEW.md).

## 22. Rank64 terminal2350: task-weighted checkpoint review

Follow-up2026-10-18. The user correctly identified that maximizing field F1 alone
can reject a useful checkpoint: terminal2350 retains nearly the same field score
as selected587 while substantially improving relations. The section21 selected-
export comparison remains numerically correct, but its original recommendation
was too narrow to decide which checkpoint is best for the actual task.

### 22.1 Exact native aggregate comparison

The selected columns are their separate saved final prediction passes. Terminal
is the original epoch10 logged evaluation, independently matched between
`checkpoint-2350/trainer_state.json` and `logs/events.jsonl:489`.
All three share the exact frozen R27 contract and validation references.

|Metric|Rank48 selected1174|Rank64 selected587|Rank64 terminal2350|
|---|---:|---:|---:|
|Field precision|0.905300|0.917538|0.914228|
|Field recall|0.900737|0.884451|0.884451|
|Field F1|0.903013|0.900691|0.899093|
|Field accuracy|0.862927|0.855589|0.851755|
|Non-relation extraction-fact F1|0.899660|0.897580|0.892248|
|Relation precision|0.936364|0.935484|**0.964126**|
|Relation recall|0.936364|0.922727|**0.977273**|
|Relation F1|0.936364|0.929062|**0.970655**|
|Relation accuracy|0.880342|0.867521|**0.942982**|
|Exact relation graphs|48/60|48/60|**53/60**|
|Category precision|0.889286|0.878571|0.871528|
|Category recall|0.912088|0.901099|**0.919414**|
|Category F1|0.900542|0.889693|0.894831|
|Exact category graphs|45/60|44/60|43/60|
|JSON / schema valid|59 /49|58 /51|56 /48|
|Whole-document exact|5/60|7/60|4/60|
|Eval loss|0.062170|0.053186|0.071969|

Versus rank64 selected587, terminal2350 changes field F1 by**−0.001598** and
accuracy by**−0.003834**, but relation F1 by**+0.041593** and relation accuracy
by**+0.075461**. Versus rank48 selected1174, the field differences are somewhat
larger (**−0.003919 F1, −0.011172 accuracy**), while relation F1/accuracy improve
**+0.034291/+0.062641**. These are absolute metric deltas, not percentages.

The user's approximately0.001 F1/0.003 accuracy comparison is also correct:
comparing the **two original training-time evaluations**, scheduled587 versus
terminal2350, gives **−0.001025 field F1 and−0.003138 accuracy**. The slightly
larger differences above use the separate selected-export pass. Neither basis
makes the small field-score difference a sufficient overall ranking criterion.

### 22.2 What the aggregate gain actually contains

These exact integer counts are reconstructed against the known unchanged gold
support and checked against every relevant native precision/recall/F1/accuracy
value. Forty-five arithmetic checks pass, maximum difference1.11e-16.

|Facts|Rank48 selected|Rank64 selected|Rank64 terminal|
|---|---:|---:|---:|
|Correct scalar values /2579|2323|2281|2281|
|Wrong/extra scalar predictions|243|205|214|
|Correct relations /220|206|203|**215**|
|Missing/wrong relations|14|17|**5**|
|Wrong/extra relations|14|14|**8**|
|Correct category facts /273|249|246|**251**|
|Wrong/extra category predictions|31|34|37|
|Correct non-relation extraction facts /2359|2116|2077|2066|
|Wrong/extra non-relation predictions|229|192|206|

The relation gain is **not recall purchased by indiscriminately adding edges**:
relative to selected587 it recovers12 more correct relations **and removes six
extras**. Missing relations drop17→5, about71%; extras14→8, about43%.
Relative to rank48 selected, it recovers nine more correct relations with six
fewer extras. This is a substantial structural improvement on this validation
set and a sound reason to consider terminal2350 the more useful model.

The other side is not zero: unchanged total scalar correct count does not mean
unchanged facts or fields. Non-relation correct facts decrease2077→2066 and
their extras increase192→206; category recall improves while category precision
falls. JSON/schema and complete-document exactness are also lower. A
task-weighted preference for terminal2350 is reasonable, but it should be based
on inspecting those trades, not treating all non-relation metrics as identical.

### 22.3 The saved terminal adapter is the evaluated model

The original Docker image's installed Transformers5.15/ScheduleFree1.4.1 source
was inspected without a model. Evaluation calls `optimizer.eval()`, then Trainer
saves the checkpoint. Our `ScheduleFreeTrainerMixin.save_model` calls the same
idempotent `optimizer.eval()` before saving both adapter and optimizer state.
No training step occurs between terminal evaluation and checkpoint2350 saving.
Therefore the saved adapter contains the evaluated **x weights**, not a different
training y iterate. A direct adapter-only replay is appropriate; recreating an
optimizer or rerunning EVA would be unnecessary and wrong for this audit.

The original terminal predictions were not retained during training. The first
fresh replay used a different batch plan because checkpoint RNG is saved
**after** the native evaluation. The subsequent investigation in section23
recovered the original plan from checkpoints2348/2350 and reproduced all26
native quality/loss/token metrics exactly. The original training run, selected
adapter and data are kept unchanged.

Evidence: [native three-way metrics/counts and ScheduleFree lifecycle](analysis/r27-rank64-results-20261018/terminal-native-comparison.json).

### 22.4 First fresh replay: mismatched batches; superseded by exact recovery in section23

One validation-only replay loaded checkpoint2350 in the same trainer image and
verified all504 adapter tensors exactly. Model revision, dependencies, captured
source hashes, prepared dataset/cache report, prompt, targets, BF16/SDPA,
batch2 and generation settings match training. Saved2350 RNG is restored after
Trainer initialization, but it was saved after the historical terminal evaluation,
so its length-grouped batching is not a reconstruction of the original pass.
No optimizer, EVA calibration, training, MLflow run or export was created.

|Rank64 terminal|Native epoch10 log|Fresh60-document replay|
|---|---:|---:|
|Field F1|0.899093|0.888268|
|Field accuracy|0.851755|0.833396|
|Relation F1|0.970655|0.946882|
|Correct / predicted / gold relations|215 /223 /220|205 /213 /220|
|Wrong/extra relations|8|8|
|Category F1|0.894831|0.902527|
|Exact relation graphs|53/60|51/60|
|JSON / schema valid|56 /48|55 /48|
|Eval loss|0.071969|0.071997|

At this first stage the native0.970655 peak was not reproduced. Section23 now
establishes the cause experimentally and reproduces that exact peak, together
with every other native quality metric and loss. The table above remains a
diagnostic comparison of two batch plans, not evidence of damaged weights or
the authoritative terminal-checkpoint performance.

The fresh five malformed outputs contain170 gold scalar values; three carry
10 gold relations. Thus10 of its15 missed relations are behind malformed JSON.
Its own55 parseable documents have relation F1 **0.969267** (205 correct,
213 predicted,210 gold). That conditional score corroborates the learned relation
signal but is not a repaired all60 score. We cannot assign every original-to-fresh
change to those exact records because original terminal strings do not exist.
One firm difference: `6a1350a5` contains six gold relations, so it could not have
been malformed in a native pass that missed only five relations overall.

### 22.5 Identical three-way parseable cohort and granular terminal findings

This subsection retains the first, mismatched-batch replay's conditional analysis
for provenance. Use section23's recovered native pass for terminal performance;
do not substitute this older replay for the original scheduled evaluation.

All **53** documents below parse in N-selected, O-selected and fresh terminal.
The same2324 scalar values,206 relation facts and258 category facts are scored;
this avoids giving one model an easier parseable subset than another. It is
still conditional on parsing, not an estimate of constrained-decoding performance.

|Shared53 metric|Rank48 selected1174|Rank64 selected587|Rank64 terminal fresh|
|---|---:|---:|---:|
|Field F1|0.916935|0.916540|**0.920968**|
|Field accuracy|0.885939|0.882672|**0.891746**|
|Relation F1|0.946341|0.933985|**0.968675**|
|Correct / gold relations|194 /206|191 /206|**201 /206**|
|Wrong/extra relations|10|12|**8**|
|Category F1|0.908397|0.905950|**0.915572**|

|Correct values / shared gold|Rank48 selected|Rank64 selected|Terminal fresh|
|---|---:|---:|---:|
|Container IDs /113|105|104|**111**|
|Container seals /114|105|105|**114**|
|Container sizes /102|97|95|**99**|
|Container types /102|97|95|**99**|
|Per-container quantities /93|88|86|**92**|
|Goods gross-weight values /44|31|34|**37**|
|Phones /99|89|83|**90**|
|Countries /230|223|218|223|
|Descriptions /53|35|37|37|
|Party names /160|141|147|141|
|Addresses /155|126|127|125|
|Package categories /52|46|45|44|
|HS strings /46|44|42|43|

Terminal's anchored container-use F1 is0.973214 versus selected587's0.954955;
allocation F1 is0.963351 versus0.909091. Six shared-cohort documents improve in
relation F1 and one regresses. Source inspection confirms:

- **Ten-container `5cabcd4b`:** all10 IDs, seals and5-package allocations now
  appear; correct relations14→20/20 versus selected587, or16→20 versus rank48.
  It still gets the shipment package total wrong,44 instead of50. Perfect local
  relations do not establish complete-document correctness.
- **Ten-container `aa9ed483`:** relations18→20/20; all ten12-package placements
  are recovered. Gross weight also improves24348→243480. Description boundaries
  remain wrong.
- **Seven-container `13ea92fa`:** all seven goods/container links are recovered,
  5→7/8 total relation facts; its single8-pallet allocation is still omitted.
- **`217e911a`:** the `MSUD` copying error is corrected to `MSDU`, restoring both
  anchored facts. **`5fa29caa`:** the second local quantity becomes2377 rather
  than3270/3700. **`6fa0b967`:** local4015 is recovered instead of shipment7080.
- **Counterexample, `5d6b477c`:** all four containers remain recognized, but their
  placement list disappears entirely, losing four previously correct links.
  This is a genuine mapping regression, not a malformed-output penalty.

Those last four links and the refractory8-pallet allocation account for all five
missed shared-cohort relations. Eight gold-absent extras remain, including a loader
serial and `RESIN` treated as container IDs and unsupported local quantities.
The older646 allocation ownership question remains explicitly bounded; this
review does not reclassify printed numbers as fabricated without evidence.

Other trades are specific rather than universal degradation: country recall
returns to rank48's shared count, New York/Philadelphia country extras disappear,
and the DG class9/carton mistakes are corrected. But party-name/address/package
scores remain weaker; the DG's product references are now incorrectly added as
HS codes. Descriptions tie selected587's exact count with different wins/losses.
The tyre example gets its local quantity right while copying a local net mass
into the shipment net-weight field. These cases remain coverage priorities.

Shared53 paired-bootstrap relation-F1 intervals span zero: terminal minus rank48
**[−0.017139,+0.055328]**, terminal minus selected587
**[−0.006851,+0.075373]**. Improvements are observed on this reused validation
set, not established independent-test or universal superiority.

### 22.6 Revised checkpoint judgment and validation receipts

**Terminal2350 is now the designated best overall checkpoint of the latest run**
(decision finalized in section24 after exact reproduction). The original native
trade-off is favorable, and fresh shared-parseable field/accuracy/category gains
support it beyond just relation aggregates. The earlier “keep rank48” conclusion
based only on automatic selected exports was insufficient for that decision.
Do not nevertheless hide the fresh all60 regression, the syntax instability,
remaining semantic failures or the small/reused validation scope. In particular,
no constrained-decoding experiment has yet demonstrated how much syntax loss
will disappear or what content those constrained outputs will contain.

The ordinary `final-adapter` is **still587**; this audit does not promote, overwrite
or re-export2350. Future checkpoint decisions should consider field and relation
quality together with semantic error cohorts, rather than interpreting the
single configured selection metric as an overall quality ranking.

GPU prediction took759.09seconds; full driver793.55seconds; peak allocated/reserved
memory8.18/8.23GiB. All78 protected files are unchanged. Independent CPU three-way
replay takes2.963seconds after imports,411.15MiB peak RSS, reproduces every saved
pass's metrics within1e-12, and passes identity/permutation/case/format, scalar/fact
count-conservation, source-contract and hash guards. Both utilities pass Ruff.

Evidence: [detailed terminal source review](analysis/r27-rank64-results-20261018/TERMINAL_REVIEW.md),
[fresh replay receipt](analysis/r27-rank64-results-20261018/terminal-replay-2350/receipt.json),
[fresh prediction text](analysis/r27-rank64-results-20261018/terminal-replay-2350/predictions/validation.jsonl),
[three-way metrics/bootstraps/guards](analysis/r27-rank64-results-20261018/terminal-threeway-metrics.json),
[per-field scores](analysis/r27-rank64-results-20261018/terminal-threeway-fields.csv),
[per-document scores](analysis/r27-rank64-results-20261018/terminal-threeway-documents.csv),
[fact errors](analysis/r27-rank64-results-20261018/terminal-threeway-fact-errors.csv),
[scalar transitions](analysis/r27-rank64-results-20261018/terminal-threeway-transitions.csv),
[replay driver](analysis/r27-rank64-results-20261018/terminal_replay.py)
and [CPU comparison utility](analysis/r27-rank64-results-20261018/terminal_compare.py).

## 23. Evaluation non-reproduction: root cause fixed and native terminal result recovered

2026-10-18, direct investigation without Codex subagents. This supersedes the
unresolved reproduction caveat in section22; its first-replay analyses remain
retained as diagnostics, not the authoritative terminal-checkpoint score.

**The original rank64 epoch10 evaluation now reproduces exactly.** Recovering
the historical batch plan from checkpoint2348's CPU RNG produces checkpoint2350's
saved CPU RNG exactly after one60-element permutation. The full original-image
replay matches all26 native quality/loss/token metrics with zero difference.
A second full replay through the corrected production evaluator in the rebuilt
Compose image repeats every metric and the entire60-document prediction JSONL
byte-for-byte.

|Metric|Native epoch10|Both certified replays|
|---|---:|---:|
|Eval loss|0.07196877896785736|0.07196877896785736|
|Field F1|0.8990934174221521|0.8990934174221521|
|Field accuracy|0.8517550410754294|0.8517550410754294|
|Relation F1|0.9706546275395034|0.9706546275395034|
|Relation accuracy|0.9429824561403509|0.9429824561403509|
|Category F1|0.8948306595365418|0.8948306595365418|
|JSON / schema valid|56 /48|56 /48|
|Correct / predicted / gold relations|215 /223 /220|215 /223 /220|

### 23.1 Proven cause

Transformers5.15 reuses randomized training length grouping for evaluation.
Evaluation permutations consume the global Torch CPU RNG, which is checkpointed
**after** evaluation. Restoring the terminal RNG therefore changes validation
companions and padding instead of recreating the scored pass. Batch size2 alone
was not a complete reproducibility contract.

Controlled repeated real-model probes kept identical unpadded document tokens and
weights: the first-replay pair's2x2128 padding repeatedly produces malformed JSON;
the historical pair's2x1864 padding repeatedly produces valid JSON. The first
greedy token divergence is index53. The primary alone repeats the first-replay
output. This is a batch-layout-sensitive numerical decoding decision, not dropout,
random decoding, a bad label, the wrong adapter, or ScheduleFree saving y instead
of its evaluated x weights. Different batching changed six document strings in
the complete pass. Restoring the original batches restores every native metric.

The first-divergence probe directly captures the numerical mechanism with an
identical53-token decoder prefix: competing logits77.0/76.5 become76.5/76.5 under
historical batching, so the tied greedy argmax changes. This is one BF16
representable step at that magnitude; both sets of logits repeat exactly.

### 23.2 Upstream correction and validation

Evaluation now uses deterministic descending-length grouping with stable ties;
the training sampler remains unchanged. Each scheduled evaluation persists its
actual batch order, raw predictions/references, exact metrics and model/data/
precision/generation/scoring contract. A checkpoint saved at that step gets the
corresponding evaluation contract, not the later selected-best model's output.

The checkpoint evaluation command requires that contract, verifies the loaded
adapter tensors, reuses the recorded plan, and fails explicitly if any recorded
metric or prediction-file byte differs. Historical checkpoints without retained
plans require explicit, certified recovery evidence. Failed replay outputs are
retained. The tool also now honors the configured case-scoring policy consistently.

**134 relevant tests pass** in the rebuilt Compose image, including actual tiny
T5Gemma2+PEFT train/eval/save/new-model-load/re-eval checks with AdamW and ScheduleFree,
persistent workers, changed RNG seeds and exact generated-token/loss comparisons.
Ruff and diff checks pass. The60-row stable sampler costs6.58microseconds; padded
input tokens fall149,056→140,048 (−6.04%). Captured-text publication/independent
rescoring/contract persistence averaged0.0623seconds, +2.875MiB peak RSS.
The production replay takes768.65seconds, without switching precision or batch size.

### 23.3 Wider implication and preserved state

Twenty-one completed local runs use this affected Transformers5.15/BF16 grouping
setup, including all15 A–O runs. This is an exposure inventory, not a claim that
every result materially changes. Close comparisons and automatic best-checkpoint
selection could include batch-layout effects; future comparable-policy checkpoint
comparisons should use the same fixed evaluation plan. Logged historical metrics
are not erased or rewritten. The original terminal's relation advantage is now
confirmed rather than discounted by the first replay.

All original protected training files, datasets, adapters and selected587 export
remain unchanged. The latest preservation pass checks79 immutable inputs
(the original78 plus the recovered-order evidence). No training was launched.

Full diagnosis, code boundaries, measurements and evidence:
[evaluation reproducibility report](analysis/r27-rank64-results-20261018/EVALUATION_REPRODUCIBILITY.md),
[all native/recovered metric deltas](analysis/r27-rank64-results-20261018/terminal-replay-2350-historical-order/native-metric-comparison.json),
[production exact-replay certification](analysis/r27-rank64-results-20261018/terminal-replay-2350-production-fixed/manifest.json),
[affected-run inventory](analysis/r27-rank64-results-20261018/evaluation-sampler-exposure.json).

## 24. Preferred rank64 terminal checkpoint and terminal rank48 comparison

2026-10-18 follow-up. **Run O's terminal checkpoint2350 (epoch10) is designated
the best overall checkpoint of the latest run.** This is the task-weighted
decision requested by the user, prioritizing stronger relation extraction while
retaining near-selected field accuracy/F1. It is distinct from the automatic
field-F1-selected checkpoint587 and does not silently replace the existing export.

### 24.1 Terminal versus terminal: identical native evaluation batches

This is N/rank48 checkpoint2350 versus O/rank64 checkpoint2350, not N's selected
epoch5 model versus O's selected epoch2.5 model. Both train on exactly the same
6,900 synthetic +600 anonymized real samples and evaluate the same60 validation
inputs/targets, including the same description, package, country and HS policies.

The saved **pre-terminal CPU RNG states are identical** between N and O. In each
run, applying the installed60-element length-grouping permutation to checkpoint
2348's state yields checkpoint2350's state exactly. Frozen validation input
lengths and source ordering are identical, so all30 two-document terminal batches
are identical too. Therefore the terminal rank comparison is **not confounded by
the randomized evaluation-batch defect** diagnosed in section23. The
[batch-equality certificate](analysis/r27-rank64-results-20261018/terminal-native-batch-equality.json)
records both state hashes and complete ordered IDs.

|Native terminal metric|Rank48 N,2350|Rank64 O,2350|O−N|
|---|---:|---:|---:|
|Field precision|0.909358|0.914228|+0.487 pp|
|Field recall|0.840248|0.884451|+4.420 pp|
|Field F1|0.873438|0.899093|**+2.566 pp**|
|Field accuracy|0.809791|0.851755|**+4.196 pp**|
|Non-relation extraction-fact precision|0.906006|0.909331|+0.332 pp|
|Non-relation extraction-fact recall|0.837643|0.875795|+3.815 pp|
|Non-relation extraction-fact F1|0.870485|0.892248|**+2.176 pp**|
|Relation precision|0.945545|0.964126|+1.858 pp|
|Relation recall|0.868182|0.977273|**+10.909 pp**|
|Relation F1|0.905213|0.970655|**+6.544 pp**|
|Relation accuracy|0.826840|0.942982|**+11.614 pp**|
|Exact relation graphs|48/60|53/60|+5 documents|
|Relation-supported documents|54/60|54/60|Unchanged|
|Category precision|0.898785|0.871528|−2.726 pp|
|Category recall|0.813187|0.919414|**+10.623 pp**|
|Category F1|0.853846|0.894831|**+4.098 pp**|
|Category accuracy|0.744966|0.809677|+6.471 pp|
|Exact category graphs|44/60|43/60|−1 document|
|Category-supported documents|59/60|59/60|Unchanged|
|JSON valid|56/60|56/60|Unchanged count|
|Schema valid|45/60|48/60|+3 documents|
|Whole-document exact|5/60|4/60|−1 document|
|Eval loss|0.07686724|0.07196878|−0.00489846 absolute|
|Generated non-pad tokens mean|606.9667|582.2833|−24.6833 tokens|
|Generated tokens maximum|3072|1391|−1681 tokens|
|EOS reached|59/60|60/60|+1 document|

The validity counts being equal does not mean the same documents parse. The
shared-parseable analysis below separates genuine within-document extraction
changes from the redistribution of malformed outputs. Likewise, improved micro
category F1 does not imply improved exact category graphs or category precision.
One rank48 output reaches the generation limit, so the shorter rank64 mean is
not by itself evidence of uniformly more concise or faster decoding.

### 24.2 Integer facts behind the gain

|All60 facts|Rank48 terminal|Rank64 terminal|Change|
|---|---:|---:|---:|
|Correct scalar values /2579 gold|2167|2281|+114|
|Wrong/extra scalar predictions|216|214|−2|
|Missing/wrong scalar gold values|412|298|−114|
|Correct relations /220 gold|191|215|+24|
|Wrong/extra relations|11|8|−3|
|Missing/wrong relations|29|5|−24|
|Correct category facts /273 gold|222|251|+29|
|Wrong/extra category predictions|25|37|+12|
|Missing/wrong category facts|51|22|−29|
|Correct non-relation extraction facts /2359 gold|1976|2066|+90|
|Wrong/extra non-relation predictions|205|206|+1|

The structural improvement is strong recall **and** better precision, not merely
more permissive relation emission: missed relations fall29→5 (−82.8%), and
extras11→8 (−27.3%). Category expansion is a different trade:29 more correct
facts, but12 more extras. This is why category precision/exactness remain a
specific weakness even though category recall/F1 improve substantially.

The scientific configuration change remains rank48→64 with fixed alpha32 and
all other training settings unchanged. Trainable LoRA parameters rise33.33%,
22,781,952→30,375,936. Fixed-alpha rsLoRA scale changes4.6188→4.0000 and EVA
calibration is rank-dependent. Both runs also have their documented restart
histories. The observed comparison supports the **higher-rank training recipe**
on this validation set; it does not isolate capacity independently of scaling/
initialization, establish a population-wide improvement, or diagnose capacity as
the cause of every remaining error.

### 24.3 Preferred terminal versus both automatically selected exports

These selected-export rows are the saved final prediction passes; terminal rows
are the original scheduled epoch10 evaluations, now recovered. Keep the stages
explicit: selected587 and selected1174 used different historical batch plans
from the terminal plan. Their score differences are observations, not a
batch-controlled attribution to extra training updates.

|All60|Rank48 selected1174|Rank64 selected587|Rank64 preferred terminal2350|
|---|---:|---:|---:|
|Field F1|0.903013|0.900691|0.899093|
|Field accuracy|0.862927|0.855589|0.851755|
|Non-relation extraction-fact F1|0.899660|0.897580|0.892248|
|Relation F1|0.936364|0.929062|**0.970655**|
|Relation accuracy|0.880342|0.867521|**0.942982**|
|Category F1|0.900542|0.889693|0.894831|
|Exact relation graphs|48/60|48/60|**53/60**|
|JSON / schema valid|59 /49|58 /51|56 /48|
|Whole-document exact|5/60|7/60|4/60|

Versus the rank64 selected export, terminal sacrifices0.001598 field F1 and
0.003834 field accuracy, while gaining0.041593 relation F1,0.075461 relation
accuracy and five exact relation graphs. Comparing instead with the original
training-time checkpoint587 evaluation gives the user's smaller field deltas,
−0.001025 F1 and−0.003138 accuracy. Both comparison bases favor2350 for the
stated overall task priorities, but neither is a claim of strict dominance on
every metric.

The recovered terminal changes the old mismatched-replay shared cohort from53
to **54 identical parseable documents**, containing2374 scalar gold values,
212 relation facts and259 category facts. No description/address/category field
is removed to inflate the result.

|Same54 parseable in all three passes|Rank48 selected1174|Rank64 selected587|Rank64 terminal2350|
|---|---:|---:|---:|
|Field F1|0.917874|0.916331|**0.922401**|
|Field accuracy|0.887130|0.881777|**0.893436**|
|Correct scalar values /2374|2185|2163|**2205**|
|Relation F1|0.947867|0.935867|**0.979118**|
|Relation accuracy|0.900901|0.879464|**0.959091**|
|Correct relations /212|200|197|**211**|
|Wrong/extra relations|10|12|**8**|
|Exact relation graphs|45/54|44/54|**49/54**|
|Category F1|**0.908745**|0.896030|0.905730|
|Correct category facts /259|239|237|**245**|
|Wrong/extra category predictions|28|33|37|

This confirms improved scalar and relation extraction inside a fixed parseable
cohort, not only changed JSON coverage. It also corrects the older “strongest
on all four shared-cohort metrics” wording: with the certified native outputs,
terminal category F1 is slightly below rank48 selected, because its higher
category recall comes with lower precision.

|Correct scalar values /same54 gold|Rank48 selected|Rank64 selected|Rank64 terminal|
|---|---:|---:|---:|
|Container IDs /116|108|107|**114**|
|Container seals /117|108|108|**117**|
|Container size /102|97|95|**99**|
|Container type /102|97|95|**99**|
|Per-container package quantities /96|91|89|**95**|
|Phone values /102|92|86|**93**|
|Country values /236|**229**|224|**229**|
|Descriptions /54|36|**38**|**38**|
|Party names /164|145|**151**|145|
|Postal addresses /159|130|**131**|130|
|Declared package quantity /53|47|**49**|48|
|Package categories /53|**47**|46|45|
|HS strings /47|**45**|43|44|

These are aligned scalar counts, not a replacement for identity-anchored
category/relation facts. The residual trade-offs are specific: party-name and
package-category exactness do not improve uniformly; HS recall improves versus
rank64 selected but additional wrong/extra HS strings lower its precision.

### 24.4 Rank64 terminal's own parseable support and corrected relation diagnosis

On its own56 parseable documents: field F1 **0.920872**, field accuracy
**0.891712**, relation F1 **0.979499**, relation accuracy **0.959821**, category
F1 **0.907776**. These are conditional scores, not a simulated constrained-
decoding result; the four missing outputs and their gold remain in all60 metrics.

The relation split is particularly informative:

|Relation kind, own56 parseable|Correct|Predicted|Gold|F1 / recall|
|---|---:|---:|---:|---:|
|Goods uses container|118|120|118|0.991597 / **1.000000**|
|Per-container package quantity|97|103|98|0.965174 / **0.989796**|
|Combined|215|223|216|0.979499 / **0.995370**|

The all60 five missed gold relations comprise **four behind malformed JSON**
(two each in `5e03e6b2` and `cf7299a1`) and **one genuine semantic omission**:
the printed8-pallet allocation for `CMAU2325606` in `13ea92fa`. The other two
malformed documents, `e5a37f5a` and `da651f2f`, have no gold relations.

**Withdraw the old `5d6b477c` terminal-regression claim:** the certified recovered
terminal retains all four expected placements. Their omission belonged only to
the first mismatched-batch replay. All118 supported goods/container links are
recovered on the parseable inputs; the eight extra relation facts are two
unsupported links and six unsupported local quantities. Those extras remain
real precision failures to inspect, not eliminated by correct JSON structure.

### 24.5 Terminal rank comparison on the same52 parseable documents

Both terminal models parse56 documents, but **the four malformed documents are
different in each**. The intersection is52 documents, with2247 identical scalar
gold values,190 relations and235 category facts. Those52 also parse in both
selected exports, so the complete four-checkpoint comparison is:

|Same52 parseable in all four passes|Rank48 selected1174|Rank48 terminal2350|Rank64 selected587|Rank64 terminal2350|
|---|---:|---:|---:|---:|
|Field F1|0.918751|0.917814|0.918859|**0.920256**|
|Field accuracy|0.888651|0.886695|0.887215|**0.890171**|
|Relation F1|0.955381|0.973958|0.945170|**0.976744**|
|Relation accuracy|0.914573|0.949239|0.896040|**0.954545**|
|Category F1|0.908714|**0.911017**|0.899384|0.896552|
|Correct scalar values /2247|2075|2066|2061|**2083**|
|Correct relations /190|182|187|181|**189**|
|Exact relation graphs|44/52|46/52|43/52|**47/52**|

The terminal-rank gain inside the matched cohort is **+0.244 pp field F1,
+0.348 pp field accuracy, +0.279 pp relation F1**, and−1.447 pp category F1.
That is a modest semantic improvement with a real category-precision regression,
not the uniform6.54-point relation improvement seen in the all60 score.

The exact decomposition of rank48-terminal→rank64-terminal is:

|Parseability group|Documents|Correct scalar delta|Correct relation delta|Correct category delta|
|---|---:|---:|---:|---:|
|Parseable in both|52|+17|+2|+6|
|Newly parseable|4|+198|+26|+30|
|Newly malformed|4|−101|−4|−7|
|All60 net|60|**+114**|**+24**|**+29**|

Thus97/114 of the net correct-scalar gain and22/24 of the net correct-relation
gain come from which documents successfully parse, despite the same total JSON
validity count. This is an observed output-level gain, not evidence that all of
it would survive a constrained-decoding comparison. However, the largest rescued
case is also a substantive repeated-inventory failure, not just a missing brace:

- `5cabcd4b`: rank48 emits the **same `ECMU8103229` inventory object64 times** and
  hits the3072-token ceiling without EOS. Rank64 instead emits all10 actual
  container IDs/seals/45-foot-high-cube attributes and all ten5-package placements:
  **85/89 scalar values and20/20 relations correct**. It still emits total44
  rather than50 and imports a PO prefix into the description. JSON constraints
  alone do not establish recovery of the correct inventory or prevent repetition.
- `45ed05eb`, `684f442c` and `f67e893f` also become parseable; together with that
  ten-container case they supply198 correct scalars and26 relations. Rank64 newly
  breaks `5e03e6b2`, `cf7299a1`, `e5a37f5a` and `da651f2f`, losing101 previously
  correct scalars and four relations. Those regressions remain in all60 scoring.

Do not overinterpret the all60 precision gain either. Inside the common52,
wrong/extra relations **increase7→8** and precision0.963918→0.959391. Some
rank48 extras disappear outside that cohort because rank64's malformed outputs
receive no fact credit. All60 precision improvement is real as a scored result,
but it is not proof of uniformly stronger semantic restraint.

Across all60 documents, field F1 improves in28, regresses in19 and ties in13.
Both field F1 and accuracy exceed0.90 in30→33 documents; both are at least0.95
in15→17. These are distribution improvements, not every-document dominance.

Ten-thousand paired-document bootstrap95% intervals for the terminal rank delta
remain broad: all60 field F1 **[−0.01913,+0.07791]**, relation F1
**[−0.00482,+0.17751]**; matched52 field F1 **[−0.00602,+0.01056]**, relation F1
**[−0.01252,+0.01946]**. These describe sensitivity to this small reused validation
set, not independent significance or a universal rank64 advantage.

### 24.6 Where terminal rank64 improves and where it still regresses

All rows here use the **same52 terminal-parseable documents**. Scalar category
counts are list-aligned; the identity-anchored category facts remain separately
reported in the comparison inventory.

|Field group / shared gold|Rank48 terminal correct|Rank64 terminal correct|F1 rank48→rank64|
|---|---:|---:|---:|
|Container IDs /105|100|103|0.9756→0.9904|
|Container seals /105|104|105|0.9498→0.9722|
|Phone values /102|87|93|0.8657→0.9163|
|Party names /159|136|140|0.8635→0.8805|
|Other locality fields /198|188|190|0.9447→0.9620|
|Declared package quantities /51|46|47|0.8932→0.9126|
|Allocated quantities /85|82|84|0.9591→0.9600|
|Container sizes /91|85|88|0.9290→0.9119|
|Container types /91|85|88|0.9290→0.9119|
|Package categories /51|44|43|0.8544→0.8350|
|Country values /225|221|219|0.9693→0.9648|
|Postal addresses /154|126|125|0.8208→0.8039|
|HS strings /45|42|42|0.9438→0.9231|
|Descriptions /52|38|37|0.7308→0.7115|

More correct size/type values coexist with worse F1 because unsupported
assignments increase. Matched52 category facts improve215→221, but extras
**22→37**; container-size and type extras each grow7→14. This is a coverage/
restraint problem still worth targeting, not evidence that larger rank resolves
the packaging/category problem by itself. Exact descriptions decrease40→39
over all60 (three newly exact, four newly nonexact), and38→37 in the shared52.

Source-checked examples establish concrete wins and counterexamples:

- `5fa29caa`: rank64 recovers both printed local tyre counts3270 and2377 that
  rank48 omitted, accounting for the two extra correct shared-cohort relation
  facts. However, it also changes a correct shipment net total40928.95 to local
  net14650.6. The OCR explicitly prints both local and total NET sections.
- `5222fa98`: consignee/notify names stop at the company rather than swallowing
  the following address line; vessel/voyage separation correctly becomes
  `PANDA 002` / `XA519R`, and `ABERDEEN` is copied correctly. The native target's
  `PACKAGE_RACK` nevertheless becomes generic `PACKAGE_PACKAGE`.
- `7a02e126`: both consignee and notify retain the previously dropped trailing
  `EXPORT`. `66feaac3` recovers the printed shipper email, phone and website.
- `5d6b477c`: all four placements remain correct in **both terminal models**.
  Rank64 repairs the B/L number from an unrelated top reference, but adds eight
  unsupported size/type values, mixes `EXC/ESENYURT` into payment arrangement
  and puts a printed phone into an address as well as the contact field.
- `ff146232`: both true container links remain correct, but rank64 adds local
  quantities1 and48 where gold has none. OCR shows two container declarations
  and a separate `TOTAL 48-PALLET(S)`, not those per-container allocations.
  This remains a shipment-versus-local ownership failure.

### 24.7 Certification and preferred-checkpoint handoff

The rank48 terminal replay verifies all504 adapter tensors against saved weights
and reproduces **all26 original quality/loss/token metrics exactly**, just as the
rank64 replay does. Their complete batch plans are certified identical. No
training, optimizer/EVA initialization, MLflow run, data repair or historical
artifact/export mutation was performed. The original immutable trainer image was
used through a separate analysis-only Compose override; the normal corrected
production-image tag was not reverted.

The four-pass CPU audit replays every saved structured metric within1e-12,
checks all60 IDs/reference strings and native constraints/shard hashes, conserves
field/category/relation counts and runs the ten existing permutation/case/
malformed/identity regression guards. It took5.37seconds after imports with
411.15MiB peak RSS. Ruff and diff checks pass. The native terminal comparison
and all conditioned tables use the original frozen targets, not revised gold.

**Use O/rank64 checkpoint2350 as the preferred overall latest-run baseline.**
The existing `final-adapter` still contains automatically selected587; this report
does not copy, rename or overwrite model weights. The preferred named adapter is
under the O run's `checkpoints/checkpoint-2350/` directory. A future inference or
explicit export must choose that checkpoint rather than assuming `final-adapter`
means the user-designated overall best.

Higher rank improves this run's terminal extraction outcome substantially over
rank48 terminal and strongly recovers the difficult repeated inventory. The
matched-parseable gains are smaller and categories still trade precision for
recall. Retain both observations when planning R28; neither dismiss the improved
terminal result nor turn it into a claim that every extraction weakness is solved.

Evidence and reusable audit outputs:
[certified four-pass metrics, cohorts, bootstrap and protected hashes](analysis/r27-rank64-results-20261018/terminal-certified-metrics.json),
[matched field groups](analysis/r27-rank64-results-20261018/terminal-certified-groups.csv),
[individual fields](analysis/r27-rank64-results-20261018/terminal-certified-fields.csv),
[category values](analysis/r27-rank64-results-20261018/terminal-certified-category-values.csv),
[document-level changes](analysis/r27-rank64-results-20261018/terminal-certified-documents.csv),
[scalar transitions](analysis/r27-rank64-results-20261018/terminal-certified-transitions.csv),
[relation/category errors](analysis/r27-rank64-results-20261018/terminal-certified-fact-errors.csv),
[schema errors](analysis/r27-rank64-results-20261018/terminal-certified-schema-errors.json),
[rank48 replay, model/data preservation and environment receipt](analysis/r27-rank64-results-20261018/rank48-terminal2350-historical-order/receipt.json),
[all rank48 original/replayed metric deltas](analysis/r27-rank64-results-20261018/rank48-terminal2350-historical-order/native-metric-comparison.json),
[rank48 terminal predictions](analysis/r27-rank64-results-20261018/rank48-terminal2350-historical-order/predictions/validation.jsonl),
[identical terminal-batch proof](analysis/r27-rank64-results-20261018/terminal-native-batch-equality.json)
and [reusable CPU comparison](analysis/r27-rank64-results-20261018/terminal_compare.py).

## 25. P/R28: completed 10,000-record rank64 run

Audit date: **2026-11-02**. This section supersedes the opening checkpoint
assessment for the latest run, not the recorded R27 results or its preferred
checkpoint. Scope is read-only result analysis and report updates. No training,
inference rerun, dataset correction, checkpoint replacement or Paddle migration
was performed.

### 25.1 Completion, checkpoint identity and reproducibility

The completed run is
`t5gemma2-270m-mpci-bl-real600-synthetic9400-r28-positions-inputonly-v7-e10-compact-eva-a32-r64-local-sfadamw-v1`.
It contains **9,400 synthetic +600 anonymized real training documents** and the
same60 validation identities. The recorded run starts November1 and completes
November2; there is one training-start event, not a resumed partial run.

**The automatic best and terminal checkpoints coincide: checkpoint3130,
epoch10.** Their detailed error inventory is therefore one inventory, not two
independently evaluated models. The immediately preceding checkpoint3128 is
also analyzed because it presents a real field-versus-relation trade-off.

Verified evidence:

- All **85 manifest-listed artifacts** match recorded byte counts and SHA-256.
- All **504 final-adapter tensors** exactly equal saved checkpoint3130 tensors.
  The safetensors are F32; the CPU comparison uses NumPy without dtype coercion.
- The terminal scheduled prediction JSONL and the final selected-model
  prediction JSONL are **byte-identical**.
- All **26 recorded quality/loss/generation-token metrics** from those two
  evaluations agree exactly. Runtime and live GPU-memory counters are not
  expected to be identical and are not part of this equality claim.
- All five scheduled R28 evaluations use the same recorded document order and
  batch size2 under `stable_length_descending_v1`.
- Every saved structured metric is independently replayed against its own
  reference to within1e-12; all60 prediction/reference identities match.
- **91 protected files** remain hash-identical across the audit.

This is materially different from the earlier R27 reproducibility incident:
there is **no observed saved-checkpoint/final-evaluation discrepancy here**.
The existing repeated final inference supplies the confirmation; no additional
GPU replay was needed in this turn. This is not a claim of reproducibility under
arbitrary batch sizes, kernels or hardware.

### 25.2 Actual experiment and fair comparison boundary

Rank64, alpha32, EVA, ScheduleFree AdamW, learning rate1e-4, ten epochs,
microbatch1 and accumulation32 remain the local setup. The larger dataset changes
the amount of optimization:

|Quantity|O/R27 rank64|P/R28 rank64|
|---|---:|---:|
|Training documents|7,500|10,000|
|Synthetic / real|6,900 /600|9,400 /600|
|Optimizer updates|2,350|3,130|
|5% warmup updates|118|157|
|Scheduled evaluation interval|587|782|
|Recorded input tokens seen|181,761,022|247,057,200|
|Mean training input tokens|2,423.485|2,470.572|
|Mean training target tokens|643.4124|626.8927|
|Maximum training input / target tokens|8,077 /3,145|8,077 /3,145|

Training input-token exposure increases **35.92%**, while document count grows
33.33%. Inputs average1.94% longer; targets average2.57% shorter. Validation input
length remains mean2,290.8/max5,698. The inspected dataset summaries report no
input truncation or filtered overlength targets. R28 records45,377.28seconds
(12h36m17s) for the training phase, including scheduled evaluation, followed by
the separate final evaluation. R27's resumed runtime is not a whole-run
throughput baseline, so these numbers do not establish a speed regression.

The model/runtime package versions and RTX4090 hardware match the prior local
run. The scoring implementations for scalar matching and metrics are unchanged.
However, **evaluation/prediction infrastructure changed** with the reproducibility
repair: R28 uses stable batches, while the certified R27 terminal outputs use
their recovered native historical batches. R27's `production-fixed` replay also
uses that historical order, not a fresh stable-order comparison. Removing
randomized evaluation sampling also changes evaluation's consumption of RNG
state, which can change subsequent training's stochastic trajectory.

Therefore this is a comparison of observed completed runs, **not a controlled
ablation of synthetic-data quality alone**. R28 combines cleanup, refreshes,
donor-free expansion, package aliases/multi-row coverage, thermal/DG/party
coverage, more optimizer steps, and corrected evaluation behavior. Adapter
capacity, data poisoning or any individual synthesis change is not established
as the cause of a particular metric delta.

The label comparison is narrower and directly verified:

- Validation `positionedText` is **byte-identical for all60 documents**.
- Exactly **14 documents change targets**, exclusively in
  `handlingInstructions`.
- All non-handling targets, including HS punctuation, countries, package
  categories, identifiers and allocations, are identical.
- The full schema hash difference is exactly the revised handling field
  description. Sparse prompt-schema hash, package vocabulary and container
  vocabulary remain identical.

Historical predictions are replayed against **their historical gold**, never
R28 replacement labels. For the policy-controlled scalar comparison below,
**all handling fields are excluded symmetrically from both sides**. The
complete native comparison is retained separately. Relation/category facts are
unchanged by this handling exclusion.

### 25.3 Training progression: loss, parsing and content

|Epoch|Step|Eval loss|Field F1|Field accuracy|Relation F1|Category F1|JSON / schema|Whole exact|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|2.4992|782|0.053153|0.893323|0.845550|0.911964|0.877514|58 /48|5|
|4.9984|1564|0.063196|0.880524|0.819497|0.906542|0.889720|54 /47|5|
|7.4960|2346|0.067690|0.876813|0.814421|0.937063|0.892655|54 /49|8|
|9.9952|3128|0.070187|0.902180|0.859048|0.937355|0.930582|58 /53|8|
|10|3130|0.070072|0.902930|0.862200|0.928736|0.930582|58 /52|8|

All counts are out of60. All five evaluations reach EOS for every document;
maximum generated lengths are1,222/1,203/1,260/1,284/1,299, below the3,072 limit.
Thus the malformed outputs are **not output-cap truncations**.

Training loss declines while validation loss rises after the first evaluation.
Mean logged training loss is about0.01825 in epoch2,0.00626 in epoch5 and0.00294
in epoch10. This shows a widening training/validation likelihood gap; it does
not establish that every extraction capability worsens after epoch2.5.

On the **same49 documents parseable at all five R28 checkpoints**, field F1
progresses monotonically:

|Step|Field F1, fixed49|Field accuracy, fixed49|
|---:|---:|---:|
|782|0.924651|0.896303|
|1564|0.930643|0.908926|
|2346|0.931323|0.910204|
|3128|0.932271|0.910297|
|3130|0.933022|0.914037|

The apparent mid-run all60 field decline is therefore strongly affected by
which documents fail JSON parsing. Content on this fixed parseable subset
improves despite rising loss. This is a conditional diagnostic, not replacement
headline performance or a prediction of what constrained decoding would achieve.

### 25.4 Native comparison: selected and terminal models

The table uses the **saved prediction pass being analyzed**, against each run's
own frozen targets. N/O selected-export scores differ slightly from their
original scheduled-best scores for the already diagnosed historical batching
reason. P's scheduled/final scores agree exactly.

|Model / prediction pass|Field F1|Field accuracy|Relation F1|Relation accuracy|Category F1|JSON / schema|Whole exact|
|---|---:|---:|---:|---:|---:|---:|---:|
|N/R27 rank48 selected1174|0.903013|0.862927|0.936364|0.880342|0.900542|59 /49|5|
|O/R27 rank64 selected587|0.900691|0.855589|0.929062|0.867521|0.889693|58 /51|7|
|O/R27 rank64 preferred terminal2350|0.899093|0.851755|0.970655|0.942982|0.894831|56 /48|4|
|P/R28 rank64 near-terminal3128|0.902180|0.859048|0.937355|0.882096|0.930582|58 /53|8|
|P/R28 rank64 selected = terminal3130|0.902930|0.862200|0.928736|0.866953|0.930582|58 /52|8|

Relative to O's preferred terminal, P's native field F1 gains0.003837 and
accuracy0.010445; category F1 gains0.035751. Relation F1 loses0.041919 and
relation accuracy0.076030. R28 is essentially level with N's selected native
field score, not a decisive new field-performance ceiling.

Integer counts explain the trade-off:

|Fact family|R27 terminal correct / predicted / gold|R28 terminal correct / predicted / gold|
|---|---:|---:|
|All scalar values, native handling|2,281 /2,495 /2,579|2,265 /2,450 /2,567|
|Scalars excluding handling|2,274 /2,480 /2,558|2,260 /2,444 /2,558|
|Cargo relations|215 /223 /220|202 /215 /220|
|Categories|251 /288 /273|248 /260 /273|

For categories, **extra/wrong projected facts fall37→12**, while correct facts
fall251→248. This is primarily improved precision/restraint, not universally
better category recall. For relations, missed facts rise5→18 and extras8→13.

### 25.5 Policy-controlled and shared-parseable comparisons

|Cohort and metric|R27 terminal2350|R28 terminal3130|
|---|---:|---:|
|All60, field F1 excluding handling|0.902739|0.903639|
|All60, field accuracy excluding handling|0.857790|0.863254|
|Same54 parseable, field F1 excluding handling|0.926374|0.927468|
|Same54 parseable, field accuracy excluding handling|0.901321|0.903428|
|Same54 parseable, relation F1|0.978723|0.945107|
|Same54 parseable, relation accuracy|0.958333|0.895928|
|Same54 parseable, category F1|0.916512|0.939571|

The all60 policy-controlled field gain is **0.000899 F1**, or0.090 percentage
points, rather than the native0.003837. Correct non-handling scalar values
decrease14; fewer wrong/extra predictions compensate. On the shared54, correct
non-handling scalars are2,183→2,161 against the same2,342 gold values, while
predicted values fall2,371→2,318. Again, the gain is precision-led.

The shared54 comparison confirms that **the relation regression is not solely
caused by different JSON failures**. Correct relations fall207→198 against208
gold facts; exact relation graphs fall49→46. Category exactness remains42/54,
despite higher category F1.

Descriptive paired bootstrap,10,000 draws, gives an all60 handling-excluded
field-F1 delta interval of[-0.03167,+0.03314]; on shared54 it is
[-0.01050,+0.01162]. These reused validation documents do not support a claim
that the small overall field improvement is statistically secure or causal.
They do support identifying concrete changed errors and preserving both models.

### 25.6 Field and category breakdown: what actually improved

Correct counts below are aligned scalar exact matches unless marked as
category facts. Whole descriptions and addresses each count as one string;
case normalization follows the unchanged case-insensitive metric.

|Field/group|R27 terminal correct|R28 terminal correct|Gold support / interpretation|
|---|---:|---:|---|
|Party names|152|156|182 values; improvement|
|Party addresses|134|134|175; no exact-count gain|
|Goods descriptions|39|39|60; no exact-count gain|
|Phone values|98|97|119; fewer predictions improve F1 slightly, not recall|
|All country values|236|237|256; essentially stable aggregate|
|HS values|46|45|51; predictions51→46, so F1 improves despite one fewer correct|
|Container identifiers|116|111|120; worse inventory recovery|
|Seals|120|111|122; important regression|
|Package quantities|50|54|59; improvement|
|Gross-weight numeric values|40|40|48; stable correct count|
|Net-weight numeric values|13|12|15; slight regression|
|Volume numeric values|26|29|33; improvement|
|Anchored package categories|47|50|59; F10.817391→0.854701|
|Anchored container type categories|101|98|106; F10.914027→0.951456 through fewer extras|
|Anchored container size categories|101|98|106; same precision/recall trade-off|

For container type and size, each has predicted115→100, precision0.8783→0.9800
and recall0.9528→0.9245. This is useful reduction of unsupported equipment
attributes, but it must not be described as improved inventory completeness.

Package-category details:

- PALLET improves10/12→12/12; RACK0/1→1/1; generic PACKAGE15/18→17/18.
- BALE remains2/2; CASE remains3/3. Their coverage repairs do not show a new
  improvement on this small validation support because these examples were
  already correct.
- STEEL DRUM remains0/1: the model outputs generic DRUM despite the source's
  material-bearing packing statement.
- IBC remains0/1 through generic PACKAGE substitution; UNPACKED SINGLE remains
  0/1 through VEHICLE substitution; VEHICLE remains0/1 through PACKAGE substitution.
- The one DG document, `d9be84a2`, retains correct UN3077/class9/PGIII but
  predicts fibreboard BOX rather than the declared CARTON category.
- There is **no SET package gold** in this validation split. A product named
  `TUBING SET` is not a declared SET package row and must not be counted as one.

R28 terminal's overall category F10.9306 therefore does **not** mean package
classification itself reaches that score: its package-category F1 is0.8547.
The larger container type/size support dominates the combined metric.

Country performance is not uniformly improved. Shipper countries improve42/45→
44/45 and consignee countries47/52→49/52; delivery-agent countries regress15/16→
12/16, notify34/36→33/36 and discharge countries31/32→30/32. Loading-country
correctness remains31/33, with one more prediction. These are role-level
presence/ownership distinctions, not evidence of a general country-mapping
breakthrough or a general failure of alias rendering.

The shared54 content control also retains weaknesses: correct descriptions
37→37, addresses127→127, phones92→87, container IDs112→109 and seals115→109.
Party names improve144→150 and package quantities49→50. Thus descriptions and
address boundaries remain substantial work even after the broader synthesis.

### 25.7 Source-reviewed relation failures and absence errors

There are **18 missing relation facts** in R28 terminal. The following inventory
accounts for all18, with current positioned OCR and saved gold/predictions
checked together:

|Document|Missing facts|Observed failure|
|---|---:|---|
|`684f442c`|2|Malformed JSON: missing comma after the description prevents the entire output being scored.|
|`6a1350a5`|6|Malformed equipment-size string (`"sizeCategory":"0.0}`); no recovered relation credit. Source does not establish that size.|
|`5cabcd4b`|6|Ten-container inventory collapses to seven distinct containers: three missing links and their three allocation quantities.|
|`5fa29caa`|2|Both container links are present, but their printed tyre allocations3,270 and2,377 are omitted.|
|`9f97c8ed`|1|Single-container allocation is1,466 instead of printed1,467; the shipment package total is correctly1,467.|
|`13ea92fa`|1|All seven links are retained, but the one printed eight-pallet allocation for `CMAU2325606` is still absent.|

The first two account for eight misses through parsing. The other ten are real
inventory/allocation errors in parseable outputs. No unsupported per-container
split should be filled merely to increase scores; the partial-allocation case
requires only its actually printed quantity.

The repeated inventory is particularly informative. `5cabcd4b` prints ten45HC
containers, five packages each, total50. R27 terminal recovers all ten container
links and allocations but gets the shipment total wrong at44. R28 fixes that
total to50, yet terminal output has **nine container rows representing only
seven unique containers**, and11 allocation rows representing those same seven.
It omits `ECMU8155217`, `ECMU8122430` and `ECMU8126040`, and repeats
`ECMU8103300`/`ECMU8134873`. This is not evidence that package total and container
enumeration are interchangeable learning objectives.

There are also **13 extra relation facts**:

- Four repeated allocation facts in `5cabcd4b`.
- Two equipment/link facts in `53e77979`, where text from the shipper/postcode
  becomes the invented ID `SILVERLINE_SERVICES_WITH_BEGIN_33647`, and the loader's
  VIN is moved into a seal field. The source does not declare carrying equipment.
  R27 also failed this boundary, using the VIN as a container ID instead.
- Two in `ff7f0853`, where a company name becomes a container identifier.
- One wrong1,466 allocation in `9f97c8ed`.
- Two646 allocations in `e42f749e` where gold leaves local allocations absent.
  This is a longstanding allocation-ownership adjudication case, not proof that
  the numeric646 never occurs in source. Gold was not changed in this audit.
- Two local48 allocations in `ff146232`, where the shipment prints a separate
  total48 pallets without those individual allocations.

Relative to R27 terminal, relation F1 worsens on five documents (the two new
malformed outputs, repeated inventory, tyre quantities and off-by-one carton
allocation), improves on two newly parseable documents, and is unchanged on53.
The correct-relation difference reconciles exactly:
**−8 parsing −6 inventory −2 tyres −1 carton +4 newly parseable = −13 facts**.

A separate absence error remains in `5d6b477c`: the goods description includes
`TUBING SET`, but no package count/type is declared. R28 terminal invents
33,522 IBCs. Its earlier3128 output invents36 generic packages. Neither is
supported and neither is an example of legitimate SET-package extraction.

These cases define concrete future evaluation targets: enumeration completeness,
duplicate suppression, scalar-total versus local-allocation ownership, and
absence of equipment/package declarations. They do not by themselves identify
which training examples caused the behavior.

### 25.8 Why checkpoint3128 must remain in the comparison

Only two optimizer updates separate3128 from3130. Both use the same stable
evaluation plan, so the observed difference is **not the old batch-order replay
problem**. Nine generated strings change; three documents change aligned scalar
correctness:

|Document|Change at3130|Effect|
|---|---|---|
|`5222fa98`|`ABERDEIN` becomes `ABERDEEN`|One corrected scalar.|
|`aa9ed483`|Recovers volume53.9 and its unit|Two corrected scalars.|
|`5cabcd4b`|Adds repeated equipment/allocation rows|Aligned correct scalars69→76, predicted71→87; distinct correct relation facts remain14 while extra relations increase by4; schema-valid becomes invalid.|

The scalar alignment metric can credit matching size/seal values in repeated
rows even when inventory identity is wrong or duplicated. Accordingly, the
last row's seven extra scalar matches do **not** demonstrate seven recovered
distinct shipment facts. Relation and schema checks expose what field F1 alone
does not.

The remaining six changed outputs include the unsupported package-count/type
change in `5d6b477c`; their primary scalar correctness does not improve. The
full before/after inventory is saved, rather than treating all changed wording
as progress.

Overall,3130 gains **0.000750 field F1** and0.003153 field accuracy over3128,
but loses **0.008619 relation F1**,0.015143 relation accuracy and one schema-valid
document. Category metrics and whole-document exactness are identical.

**Recommendation:** retain3130 as the correctly exported automatic field-F1
winner, and keep3128 as the relation/schema-oriented alternative. On the user's
previous task-weighted priorities,3128 merits consideration before declaring
3130 the best overall R28 checkpoint. Neither replaces R27 terminal's superior
relation result. No weights or automatic-selection rule were changed here.

### 25.9 Handling cleanup, syntax and limits of this validation set

R27 has21 handling strings; R28 has nine across eight documents. R28 terminal
matches five exactly, predicts six, and has nine gold strings. No final
prediction retains the removed free detention/demurrage terms. Reviewed stowage
instructions are recovered in `a44bb232` and `891555b9`.

Remaining handling issues are concrete:

- `40e523d2`: source/gold retains the literal doubled `TO TO` in its transit
  instruction. Prediction removes the repetition and appends a responsibility
  clause. That fails literal extraction and the intended instruction boundary.
- `48c177a3`: humidity85% and ventilation15 instructions are omitted.
- `9f97c8ed`: the complete mixed4°C/closed-ventilation clause is omitted.

This is evidence that the commercial-term cleanup is reflected in outputs,
not proof that handling is solved. Its new F1 must not be compared directly with
the former broader-field F1 as though gold were unchanged.

Of the terminal eight schema-invalid documents, two are malformed JSON
(`684f442c`, `6a1350a5`). The six parseable failures are:

- `53e77979`: invented invalid container identifier.
- `5cabcd4b`: duplicate equipment identifiers.
- `5e03e6b2`: unsupported notify `sameAs: "non_negotiable"`; only `shipper`
  or `consignee` is permitted.
- `891555b9`: placement references without the required container inventory.
- `a44bb232`: email included in the postal address.
- `ff7f0853`: company name used as container identifier.

JSON-constrained decoding might prevent malformed syntax, but it cannot be
assumed to fix these semantic/ownership failures. No such decoding experiment
was run in this audit.

The current validation set is also too narrow to certify several R28 additions:

- **Zero multi-public-package-row targets**, although synthesis now supports
  and includes them.
- Only **one DG document**, with one UN/class/PG declaration.
- Four thermal documents; insufficient to establish broad temperature,
  ventilation and humidity generalization.
- Six forwarding-agent and25 delivery-agent documents.
- Twenty-four multi-container,30 single-container and six no-container targets.

Thus broad multi-row, multi-DG and new physical/thermal coverage cannot be
judged from these60 alone. A separately held-out, source-reviewed challenge set
would be useful for those capabilities; training descendants must not simply
be moved into validation and called unseen evaluation.

### 25.10 Validation, evidence and handoff to the Paddle discussion

The eight-pass CPU audit replays all five R28 checkpoints and three historical
prediction sets, checks original gold/constraints, retains matched cohorts and
reconciles scalar/category/relation counts. Its ten guards cover identity,
recursive list permutation, one wrong scalar, malformed-output zero credit,
case normalization, formatting negatives, native replay, count conservation,
input/target-contract checks and protected hashes. Two complete replays took
**14.82 and15.39seconds,529.35MiB peak RSS**. This measures analysis cost, not
inference throughput. No production hot path was changed.

Targeted existing tests: **41 passed,3 skipped in3.90seconds**. The skipped
best-checkpoint test module and two real-Trainer tests require Torch, which is
not installed in this host virtualenv. They are explicitly not claimed as run.
The NumPy504-tensor equality check, saved native/final prediction equality and
CPU metric replay above are separate successful checks of this actual run.
Ruff passes for the audit script.

Reproduction command from the repository root:

```bash
.venv/bin/python docs/analysis/r28-training-results-20261102/audit.py
```

Evidence:
[complete metrics, cohorts, integrity and hash receipt](analysis/r28-training-results-20261102/summary.json),
[native progression](analysis/r28-training-results-20261102/progression.json),
[exact changed validation handling values](analysis/r28-training-results-20261102/validation-label-changes.json),
[field groups](analysis/r28-training-results-20261102/groups.csv),
[individual fields](analysis/r28-training-results-20261102/fields.csv),
[category values](analysis/r28-training-results-20261102/categories.csv),
[document results](analysis/r28-training-results-20261102/documents.csv),
[scalar errors](analysis/r28-training-results-20261102/errors.csv),
[relation/category error facts](analysis/r28-training-results-20261102/fact-errors.csv),
[schema diagnostics](analysis/r28-training-results-20261102/schema-errors.json),
[3128 versus3130 outputs](analysis/r28-training-results-20261102/last-two-updates.json)
and [replay implementation](analysis/r28-training-results-20261102/audit.py).

**Handoff:** R28 provides a verified modern baseline with improved category
precision and valid-output coverage, but no decisive overall field improvement
and a specific relation/inventory regression relative to R27 terminal. Preserve
R28's3130 and3128 plus R27's2350 for the next discussion. A Paddle-only experiment
should distinguish text-capture gains, coordinate representation changes and
label additions; this audit does not establish that Paddle would resolve the
observed output-side duplication or ownership failures. Paddle migration remains
unimplemented and is intentionally deferred to the user's next discussion.

## 26. R28 cargo relation regression diagnosis

Follow-up audit, 2026-11-02. **There is no broad loss of the relation representation
or detected structural corruption of the training targets.** The observed loss
combines syntax failures, repeated generated rows, and three localized content
regressions. Some relevant training combinations are sparse or absent, but their
causal effect has not been isolated by a training ablation.

This section analyzes diagnostic copies of saved predictions. Original metrics,
model outputs, checkpoints, labels and production inference remain unchanged.
No new GPU inference was required for the measurements below.

### 26.1 Recoverable JSON versus schema invalidity

The metric's behavior is important: malformed JSON receives no recovered-fact
credit, whereas **parseable but schema-invalid output already receives partial
field and relation credit**. Fixing an unrelated country, enum or notify field
does not automatically restore additional relations.

Both malformed terminal R28 outputs contain correct cargo relations:

- `684f442c`: insert the missing comma/opening quote before `grossWeight`.
  The already-generated container link and quantity 1,400 then become readable.
- `6a1350a5`: repair three malformed string/object boundaries around
  `sizeCategory: "0.0"`. All three links and quantities 990, 891 and 539 were
  already generated correctly. The illegal `0.0` category remains illegal:
  the diagnostic does not substitute a valid size or infer absent equipment facts.

Together these recover **eight correct relation facts**. Equivalent review was
applied to all four malformed R27 terminal outputs: a missing goods-array
closure, an unclosed locality string, malformed phone-list punctuation, and a
space inside a numeric literal. That recovers four correct relations and exposes
one additional unsupported VIN-as-container relation. No words or digit
sequences are added, removed, reordered or replaced by these edits. The malformed
phone list remains assigned to its generated address field, even though that
field/type is semantically invalid; it was not silently corrected into contact data.

The separate duplicate diagnostic removes only byte-equivalent structured rows
within the same list. It does not merge conflicting quantities, infer missing
containers or transfer rows between goods. Only R28 terminal's `5cabcd4b` changes:
two duplicate equipment rows and four duplicate allocation rows are removed.

|Comparison on all 60 documents|R27 terminal relation F1 / accuracy|R28 terminal relation F1 / accuracy|
|---|---:|---:|
|Original outputs|0.970655 / 0.942982|0.928736 / 0.866953|
|Reviewed syntax repairs only|0.977679 / 0.956332|0.948081 / 0.901288|
|Syntax plus exact-duplicate removal|0.977679 / 0.956332|0.956720 / 0.917031|

The native relation-F1 difference is 0.041919; after both diagnostics it is
0.020959. Thus roughly half of the measured gap disappears under these reviewed
transformations, **but half remains**. This is a diagnostic of information
already generated, not a measured constrained-decoding or production-repair result.

For context, syntax-only native field F1/accuracy becomes 0.914726/0.883271 for
R27 and 0.918634/0.891667 for R28. Excluding handling on both sides, field F1 is
0.918395 versus 0.919374. The slight field advantage remains, alongside a larger
relation disadvantage. Removing exact duplicates from R28 raises relation F1,
but scalar accuracy slightly decreases: repeated rows had received some aligned
scalar credit. This reinforces the need to track both metrics.

On the **same 45 documents whose original outputs pass schema validation in both
models**, relation F1 is 0.985423 versus 0.973607. Filtering to schema-valid output
does not remove the content regression and must not replace the all-document result.

### 26.2 The remaining content difference is concentrated in three documents

After syntax recovery and exact deduplication, R27 has **219 correct / 228
predicted / 220 gold** relation facts; R28 has **210 / 219 / 220**. Both have nine
extra facts. Their nine-correct-fact difference is fully accounted for by:

|Document|R28 loss relative to R27|Observed mechanism|
|---|---:|---|
|`5cabcd4b`|6 facts|Three omitted container identities, each losing its membership link and allocation quantity.|
|`5fa29caa`|2 facts|Both container identities are correct, but the printed per-container quantities are omitted.|
|`9f97c8ed`|1 fact|The package total is correct at 1,467, but the repeated allocation value is 1,466.|

The long-standing missing eight-pallet allocation in `13ea92fa` affects both
terminal models; it is not an additional R28 regression.

For localization only, excluding those three cases from the repaired comparison
leaves the same **193 correct of 194 gold relations** in both models. Relation F1
is 0.974747 for R27 and 0.977215 for R28 over those remaining 57 documents.
This does not excuse the three failures; it establishes that the remaining
relation recall is not broadly degraded.

#### Repeated inventory and downstream repetition

`5cabcd4b` prints ten distinct `ECMU` container IDs, all with five packages. The
missing R28 IDs occur in the actual input, including positioned lines; they were
not lost by OCR or input truncation. The source spans pages and repeats a highly
similar row structure.

R28 checkpoint behavior is revealing:

|Checkpoint|Inventory / allocations|
|---:|---|
|782|Eight distinct containers; two omitted.|
|1564|Nine distinct containers; one omitted.|
|2346|All ten containers and all ten quantities correct.|
|3128|Seven distinct containers; three omitted.|
|3130|The same seven distinct containers, plus repeated equipment and allocation rows.|

The capability was present at epoch 7.5 and was not retained at the selected
terminal checkpoint. This rules out the stronger claim that the architecture,
schema or input could never represent this document correctly. It does not
identify which later gradient updates or training examples caused the loss.

The allocation list repeats the same reduced identity set as the earlier
generated equipment inventory. At terminal, it has eleven rows summing to 55,
while the correctly extracted shipment total is 50. After deduplication there
are only seven allocations summing to 35. Deduplication fixes repetition, not
inventory completeness. The observed output is consistent with propagation of
an earlier generated inventory error into later allocations; no claim about
internal attention or a proven neural copying circuit is made.

The selected field metric also provides an imperfect checkpoint criterion for
this failure. The last two updates add matching scalar fields in repeated rows
without recovering distinct container relations. That contributes to selecting
3130 over the more relation-consistent 3128. It explains the selection outcome,
not why the weights started producing duplicates.

#### Spatial association and numeric distractors

In `5fa29caa`, the exact input includes:

```text
TCLU5473302/40HC || 104,430
3270 PACKAGE    || 297,429

TLLU5170541/40HC || 103,558
2377 PACKAGE    || 297,557
```

The left-column container blocks occur earlier in text order than the
right-column goods/quantity blocks. Their nearly equal y coordinates provide
the association. The document also prints equipment tare weights, local net
weight and shipment totals.

At checkpoint 782, the model assigns 3,840 to the first container—exactly its
printed **tare weight**, not its package count—and assigns an incorrect 3,847
to the second. From checkpoint 1564 onward, it omits both quantities instead.
Thus the local problem is not absence of the numbers: it is identifying the
correct numerical role and associating separate columns. R28 nevertheless fixes
the shipment net weight to 40,928.95 where R27 used one container's 14,650.6.
This is a selective ownership trade-off, not blanket loss of numerical extraction.

The input already contains useful coordinates. Paddle or a row-oriented
presentation could make association easier, but no changed-input inference has
yet demonstrated that effect for this case.

#### Inconsistent repetition of a correct scalar

`9f97c8ed` is a one-container document. R28 extracts package total 1,467 and the
correct CARTON category but writes allocation 1,466. This needs neither a new
package schema nor an unknown split. It is an inconsistent repeated value.
Earlier checkpoints 782 and 1564 produced the correct quantity; late checkpoints
3128 and 3130 do not. A conditional consistency check can flag it, but should
not be generalized into forced conservation for documents with incomplete or
contradictory allocations.

For example, `13ea92fa` has only one printed local quantity. At epoch 5, R28
assigns eight to all seven containers; later it omits every local quantity.
Neither behavior is correct. Requiring every allocation to be populated or
balanced would teach the wrong policy for this source.

### 26.3 What the training audit rules out

The paired scan covers both training versions and their original validation
references. Every target passes its relevant task contract. For R28 synthetic
training, it finds:

- No duplicated container identifiers or duplicated placement identifiers.
- No placement referencing an identifier outside the equipment inventory.
- No inventory identifiers left unlinked to the single goods item.
- All target container IDs found on at least one input line after stripping
  presentation separators. This is a text-presence check, not a proof of every
  spatial association.
- **8,728 complete allocation sets**, all with allocated quantities summing to
  the declared public package quantities.
- Partial and missing quantities excluded from that conservation check.

All **6,900 retained synthetic identities** preserve their container count,
placement count and quantity-visibility pattern between R27 and R28. Refreshes
change identifiers, values and package rows, but do not systematically delete
local quantities. The 600 real records retain exactly the same equipment-ID
lists and placement values/order. Eight existing real arithmetic discrepancies
and one validation discrepancy remain visible in the audit; they are not new
R28 changes and were not relabeled to force conservation.

Nor did relation supervision disappear under other output fields. Using the
identical saved tokenizer and canonical decoder serialization, tokens overlapping
`splitGoodsPlacement` occupy **6.693% of target content tokens in R27 versus
6.757% in R28**, excluding special tokens. This measures token representation,
not the size or direction of gradient contributions. It does not support a
general claim that more description/package text displaced allocation supervision.

### 26.4 Concrete coverage weaknesses and distribution changes

|Synthetic training property|R27|R28|
|---|---:|---:|
|Documents|6,900|9,400|
|Multi-container documents|2,104|2,904|
|Multi-container, complete quantities|1,770|2,446|
|Multi-container, all quantities absent|327|427|
|Multi-container, partially visible quantities|7|31|
|Equal quantities among complete multi-container documents|393 / 1,770 = 22.2%|357 / 2,446 = 14.6%|
|Documents with at least ten containers|105|133|
|Parents supplying those high-count documents|3|3|
|At least eight containers, one ID prefix, equal quantities|0|0|

The extra 2,500 documents contribute only **49 equal-load cases among 676
fully allocated multi-container examples**, or 7.2%. Within the retained 6,900,
the number of equal-load examples decreases from 393 to 308 during regeneration.
By comparison, complete multi-container real training has 73/168 equal-load
cases (43.5%), and validation has 11/19 (57.9%). Unequal allocations are valuable;
the issue is preserving coverage of both patterns rather than replacing one
with the other.

Every synthetic shipment with at least ten containers still comes from
`e827ecdd`, `af151047` or `f36409a0`; their current inventories have 13 or 14
containers. There are no synthetic ten-container inventories, despite two in
validation. Exact length ten is not inherently necessary, but the concentration
in three source layouts shows that more total samples did not broaden this
part of the task much.

More specifically, neither version's real or synthetic training has the joint
pattern **eight-or-more containers + identical four-letter ID prefix + equal
allocations**. `5cabcd4b` does. This is a defensible challenge-set and coverage
target, but a **pre-existing gap**, not proof that R28 introduced the failure.
Its successful epoch-7.5 output demonstrates some generalization beyond that gap.

The tyre layout has another narrower imbalance. A typography-tolerant screen
for multi-container documents with `TARE WT`/`TARE WEIGHT` plus a number finds:

- R27 synthetic: **43 complete / 7 partial** allocation examples.
- R28 synthetic: **55 complete / 31 partial** examples.
- Real training: **five complete** examples in both versions.

The 31 current partial examples all derive from `b8544221`, correctly printing
only the fifth container's package quantity. The exact `Tare Wt :` presentation
used in the failed validation case occurs in that partial synthetic family,
while the real two-container source `4a48581c` supplies a close complete-allocation
positive example but no synthetic descendants. Broader typography variants do
have complete examples, so it would be wrong to say the model has seen none.

This is a plausible opportunity for balancing complete versus partial layouts
with the same numeric distractors. It is **not** evidence that the correctly
labeled partial examples should be removed or filled with invented quantities.

### 26.5 What is established and what remains causal speculation

Established mechanisms at the output/evaluation level are:

1. Syntax failures hide correctly generated relations.
2. Duplicate rows create extra facts; removing them cannot restore missing IDs.
3. Late training loses a previously successful long-inventory realization.
4. Some quantities are confused with unrelated weights, then omitted.
5. Correct totals are not always repeated consistently in allocation fields.
6. Field-F1-only checkpoint selection can prefer a model with worse distinct
   relations and structural validity.

The training scan identifies plausible contributing coverage problems, but does
not prove that equal-load dilution, partial-allocation expansion, multi-row
synthesis, alias changes, adapter capacity or any individual batch caused the
regression. R27/R28 also differ in total optimizer exposure and native evaluation
batch plans. No claim of a universal causal explanation is warranted from this
one comparison and three concentrated semantic regressions.

### 26.6 Recommended bounded experiments

**First, separate generation robustness from content.** Test JSON/schema-constrained
generation or an explicit validation-and-retry path on the same frozen inputs,
with reproducible batching. Measure all fields and relation facts, not only JSON
validity. The syntax-repair scores quantify recoverable existing content; they
do not predict how a constrained decoder would actually complete its outputs.

Add source-aware consistency diagnostics for unique equipment, repeated identical
placements, inventory/placement membership and provably complete quantity
conservation. Flag disagreements for review/re-extraction; do not silently fill
missing allocations, guess container identities or force totals on partial cases.
No such serving behavior was implemented by this audit.

**Second, test the concentrated semantic patterns.** A small independently held-out
set should cross inventory length, shared versus different ID prefixes, equal
versus unequal counts, page breaks, column-separated quantities, tare/weight
distractors, and complete versus partial visibility. Keep package family and
goods variety broad. Do not regenerate validation documents into training.

For a training intervention, target those combinations on suitable training
sources, rather than adding arbitrary multi-container volume. The complete
two-container tyre source is one candidate; the long inventories need more
source/layout diversity or an explicitly validated variable-row presentation.
Maintaining both equal and unequal allocations is preferable to a global
quota change based solely on 19 validation examples.

**Third, make checkpoint selection reflect the task.** Keep the original native
field-F1 winner for reproducibility, but compare a prespecified relation/schema
guardrail or task-weighted selection criterion. Checkpoint 3128 already supplies
a useful candidate without retraining. Do not retrofit a score solely to make
one inspected validation example win.

Only a matched ablation can determine whether targeted sampling restores these
relations while preserving R28's category gains. Paddle input experiments remain
separate: better text capture or geometry may help spatial association, but the
enumeration and repeated-value failures also exist with all necessary text present.

### 26.7 Evidence and validation

[Diagnostic scores, full dataset distributions and protected hashes](analysis/r28-training-results-20261102/relation-diagnostics.json),
[every reviewed punctuation edit and exact duplicate removal](analysis/r28-training-results-20261102/relation-repair-inventory.json),
[checkpoint-by-checkpoint cargo outputs](analysis/r28-training-results-20261102/relation-trajectories.json),
[six original validation inputs and targets](analysis/r28-training-results-20261102/relation-validation-focus.json),
[per-record structural inventory](analysis/r28-training-results-20261102/relation-training-records.csv),
[retained-record structural changes](analysis/r28-training-results-20261102/relation-training-structural-changes.json),
[parent distributions](analysis/r28-training-results-20261102/relation-parent-distributions.json),
[read-only diagnostic implementation](analysis/r28-training-results-20261102/relation_diagnosis.py)
and [diagnostic regression tests](analysis/r28-training-results-20261102/test_relation_diagnosis.py).

The final diagnostic pass validates all 17,620 versioned train/validation targets,
replays nine model/stage combinations, tokenizes the original targets and checks
ten protected source artifacts. Runtime is **27.79 seconds**, with **599.64 MiB
peak RSS**. This is analysis cost, not inference performance. No production hot
path, training sample, schema, checkpoint or original prediction was modified.

The seven new diagnostic guards plus existing metric/output-order tests pass:
**44 passed in 2.93 seconds**, with no skips. Guards verify repair scope,
unchanged already-parseable outputs, fail-closed handling of unreviewed malformed
text, non-mutating exact deduplication, preservation of conflicting rows and
partial visibility, schema-invalid relation credit, and receipt reconciliation.
Ruff, report-table/link checks and `git diff --check` pass.

## 27. Cargo refresh scope and Paddle migration audit — 2026-11-02

The [detailed follow-up](analysis/paddle-migration-scope-20261102/REPORT.md)
separates narrow coverage refreshes from a Paddle-only dataset migration.
No datasets, templates, production code or model artifacts changed in this audit.

- **Smallest missing-combination repair:** 24 older records across seven parents
  already have at least eight containers and equal allocations. A same-prefix
  identifier refresh can preserve goods, package totals and allocation quantities.
- **Equal-load candidate pool:** 345 older unequal, complete-allocation records
  have divisible totals; 183 have one public package row. These are arithmetic
  candidates, not capacity/binding-certified edits. Repeated masses, volumes and
  nested counts must be updated consistently where the allocation changes them.
- **Long-layout coverage:** the admitted registry has only the three existing
  13/14-container layouts. Nine further >=10-container real-training sources
  remain unadmitted, including five ten-container candidates.
- **Tyre source confirmed:** its left-column equipment and right-column counts
  are aligned in the real PDF. GLM orders the columns separately; Paddle places
  each count within two region entries of its container. This is not a synthetic
  rendering defect or proof of Paddle-model improvement.
- **Paddle rebinding scope:** 24,513/31,151 existing binding occurrences have
  literal region candidates; 6,638 need segmentation/recognition/geometry review.
  Additional unowned regions still need field/privacy dispositions. Source review
  spans 707 unique identities when real600, validation60 and template parents are
  combined, with 409 overlapping real/template identities.
- **Existing-state preservation:** all9,400 synthetic provenance references were
  located. Direct current raw/position receipts match4,306 records; an additional
  2,054 match the alias publication endpoint exactly. The other3,040 need further
  correction-chain reconciliation, not automatic rollback to old candidates.
- **Geometry:** 948 generated logical envelopes across200 distinct parents
  reproduce current centroids. Wrapped logical envelopes are not individual
  Paddle regions. The isolated centroid/bbox serializer passes all660 sources;
  production physical-fragment output and full Paddle migration remain work ahead.

The audit also separates true recognition changes from interleaved-region lookup
failures, inventories transfer of existing anonymization decisions, and defines
the paired20-template/60-descendant full-document gate before broader conversion.
Its60 region-level transfer controls are not full-document certification.
The new and existing position/OCR tests pass **117 tests in 41.52 seconds**.

## 28. R29 intermediary equal-allocation refresh — 2026-11-02

The [refresh report](analysis/r29-equal-load-refresh-20261102/REPORT.md)
documents a new R29 copy; R28, real600, validation60 and model artifacts stay
unchanged. No R29 training or Paddle conversion is scheduled by this work.

- **432 older synthetic records refreshed/replaced**, keeping 9,400 synthetic
  plus 600 anonymized real training records. Both overlapping UAE sidecars are
  synchronized; they are not additional membership.
- **400 donor-free cargo refreshes across 84 parents**, including 112 mixed and
  288 uniform equipment examples. Repeated counts, package nouns, cargo
  measurements and DG/thermal facts move together. Noncargo facts and existing
  container/seal identifiers stay protected in this cohort.
- **24 shared-prefix-only refreshes** supply the formerly absent long/equal/shared
  ID-prefix pattern without resampling cargo.
- **Six source-preserving identifier variants**, three each from the anonymized
  11-flexitank and 40-container layouts, replace six other older records. Neither
  source has a published production template contract; the historical 40-container
  compilation was rejected. The user-authorized fallback was used, **not presented
  as donor-free admission**. Complete independent admission remains separate work.
- Two older source-only tonnage repetitions were brought into agreement with
  their already-correct net-weight targets; labels did not change.

Equal allocations rise **357/2,446 (14.60%) → 763/2,446 (31.19%)**; their parent
coverage rises **83 → 100**. Equal/mixed cases rise **125 → 237** and equal/multi-row
cases **31 → 227**. At least eight containers plus one shared prefix plus equal
allocations rise **0 → 30 across nine parents**. At least ten-container inventories
rise **133 → 139**, with parent coverage **3 → 5** (the two new parents are the
explicit source-preserving cohort). Overall mixed-equipment examples rise
**1,067 → 1,129**, and multiple-public-package-row examples **1,900 → 1,967**.

The production extension exposes an explicit equal-allocation preference, uses
the multi-group ledger with capacity/indivisibility checks, and does not impose
equality on ordinary synthesis. Scoped refresh now follows explicit private
cargo-owner dependencies, preventing stale repeated nouns/chemical identities.
Two campaign-construction issues (repeated rows mistaken for distinct groups;
mix probability without an eligible source scope) were caught before publication,
corrected and regression-tested. One gas-as-solid/drum proposal was excluded by
main-agent review; no substance-specific exception was added.

Validation includes exact native/scoped replay, full hybrid text review, schema,
literal values, arithmetic, equipment capacity, existing field absence, geometry,
sidecars and frozen hashes. **559 tests pass.** Prepared corpus readback took
17.27 seconds/758.64 MiB peak RSS; explicit equalization adds 12.36 µs in the
planner microbenchmark. Total recorded provider spend, including retries and
unselected attempts, is **$2.49876070**. This is not an accepted-only unit price.

Cargo page reflow can move unchanged neighboring text; it is not a promise of
unchanged centroids. All previously anchored unchanged lines retain anchors and
the generated layouts pass the existing bounds/clearance checks. Identifier-only
cohorts retain exact coordinate suffixes. These are still logical-line centroids,
not Paddle-region or bbox certification.

The longest new target is 3,129 tokens including EOS: within the unchanged 5,500
training limit, but above the current 3,072 evaluation-generation cap. These are
training replacements, not new validation records. Future evaluation of these
40-container inventories needs adequate output length; no target was truncated.

DG, thermal, forwarding, delivery and transshipment counts remain unchanged.
The complete before/after census records smaller cargo-frequency shifts as well.
Improved relation-model performance remains a hypothesis for the next training
comparison. Section 27 and its linked reports already preserve the Paddle audit
and probe results; this intermediary does not perform that migration.

## 29. R30 Paddle-centroid preparation — 2026-11-02

The [R30 preparation specification](../configs/training/preparation/r30_paddle_centroids.yaml)
sets **evaluation.generation_max_length: 5500** for both local and RunPod profiles,
matching the existing training target budget. It is deliberately not a runnable
training config: certified Paddle files, target views and their final hashes do
not yet exist. Preserve the override when materializing the final configs; do not
silently run the R29 GLM data under an R30/Paddle name. Normal EOS termination is
unchanged; long/nonterminating generations can cost more, so benchmark the final
Paddle inputs and decoding cap before launch.

The preparation override validates against both actual training-config schemas;
only the evaluation length differs from each inherited hardware profile. Three
new preparation guards plus the existing configuration suite pass **30 tests in
18.64 seconds**. The preparation file itself is rejected by the trainer, preventing
an accidental launch with unmigrated data. No generation-throughput or VRAM
measurement is claimed by these configuration-only checks.

The migration starts from **published R29**, including its 432 replacements and
their receipts—not an old R28 generation endpoint. R29 remains a frozen GLM-view
intermediary. The prior all660/456/9400 audit counts describe the R28 snapshot;
refresh its provenance inventory for the R29 changes before applying conversion.
The two identifier-only long layouts also need Paddle ownership review; they
were not independently admitted by the R29 refresh.

First agree/pin native Paddle page/region order, existing page markers and the
centroid serializer. Then complete a paired real-training pilot (20 representative
sources, favoring overlap with template parents) and the separate validation60
target view. Transfer the approved anonymization choices onto owned Paddle spans;
review newly visible private data and preserve the validation set's separate,
currently non-anonymized policy. For each field record retain/change/add/remove
with its exact evidence and policy. PDF review adjudicates ownership but does not
silently insert corrected/GLM text into a Paddle-only input.

After that pilot establishes the contract, scale the remaining real/source review
and rebase a 20-template/60-descendant full-document synthesis pilot. Implement
physical text fragments with full-precision boxes underneath both centroid and
bbox serializers. Existing donor-free sampling, cargo ledgers, aliases, multi-row
packages and DG/thermal capabilities stay in place. Source certification proceeds
in batches of 20; current-state descendant conversion in batches of 100 follows
accepted source bindings and resolved correction chains. Do not recreate current
samples from obsolete candidates or append unmatched private/source cargo text.

The prior 60 transfer controls were region-level probes, not fully migrated
documents. Full migration, rendered-image/re-OCR controls, throughput measurement
and Paddle-trained model parity remain unperformed. Package factorization remains
deferred; it should not be coupled to this OCR-input experiment.

## 30. R30 Paddle baseline completed — 2026-11-02

The [bounded pilot report](analysis/r30-paddle-pilot-20261102/REPORT.md) and
[certificate](analysis/r30-paddle-pilot-20261102/certification.json) now cover
20 anonymized real sources, all 60 validation sources, and 20 rebased Paddle
template overlays with three complete descendants each. No holds remain in this
cohort. Immutable pilot shards are published under the certificate, not as the
full R30 training dataset. All 37 protected R29/source pins remain unchanged.

Targets changed in 18/20 real and 56/60 validation records. The real inputs have
650 explicit privacy replacements, including newly visible native repetitions;
validation retains its original non-anonymized policy. Registered-office versus
factory ownership in `684f442c` is resolved without borrowing another party's
country or concatenating two addresses. Detailed decisions and final readback
hashes accompany the report.

The 60 descendants comprise 58 current-shipment paired replays and two fresh
donor-free thermal generations, not 60 fresh LLM draws. They exercise multi-row
packages, DG/thermal, long inventories, tank/NOR/vehicle and nested presentations.
Native repeated measurements/identities, literal phone projection, source-equal
ventilation and physical ownership were repaired and rerun through the combined
path. Both centroid and bbox views share full-precision physical boxes.

All 146 targeted and existing position/configuration tests pass. The full
60-document double replay takes 3.63 seconds after setup. On all 660 original
sources, centroid serialization is 9.1% faster than the audit prototype (median);
bbox serialization is 2.8% faster, with slightly lower incremental Python heap.
This is preprocessing evidence, not a model-quality or re-OCR result.

Next scope remains the other 580 real sources, remaining templates and full
synthetic conversion, followed by final input/token/VRAM checks and training
publication. The prepared 5,500-token generation limit remains required for both
hardware profiles. No training was launched and package factorization was not
introduced into this migration.

### R30 full-source rebase and newly exposed privacy — 2026-11-02

The full600-real/60-validation/456-template scope is pinned and unchanged; the
191 real sources without admitted templates remain source-only. The user's
Paddle privacy clarification applies to newly exposed input-only information as
well as new targets, while preserving goods, public information and localities.
See the [full-rebase working report](analysis/r30-full-rebase-20261102/REPORT.md).

Historically, the first additional20-source batch had788 private-span replacements, including
40 explicit extra entries across12 sources, and passes structural/preservation
checks. Five of its12 admitted templates have15 passing full shipment replays.
At that interim point it was not certified; it has since been published. Newly
visible repeated B/Ls, headers, private Marks and split contact names receive
explicit ownership; known private residues now fail synthetic replay too.
All60 prior pilot renderings remain exactly reproducible;45 focused tests pass.
R29, original PDFs/Paddle artifacts, and published pilot data remain unchanged.

### R30 rebase checkpoint after real-13 — 2026-11-03

At the user's request, work stops after completing the current batch, before
starting the next one. Published status is now **280/600 anonymized real sources,
60/60 validation sources and180/456 reviewed Paddle template overlays**, with
540 reviewed replay controls. The full dataset remains explicitly not trainable.
These template certificates cover native rendering/target projection over pinned
accepted shipment facts; standalone fresh-generation admission is still required.

The completed batch adds20 sources (38 pages,2,837 native regions),587 audited
private-span replacements,14 overlays and42 paired full shipment controls.
New B/L/party/net/date fields, repeated counts/masses, private source-only owners,
split contacts and literal target boundaries are bound and reviewed. All89 focused
tests and Ruff pass; all618 protected pins remain unchanged. The final checksum
audit verified1,951 pinned files. No9400-synthetic conversion or training occurred.

The [checkpoint report](analysis/r30-full-rebase-20261102/batches/real-13/REPORT.md)
records the per-template baseline and targeted continuation: one unified native
review packet, existing-fact reuse for exact matches, explicit exception queues,
new wording only for genuinely new linguistic owners, and complete three-shipment
replay with changed-span/target readback. Earlier real-12 templates remain openly
unpublished, including a source needing newly owned nested carton/piece facts.
The remaining source/template scope has not been removed.

### R30 match-first efficiency probe — 2026-11-03

The next20 real sources are reviewed, validated and published: **300/600 real,
60/60 validation,180/456 template overlays**. R30 remains not trainable. This batch
has35 pages/3,072 native regions and471 private-span replacements, including52 new
privacy decisions across15 records. Target leaf changes are51 added,57 changed
and13 removed; frozen R29 data are not rewritten.

The [efficiency report](analysis/r30-full-rebase-20261102/batches/real-14/REPORT.md)
distinguishes the historical84.93% GLM-centroid matching measurement from actual
Paddle rebase coverage. The new packet finds1,976/3,072 native regions with unique
literal matches and495/624 used template owners with unique-span proposals.
These are review hints, not automatic acceptance. Full native boxes, boundaries,
target paths and privacy context remain alongside the matches. A historical
backtest matched590/598 proposed spans to accepted spans; eight still needed
semantic/boundary correction.

Three layouts have nine passing, directly reviewed paired shipment controls;
the other11 drafts remain incomplete with explicit owner-review failures. The
14-template batch is **not certified or published**. Completed probe repairs are
retained for continuation; no fresh-generation readiness or full synthetic
migration is claimed. A newly exposed two-HS/two-mass presentation also needs
coherent independent product facts, not simply text matching.

All88 focused tests pass in18.80s and Ruff passes. Packet construction0.901s,
paired probe0.880s, source staging/audit/publication1.108/0.404/1.900s. These measure
local execution, not human review throughput; no end-to-end speedup has yet been
established. No inference/generation hot path changed and no new provider calls
were needed for this deterministic probe. Continue with consolidated exception
review and unchanged publication gates, not LLM-only diffing or auto-acceptance.

### R30 streamlined continuation through real-17 — 2026-11-03

The previously incomplete real-14 overlays and subsequent real-15/16/17 batches
are now published after direct review and three paired controls per template.
Current verified totals: **360/600 real, 60/60 validation, 238/456 overlays,
714 reviewed controls**. These remain native presentation/replay acceptance,
not standalone fresh-generation admission; R30 is still explicitly not trainable.

The [real-17 completion report](analysis/r30-full-rebase-20261102/batches/real-17/REPORT.md)
records newly visible contacts, route roles, numeric repetitions, literal
description continuations and privacy owners. A mixed-equipment tare remainder
uses the actual sampled ledger rather than copying a neighbouring tare. All73
focused tests and Ruff pass; measured common recipe overhead is0.110ms/20,000
calls. All618 protected inputs verify. No synthetic9400 conversion is claimed;
the remaining source/template membership stays in scope.

### R30 completed real-18 checkpoint — 2026-11-03

At the user's requested stopping point, real-18 is complete:20 source records,
11 native-template overlays and33 reviewed paired shipment controls are published.
Verified totals are **380/600 real,60/60 validation,249/456 overlays and747
reviewed controls**. Since the real-13 checkpoint this adds100 real records,
69 overlays and207 controls. No real-19 processing has begun.

The [full checkpoint breakdown](analysis/r30-full-rebase-20261102/batches/real-18/REPORT.md)
documents42 pages/4,065 native regions,618 private-span replacements, and22 added,
48 changed/11 removed target leaves across18 records. Corrections include fax vs
phone ownership, complete postal/handling continuations, new bill/date/reference
and repeated numeric owners, corrupted unchanged equipment aliases, narrow-cell
package rendering, parenthesis framing and preferred-country-name projection.
Native real text is retained except reviewed anonymization. All33 official replay
artifacts equal their manually reviewed probes before provenance pins are added.

All87 focused tests pass in16.12s. Official replay took5.183s, template publication
2.633s and process peak543.70MiB. Final verification checked2,028 pinned files in
5.508s; all618 protected inputs verify with only the already approved policy
subsection amendment. No shared hot-path code changed in this final batch.

R30 is still not trainable. Remaining:220 real records,148 associated templates,
12 earlier real-12 overlays and47 wider-only templates; then standalone fresh-
generation/aggregate admission. The9,400 synthetic migration is separate future
work. Overlay/replay certificates are not silently treated as fresh-generation
certificates or migrated training descendants. The match-first process saves
duplicate mapping work but still requires semantic/private-content readback;
no measured human-throughput multiplier is claimed.

### R30 reusable-workbench investigation: real-19 and real-20 — 2026-11-03

Both requested cohorts are complete and published:40 real records,27 native
template overlays and81 reviewed paired controls. Totals are420/600 real,
60/60 validation,276/456 overlays and828 reviewed controls including the pilot.
No Codex subagents were used. R30 remains not trainable; the9,400 synthetic
migration and standalone fresh-generation admission remain in scope but unfinished.

The [efficiency investigation](analysis/r30-full-rebase-20261102/EFFICIENCY-REAL19-20.md)
documents the reusable workbench, failed combined privacy/field-proposal approach,
separate fields/bindings proposals, declarative decisions, all-owner preflight,
exact source-span target projection, separate native-line geometry frames and
pinned field-shape readback. These retained all source/privacy/rendering gates.
The two cohorts contain995 private replacement operations and target changes in
38 records. Two fresh DG controls required scoped wording regeneration; their
numbers remained authoritative.

All92 focused tests pass.726 prior full-rebase controls reproduce exactly; median
30-control rendering is0.73537s before/0.73058s after (within noise), with no
meaningful runtime or memory regression.2,186 pinned files and618 protected inputs
verify. The requested15–30-minute end-to-end batch throughput is **not yet
demonstrated**: internal stage timings must not be presented as full batch times.
Remaining:180 real records and180 overlays (121 associated,12 unfinished real-12,
47 wider-only), then complete generation/aggregate admission.

### R30 continued source rebasing: real-21 and real-22 — 2026-11-03

Two further batches are published:40 real records,26 overlays and78 reviewed
paired controls. Current totals:460/600 real,60/60 validation,302/456 overlays,
906 controls including the pilot. Reports:
[real-21](analysis/r30-full-rebase-20261102/batches/real-21/REPORT.md),
[real-22](analysis/r30-full-rebase-20261102/batches/real-22/REPORT.md).

These cohorts cover81 pages,8,282 native regions and1,288 surgical privacy edits.
Newly visible fields and repetitions include complete allocations, gross/net
measurements, party/contact blocks, seals, IMOs, signatures and commercial refs.
Unreliable native OCR remains literal; PDF inspection establishes ownership, not
replacement text. Input-only references remain outside the reduced target schema.

A source-only target amendment corrects nine locality leaves in four real-21
records without changing inputs or historical certificates. A second finding
restores explicit frozen-schema checks to shortcut replay and template publication;
only three unpublished candidates were affected. All879 saved full-rebase controls
now validate.95 focused tests pass. The added offline schema gate measures about
0.87ms/control and94,034bytes incremental traced peak; no generation/inference
hot path changes. The country projection uses the pinned preferred-name registry,
not ISO display-name strings. R30 remains not trainable. Remaining:140 real and154
overlays; real-23 is prepared and under direct review, followed by full admission.

### R30 complete source/template baseline — 2026-11-03

The complete 600 anonymized real-training and 60 validation records are now
rebased and published, with all 456 existing templates (409 real-training
parents and 47 wider references). The 191 real sources without templates remain
input/label-only, as instructed. No membership was added or removed.

All 1,368 reviewed controls reproduce through the shared native renderer and
pinned catalog, with zero validation failures. They include 39 fresh controls
across 17 templates tested without old-dataset targets. Eleven overlapping fresh
controls also pass the standalone CLI, including the two newly exposed nested
packing ledgers and thermal/equipment variation. All 456 sampler prerequisites
pass in the combined configuration. Generated wording and final documents
still require semantic audit; passing controls do not auto-accept future draws.

The source aggregate changes targets in 553/600 real and 56/60 validation records;
these are OCR-conditioned projections and scope/privacy corrections, not model
errors. Additional private owners exposed by Paddle are anonymized and bound.
ICC ISBP821 establishes KOS as kilograms in the reviewed mass contexts; seven
existing source labels and four overlays were amended, while tare and unitless
volume are not silently relabeled as cargo measurements. The wider4d47f5e7
reference is also resolved without adding it to training membership.

Shared rendering is output-identical and measures 36.85s versus37.16s median
over1,368 controls. Full catalog validation, including I/O, schema/privacy gates
and fresh double-controls, takes142.96s with639.08MiB peak RSS. No R29 artifact
was changed. The 9,400 synthetic migration is still the next separate task;
R30 is not a complete training mixture yet.

See the [complete report](analysis/r30-full-rebase-20261102/REPORT.md),
[completion certificate](analysis/r30-full-rebase-20261102/completion.json), and
[synthesis handoff](analysis/r30-full-rebase-20261102/HANDOFF.md).

### R30 independent source/template audit — 2026-11-03

The complete 600-real/60-validation baseline and all 456 native templates were
independently re-audited. Corrections are published for 86 real and six validation
records, including ownership, literal projection, privacy and one cross-column
reflow defect. Ninety-three template overlays were corrected; one additional
template's retained control uses a narrowly regenerated goods description after
an older product response leaked party-field instructions.

All 1,368 controls passed the final audited catalog replay (156.98 s total,
83.32 s rendering/validation, 645.75 MiB peak RSS). The paired amended-template
render benchmark was 20.02 s versus 21.45 s baseline; no throughput regression
was observed. The relevant regression suite passed 178 tests; three additional
receipt-composition guards also pass. R29 and old batch evidence remain unchanged.

At this audit publication, the active catalog was
`artifacts/synthesis-templates/mpci-bl-v7-paddle-reviewed/r30/audit-20261103/catalog.json`,
SHA-256 `710994eb29e9293cbe4f395ad56aba46edabd29366a5dece2b07b948869a9713`.
The dataset manifest pins this revision and its complete controls. The 9,400
synthetic migration is the next distinct publication step, in batches of 100;
this audit does not yet make R30 a complete training mixture.

See [audit findings and exact repair scope](analysis/r30-independent-audit-20261103/REPORT.md)
and [publication receipt](analysis/r30-independent-audit-20261103/publication.json).

### R30 synthetic migration checkpoint — 2026-11-03

The first 100/9,400 synthetic records are certified and published alongside the
600 real and 60 validation records. **9,300 synthetic records remain uncertified;
R30 is not trainable yet.** OpenRouter returned HTTP 402 (insufficient credits)
during semantic review; further provider work requires restored credit.

Migration readback additionally corrected two real targets (one previously
repaired), bringing the independent-audit total to 93 distinct sources: 87 real
and six validation. Native repetitions now consume semantic shipment authorities
instead of assuming every historical receipt has newer compound-owner keys.
Existing sampled equipment tare is retained even when its unchanged value produced
no text edit. All 456 templates / 1,368 controls pass final replay; 147.26 s total,
81.99 s rendering, 647.90 MiB peak RSS. The active catalog is pinned by the manifest
under `r30/migration-repetitions-20261103`; older certificates retain their own pins.

The next three 100-record batches mechanically render 99 each. Their remaining
three `5c2fdc31` cases need coherent three-level packing for the retained cargo,
or a narrowly scoped cargo refresh—not invented counts or omitted hierarchy.
Separate historical receipt composition and additional-owner wording remain for
later batches. Exact current-text transport receipts are not dataset acceptance.

See the [migration report and resume sequence](analysis/r30-synthetic-migration-20261103/REPORT.md).

### R30 resumed migration and numeric-layout correction — historical 2,800-record checkpoint

Current publication: 600 anonymized real, 60 validation, 2,800 synthetic records
(batches 001–028). **6,600 synthetic identities remain unpublished; R30 is not
ready for training.** A new OpenRouter HTTP 402 stopped paid semantic reviews.

The user's goods-origin boundary is documented and applied: unambiguous attached
country qualifiers can establish origin without an ORIGIN heading; genuinely
ambiguous brand/market qualifiers remain description-only when inside the
policy-owned passage. Detached text is not appended to descriptions.

A complete numeric-fragment screen found 105 published documents with scalar
numbers split across physical lines. The scoped repair and repeated screen now
give zero such documents. An incorrectly bound demurrage tariff was restored in
39 descendants, without changing cargo volume targets. Two source-target repairs
were propagated to six synthetic targets. Full relevant tests: 390 passed;
full template control replay: 1,368/1,368, followed by six passing controls for
two additional exact numeric/retained-owner corrections. All 400 records in
batches 029–032 mechanically pass, but remain unpublished pending exact-current
semantic review. The working migration report records
the exact changes, rejected intermediate candidates, timing/memory evidence,
publication pins and remaining provider boundary. These checks are not evidence
that the unfinished synthetic dataset is fully accepted.

### R30 continuation — 4,000 published synthetic records, 2026-11-04

Publication has advanced through complete batches 001–040: **4,000 synthetic,
600 anonymized real-training and 60 validation records**. The remaining 5,400
synthetic identities are not certified. R30 remains **not ready for training**.

Fresh full remaining-set staging renders 4,843 of those 5,400 records; 557 are
explicitly held. The batch-100 renderer scan takes about 3.5 minutes for the
whole pending set, but semantic review and missing-data/ownership repairs are
separate requirements. The active-publication audit passes for all 4,660
published records, and no fragmented numeric scalar remains among the 4,000
published synthetic records.

The current catalog includes reviewed retained-owner and compound-party
boundaries, exact physical-measure rounding checks, marked consignee email
continuations, complete repeated private signatures and explicit equipment
authority for new tare occurrences. Full control replay passed 1,368/1,368
unchanged after the shared rounding correction; later source-specific edits
passed their scoped controls and published-record readbacks. The final broader
regression suite passed 603 tests in 126.04 s. Existing semantic reviews are reused only
when the entire reviewed input/target and request are exactly unchanged.

The synthesis account again reports exhausted credits, confirmed by a read-only
check using the same credential loader at 21:37 UTC. Paid calls stopped; local
repairs and validation continued. This is separate from the remaining local
binding/private-data/numeric holds, which are explicitly inventoried, not
attributed to billing. See the [current migration report](analysis/r30-synthetic-migration-20261103/REPORT.md)
for exact correction scopes, rejected proposals, costs, timings, publication
pins and resume requirements. Rendered candidates and repaired receipts are not
counted as certified records.

### R30 Paddle migration completed — 2026-11-05

The historical incomplete checkpoints above are superseded. **All 9,400 synthetic
records, 600 anonymized real-training records and 60 validation records are now
certified and published**, with all 456 native templates covered by current
controls. The frozen membership was preserved. This completes the data rebase,
not a training run or a demonstration of Paddle model parity.

The final independent audit verifies all 10,060 records, exact active
certificate/aggregate equality, schemas, native order and raw/centroid/bbox views.
A fresh replay of all 9,400 synthetic records through the current native path
has zero differences (169.78 seconds, no provider calls). The whole-dataset
publication audit takes 106.15 seconds and 191.48 MiB peak RSS. The regression
suite passes 810 tests; every exact current template is covered by three passing
controls. Completion evidence is pinned in the R30 dataset manifest.

The remaining 5,400 records after the 4,000 checkpoint required **$7.22183221
additional accounted provider cost**, including $0.0423225 retained as an unknown
request-cost reserve. This is not the total historical migration cost and not
a fresh-generation price forecast. Readiness-first paid review, exact evidence
reuse, scoped generation and individual adjudication replaced front-loaded
review of mechanically unfinished records. Final batch certification took
51.29 seconds without provider calls; manual exception review is separate.

See the [completed migration report](analysis/r30-synthetic-migration-20261103/REPORT.md)
and [completion receipt](analysis/r30-synthetic-migration-20261103/completion595.json)
for final counts, cost/attempt ledgers, correction scopes, benchmarks and pins.
R28/R29 remain unchanged. Next: materialize R30 training configs with the current
dataset/task pins, retain the 5,500-token evaluation generation limit, and run
the final input/target-length and VRAM preflight before training.

### R30 independent training-readiness review — 2026-11-06

The current dataset and native registry are **approved for the first controlled
Paddle-centroid training experiment**, subject to materializing the training
configs and measuring GPU memory/throughput. This is not a guarantee of model
parity. No dataset/template correction was warranted by this review; no provider
calls or Codex subagents were used.

Fresh checks cover all 10,060 records, the pinned trainer tokenizer and exact
raw/centroid/bbox serialization. Every input/target fits 19,200/5,500 tokens.
Template replay passes 1,368/1,368 controls across all 456 templates, unchanged;
816 Paddle/R30 tests pass. Real and synthetic median line heights are both 9.192
on the 0–1000 grid, with closely matched text-density distributions and no
quantized-centroid collisions. Inputs are approximately 40–46% longer than R29.

Synthetic equal-load coverage is 860/2,664 complete labeled multi-container cases
(32.3%), across 106 parents; 56 examples combine eight-plus containers, one ID
prefix and equal quantities. The report also records limitations without hiding
them: long inventories remain concentrated in five parents, mixed equipment
decreased 1,129→1,068 during coupled migration resampling, a narrow goods-slot
family wraps awkwardly, and synthetic descriptions remain longer than real ones.
Screened arithmetic mismatches trace to approved literal OCR conflicts or missing
valid container identifiers, not a reason to balance targets by invention.

**56/60 validation targets changed with the input evidence.** Next-run reporting
must distinguish the full R30-native task from comparisons on unchanged evidence;
raw aggregate scores alone are not a clean R28-vs-R30 model comparison. The R28
per-record hash-field setting also cannot be copied blindly: current R30
real/validation records lack `positionedTextSha256`.

See the [independent readiness report](analysis/r30-training-readiness-20261106/REPORT.md)
for exact pins, examples, plots, measured audit costs, limitations and launch
requirements. The next useful evidence is controlled training, not another
unrestricted paid review sweep.

### R30 runnable config and full-coordinate follow-up — 2026-11-06

Both local and RunPod R30 training configs are now materialized with current
dataset/constraint hashes and fresh run IDs. Local retains R28 settings; RunPod
now uses the requested **effective batch 48** (microbatch 4, accumulation 12):
209 updates/epoch, 2,090 total, warmup 105, eval/save every 522 updates and logging
every 3 updates. Rank, learning rate, epochs and other optimizer settings remain
unchanged. Generation is capped at **3,329**, replacing the planned
5,500: the longest current label is 3,200 tokens including EOS, plus one decoder
start and 128 spare tokens. Training target-length budget remains unchanged.

The actual frozen-vocabulary build caught a previously missed registration for
the reviewed real-only `PACKAGE_PALLET_PLASTIC` category. R30's registry now
includes it as an extraction-only semantic extension, with no invented MPCI
code and no target changes. The bound trainer inspection passes all 10,060 rows.

Coordinate coverage is **100%**: 104,694 real-training regions, 1,662,608 synthetic
regions and 9,584 validation regions all have valid boxes and exact centroid/bbox
views. Headers/separators are not OCR regions. Sixty-six focused tests pass; no
training or GPU-memory benchmark was launched. See the follow-up section of the
[readiness report](analysis/r30-training-readiness-20261106/REPORT.md).

## 31. Q/R30: completed Paddle-only RunPod run

Audited2026-11-08. [Detailed report and evidence](analysis/r30-training-results-20261108/REPORT.md).

### 31.1 Completion, artifacts and experiment import

The complete RunPod run is downloaded to its normal `artifacts/kie-training/`
directory: **87 files,2.15GB, all SHA-256 verified**, including every full checkpoint
(522,1044,1566,2088,2090), optimizer/RNG/scheduler state, predictions, logs,
tokenizer, frozen configurations and selected adapter. All85 internal-manifest
entries also match. The selected export's504 tensors match1044 and its two
evaluation prediction files are byte-identical;26 quality/loss/generation metrics
match exactly. CPU replay reproduces all five checkpoints' structured metrics.

Both RunPod MLflow SQLite stores were snapshot-backed-up, and the completed run
was imported into the existing local `document-ocr-kie` experiment3. All279,607
metric records retain their timestamps/steps, and all312 parameters and source
tags were verified. A pre-import local database backup is retained. Existing
experiments were not replaced. The entire87-file run is also attached to MLflow
under `training-run/`, with stored hashes checked. The source had no registered
models to import; this is a complete experiment/history/artifact import.

[Local MLflow result](http://127.0.0.1:5000/#/experiments/3/runs/73ff4e9efee948bda3bc46e37bf1136f).
Raw tracking snapshots and receipts are under
`artifacts/kie-training/runpod-downloads/20261108-r30-complete/`.

The final manifest and tracking status confirm completion. An earlier pending
manifest status file was not rewritten. EVA completed239 forwards in771.74s;
logged training runtime is6h47m06s including scheduled evaluations, with EVA and
the separate selected-model evaluation additional. No new training or GPU
evaluation was necessary for this audit.

### 31.2 Best versus final and previous preferred checkpoints

|Checkpoint|Field F1|Field accuracy|Relation F1|Relation accuracy|Category F1|JSON/schema|Whole exact|
|---|---:|---:|---:|---:|---:|---:|---:|
|R27 rank64 preferred2350|0.899093|0.851755|0.970655|0.942982|0.894831|56/48|4|
|R28 selected/final3130|0.902930|0.862200|0.928736|0.866953|0.930582|58/52|8|
|R30 selected1044|0.902204|0.865176|0.948936|0.902834|0.912106|58/55|6|
|R30 terminal2090|0.895197|0.851211|0.889344|0.800738|0.913793|58/54|9|

Counts are out of60 and scores use each run's own gold. Best1044 has223/238
correct relation facts, nine extra and15 missed; terminal has217correct,33extra
and21missed. Best has55 exact relation graphs versus52 at terminal. Terminal
nevertheless improves exact descriptions40→45 and HS F10.941176→0.969697.

**Recommend1044 for unmodified R30 inference.** Later training is not uniformly
better, and2090's slightly higher category/extraction-fact F1 does not offset its
native relation deficit. Neither export nor historical result was changed.

### 31.3 Control changed gold and parseability

Paddle inputs change the evidence, and **56/60 targets differ** from R28. Scalar
gold facts grow2,567→2,777 and relations220→238. Effective batch also changes32→48,
and the data includes the R29 refresh. This is not a pure OCR or hardware ablation.

Exclude changed aligned field families per document in both models:

|Comparison|Unchanged field F1|Unchanged field accuracy|
|---|---:|---:|
|R28→R30 best, all60|0.918372→0.917309|0.880663→0.885404|
|R28→R30 best, same56 parseable|0.936953→0.925797|0.915312→0.901402|
|R27→R30 best, all60|0.920615→0.918231|0.880032→0.886785|
|R27→R30 best, same54 parseable|0.945423→0.930531|0.926261→0.906466|

Thus overall field performance is near the previous range, but common-parseable
semantic extraction regresses. Names, addresses and localities account for clear
losses; they are not all harmless whitespace. Phones, container identifiers,
allocations, goods descriptions, HS and package categories improve in the
all60 unchanged-field comparison.

On55 documents with identical relation gold, R27/R28/R30-best/terminal relation
F1 is **0.980198 /0.934343 /0.975369 /0.900943**. R30 best recovers198/202facts,
the same number as R27 and13 more than R28, with two more extras than R27.
Restricting R28/R30 to the same51 parseable identical-relation-gold documents
still improves0.958549→0.974359. Common-gold category F1 also improves over R28,
0.942801→0.957198 on54documents, despite the lower native category score.

### 31.4 Error mechanisms and bounded follow-ups

- **Repeated document content:** `cda61263` repeats the same two container rows in
  its input. Best is exact; terminal outputs six inventory rows and24 allocations.
  Those duplicates produce22/33 terminal extra relations. Exact-row deduplication
  on diagnostic copies yields terminal field F1**0.905077**, relations**0.931330**,
  schema55/60. These are not native scores or a deployed repair; even after this
  correction, relations remain below1044.
- **Long inventories:** `5cabcd4b`'s ten equal/shared-prefix containers are all
  recovered by best but only six by terminal. `13ea92fa` loses four of seven in
  both models. The complete evidence fits the input budget.
- **Quantity restraint:** `891555b9` lacks per-container quantities, but both
  models populate them from shipment-total/commercial numbers such as3,700 and66.
- **Mixed package aggregation:** `5d6b477c` requires shipment88pallets+1,200boxes
  and scalar allocations620,620,24,24. Terminal copies group20/600 into shipment
  totals and misassigns quantities. More public package rows alone do not solve
  this numerical-role problem.
- **Paddle literal boundaries:** company/address shifts and extra locality words
  coexist with punctuation-spacing errors. Only7/41 wrong-valued best-checkpoint
  addresses are whitespace-only; metric normalization is not a complete remedy.
- **Malformed/schema-invalid outputs:** best has one no-container repetition loop
  reaching the3,329 cap; terminal's relation failures occur with EOS reached on
  all60 and maximum1,298tokens. No-container structures, invented category names,
  HS fragments and inventory/reference inconsistencies remain review priorities.

The same57 documents parseable at best/terminal still regress in relation F1
0.948498→0.893004. This is not merely malformed JSON. Validation includes only
one DG declaration and six temperature examples, so broad DG/thermal conclusions
remain unsupported even though those structured values are correct in R30.

The next bounded work should investigate those error families and evaluate a
separate duplicate-output guard if desired, without rewriting validation to fit
predictions or importing its sources into training. Paddle-only has established
a viable performance range, not universal superiority or field accuracy>0.90.
Full per-field, per-source, category and controlled comparison tables accompany
the detailed report. Three new comparison-guard tests pass; the full CPU audit
takes12.08s/~352MiB. No datasets, training settings or model artifacts were changed.

### 31.5 Private inference release — 2026-11-08

Selected checkpoint1044 is published privately as
[Samandarine/t5gemma2-270m-mpci-bl-r30-lora](https://huggingface.co/Samandarine/t5gemma2-270m-mpci-bl-r30-lora),
revision `c10daac2995fbaacbfb28c5a0c6415b7de91cba5`. The15-file inference bundle
follows R28's structure, updated for Paddle-native centroid inputs,49 package
categories and the3,329 generation cap. It excludes base weights, datasets,
predictions and optimizer state. Private metadata, anonymous401 denial and all
fresh-download hashes pass; the downloaded adapter reproduces the source's272
generated token IDs in the bounded offline smoke test. Full evidence is in the
[release report](analysis/r30-huggingface-release-20261108/REPORT.md).

### 31.6 Epoch-five boundary audit, fresh synthesis and rank96 — 2026-11-08

Detailed [audit and next-run report](analysis/r30-party-boundary-audit-20261108/REPORT.md)
contains the complete87-field discrepancy taxonomy, full10,060-record mechanical
census, PDF checks and reproducible probes. No review-agent sweep was used.

- **Party regression is not principally target newlines:** all64,686 party
  name/address values have no embedded newlines/tabs, coordinate markers or
  detected prompt leakage. All14 lexical-support screens resolve to linked word
  fragments/interleaving or Unicode dotted-I equivalence. The model has24 boundary
  discrepancies across12 role blocks/eight documents,14 spelling/value rewrites,
  10 wrong-role/owner errors, plus formatting, omissions and JSON failures.
- **Bounded data findings remain:** the same UNIFREIGHT Alexandria-branch qualifier
  is assigned inconsistently in two validation documents; one old synthetic
  shipment has locality-like party names; a stale guidance paragraph conflicts
  with the current no-added-commas address policy. Frozen labels/scores unchanged.
- **Fresh synthesis found an integration gap:** seven of eight new shipments
  pass native rendering/replay; the eighth (`00a165a2`, multi-package rows) lacks
  the compound package-expression binding, although separate count/type bindings
  exist. It fails closed. Static screening identifies62 parents with missing
  exact compound owners (four are deliberately single-row-only); all695 existing
  descendants are single-row, so successful historical replay did not test this
  mode. Review/complete the58 remaining parents' mode coverage before claiming
  unrestricted fresh synthesis. No dataset repair or silent single-row fallback.
- **Epoch shuffling already occurs:** actual length-grouped Trainer/Accelerate
  loader covers all10,000 records exactly once and changes order each epoch;
  evaluation is stable by design. No runtime extension needed.
- **Rank96 configs ready:** same frozen data, alpha32/rsLoRA/EVA and respective
  schedules. Local effective batch32; RunPod48. LoRA parameters30.376M→45.564M.
  All4 new rank96 tests pass inside the trainer image; full dataset inspection
  passes. No training launched or GPU performance/memory claim made.

Follow-up priorities are bounded: adjudicate branch-qualifier ownership before
any future gold revision; repair native compound-expression mode coverage;
compare rank96 with1044 and2090 on unchanged R30 before attributing the boundary
regression to insufficient capacity. Public package factorization remains separate.

### 31.7 Package-mode repair and scoped party amendments — 2026-11-08

The [detailed repair report](analysis/r30-posttraining-repairs-20261108/REPORT.md)
supersedes the outstanding repair items in 31.6, not its historical model scores.

- **Compound binding gap repaired:** 95 complete declarations in 58 templates,
  plus one additional dependent-repetition template; 59 template files changed.
  All456 mode admissions pass. The four intentionally single-row-only layouts
  remain explicit restrictions, not silent fallbacks.
- **Related modes covered:** scalar extras inside compound owners, unchanged
  aliases, dependent counts/types, the CHEREAU vehicle-count owner and all nested
  group repetitions. 134 fresh mechanical controls pass; every affected
  multi-row-capable template has both single- and multi-row generation coverage.
  Normal content review remains required; questionable fresh wording was rejected
  and resampled rather than admitted to training.
- **Historical records preserved:** 1,088 paired outputs are identical, including
  all695 originally affected descendants. No cargo dataset rewrite was necessary.
- **Seven party records fixed:** three real, two validation, two synthetic;
  own-branch qualifiers consistently belong to names. One old synthetic naming
  outlier was re-rendered in its owned fields. Punctuation/physical-line guidance
  and generation instructions now agree. Fifteen target leaves changed.
- **Published readback:** all10,060 records and1,776,886 regions pass;10,053 rows
  are byte-exact unchanged. Three previously adjudicated source-preserving OCR
  accounting conflicts remain literal and unchanged, with receipt verification.
- **Reproducibility:** original splits retained under the November8 audit-history
  directory. Rank64 configs use those bytes; rank96 configs use corrected R30.
  Hyperparameters/schedules/schema are unchanged. Compare on common gold or
  explicitly rescore; do not claim a perfectly unchanged-data rank ablation.
- **Cost/performance:** $0.19601896 all-attempt provider spend. Paired rendering
  69.15s before /68.35s after; no observed slowdown. Complete readback48.72s.
- **Final regression:**4,188 synthesis/render/package tests pass in196.21s;
  17 host config/audit tests pass, with the host-Torch skip covered by a direct
  sampler assertion in the pinned trainer image. A stale non-runtime ontology
  input pin was reverified/repaired; no public factorization was activated.

Current synthesis catalog is the dataset manifest's `package-modes-20261108`
pin. Updated commands are in the native synthesis handoff. No training or model
publication was performed in this repair pass.

## 32. R/R30 rank96: completed RunPod run

2026-11-08. Full evidence: [rank96 report](analysis/r30-rank96-results-20261108/REPORT.md),
[machine-readable comparisons](analysis/r30-rank96-results-20261108/summary.json),
[field groups](analysis/r30-rank96-results-20261108/groups.csv),
[document scores](analysis/r30-rank96-results-20261108/documents.csv),
[schema failures](analysis/r30-rank96-results-20261108/schema-errors.json).

### 32.1 Archival, import and evaluation integrity

- 87 run files, **3,121,662,754 bytes**, remote/local SHA-256 identical;85 manifest
entries verified. Full522/1044/1566/2088/2090 checkpoints, training state, all saved
predictions, tokenizer/configs/logs and selected adapter retained locally.
- Both remote MLflow DBs backed up consistently; the separate root DB is empty.
Runtime code/lock/package/git/GPU/filesystem provenance also retained. Archive:
`artifacts/kie-training/runpod-downloads/20261108-r30-r96-complete/`.
- Local MLflow experiment3 run **375f73658c1a4f0cad030423a909bde2** contains315,032
exact metric-history rows,312 parameters, original tags/times and all87 run
artifacts under `training-run/`. Existing61 runs unchanged; backend backed up first.
[Open MLflow](http://127.0.0.1:5000/#/experiments/3/runs/375f73658c1a4f0cad030423a909bde2).
- CPU replay reproduces all ten scheduled rank64/rank96 evaluations. The rank96
post-training selected evaluation repeats1566 byte-for-byte;504 export tensors
match1566. Terminal metrics come from2090's own receipt, not the selected export.
- Same recorded training-source hashes, libraries, GPU class, effective batch48,
ten epochs and schedule. Rank increases64→96, parameters30.38M→45.56M, rsLoRA
scaling4→3.266. Five training targets, one synthetic input and two validation
party-boundary targets had been amended; validation inputs/cargo gold are unchanged.

### 32.2 Best and terminal outcomes

|Metric|Q/r64 best1044|Q/r64 final2090|R/r96 best1566|R/r96 final2090|
|---|---:|---:|---:|---:|
|Native field F1|0.902204|0.895197|**0.910305**|0.903769|
|Native field accuracy|0.865176|0.851211|**0.879665**|0.866059|
|Relation F1|**0.948936**|0.889344|0.937500|0.937500|
|Relation accuracy|**0.902834**|0.800738|0.882353|0.882353|
|Category F1|0.912106|0.913793|0.925865|**0.931092**|
|Extraction-fact F1|0.897829|0.899719|**0.907331**|0.900521|
|JSON /60|58|58|**60**|58|
|Schema /60|**55**|54|53|51|
|Canonical exact /60|6|**9**|6|7|

**Prefer1566 for this run.** It improves usable field extraction, but does not
dominate rank64 on all tasks. Rank64 remains the agreed default unless separately
changed; neither config nor public/private release was changed by this audit.

### 32.3 What changed after controlling the comparison

Common corrected gold: rank64 best field F1 **0.902568→0.910305**, accuracy
**0.865526→0.879665**. Correct fields2,478→2,522. **41 of44 net gained correct
fields come from two formerly malformed outputs** (`61d1d525`, `ff7f0853`). On
the same58 parseable documents, F1 improves only **0.910360→0.911797**. Both
recovered outputs still contain schema/semantic problems. Best-to-best paired
bootstrap95% interval for field-F1 delta is−0.00631…+0.02406, not a statistically
secure population-level win on this small repeatedly used validation set.

Rank96 terminal newly breaks JSON in `45ed05eb` and `48c177a3`, losing68 correct
fields; it gains16 net on the other58. Both failures reach EOS well below the
generation cap. On shared58 parseable documents, best→terminal field F1 actually
improves0.910406→0.916512. This distinguishes serialization reliability from
semantic extraction; it does not justify ignoring malformed documents.

### 32.4 Relations: the main residual mechanism

Best rank64 has223/232/238 correct/predicted/gold relation facts; rank96 has
225/242/238. Recall improves, but precision drops. On **54 actual-container
documents**, relation F1 is **0.948936→0.949367** (rank96 terminal0.953390).
On six no-container documents, rank64 best emits no false relations; rank96 best
emits six, terminal eight. These invented relationships explain the aggregate
regression more specifically than a broad loss of allocation ability.

- `da651f2f`: party text MAHER becomes a container; volume is also mistaken for mass.
- `e5a37f5a`: vehicle VIN becomes carrying equipment; invalid PACKAGE_CARGO remains.
- `ff7f0853`: FIOSORIGINAL becomes a container; description/unit fidelity also wrong.
- `13ea92fa`: seven-container inventory improves3→6 retained, still incomplete.
- `5cabcd4b`: ten shared-prefix containers: rank64 best10/final6, rank96 both9.
- `cda61263`: the old final's six inventory rows/24 allocations become the correct
two/two. Exact-duplicate diagnostic alone raises old final relation F1 to0.931330,
explaining most, but not all, of rank96's final-to-final gain to0.937500.
- `5d6b477c`: mixed shipment totals/per-container package groups still conflated.
- `891555b9`: six absent allocation quantities are still invented.

No exact duplicate container/allocation rows occur in either rank96 headline
output. Package-row repetition in the mixed-accounting document is a different
problem, not silently fixed by that diagnostic.

### 32.5 Other fields, limits and next priorities

Common-gold best-to-best: party names0.8454→0.8542, addresses0.7520→0.7619,
countries0.9373→0.9487, descriptions0.6780→0.7167, HS0.9412→0.9703. Phones regress
0.8976→0.8933. Name/address boundaries and secondary-notify omissions remain;
larger rank did not eliminate them. Package categories improve, with RACK recovered,
but steel-drum specificity, IBC, vehicle/unpacked distinctions and mixed accounting
remain weak. Handling correct values3→5 of11. The sole DG case loses packing
groupIII while retaining UN/class; best retains six temperature settings. This
validation set is too small for broad DG/thermal conclusions.

Priorities: no-container/vehicle-ID negative cases; literal inventory completeness;
missing allocation evidence; mixed group-versus-shipment accounting; party/contact
ownership; and independently measured JSON/schema reliability. Broader held-out
coverage is needed before another generalization claim. No dataset changes or
generation campaign were performed during this result audit.

EVA239→401 forwards and771.74→2,102.98s; reported trainer runtime24,426.28→22,464.40s.
These are observed different-pod timings, not a controlled performance benchmark.
Audit42.73s, peak RSS3,265,436KiB; six targeted tests and ten metric-guard families pass.

### 32.6 Literal text fidelity and boundary follow-up

Recommend rank96 checkpoint1566 overall, retaining rank64 checkpoint1044's stronger
all-document relations/schema validity as an explicit trade-off. No serving default
or model release changed. See [detailed fidelity analysis and primary research](analysis/r30-rank96-results-20261108/REPORT.md#11-checkpoint-recommendation-and-text-fidelity-research-follow-up).

Saved-output review separates altered spelling, plausible rewriting, numeric edits,
repetition/omission, incorrect spans and party-boundary errors. Rank96 best has84
wrong values and8 missing among446 party-name/address/description fields, plus5
extra;10 of84 mismatches are only spacing/punctuation. All60 input strings and446
target strings round-trip through the exported tokenizer without loss/unknown
tokens. Greedy decoding and non-truncated rank96 generations exclude random
sampling/output-cap explanations. The uppercase target policy changes subword
sequences in some cases; it is intentional, not a newly found corruption, and
its contribution to errors is untested.

Research supports scoped source-constrained decoding, span extraction and targeted
copy/boundary training experiments; a close Flan-T5 study also finds that adding
a pointer-generator can hurt, so no architecture change is presumed beneficial.
Next diagnostic: gold-prefix/free-prefix controls and independent held-out fidelity
cases. Separate JSON validity from lexical fidelity and semantic role correctness.
No dataset, training, model or inference changes in this follow-up; local audit
1.25s/377.1MiB, no paid model calls.

### 32.7 Training tweaks: casing and party-boundary coverage

[Detailed training-side follow-up](analysis/r30-rank96-results-20261108/REPORT.md#12-training-side-follow-up-casing-learning-rate-and-party-boundaries)
separates representation, data coverage and optimizer hypotheses. No training or
dataset changes made. All10,060 inputs screened; all189 validation name/address
presentations reviewed using saved predictions and the config-pinned dataset.

Rank96 best1566 gets both fields correct for6/14 shared-line party roles versus
131/175 separate-line roles (42.9% vs74.9%). Repeated consignee/notify details and
role/layout mix qualify this small diagnostic panel; deduplicating within documents
gives6/11 versus115/153. Shared-line outputs all parse, and punctuation relaxation
does not rescue them. Rank64 best was8/14, so higher rank has not solved the pattern.

Training screen finds71 real and991 synthetic shared-line roles, the latter in689
documents/32 parents, after rejecting one suffix-collision false positive. These
are not exhaustive semantic census counts. Of991,850 put the whole company name
on the boundary line; only141 are name continuations and76 of those end in AGENCY.
Validation has13/14 continuations. Prioritize varied semantic name endings and
realistic split/merge layouts, not merely more company-plus-street rows. Keep
clear-boundary controls and independent identities; no validation leakage.

Source-case targets are a reasonable separate experiment. In matched-witness
validation names, rank96 is correct144/161 where case already matches, versus10/17
where transformation is needed. Addresses show no analogous disadvantage. This
is confounded observational evidence, not a demonstrated casing effect. A trial
must retain spans/punctuation/line joining and canonical non-literal fields,
project actual source case rather than guessing it, and keep comparable metrics.

Current encoder/decoder adapters share1e-4 LR. Lower global LR and lower encoder
LR are testable but no encoder-overadaptation cause was established. BERTSUM's
pretrained-encoder/random-decoder precedent differs from our both-pretrained LoRA
setup. Suggested independent optimization controls:5e-5 globally, or2.5e-5 encoder
with1e-4 decoder, keeping other settings fixed; not asserted optimal values. The
latter needs tested optimizer-group support, not an existing YAML switch.

Eight audit tests pass; full screen32.78s/352MiB, no paid model calls. Serving
constraints, architecture changes and production implementation are outside this
training-focused diagnosis.

## 33. R31 Stage A: source-casing migration; Stage B boundary plan — 2026-11-09

[Full migration and boundary-coverage report](analysis/r31-casing-migration-20261109/REPORT.md).

Published `data/curated/mpci-bl-r31-paddle-centroids-v1`: same600 real +9400
synthetic training records, same60 validation records. Only party names, party
addresses and goods descriptions now preserve their final anonymized Paddle
casing. No spelling, punctuation, field boundary, input, coordinate or other
target changes. R30 remains unchanged.31,000 of74,727 selected fields change case,
in182 real,5251 synthetic and25 validation documents. All fields resolved;101
batches of at most100 records, full strict-schema and independent readback checks.

The explicitly versioned R31 native catalog retains456 templates. All pins and
ownership contracts loaded;11 overlays clarify primary occurrences. Twenty paired
rendering controls, including all11, preserve input/geometry/proof exactly and
change only permitted casing.141 related tests passed. Full publication258.32s/
518.45MiB; independent readback plus paired replay41.29s/351.91MiB. After profiling
and lazy case indexing, warm renderer medians42.44ms→42.57ms (not an overall
synthesis speedup claim). No paid calls, subagents or training launch.

Training target tokens fall1.70%; maximum3200→3182 including EOS. Current3329
generation cap remains sufficient. Wire schema/frozen hashes stay unchanged;
the dated R31 policy overrides historical uppercase annotations for these fields.
Current training uses input-only prompts, not those schema descriptions.

Stage B remains **planned, not generated**. New readback rejects one additional
training-screen role collision:49feedab notify matched the consignee's shared
line. Real continuation candidates43→42 (2.19% of1917 paired roles); synthetic
141/29753 (0.47%); combined183/31670 (0.58%); reviewed validation13/189 (6.88%).
These training rates remain screened estimates, not a complete semantic census.
There are22 admitted real-source continuation parents and48 admitted parents
with any training shared-line evidence; current synthetic continuations occupy
only110 documents/six parents. Actual rendering can turn a source's shared
continuation into separate lines, so parent choice alone does not supply coverage.

Proposed next step: a24–30-parent ×3 pilot, then about750 **new** samples across
30–40 diverse parents if the realized controls pass. Aim for600 difficult
documents plus150 easier/clear-boundary controls, checking at least600 actual
continuation-role examples and separately counting distinct repeated blocks.
500 can suffice with adequate measured yield;1000 is not required merely to
reach real-training prevalence. Keep validation fixed, preserve cargo/role
coverage, and recalculate schedules/token caps only when Stage B size is settled.
No Stage B presentation changes, refresh, additional generation or R31 launch
config have been made; Stage A remains an independently testable snapshot.

## 34. R31 Stage B complete: 750 boundary-focused descendants — 2026-11-09

[Full extension report](analysis/r31-boundary-extension-20261109/REPORT.md) and
[final replay certificate](analysis/r31-boundary-extension-20261109/final-certification.json).
This supersedes section33's Stage-B planning status, not its frozen Stage-A evidence.

Published seven100-record batches plus50: **10,750 training records =600 real
+10,150 synthetic**, validation60 unchanged. The preceding10,000 training payloads
are unchanged; a separately certified amendment repairs only two stale text-hash
metadata fields in9399 older synthetic records. Real/validation records remain
byte-identical to Stage A. R30 data and the frozen Stage-A snapshot are unchanged.
All750 new records preserve actual printed casing for party names, addresses and
goods descriptions. No new training config or run was launched.

Measured after rendering, not inferred from sources:442 name-continuation/shared
address documents,207 whole-name/shared-line documents,101 clear/unresolved
controls; **649 difficult documents**. Forty parents overall,33 with difficult
outputs and17 with continuations.799 continuation-role examples include repeated
consignee/notify roles; they are not799 independent shipments. Only52/799 end in
AGENCY, versus76/141 in the previous synthetic screen. There are2,233 distinct
party names in2,873 role occurrences. Fourteen native overlays preserve the source's
name fragments so generation no longer unnecessarily removes the difficult boundary.

Other coverage:193 multi-container,67 mixed-size (not mixed-type-category),363
multi-package-row,132 thermal,65 DG including22 multi-DG,145 forwarder and506
delivery-agent records.47 package categories and635 distinct six-digit HS prefixes.
The final training corpus retains thermal10.56%,DG9.99%,forwarder11.55% coverage.

Proper-layer fixes cover shared-unit nested support statements, one-wrapper
compound packing rendering, complete locality name/country ownership and a
static address-owned C/O casing scope. New active catalog still has456 templates;
production ownership references are updated. Old evidence catalogs remain immutable.
One-off generated problems were corrected/regenerated or excluded; no gates relaxed.
Of1,000 planned proposals,825 native renders led to794 semantic approvals;750 were
published,44 retained as reserves.31 semantic holds are outside the dataset.

**958 tests pass**. All750 exact pinned replays and active-catalog public replays
pass; all456 catalog references load.124,766 new lines have complete coordinates.
Original-data/hash/ID/validation-parent/casing/schema checks pass. Full final replay
148.25s/351.9MiB; token-length audit16.27s. Maximum training target remains3182,
validation1356; current3329 generation cap is sufficient. Recompute schedules for
10750 records when configuring the next run.

Provider-reported all-attempt cost **$1.61745**, including retries and unused draws;
about$2.16/1000 published samples for this campaign, not a future cost guarantee.
No Codex subagents or paid review sweep. These are validated supervision/coverage
improvements; their model-performance effect still requires training comparison.

The incidental checksum finding traces to R30 publishers replacing Paddle text
while retaining R29 per-record digest fields. Actual text/pages/targets were
correct; current training uses whole-file hashes and disables that optional
per-record field, so trained inputs/results were unaffected. R31 now recomputes
the derived fields, with before/after receipts; common digest generation and
publication checks prevent recurrence. No historical datasets were rewritten.

## 35. R31 rank-64 training configurations prepared — 2026-11-09

Prepared fresh-run local and RunPod profiles under `configs/training/production`,
named `t5gemma2_270m_lora.mpci_bl_real600_synthetic10150_r31_paddle_centroids_inputonly_v7_e10_compact_eva_a32_r64_{local,runpod}_schedulefree_v1.yaml`.
They derive from the completed R30 rank-64 profiles, **not rank96**, and pin the
final R31 aggregate files after the text-checksum metadata amendment:10,150
synthetic +600 real training records and60 validation records. New run/adapter
identities and no checkpoint/MLflow resume prevent accidental continuation.

| Setting | Local | RunPod 96GB |
|---|---:|---:|
| Microbatch | 1 | 4 |
| Gradient accumulation | 32 | 12 |
| Effective optimizer batch | 32 | 48 |
| Updates per epoch | 336 | 224 |
| Total updates, ten epochs | 3,360 | 2,240 |
| ScheduleFree warmup updates | 168 | 112 |
| Eval/save interval | 840 | 560 |
| Final partial optimizer batch each epoch | 30 records | 46 records |

All other training behavior is preserved:rank64/alpha32,rsLoRA,EVA,LR1e-4,
ScheduleFreeAdamW,dropout0.05,bf16,gradient checkpointing,length-grouped sampling,
seeds424,greedy evaluation and case-insensitive scoring. Eval/save still occurs
every2.5 epochs, with final evaluation enabled. Generation cap3329,input limit19200
and target limit5500 are unchanged. The V7 schema/constraint artifact is unchanged;
the input-only prompt injects no historical uppercase instructions, and canonical
target processing preserves R31 printed casing. Cache identity includes the new
dataset hashes. MLflow tags identify the casing policy and750-record extension.

The installed Trainer's final partial-accumulation flush was checked when computing
the schedules. Full `inspect-dataset` readback verifies all10,810 records, source
file SHA-256 values and target schemas:18.41s,587 records/s. RunPod strict config,
prompt and task-constraint validation also passes. Regression tests compare both
new profiles against R30 with only the deliberate data/identity/schedule/tag
differences and check printed-case canonicalization: **42 tests passed in20.48s**.

No dataset, runtime implementation, existing config, training job or pod state
was changed. No GPU benchmark was run:hardware/batching/model settings are unchanged.
When deploying RunPod, continue using pod-local code/venv/caches and persistent
`/workspace` outputs; the old R30 launch script still points to its historical run
and must not be used to launch R31 unchanged.

## 36. R31 AGENCY-ending follow-up — 2026-11-09

[Detailed audit and scoped amendment](analysis/r31-agency-diversity-20261109/REPORT.md).
The earlier76 AGENCY-ending continuation documents contain72 distinct company
names, but come from only two delivery-agent parents (31/45). Their wording
instructions explicitly required the ending;39 use exactly `(Agency)`.
Different company stems alone therefore did not remove the boundary-token cue.

Manually varied40 older terminal words (16/24 per parent) across eight plausible
company/branch endings, retaining36 AGENCY examples. Only the name target and one
physical line per record change; its geometry is recalculated by the renderer.
All40 retain the difficult continuation boundary. Addresses, contacts, company
stems, other targets/coordinates, real/validation and the newer750 are unchanged.
Old-screen terminal-AGENCY share becomes36/141 (25.5%); new52/799 (6.5%) is unchanged.
Both template instructions no longer force AGENCY. No paid generation was used.

Immutable before/after wording/rendering/review receipts and an explicit manifest
amendment preserve provenance. Both local and RunPod rank64 config file hashes
are refreshed; counts/schedules/settings are unchanged. Targeted replay, boundary,
coordinate, unrelated-change rejection, config and publication tests pass.
Full recertification also passes all750 exact and active-catalog replays and the
explicit40-record amendment chain (150.96s). The new cohort's coverage is unchanged.
