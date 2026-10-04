# Real-data foundation and model-ceiling audit

Date: 2026-10-01. Scope: real training/validation labels, historical real-only runs, and comparison with later saved model results. This is an **analysis-only** audit: no dataset, template, prompt, training configuration, model, or production code was changed. No training or model inference was launched; no paid model calls were made.

All new executable analysis, measurements, per-field/per-document inventories, example OCR/labels, and eight plots are together in [real-data-baseline-audit-20261001](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/). This report is outside the repair-artifact hierarchy so it can serve as a visible project decision record.

## 1. Answers to the main questions

1. **The real labels were not a demonstrably clean foundation before synthesis.** The historical real-only file itself contains product text assigned to `additionalInformation`, an aggregate/portion cargo duplication, and addresses assembled by omitting interior locality words. Some are actual ownership/topology errors; some are target-policy choices that make extraction harder or inconsistent. They must not all be called hallucinations or assigned one blanket error rate.
2. **The exact historical baseline has survived.** Its original 1,157-record file matches the training configuration's SHA-256, and the 100 references embedded in the saved checkpoint-900 prediction file match those labels exactly. We are not guessing what the baseline learned or evaluating it against today's labels by accident.
3. **The best retained real-only adapter is already close to the headline later scores:** saved replay F1 0.8324 / accuracy 0.7618; latest 30k run's best logged F1 checkpoint reaches 0.8512 / 0.7933. These native scores use different target contracts and are not a controlled estimate of synthesis benefit.
4. **On a fixed common subset, the saved 30k v5 model scarcely improves overall extraction:** F1 0.8390 → 0.8406; paired 95% interval for the difference −0.0176 to +0.0213. Relations and equipment improve, while party and description extraction do not. This comparison uses the same 100 OCR inputs and same gold values, not merely the same nominal validation size.
5. **Address and AAI repairs alone cannot account for the whole numerical gap.** A hypothetical perfect-prediction replacement of just those fields raises common-field F1 to 0.8731 for the real-only model and 0.8780 for the saved 30k v5 model. These are diagnostics, not predicted results from retraining or from changing the target schema.
6. **The evidence does not establish a 270M-model or text-only ceiling.** It also does not establish that a clean synthetic rebuild will reach 0.95. Training size, supervision, optimizer, effective batch, epochs, representation, and real-example exposure changed together. Dataset size cannot be ruled out from these runs; neither can model capacity or lost layout information.
7. **A synthesis rewrite is not the next justified irreversible decision.** First establish a fixed, source-adjudicated real evaluation contract and a bounded clean-real training baseline. The existing artifacts are useful evidence and should be preserved.

## 2. What data and runs actually exist

### 2.1 The three real-label views

| View | Real train | Real validation | What it represents |
|---|---:|---:|---|
| Historical real-only v3 | 1,057 | 100 | Exact labels used by the principal real-only experiments |
| Frozen v6, used by latest r48/a32 run | 1,056 | 100 | Same real corpus minus one held ambiguous package record; earlier repairs and schema migration applied |
| Current GROUND-015 working view, including holds | 1,056 | 100 | Address-line/cargo edits in progress; not a uniformly finalized training dataset |

Historical file: [records.jsonl](../artifacts/kie-training/datasets/mpci-bl-combined1157-task-facing-package-categories-v2/records.jsonl).

Its hash is `2a3e2ea3231cfff7674e85b54a52f66d0fee98b59b207bc7b04fe0f9916dfc42`. It matches the saved real-only resolved configuration and original dataset report. All 1,157 IDs are unique; there are 1,157 distinct exact OCR hashes. This rules out exact duplicate text in that file, not near-duplicate layouts.

Frozen later file: [v6 archive](../artifacts/kie-training/datasets/mpci-bl-real1057-synthetic29910-recovered-mpci-aligned-v6-v1_pre_ground015_archived/). Its train hash is `7e68d4aea96aa70eff31c6bc9464637761262f3bdc59c7338e1499987cb62f2f`; validation hash is `44975df26f24f8176fc4fb90252a6c4edbd058bd42760aa91ab8f81d10001d33`. Both are exactly the latest completed RunPod run's pinned inputs.

The older 30k v5 run's original dataset path has subsequently changed. Its **archive**, however, matches both original hashes exactly: train `e633571fa9d2f303c4a7990b31da7a5ee7f80f27ba0f9a451b2f666296ac07f6`, validation `ad8b91101922365fd7314d2030e230ab71c3592893095a0d0897fa8d76969dfe`. A failed hash check on its mutable original path therefore does not mean that the historical labels are lost. See [archive verification](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/v5_archived_run_dataset_verification.json).

Current working real records are distributed across:

| Current file | Real records |
|---|---:|
| `train.jsonl` | 1,033 |
| `validation.jsonl` | 95 |
| `address-fixed-cargo-held.jsonl` | 17 |
| `held-original-records.jsonl` | 11, including the other 5 validation IDs |

The audit reunited these **for analysis only**. It did not quietly reduce validation from 100 to 95. Current partition status is not semantic certification of every field. The one historical record absent from this working corpus, `doc_d9deb738…`, is preserved separately by GROUND-009D: its OCR prints 290 drums and 290 pails without enough evidence to determine nesting versus aliases. See the [package decision](../artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground009d_packages/RESULT.md).

All 100 validation OCR inputs are byte-identical across these versions. Two current real **training** inputs differ from the historical OCR, `doc_a0f7d986…` and `doc_6ce54cdf…`; these are earlier explicit PDF-backed cargo/OCR derivatives, not newly generated addresses or edits made in this audit. Their original and derivative texts are preserved separately in the example directory.

The wider compilation source universe is 2,175 distinct real IDs. It is **not** the same population as the 1,157-record real-only training/evaluation dataset. Prior complete-source inventories are in [REAL_SOURCE_MAPPING_REPORT.md](../artifacts/kie-training/analysis/schema-label-contract-audit-20260924/REAL_SOURCE_MAPPING_REPORT.md). This audit's principal numerical denominators are 1,157 historical / 1,156 retained real records.

### 2.2 Recovered real-only run inventory

These are best **logged** checkpoints; rows with different validation sets or schemas must not be interpreted as a learning curve over sample size.

| Run | Best saved/logged step | Field F1 | Field accuracy |
|---|---:|---:|---:|
| Pilot106 v5 | 80 | 0.6531 | 0.5189 |
| Combined487 v2 | 450 | 0.7940 | 0.6900 |
| Relation-explicit v3 | 450 | 0.7964 | 0.6957 |
| Relation-v3 table-input v2 | 180 | 0.7394 | 0.6185 |
| Combined1157 task-facing r32/a64 | 900 | **0.8294** | **0.7555** |
| Combined1157 r64/a96 | 1,020 | 0.8036 | 0.7187 |
| Combined1157 embeddings variant | 1,125 | 0.7900 | 0.6990 |

All seven identified real-only run configurations' dataset paths still match their stored hashes. Full settings and trajectories: [run scoreboard](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/run_scoreboard.csv), [eval history](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/eval_history.csv).

The principal r32/a64 real-only run reached 1,125 optimizer steps but did not finalize its epoch-25 generated evaluation. Checkpoint 900 is the strongest retained evaluated checkpoint; it is **not** a recovered epoch-25 final model. Direct read-only inspection of the Docker MLflow volume still shows that historical run as `RUNNING`, with no end time and no completed post-900 evaluation. That is stale run status, not evidence of a live process. No container/service or training run was started to change it; a network-disabled, read-only Python diagnostic queried the database. [MLflow receipt](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/mlflow_real_baseline_readonly.json).

The later saved checkpoint-900 replay reaches F1 **0.832445**, accuracy **0.761795**, precision **0.840509**, recall **0.824534**. The small difference from the training-time 0.829358 belongs to a separate saved generation replay; it is not a new training result. We use that replay for document/field comparisons because it contains the actual predictions and embedded historical references.

## 3. Metrics: what is comparable and what is not

### 3.1 Definitions

The production [metric implementation](../src/document_ocr/training/metrics.py) uses:

- Field precision: exact matching `(path, value)` leaves / predicted leaves.
- Field recall: exact matching leaves / gold leaves.
- Field F1: harmonic mean of these precision and recall values.
- Field accuracy: exact matching leaves / union of predicted and gold **paths**. A wrong value at an existing path is one incorrect path, not two union entries.
- Relation/category accuracy: intersection / union of **fact tuples**, i.e. micro Jaccard. This differs from field accuracy and from whole-document exact match.
- Facts: extraction leaves with relation/scaffolding paths removed. Categories may still be extraction facts; these are not mutually exclusive partitions of supervision.

Sparse absent fields do not create a large true-negative reward. List index matters for field metrics. The relation projector removes some array ordering sensitivity but still uses goods identity/index and, in v6 quantity relations, placement index. It is not a fully optimal graph-matching score.

For fact-set Jaccard, `accuracy = F1 / (2 - F1)`. Therefore category F1 0.9387 corresponds to category accuracy 0.8845; saying “categories are almost at 0.95” must not imply both measures meet the goal. This identity does not generally apply to path-based field accuracy.

### 3.2 Native full-task scores

| Model/checkpoint | P | R | F1 | Accuracy | JSON valid | Schema valid |
|---|---:|---:|---:|---:|---:|---:|
| Real-only r32, step900 saved replay | .8405 | .8245 | .8324 | .7618 | .98 | .76 historically |
| 30k v5 r32/a32, step3220 saved replay | .8527 | .8440 | .8483 | .7854 | .99 | .87 |
| 30k v6 r32/a64, step2576 saved replay | .8374 | .8391 | .8383 | .7725 | .99 | .89 |
| Latest 30k v6 r48/a32, best-F1 step1932, logged | .8540 | .8484 | **.8512** | .7933 | .99 | .93 |
| Latest r48/a32, final step3220, logged | — | — | .8400 | .7716 | — | — |

Schema code evolved: running the old real-only predictions through today's schema accepts .79 instead of the original .76, although their field F1/P/R/accuracy and relation/category scores reproduce. This is validator drift, **not improved predictions**. Both the historic and current outcomes are retained rather than silently choosing the favorable one. The v5 training-time final score was .849226, versus .848343 in its separate saved replay; again these are not interchangeable records.

Latest r48 best **accuracy** is step2576 at .796095 (F1 .851059), rather than best-F1 step1932. The final checkpoint is worse than both. Downloaded MLflow and checkpoint log values agree exactly for the compared evaluation metrics.

| Metric family | Real-only native P / R / F1 / accuracy | Latest r48 native P / R / F1 / accuracy |
|---|---|---|
| Relations | .6985 / .7472 / .7220 / .5650 | .7688 / .8956 / .8274 / .7056 |
| Categories | .7815 / .8158 / .7983 / .6643 | .9289 / .9487 / .9387 / .8845 |
| Extraction facts | Not logged as a separate v3 metric | .8612 / .8457 / .8534 / not logged |

Old relations include coverage/package-ID scaffolding eliminated by v6. Category inventories also changed. These native family scores are useful summaries of each run, **not causal gains on identical subproblems**. Comparable projections are below.

Latest r48 per-document predictions were not saved (`write_predictions_for: []`, `run_final_evaluation: false`). There is no honest way to derive a fresh field-by-field error inventory for that checkpoint from aggregate MLflow metrics. This audit therefore uses its verified aggregate history and the two other available 30k prediction replays for paired granular analysis. A future inference-only replay can close this specific gap; none was launched here.

![Native scores](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/plots/01_native_scores.png)

### 3.3 Same-input, same-label, common-contract comparison

For a stronger comparison, the analysis:

1. Requires exactly the historical 100 validation document IDs.
2. Checks the frozen OCR identities, and verifies all old embedded gold references.
3. Maps old arrays to v6-shaped goods/packages/placements without correcting predicted fact values.
4. Excludes container size/type/category surfaces and DG fields whose schemas changed.
5. Scores all models against the same **4,375 historical reference leaves**, retaining the historical address and AAI policies.
6. Accounts for malformed JSON as empty predictions; records projection problems rather than silently repairing them.

This is a **shared-field diagnostic**, not the full MPCI task. It cannot assess distinctions intentionally lost in the newer representation. Package/list projection may alter path identity; the same mapping is applied to old gold and old predictions.

All 100 historical gold targets project without issues. Two real-only predictions and two v5 predictions contain orphaned package/allocation rows without a usable goods parent; the shape-only comparison cannot attach those rows. Their failures are recorded, and a conservative sensitivity check additionally counts their four/seven unprojectable leaves as false positives. F1 becomes **.8386 versus .8400**, leaving the conclusion unchanged. The shared-field score must not be mistaken for full old-graph validity. [Sensitivity receipt](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/projection_sensitivity.json).

| Model | P | R | F1 | Accuracy | Correct leaves |
|---|---:|---:|---:|---:|---:|
| Real-only r32 | .8494 | .8288 | **.8390** | **.7740** | 3,626 |
| 30k v5 r32/a32 | .8483 | .8331 | **.8406** | **.7795** | 3,645 |
| 30k v6 r32/a64 | .8270 | .8263 | .8267 | .7587 | 3,615 |

The v5 30k improvement is **19 additional correct leaves**, +0.00167 F1 and +0.00555 accuracy. A 10,000-resample paired document bootstrap gives F1-difference interval **[−0.01762, +0.02132]** and accuracy-difference interval **[−0.02300, +0.03332]**. This does not establish either a reliable improvement or reliable equivalence. It is sampling uncertainty over these 100 documents, not seed-to-seed training uncertainty or confidence for unseen carriers.

![Common-contract uncertainty](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/plots/02_common_contract_ci.png)

### 3.4 Common facts, relations and categories

| Model | Shared facts P / R / F1 / path accuracy | Common relations P / R / F1 / Jaccard | Package-only categories P / R / F1 / Jaccard |
|---|---|---|---|
| Real-only | .8583 / .8305 / **.8441** / .7865 | .7415 / .8114 / **.7749** / .6325 | .7881 / .8158 / **.8017** / .6691 |
| 30k v5 | .8517 / .8329 / **.8422** / .7861 | .8071 / .8451 / **.8257** / .7031 | .8545 / .8246 / **.8393** / .7231 |
| 30k v6 a64 | .8336 / .8221 / **.8278** / .7644 | .7543 / .8889 / **.8161** / .6893 | .8000 / .8070 / **.8035** / .6715 |

The common relation reference set contains 297 facts. The later models do show useful mapping capability; they are not incapable of relations. For v6 a64 the recall gain comes with lower precision than v5: it finds more correct edges/quantities but emits more incorrect extras. The modest overall improvement is not a uniform extraction improvement: **shared non-relation facts are flat or worse**.

![Facts and relations](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/plots/06_common_fact_relation_categories.png)

## 4. Real-label quality: findings and scope

Evidence classes used here:

- **Census:** exact counts/diffs over every record in the identified view.
- **Confirmed example or prior source-adjudicated repair:** directly inspected OCR/label contradiction or explicitly reviewed policy correction.
- **Screen:** a candidate generator, not an error count, clean certificate, or automatic repair instruction.

### 4.1 Addresses: the problem predates synthesis, but not every split address is wrong

There are **3,452** address values in the old real dataset: 3,155 training and 297 validation. Every one matches the corresponding `normalTarget` address value; task-facing projection changed **zero**. Comparing retained records through the frozen v6 run also finds **zero** changed old address values. Thus the latest completed training did **not** test the current `addressLine` repair: it still learned the old address strings.

The observed historical convention is generally “address excluding separately labeled city/country.” Under a deliberately componentized form contract, this can be legitimate. It becomes a different, harder sequence task when the locality is inside the printed postal line, and it is incompatible with the newly agreed whole-postal-`addressLine` contract unless migrated consistently.

Example, real training `doc_b8982e77…`:

```text
A DEC INC
2601 CRESTVIEW DR
NEWBERG OR 97132
UNITED STATES
EIN: 93-055595200
```

Historical labels:

```json
{"address":"2601 CRESTVIEW DR OR 97132","city":"NEWBERG","country":"UNITED STATES"}
```

Whole-postal extraction would instead include `NEWBERG` and `UNITED STATES` in `addressLine`, keep city/country separately, and exclude the name/EIN. No model should have to infer the city from the street. [OCR](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/examples/doc_b8982e7730b71294569caa09b69960d61ed94c28b4bf9b14d7c14838777fa4f1/historical-ocr.txt), [historical/current labels](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/examples/doc_b8982e7730b71294569caa09b69960d61ed94c28b4bf9b14d7c14838777fa4f1/labels.json).

An exact token-subsequence probe found **781 address occurrences in 559 documents** where reinserting only the separately labeled city/country internally recovers an observed OCR sequence: 712 values/510 training documents and 69 values/49 validation documents. This is a grounded measure of **non-contiguous target assembly**, not 559 proven bad labels. It checks whole-document sequences and does not independently certify party ownership for each match.

Literal locality retention is not uniform: among address values with a city label, 163 retain that city literally and 2,903 do not; among those with a country label, 16 retain the country literally and 3,091 do not. Named sites/districts, compound localities, aliases and punctuation can explain some differences. These counts motivate a consistent policy; they do not authorize blind city deletion or concatenation.

The current real working view has 3,427 `addressLine` values and 38 legacy `address` values across its accepted/held partitions. It still cannot be treated as a single finalized training contract. An exact-locality screen flags 19 city and 51 country label/addressLine pairs. In the validation subset those are nine flags in just three documents:

- `doc_45ed05eb…`: two parties print `BEDI SUEFARAB REPUBLIC OF EGYPT`. The city and country are present but glued together; literal boundary matching produces false alarms. Whether `P701` belongs in the postal address is a separate ownership question, not established by this screen.
- `doc_7f6df7bd…`: address lines contain `MALAYSIA` / `EGYPT`; separate labels include `Malaysia(MY)` / `Egypt(EG)`. These are representation differences, not absent countries.
- `doc_5222fa98…`: two postal blocks end `STREET NASR P.O.BOX 7074 , CAIRO`, followed by `Tax ID: 200026429 - Egypt`. The current line excludes that entire following line while the country label retains Egypt. This is exactly the mixed auxiliary/postal boundary policy that needs explicit adjudication; the whole-text presence of Egypt cannot decide it automatically.

These examples show why neither “city not a substring of addressLine” nor “every word appears somewhere in OCR” is a trustworthy semantic pass/fail rule. They do **not** establish that all current real address repairs are bad.

### 4.2 Product descriptions and additional information

Historical real labels contain AAI in **247 documents**: 220/1,057 training documents with 456 values, and 27/100 validation documents with 78 values. This field-role ambiguity existed **before any synthetic sample was generated**.

| Real source | Historical label shape | Source-supported problem / policy distinction |
|---|---|---|
| Veal `245db42f` — train | Leg in description; loin, forequarter and flap in AAI; four carton quantities | Parallel product identities separated by field without a consistent semantic boundary |
| Machinery `7b12b905` — train | Printing machine in description; other machines and `USED` in AAI, repeated across three container rows | More product identity/condition placed in an overflow field; grouping must respect shared versus row-owned measures |
| DUMECTIN `32ebac37` — train | `DUMECTIN` in description; `ABAMECTIN 5% EC` in AAI | Product composition/specification detached from product identity |
| Dyes `d87589f1` — validation | Generic dyestuff description; 21 brand/model/product entries in AAI | Long product list taught as a separate information array |
| Pigments `f67e893f` — validation | Generic dye description; five pigment identities in AAI | Same field-boundary inconsistency |
| Lighters `21d3ced7` — train | No description; `LIGHTERS` / `LIGHTER` in AAI | Commodity identity is present but assigned away from the principal description field |

Source OCR and old/current labels for all these cases are copied into the [example directory](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/examples/). For direct review: [veal](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/examples/doc_245db42f109575d44265e7a70107402d86283711472cf4d68b7ae7d386d12aa6/historical-ocr.txt), [DUMECTIN](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/examples/doc_32ebac37f60803fca4d6e46c05278045a5a0e8855cb30bd4ba75ae02fc5dbf18/historical-ocr.txt), [validation dyes](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/examples/doc_d87589f13c2eda1fb1e7ed8a6c48ca8af83431c6090d0f85f2287ae64d38c7c9/historical-ocr.txt).

This does **not** mean each printed product must become a separate goods row. Multiple HS codes are valid. A mixed commercial goods/accounting unit with shared totals can legitimately have a long description and multiple codes. Splitting it into rows with invented per-product weights/placements would be another defect. The issue is inconsistent product-field ownership and unsupported topology, not description length alone.

Existing source-role review artifacts cover 246 of these historical IDs. They contain 133 training and 38 validation values marked for inclusion in description, plus duplicate/omit and other decisions. These are useful review evidence, not a fresh independent certificate: the draft also contains decisions whose `sourceRole` and action differ. Accordingly, this report does not promote every draft classification to ground truth.

Compared with the frozen latest-trained labels, the working view changes goods subtrees in **234 real training and 29 validation documents**. Description leaves change in 75 training and 10 validation documents; cargo-item cardinality changes in six training documents. Current AAI remains only 13 values across four real documents. These are **edit-scope counts**, not a proof that all 263 edited records were previously wrong in every respect or are now perfectly labeled.

### 4.3 Cargo topology: one concrete defect already in real-only training

Real training `doc_8c42f0fd…` prints one commodity, `GLYCERIN TECHNICAL GRADE`, eight containers carrying one flexibag each, one shipment gross total of 201,535 kg, and one net total of 200,455 kg. The eight row gross weights sum exactly to the shipment gross.

The historical label creates **nine goods groups**: an aggregate description/weight group plus eight anonymous container portions. That duplicates the gross-weight meaning and separates the product from its placements. The source-supported repaired representation is one goods item with eight placements; a 152-CBM total can be derived from the eight explicitly printed 19-CBM rows under the approved exact-sum policy.

This is a real-only source, not a defect introduced by template rendering. [Original OCR](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/examples/doc_8c42f0fd55573b2f9df85442c231f05b1641d4ee6773e73bd67a51c413381213/historical-ocr.txt), [labels before/after](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/examples/doc_8c42f0fd55573b2f9df85442c231f05b1641d4ee6773e73bd67a51c413381213/labels.json), [prior exact accounting adjudication](../artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground015_cargo_text/dataset_only_cargo_review/REPORT.md).

Historical real train/validation have 63/10 multi-goods documents, 36/7 documents with repeated goods descriptions, and 43/2 with a goods group lacking description. These are **review strata**: repetition and absent description are sometimes legitimate. They are not an automatic split/merge rule or confirmed defect count.

### 4.4 Other real-label issues already repaired before the latest run

A shape-aligned old-to-frozen-v6 comparison, excluding changed equipment vocabularies and DG schemas, finds edits in **75 training and five validation documents**, totaling 258 leaf differences. These include both genuine corrections and legitimate unit/package policy migrations:

- Seal values: 22 documents / 46 differing leaves, including prefix/boundary decisions.
- Package quantities: 23 documents / 58 differing leaves; package categories: 21 / 43. Nested packaging and duplicate totals require source ownership, not string matching.
- Marks: 17 documents / 21 differing leaves. These do not establish that every remaining mark boundary is correct.
- Gross-weight values: 15 documents; gross units: 14. Net values: 10; net units: nine. The prior mass repair specifically addressed printed-tonne versus normalized-kilogram policy and included real validation.
- Route, date, issue-place, freight and temperature paths: smaller documented source-specific corrections.

These counts overlap. They must not be summed into a defect prevalence rate. Full before/after values are in [label_version_differences.jsonl](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/label_version_differences.jsonl); the stable-field subset summary is [here](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/stable_field_change_scope.json).

The older [GROUND-006 adjudication](../artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground006_locations/RESULT.md) is a useful causal example: a PDF's `FREIGHT COLLECT` was missing from OCR, while empty Prepaid/Payable captions encouraged an unsupported prepaid label. Other OCR headings confused receipt/loading/delivery roles. Presence of a city elsewhere in the document did not make its assigned route role correct.

Conversely, some prominent synthetic defects were **not** inherited missing real facts. The original seal pass found all 61 affected original seal paths printed in the original OCR; generated labels/text subsequently diverged. The GROUND-007 party-affiliation repair changed synthetic records but no real training/validation labels. Do not transfer synthetic defect counts into the real corpus.

### 4.5 OCR information loss is a separate limit

The BORLINK source `doc_6ce54cdf…` has all 15 container rows in the PDF but only six in its old OCR/labels. A later explicit derivative recovered them. This proves some raw-text information loss, but not its prevalence. If a fact or ownership cue is genuinely absent from the model's input, label cleaning cannot teach reliable recovery of that fact; either omit it under the input-grounded task, improve OCR, or supply additional evidence.

The goal is not to train the model to guess PDF-only truth. Distinguish:

1. Information present and unambiguous in OCR: label it consistently.
2. Present but difficult order/layout: test whether text is enough to establish ownership.
3. Missing from OCR but present in PDF: OCR/input representation problem.
4. Ambiguous even in the document: explicit task policy or abstention, not fabricated precision.

## 5. Fresh whole-real-dataset grounding audit

The existing source-grounding scanner was rerun separately on all three real views:

| View | Records | Scalar observations | High-priority documents | High-priority fields |
|---|---:|---:|---:|---:|
| Historical real v3 | 1,157 | 60,105 | 2 | 9 |
| Frozen trained real v6 | 1,156 | 54,454 | 2 | 9 |
| Current working real, including holds | 1,156 | 54,017 | 6 | 21 |

Every flagged high-priority family was cross-checked with its OCR and existing source adjudication:

- `26ccea29`: four net weights are supported across OCR line wraps, e.g. `4` / `527KG`; not invented 4,527 kg.
- `1464f450`: `20 CBM/HR` is printed in the carrying-condition context; the added word `VENTILATION` is semantic normalization, not evidence of a missing numeric value. The joined `HRACID` surface also defeats simple token boundaries.
- `8c42f0fd`: 152 CBM is eight printed 19-CBM rows, not an invented measure.
- `88c4c710`: 16,686 kg and 82.7 CBM are exact sums of the two owned printed rows (12,800 + 3,886; 63.320 + 19.380). Earlier proposed two-goods labeling was superseded by one-goods/two-placement policy; do not resurrect the old proposal as a new confirmed defect.
- `6ce54cdf`: the PDF-backed current derivative supplies all 15 rows; 300 packages, 326,340 kg gross and 600 CBM are derived from printed components.
- `a0f7d986`: 1,307 packages are 667 + 640 in the already documented corrected source.

These scanner findings do not establish a new blanket absent-value problem. The historical scan's 871-record broad review queue is also **not** 871 erroneous documents. More importantly, its two high-priority documents do not imply the other 1,155 are clean: the confirmed product-field and topology defects above contain perfectly real words and numbers, and a presence checker cannot detect their wrong meaning.

The reused scanner has **3,249/3,255 unsupported-rule observations** on the frozen/current v6 views, largely because its old category/schema aliases do not cover the newer representation. Those entries remain explicit. It must not be described as a complete v6 semantic validator. Its usefulness here is broad candidate discovery plus comparison with exact adjudications, not final certification.

Reports and complete compressed field results: [historical scan](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/scan-historical_real_v3/), [trained v6 scan](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/scan-trained_real_v6/), [current scan](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/scan-current_working_real/).

## 6. Where the real-only model is strong and weak

### 6.1 Paired granular performance

All values below are F1 under the fixed common contract; gold support is identical between columns.

| Field | Gold values | Real-only | 30k v5 | 30k v6 a64 |
|---|---:|---:|---:|---:|
| Goods AAI | 78 | .1228 | .0684 | .0820 |
| Goods description | 117 | .7281 | .6404 | .6036 |
| Marks | 136 | .4247 | .4578 | .5350 |
| Shipper address | 90 | .5698 | .5889 | .4667 |
| Consignee address | 84 | .6988 | .6386 | .5680 |
| Notify address | 70 | .6763 | .6377 | .5532 |
| Shipper city | 82 | .8272 | .7595 | .7105 |
| Consignee city | 77 | .8477 | .8408 | .7600 |
| Notify city | 65 | .7520 | .7597 | .7188 |
| Package quantity | 119 | .7438 | .7983 | .7595 |
| Placement container | 170 | .8060 | .8304 | .8476 |
| Placement quantity | 129 | .7336 | .8060 | .7682 |

The real-only model already performs strongly on loading/discharge port names (.9741/.9691), B/L number (.9605), issue date (.9814), and freight payment arrangement (.9878). These are measured performance under historical gold, not assertions that every historical reference was perfectly correct. Some strong small-support fields, such as temperature, have only seven validation values and should not be generalized to rare unseen presentations.

The strongest recurring failure is **not inability to copy identifiers generally**. It is assigning and assembling text into the requested role/row/field, and producing the correct cargo structure.

![Field breakdown](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/plots/03_field_f1_heatmap.png)

### 6.2 Error shapes rather than one “hallucination” bucket

The new screen classifies mismatched text paths by exact omission/addition, normalized equality, nested span, or other substitution. It is not a semantic truth oracle:

| Real-only field family | Nonmatching paths | Useful interpretation |
|---|---:|---|
| All party addresses | 101 | 16 normalization-only, 40 nested spans, 32 other substitutions, eight omissions, five additions |
| Goods description | 37 | 22 nested-span differences, nine omissions, three other substitutions, three additions |
| AAI | 90 | 61 omissions, 19 additions, seven nested spans, three other substitutions |
| Marks | 111 | 43 omissions, 30 additions, 34 substitutions, four nested spans |

Paired nonempty address strings have median character similarity about **0.91** in the real-only model. Many are almost right but assemble a different boundary, locality or punctuation. “Not exact” is not synonymous with “invented address.” Conversely, high similarity does not prove that a dropped house number or district is harmless.

The earlier deep audit's index-insensitive diagnostic lifts real-only marks F1 from **.4247 to .6873**, and overall native F1 by about .0196. This recovers positional agreement only; it does not license moving a real mark to the wrong goods owner. Marks remain materially weak even after that diagnostic.

AAI is highly concentrated: 34/78 validation values occur in two long-list documents. Thus micro-F1 can be strongly affected by a few product-list segmentation decisions. The new common-contract cohort results are:

| Cohort | Documents | Real-only F1 | 30k v5 F1 | 30k v6 a64 F1 |
|---|---:|---:|---:|---:|
| AAI absent | 73 | .8594 | .8648 | .8486 |
| AAI present | 27 | .7944 | .7868 | .7797 |
| One goods group | 90 | .8602 | .8566 | .8495 |
| Multiple goods groups | 10 | .7013 | .7354 | .6629 |

This is association, not proof that AAI annotations cause a specific score loss. Those documents are also longer/more structurally difficult. Nevertheless, weak-field locations and confirmed label-policy defects overlap in a concrete way.

![Cohorts](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/plots/04_validation_cohorts.png)

![Error shapes](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/plots/08_real_text_error_shapes.png)

### 6.3 How much does regrading known corrected gold change the result?

A pre-existing source-adjudicated validation candidate changes **24 fields in 15 documents**, leaving the other 85 rows byte-identical. It includes product-field and placement corrections and preserves all 100 OCR inputs. This audit did not apply it to production. [Candidate decisions](../artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground015/validation-label-repairs/REPORT.md).

Scoring the same saved predictions on the same shared fields:

| Model | Frozen-v6 reference F1 | With these 15 reviewed cargo corrections | Delta |
|---|---:|---:|---:|
| Real-only | .8388 | .8424 | +.0036 |
| 30k v5 | .8414 | .8452 | +.0038 |
| 30k v6 a64 | .8281 | .8328 | +.0047 |

This demonstrates measurable evaluation-label sensitivity. It does **not** explain the entire gap to .90/.95. It also does not measure the benefit of training on corrected labels: learned behavior stays fixed in this experiment. Some corrected labels may penalize a model trained to reproduce the old mistake; therefore small net changes do not prove training noise is unimportant.

### 6.4 Error-budget diagnostic

| Model | Actual common F1 / accuracy | Excluding address + AAI F1 / accuracy | Perfect old-contract address + AAI predictions F1 / accuracy |
|---|---|---|---|
| Real-only | .8390 / .7740 | .8611 / .7975 | **.8731 / .8138** |
| 30k v5 | .8406 / .7795 | .8665 / .8077 | **.8780 / .8233** |
| 30k v6 a64 | .8267 / .7587 | .8578 / .7898 | **.8699 / .8064** |

The final column inserts the old gold address/AAI values into predictions and removes their extras, changing nothing else. It is a hypothetical partial oracle, not a realizable model or a bound on all benefits from better training. Better descriptions/relations may improve indirectly after relabeling, but that is unmeasured. The concrete conclusion is that **the current mistakes outside those two field families are already large enough to prevent the stated target**.

![Error budget](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/plots/07_address_aai_error_budget.png)

## 7. What the training histories do and do not establish

The main real-only run's F1 rises .6488 → .7783 → .8103 → .8294 at epochs 5/10/15/20. Teacher-forced validation loss bottoms at epoch10 (.0715), then rises to .1227 while generated exactness still improves. This indicates different behavior for loss and generated extraction, with evidence consistent with overfitting the training serialization. It is not a proof that another architecture is required.

The latest r48 run reaches best F1 at epoch3 and declines by epoch5. Best relation F1 occurs at another checkpoint (epoch2.5), so a single “best model” does not maximize every family. Stronger rank/alpha/optimizer settings cannot be inferred from the real-only r64 run's lower score because other settings changed too.

| Factor | Main real-only baseline | Latest 30k run |
|---|---|---|
| Effective batch | 24 | 48 |
| Rank / alpha | 32 / 64 | 48 / 32 |
| Initialization | Default LoRA | EVA |
| Optimizer | Fused AdamW | ScheduleFree AdamW |
| Schedule | Cosine, 5% warmup | Optimizer-internal warmup 161 steps |
| Best-F1 epoch | 20 retained | 3 |
| Training contract | v3 separate cargo arrays | v6 goods-local placement |
| Address values | Original split-address labels | Same split-address values for retained real records |

The latest run retained all real records during target-length filtering; 54 excluded records were synthetic. It trained on 30,912 records, only 1,056 of which were real (**3.42%**). At epoch3 it had seen each real example about three times, versus twenty times at the retained real-only checkpoint—approximately 3,168 versus 21,140 real-example presentations. Synthetic descendants may carry useful related supervision, but these runs are not matched on clean real-data exposure.

Latest validation: maximum input 11,790 tokens, maximum reference target 1,406; best-step maximum generated output 1,404; EOS fraction 1.0; input truncation zero. A 2,048 generation cap is therefore **not the observed best-step bottleneck**. Historical baseline also used no source truncation. OCR omissions/flattening are different from tokenizer truncation.

The same 100 validation IDs have been used repeatedly for checkpoint selection, diagnosis, and policy correction. They remain a valuable development benchmark, but should not be represented as an untouched final test set. Any future benchmark regrading should preserve old scores/references, and fresh held-out evidence is needed before declaring production .95 performance.

![Learning curves](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/plots/05_learning_curves.png)

## 8. What follows for the project direction

### 8.1 Conclusions supported now

- We did scale before establishing a uniformly adjudicated real supervision contract. The product/topology defects prove this directly; no synthetic audit is needed to make that statement.
- There is still useful capability in the existing models: ports, identifiers, dates, and many placements perform well. Wholesale project disposal would throw away working components and diagnostic baselines.
- The dataset is not uniformly untrustworthy in the sense that every scalar is fabricated. Exact inventories and source examples show both supported values and concentrated ownership/policy problems. However, **this audit does not certify a clean percentage of whole documents**.
- Current cleaned labels have not yet received a controlled training test. The latest model still used pre-GROUND-015 address/cargo labels, so its result is not evidence that the new whole-address/description policy has failed.
- A small model, lost layout, label quality, exposure and output representation remain competing explanations. We can prioritize investigations, but cannot assign causal percentages from the existing runs.

### 8.2 Recommended bounded next sequence—not launched

1. **Freeze the real-data experiment boundary.** Use the recovered 1,157 historical records and explicit current derivatives, not an arbitrary accepted subset of the 30k repair workspace. Keep all 100 evaluation IDs accounted for, including five currently held. Keep the one drum/pail training hold explicit.
2. **Finish a real-only semantic reference, starting with all 100 validation documents.** Reuse valid existing source decisions, but independently resolve remaining address/description/marks and goods-accounting ownership. Preserve source spans, field roles, permitted normalization and exact derived-total justifications. Presence-only checks are insufficient. Reviewers should not be asked to imitate a model's predictions.
3. **Use one target policy across real train and validation.** Whole postal `addressLine` plus separately extracted city/country; names/taxes/contact excluded. Product identity/specifications in description; a narrowly defined AAI purpose or no AAI. One goods accounting unit can span multiple products/HS codes and containers; do not invent disaggregated quantities. Preserve consistent printed units and approved derived totals.
4. **Measure before rebuilding synthesis.** Regrade saved predictions where semantics remain comparable; perform a separate inference-only replay for latest r48 if its granular diagnosis is needed. Report old and corrected-reference scores side by side. Do not score an old split-address output as though it had been trained for a new full-address field without labeling that task change.
5. **Then one focused clean-real training comparison.** Keep a strong known setup fixed and use the adjudicated real labels. This is the least expensive way to establish whether changed supervision helps, without simultaneously rebuilding 30k synthetic documents, changing the base model, and changing the optimizer. This is a recommendation for a later authorized task, not a run launched here.
6. **Only then choose the next investment from observed residual errors.** If facts are readable but the model misses them, test representation/capacity. If correct relations cannot be determined from OCR alone, test layout-bearing input on a small explicitly identified set. If the clean-real model improves but lacks category/template coverage, carefully verified synthesis becomes a grounded scaling intervention.

No step promises .95. The purpose is to make the next result attributable to a specific intervention rather than another confounded dataset/model cycle.

### 8.3 Research context, not proof about this dataset

[VRDU](https://research.google/pubs/vrdu-a-benchmark-for-visually-rich-document-understanding/) specifically evaluates rich schemas, diverse layouts and nested/repeated fields, and reports difficulty with template generalization and hierarchical extraction. This supports treating cargo structure and evaluation matching as first-class problems rather than assuming flat-field success transfers automatically.

[LMDX](https://aclanthology.org/2024.findings-acl.899/) uses text with layout/localization and grounding for hierarchical document extraction. It makes a layout-bearing input experiment plausible without deciding here that the extractor must train on images. Its results on other benchmarks do not establish that bounding boxes will solve this Bill-of-Lading dataset.

[Northcutt et al.](https://arxiv.org/abs/2103.14749) demonstrate that test-label errors can change benchmark conclusions. That supports preserving and independently adjudicating evaluation labels; it does not quantify our noise rate or justify treating every model/gold disagreement as a gold error.

## 9. Deliverables, reproducibility, and limitations

New scripts, all scoped to the audit directory:

- [analyze.py](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/analyze.py): hashes, real-only views, exact version differences, native/common rescoring, paired bootstrap, run history and read-only latest MLflow extraction.
- [supplement.py](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/supplement.py): address provenance, cargo edit scope, reviewed-reference sensitivity and source examples.
- [deep_checks.py](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/deep_checks.py): common relation/category/fact scoring, error shapes, cohort and partial-oracle diagnostics, settings/provenance checks.
- [plots.py](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/plots.py): eight informative figures from the saved measurements.
- [projection_sensitivity.py](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/projection_sensitivity.py): explicit penalty for old predictions with orphaned graph facts.
- [validate.py](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/validate.py): dataset immutability, metric arithmetic, MLflow agreement, script and artifact integrity checks.

Run from the project root with `.venv/bin/python <script>`. `analyze.py` runs before the others. `supplement.py` also uses the three scanner outputs; reproduce them with the existing `scan_grounding.py --input <audit-view.jsonl> --output <new-directory> --workers 4`, then point to those outputs. All references/hashes and per-document outputs are retained; no opaque score-only conclusion is necessary.

The main full analysis took **16.05 s** and **269.7 MiB** peak RSS on the final measured rerun. The historical whole-real scalar scan took **2.20 s**. These are offline analysis timings, not production inference benchmarks. External provider spend was **$0**. Final validation confirmed all seven source dataset files unchanged, all 100 evaluation IDs retained, native/common scores reproduced from per-document counts, the real MLflow history and 390 latest-run metric points matching saved logs, all six scripts parsing and passing Ruff, all eight PNGs valid, and all report links resolving. Exact checks and artifact hashes are recorded in [validation.json](../artifacts/kie-training/analysis/real-data-baseline-audit-20261001/validation.json).

This was a complete computational census of the identified real views, a source-grounded investigation of the important error families, and a fresh comparison of available predictions. It was **not** a blind independent human relabeling of every field in every real document. The precise remaining evidence gaps are: unresolved semantic adjudication in the current working real view; missing per-document latest-r48 outputs; lack of a controlled clean-label training comparison; and lack of a fresh untouched test cohort. None is concealed by calling a scanner pass “correct labels.”
