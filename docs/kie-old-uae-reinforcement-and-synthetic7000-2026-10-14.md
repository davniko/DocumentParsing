# Older UAE reinforcement and the 7000 sample training dataset

The [published training dataset](../data/curated/mpci-bl-synthetic7000-old-uae-reinforcement-r22-positions-v1/README.md) contains **7,000 synthetic documents**: the current 5,500 plus 1,500 selected older UAE documents. The 600 real training documents are excluded from this experiment and preserved in the parent dataset. Validation remains the same 60 real documents, byte-for-byte. No training or paid model calls were launched.

The additions contain **519 distinct scarce or difficult cases**, exceeding the requested approximate 500. They reinforce uniform container inventories, supported allocations, transshipment, thermal and DG presentations, thin package categories and postal complexity. Selection spans 340 eligible older layouts; the complete dataset retains all 413 current template sources.

## Why the older examples are useful

The latest completed comparison is run J versus K, not a trained result on the newly refreshed dataset. On the same 58 parseable validation documents, category F1 improved **0.9049 to 0.9284**, while field recall decreased **0.9145 to 0.8979** and relation recall decreased **0.9714 to 0.9048**. Container identifiers decreased from **110 to 102 correct out of 112**, and placement identifiers from **110 to 100 out of 112**. Package categories remained **47 out of 53** correct. [The result ledger](mpci-training-results-and-next-experiments.md#16-uae-coverage-replacement-run-k) preserves the full comparison and its uncertainty.

The intervention therefore retains the current equipment-category and alias contrasts while adding more ordinary inventory and ownership examples. It does not assume that an old example caused the previous model's success. The source-exposure balancing objective and coverage quotas use training inventories, not validation identities or document-specific gold errors.

## Older pool availability and exclusions

The older UAE sidecar contains 2,500 distinct documents from 358 sources. None adds a previously unseen source layout to the current 413-template pool; none duplicates a current raw input or document ID.

Two finite exclusion inventories reduce the available selection pool to **2,347 documents from 340 sources**:

| Reason | Older documents excluded | Effect |
|---|---:|---|
| Known private names or commercial references in non-extracted text | 138 | Avoid adding these historical anonymization residues |
| Earlier country or contact repairs not reflected in older labels | 15 | Avoid inconsistent extraction supervision |
| Total without overlap | 153 | None enters the new addition |

The 15 label inconsistencies are confined to three source families:

- `0d495914`, four descendants: the explicitly `CN>`-owned country is absent from consignee country and the complete address.
- `23e145de`, seven descendants: a matched-star country continuation is absent from the consignee country and address.
- `23a49094`, four descendants: consignee phone numbers printed only in the goods-area continuation remain in the target despite the agreed direct-block contact policy.

The 50 matching current synthetic counterparts already follow the corrected policy. An independent comparison of all old target field paths with the same-source current target shapes finds exactly these three patterns and 15 documents; after exclusion there are no remaining source-relative field-presence differences. This check complements inherited reviews; it does not prove every scalar interpretation through field presence alone.

[The private residue inventory](analysis/old-uae-reinforcement-20261014/known-residue-exclusions.json) and [label exclusion inventory](analysis/old-uae-reinforcement-20261014/label-defect-exclusions.json) contain exact IDs and evidence. These older records are preserved, not deleted or silently repaired in this selection task. The current retained 5,500 still contains 225 previously preserved non-target private echoes from the bounded 15-owner inventory; assembling this run does not claim complete historical anonymization.

## Selection and realized coverage

[The exact 1,500-member selection](analysis/old-uae-reinforcement-20261014/training1500-selection.json) pins every current row, target, plain OCR and positioned input. A binary optimizer enforces cardinality, coverage floors, exclusions and at least one descendant from every eligible old source. Its objective balances squared final source exposure, with stable hashes breaking ties. It is optimal for those explicit constraints, not a model-performance prediction.

Scarce or difficult cases comprise the union of thermal, DG, transshipment, reefer without setpoint, ventilation, eight-plus-container inventories, membership-only allocations, master B/L, longer postal targets, special equipment, and named package categories occurring in fewer than one percent of the current 5,500. This is a selection definition, not a labeling rule. Counts overlap and must not be added together to derive 519.

| Trait | Current 5500 | Selected addition | Final 7000 |
|---|---:|---:|---:|
| Thermal setpoints | 439 | 120 | 559 |
| DG declarations | 355 | 97 | 452 |
| Explicit transshipment | 113 | 48 | 161 |
| Multiple containers | 1,706 | 546 | 2,252 |
| Complete uniform multi-container inventories | 1,023 | 505 | 1,528 |
| Five or more containers | 581 | 201 | 782 |
| Eight or more containers | 244 | 107 | 351 |
| Membership-only allocations | 284 | 68 | 352 |
| Quantity-bearing allocations | 5,113 | 1,415 | 6,528 |
| Mixed assigned equipment | 617 | 0 | 617 |
| Negotiable consignee wording | 407 | 153 | 560 |
| Notify same-as reference | 1,404 | 325 | 1,729 |
| Explicit net weight | 1,090 | 407 | 1,497 |
| Master B/L | 36 | 25 | 61 |
| Ventilation | 63 | 29 | 92 |
| Reefer without numeric setpoint | 53 | 27 | 80 |
| Documents with an address of at least 20 words | 30 | 25 | 55 |

The additions include **937 single-container**, **546 multi-container**, and **17 no-container** documents. The no-container presentations are inherited accepted documents, not empty or fabricated inventories. Descriptions are 935 under 30 words, 506 at 30–59 words and 59 at 60 or more words. These are diagnostic bins; no wording was shortened or changed. **1,432 additions contain exact normalized description wording not already present in the current pool.** Literal novelty is distinct from a new commodity or layout category.

Final multi-container prevalence is **32.17%**, matching the real training reference of 32.17%. Negotiable prevalence is **8.00%** versus real 8.00%; transshipment is **2.30%** versus real 1.83%. Thermal and DG remain intentionally oversampled. The repeated source layouts are reinforced rather than represented as newly discovered layouts.

## Rare packaging and equipment

The nine newly added rare package categories already have explicit positive supervision: eight categories have 20 documents each and unpacked single units have 40; fibre drums have another six DG examples. The same ten compatible layouts support those categories. A source need not originally print that exact package type when its owned package fields, quantities, goods form and equipment remain compatible. Count-less bulk liquid is different: it requires a count-less presentation and must not receive an invented package count of one.

The older addition restores or strengthens familiar thin categories:

| Package category | Current | Added | Final |
|---|---:|---:|---:|
| Ordinary natural-wood boxes | 16 | 34 | 50 |
| Steel drums | 35 | 21 | 56 |
| Generic unpacked goods | 59 | 18 | 77 |
| Wooden crates | 14 | 6 | 20 |
| Reels | 4 | 3 | 7 |
| Slabs | 3 | 4 | 7 |
| Sacks | 10 | 0 | 10 |

Reels and slabs remain thin and do not gain new source layouts. The older pool cannot add positives for the newly covered nine categories. There is no blanket need to regenerate those good examples; future expansion can target more product forms, presentations and compatible allocation layouts if diagnostics show a weakness.

Owned-equipment inspection found three older examples with additional literal presentations, **`20'UT` and `20'UT FCL/FCL`**, absent from current OCR after casing and whitespace normalization. All three are selected, including a thirteen-container open-top inventory. They reinforce an existing canonical type rather than introduce a new category. [The owned alias evidence](analysis/old-uae-reinforcement-20261014/owned-alias-novelty.json) binds candidate ownership and current text; public tariff clauses are not counted as positive alias supervision.

The addition supplies 1,278 twenty-foot standard, 1,906 forty-foot high-cube, 99 forty-foot standard and 12 forty-five-foot high-cube container rows; it adds no twenty-foot high-cube rows. Current rare/mixed equipment is retained in absolute numbers, but diluted proportionally. This is a deliberate reinforcement of familiar inventory shapes, not another equipment-category rebalance.

## Coverage still requiring fresh examples

The older pool introduces no missing scalar field, finite category, package category or HS digit width. It supplies one additional broad HS chapter context, chapter 89, preserved in the selection. Exact HS-to-product taxonomy is not the extraction objective. Formatting and full code widths remain supported.

Four real-only supports remain outside this synthetic-only training set: original-B/L reference identifier, freight `payable_elsewhere`, DG primary `GASES`, and count-less bulk liquid. The older sidecar cannot close them. DG packing groups, flash points, subsidiary hazards and free-text package fallback remain at **138, 54, 22 and 25** positives respectively; older DG examples do not print these fields, so adding them would create unsupported labels.

Postal complexity and description distribution remain important future sampling dimensions. Only 55 final documents have an address of at least 20 words, compared with 23 of 600 real documents. Half the final synthetic descriptions are under 30 words versus 589 of 600 real descriptions. These counts identify distribution differences, not permission to truncate sound labels. The existing 60-document validation set contains only one of the nine newly covered rare package categories, unpacked single units; it cannot establish per-category learnability for all nine. A separate held-out diagnostic should preserve that distinction without changing the main holdout.

## Training configuration and validation

The [new configuration](../configs/training/production/t5gemma2_270m_lora.mpci_bl_synthetic7000_old_uae_reinforcement_r22_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml) retains rank 48, alpha 32, rsLoRA with EVA, LR 0.0001, ScheduleFree AdamW, microbatch 1, accumulation 32, ten epochs, input-only positioned OCR, compact direct JSON targets, case-insensitive/list-order-invariant scoring, all-checkpoint retention and best-checkpoint export.

The 7,000-row loader produces 219 updates per epoch and **2,190 total updates**. Warmup is **110 updates**, the rounded-up first five percent. Evaluation and saving use **548-update intervals**, approximately epochs 2.5, 5 and 7.5, with the configured final evaluation at 10. No new scheduling callback or optimizer change was added. The native Transformers startup calculation and complete configuration-difference check passed.

Dataset publication and independent readback verify exact original row bytes, all 7,060 direct targets, unchanged validation, zero orphan placements, no validation source/PDF leakage, all exclusions and unchanged inputs/labels/coordinates. **1,324 original protected files remain unchanged.** Final coordinate coverage is **83.7912%**, versus 83.9545% before, a composition change of 0.1633 percentage points; no known coordinate was removed or edited.

The rebuilt Compose training image passed `validate-config`, `inspect-dataset` and `prepare-dataset`. Tokenization retained all 7,000 training and 60 validation examples, with **zero truncations or exclusions**. Training maximum input/target lengths are **8,073/1,695 tokens**; validation maxima are **5,698/1,334**, within the unchanged 19,200/5,500 limits and validation generation cap of 3,072. These checks loaded only the tokenizer, not model weights.

Selection/coverage took 21.21 seconds with 811.10 MiB peak RSS. Publication took 19.81 seconds with 920.65 MiB peak RSS; independent readback took 12.23 seconds with 110.07 MiB. Native container inspection took 4.43 seconds and tokenizer preparation 47.58 seconds, excluding container startup. **50 targeted tests passed in 10.77 seconds; Ruff passed.** No production code or templates changed in this work. Paid API cost was zero.

Evidence: [coverage](analysis/old-uae-reinforcement-20261014/coverage-and-proposals.json), [publication](analysis/old-uae-reinforcement-20261014/assembly-publication.json), [independent readback](analysis/old-uae-reinforcement-20261014/assembly-independent-readback.json), and [native training readiness](analysis/old-uae-reinforcement-20261014/native-training-readiness.json).

The next run combines removal of real training examples, repaired labels/direct output formatting and the selected reinforcement. It can establish whether this practical dataset is stronger, but cannot isolate the effect of each change from one run. Keep rank unchanged for this comparison; rank 64 remains a separate later experiment.

## Launch command

Both the Docker flag and environment variable are supplied so the Compose build child also bypasses the broken credential helper for public images.

```bash
mpci_docker_config="$(mktemp -d /tmp/mpci-docker-public.XXXXXX)"

DOCKER_CONFIG="$mpci_docker_config" \
docker --config "$mpci_docker_config" compose --profile training run --rm --build kie-trainer train \
  --config configs/training/production/t5gemma2_270m_lora.mpci_bl_synthetic7000_old_uae_reinforcement_r22_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml \
  --project-root /workspace
```
