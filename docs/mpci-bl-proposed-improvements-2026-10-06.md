# MPCI Bill-of-Lading extraction: proposed improvements and experiment baseline

Date: 2026-10-06.

Status: **proposals and measured baseline, not authorization to change the dataset during the compact-versus-pretty JSON experiment.** The immediate experiment uses the latest R14 dataset, including uppercase normalization, completed container classification, and the source-specific 40EC → 40HC adjudication. No new address/description separator changes are applied here.

## 1. Objective and current decision

The objective remains field F1 and field accuracy both above 0.90, ideally at least 0.95, on the complete intended extraction task. Strong performance on selected stable fields does not establish full-target performance.

The next development directions are:

1. **Expand training data through audited augmentation and synthesis**, with coverage targeted at observed extraction weaknesses.
2. **Systematize formatting of addresses and descriptions**, particularly inserted comma/semicolon conventions, without changing or losing source-supported information.
3. Continue separating JSON generation failures from content extraction failures when judging experiments.

First, run the planned ten-epoch compact-JSON experiment without introducing additional data, label, optimizer, or prompt changes. Use its matched R14 pretty-JSON config for a controlled formatting comparison. Comparing only against the older 20-epoch R12 run is informative but confounds formatting with updated annotations, epoch budget and generation settings.

## 2. Evidence: where performance is already strong

The detailed source of these results is the [full-target audit](analysis/real660-fulltarget-audit-20261006/REPORT.md). Predictions are from the real-only model's final checkpoint 380 and the earlier 30k model's final checkpoint 3220.

### 2.1 Controlled comparison on unchanged annotations

| Scope, case-insensitive | Current 600-training model F1 / accuracy | Earlier 30k model F1 / accuracy |
|---|---:|---:|
| All 52 shared validation inputs, unchanged components | 0.893 / 0.819 | 0.964 / 0.942 |
| Original 47 current-parseable inputs | 0.959 / 0.938 | 0.967 / 0.947 |
| 46 inputs where both models' outputs parse | **0.958 / 0.937** | **0.980 / 0.972** |

The original 47-document cohort was conditional only on the current model parsing; one older output did not parse. The both-parse row is the corrected strictly matched parseability comparison.

The original unchanged-component mask retained **1,001/1,928 scalar values (51.9%)**. It excluded all 140 complete addresses, retained descriptions in only three documents, and retained only 4/91 goods-to-container placement identifiers and 4/69 allocation quantities. Thus this is strong evidence for extraction of many stable facts, not for complete addresses, descriptions or relation coverage.

### 2.2 Full-target results

These include every field in each run's historical target. The older target has address/city/country rather than today's complete addressLine and has several since-removed fields. Cross-run numbers are therefore not a pure comparison on identical labels.

| Population | Ignore case: F1 / accuracy | Also ignore commas/whitespace in human-readable text: F1 / accuracy |
|---|---:|---:|
| Current, all 60 validation documents | 0.812 / 0.728 | 0.825 / 0.739 |
| Current, 54 parseable outputs | **0.865 / 0.818** | **0.879 / 0.831** |
| Earlier, all 100 validation documents | 0.860 / 0.796 | 0.862 / 0.798 |
| Earlier, 99 parseable outputs | **0.865 / 0.804** | **0.867 / 0.807** |

With case/comma/whitespace tolerance, **23/60 current** and **29/100 earlier** documents exceed both 0.90 thresholds. On the 52 common inputs, ten pass that threshold in both runs, ten only in the current run, five only in the earlier run, and 27 in neither. Exact membership is recorded in the [document score inventory](analysis/real660-fulltarget-audit-20261006/COHORTS.md).

These are outcome-selected cohorts. They demonstrate successful complete-target extraction on some documents, not a validated rule for deciding in advance which documents will be safe. A long description is only one scalar field; nine of the 23 strong current documents have every address and description correct under this matching rule.

### 2.3 Granular current strengths and weaknesses

Current model, 54 parseable outputs, case/comma/whitespace tolerance:

| Field | F1 | Accuracy | Development implication |
|---|---:|---:|---|
| B/L number | 0.968 | 0.957 | Preserve established identifier behavior |
| Vessel name | 0.981 | 0.981 | Strong baseline |
| On-board date | 0.987 | 0.974 | Strong baseline |
| Freight payment arrangement | 0.989 | 0.978 | Strong baseline |
| Negotiability | 0.981 | 0.963 | Strong baseline |
| Container identifier | 0.959 | 0.922 | Strong overall, but repeated-list omissions remain |
| HS codes | 0.944 | 0.894 | Near target; retain multiple supported codes |
| Complete addressLine | 0.763 | 0.747 | Formatting plus boundary/ownership mistakes |
| Goods description | 0.481 | 0.481 | Completeness and semantic boundaries need attention |
| Split placement package quantity | 0.807 | 0.683 | Identifier/quantity role confusion and missing allocations |
| Gross weight value | 0.736 | 0.696 | Scope and numeric interpretation |
| Package type | 0.854 | 0.815 | Category/packaging-level interpretation |
| Handling instructions | 0.556 | 0.435 | Sparse or difficult instruction extraction |

The latest container labels were not used to train this evaluated model. Rescoring its saved predictions against R14 yields 0.882 F1 / 0.837 accuracy on parseable outputs with formatting tolerance; that is a changed-gold diagnostic, not a newly improved trained model.

## 3. Proposed data expansion: augmentation and synthesis

The ten-container failure is consistent with insufficient training coverage, and the earlier model's successful extraction supports testing broader training exposure. It does **not**, by itself, isolate dataset size as the cause: the runs differ in data composition, labels and training setup. Data expansion is a grounded experiment, not a guaranteed cure.

### Targeted coverage

- Repeated equipment lists spanning continuation pages; vary container count, including larger lists, without losing later rows.
- Single goods items placed across multiple containers, including membership-only placements when per-container quantities are not printed.
- Distinguish seals, equipment digits, package counts and masses in adjacent table rows.
- Long product-owned descriptions containing models, grades, specifications, lots and package capacities, excluding payment/customs/shipment-total material.
- Different postal-address layouts and interspersed non-address lines, while retaining one fixed target formatting convention.
- Rare but useful categories already resolved in the real dataset; preserve instruction/setpoint support rather than inventing thermal settings from equipment type.

### Constraints for the first synthesis expansion

1. Start from a small, explicitly audited subset of the **new real sources and labels**. Do not reuse the entire old template catalog merely because it compiles.
2. Review the exact fields and relationships that a selected template can render. Rendered values and labels must come from the same sampled entities and allocations.
3. Keep the validation split and its source families out of training augmentation. Track source identity and near-duplicate lineage so expanded data does not turn the validation comparison into a memorization test.
4. Use source-preserving formatting augmentation where it is unambiguous: line wrapping/spacing and presentation variations with unchanged meaning. Do not reorder ownership-bearing blocks or introduce semantic ambiguity as an unexamined augmentation.
5. Check complete equipment-list cardinality, seals/identifiers, source-supported quantity placement and full product content on rendered examples. A valid JSON schema alone does not establish these facts.
6. Add training data in a measured stage with a frozen validation set and the same evaluation rules. Report changes in JSON validity, complete-target scores, long-text performance and multi-container recall, not only aggregate loss.

This section is a proposal. No augmentation, regeneration, compilation, or paid synthesis was started.

## 4. Address and goods formatting: systematize before further label rewriting

### Measured state

- Current R14 contains **2,034 addressLine labels and 660 descriptions**, all uppercase, without embedded newlines, tabs or repeated spaces.
- Casing intentionally normalizes OCR to uppercase. It is not meant to reproduce each source's mixed casing.
- Identical full OCR address blocks have different comma placement in **20 groups / 71 labels / 48 documents**.
- **25 descriptions contain semicolons**. Exact local OCR witnesses identify **86 inserted semicolon separators in nine documents**, despite the current description rule saying to join lines with spaces. Other boundaries remain unclassified or may be printed punctuation; do not blanket-replace them.
- Ignoring address commas/whitespace improves current address F1 from **0.587 to 0.763**, recovering 28 exact matches. Description F1 stays **0.481** under the same tolerance, so description content failures must not be dismissed as formatting.

### Proposed convention choices to discuss after the experiment

For addresses, the instruction to insert commas between vaguely defined semantic components makes the model decide whether, for example, a postcode deserves another comma. Two plausible conventions are:

1. Preserve source punctuation and join ordinary OCR line wraps with spaces; avoid requiring new semantic comma boundaries.
2. Use comma-space only at explicitly established address-component boundaries, with a deterministic join rule and identical treatment for equivalent source blocks.

The first reduces the extra annotation convention, but line wraps can split words or identifiers and still require correct rejoining. The second keeps visually structured address lines but requires reliable boundaries. Neither proposal authorizes deleting printed commas inside numerical expressions, address numbers, model codes or specifications.

For descriptions, prefer the existing rule: complete product-owned wording, source order, ordinary line breaks joined with spaces, and printed punctuation retained. Product lists must not arbitrarily alternate between inserted semicolons and spaces. A separator-only repair should preserve the same ordered content and numbers; it must not regenerate the description or redistribute goods.

The decision should be applied consistently to field descriptions, labeling finalization, dataset targets and diagnostic metrics. Keep exact scoring alongside a clearly identified text-format-tolerant view. Do not globally erase punctuation from identifiers or numeric/specification content.

**Timing:** defer these changes until the current compact/pretty experiment is finished. They are not part of today's training-start correction.

## 5. Concrete error clarification: seals predicted as package quantities

Document `doc_13ea92fad1244f29e6f7e2e71d788f348b757360ed1b40574a31f6336131bbdf`.

This is an error in **model-generated JSON**, not an OCR declaration and not the current label. The [saved prediction](../artifacts/kie-training/analysis/real660-vs-30k-20261006/predictions/validation.jsonl:1) contains this excerpt at `documentPatch.goodsItemDetails[0].splitGoodsPlacement`:

```json
[
  {"equipmentIdentifier": "CMAU3038703", "packageQuantity": 14865},
  {"equipmentIdentifier": "CMAU2830803", "packageQuantity": 14888},
  {"equipmentIdentifier": "CMAU2325606", "packageQuantity": 8}
]
```

The [OCR](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_13ea92fad1244f29e6f7e2e71d788f348b757360ed1b40574a31f6336131bbdf/ocr.txt:63) prints `SEAL 14865`, `SEAL 14888` and, at lines 85–86, `CMAU2325606 / 1 x 20ST 8 PALLETS`. The model copied the first two seals into the quantity field. It also emits other unsupported allocation quantities, including `1632452`, taken from an equipment-number suffix.

The [current gold label](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_13ea92fad1244f29e6f7e2e71d788f348b757360ed1b40574a31f6336131bbdf/labels.json) leaves quantities unset for containers without printed quantities and retains `8` for `CMAU2325606`. The OCR separately prints **TOTAL NO OF PACKAGES: 54** at line 261; that is a goods-level total, not evidence to populate every container's package quantity. The prediction also calls the total 54 pallets, whereas the gold correctly uses packages.

This illustrates the desired synthesis coverage: many neighboring numbers are present, but only quantities owned by the corresponding container should be allocated to it.

## 6. Latest-dataset training startup: diagnosis and verified command

The failing command selected the old config without `r14` in its filename. That config pins R13 and the R13 task contract. Its target-schema hash is `e4efe1beab6ba86dbbc5a56f39872e422eee5e81d3963cdd88b2bb2698de11bc`, while the current code has `041475bda7cef5b37056bfefe352d936baeb073d13cfa7371e2f54f1a3c348b6`. The strict startup check is working correctly. The Docker build succeeded; weakening or bypassing this check is not needed.

The existing R14 configs already match the current schema, category vocabulary and latest dataset hashes:

- Training: `be20e81e9052702daeaac000914b0dfe43cc053e2fec0a9a4a7d650647a561fe` — 600 records.
- Validation: `e421d1ac7b98da0695d9ab1f9472e7c72a53c6ec65ab8a9d4d308d580786b400` — 60 records.
- Task-constraint artifact: `9f56457ba2e44fadcabd01f60e5ffe501479be80f8ea07bfd0373a00a5ccef5f`.

The R14 record for `CMAU7204660` was explicitly checked: it now has `FORTY_FOOT_HIGH_CUBE` / `GENERAL_PURPOSE`, reflecting the latest 40EC source adjudication.

Both existing R14 configs passed `inspect-dataset` through the actual `kie-trainer` Docker image and `/workspace` mount. All 660 rows, pinned input files, target validation, constraint binding and rendered prompt loaded successfully. Inspection took 0.329 seconds for compact and 0.360 seconds for pretty, excluding container startup. No model weights were loaded and no training run was launched.

The configs differ only in `target_format`, run ID, adapter name and the format tracking tag. They share the dataset, prompt, seed, optimizer, ten-epoch budget, source/target limits and generation budget. Evaluation/checkpoint steps are 95 and 190, approximately epochs five and ten. Their output directories did not yet exist at validation time.

Run compact JSON:

```bash
docker compose --profile training run --rm --build kie-trainer train \
  --config configs/training/production/t5gemma2_270m_lora.mpci_bl_real660_r14_reduced_v7_e10_compact_eva_a32_r32_local_schedulefree_v1.yaml \
  --project-root /workspace
```

The matched pretty arm is `configs/training/production/t5gemma2_270m_lora.mpci_bl_real660_r14_reduced_v7_e10_pretty_eva_a32_r32_local_schedulefree_v1.yaml`. Do not use the non-R14 config for this latest-dataset experiment.

No dataset or config edits were needed for this diagnosis. The existing corrected R14 pair is the intended launch path.
