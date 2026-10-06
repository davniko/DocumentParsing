# PaddleOCR positional input experiment

The current 660-document training dataset has a separate positional-input variant. Its labels, original OCR, record order and 600/60 split are unchanged from R15. The new training input adds line-centre coordinates from the completed PaddleOCR run. This is an experimental representation; whether it improves extraction is to be measured by training, not inferred from alignment coverage.

## Data and representation

- Source: `data/curated/mpci-bl-real-v7-reviewed-r15-address-punctuation-660`.
- Variant: `data/curated/mpci-bl-real-v7-reviewed-r16-paddle-positions-660`.
- Paddle source: `artifacts/paddle-ocr/filtered-bl-swb-max5-1918-ppocr6-v1`.
- Build configuration: `configs/positional_inputs/real660_r15_paddle_v1.yaml`.

Each nonblank content line ends with ` || x,y`, using integer coordinates on a 0–1000 page grid, measured from the top-left. An unresolved line ends with ` ||`. Page markers and blank lines are unchanged. Original characters, punctuation, whitespace and line endings remain intact before the suffix.

These Paddle results contain text-region boxes, not individual word boxes. The position is the midpoint of the enclosing box of the matched complete regions. For a wide table row this is a coarse row centre, not the location of each individual value. No word positions are interpolated. Full regions and boxes remain available in alignment receipts.

The output separates `inputs/original/`, `inputs/positioned/` and `labels/`. JSONL records retain every original field and add `positionedText` and `positionedTextSha256`; the new training config explicitly selects these fields. Label files are byte-identical copies. `alignments/` records assignments, matching methods, context anchors and unresolved reasons per source line.

## Alignment and ambiguity handling

PDF identity is joined through the source-selection manifest and SHA-256, then matched page-locally. Source record hashes, Paddle artifact hashes, page counts, coordinates and label schema are checked before publication.

Alignment uses whole-region equality after Unicode/case/whitespace normalization, then punctuation-normalized equality. A line spanning several regions requires complete matching text along a spatially neighboring path. Another column interleaved in Paddle's array does not force an incorrect grouping.

Repeated text requires nearby, uniquely matched exact anchors in the same source paragraph. Contextual matches never become anchors for further contextual inference. Competing paths, insufficient separation between repeated candidates, overlapping claims, substrings of larger boxes and OCR wording differences remain explicitly unresolved. Search limits also produce an unresolved result, never acceptance of an incomplete search. Gold labels are not used to choose positions.

All geometry tolerances and search limits are in the build configuration. Geometry tolerances use line heights rather than fixed pixels. A new dataset is staged and published by directory rename only after persistence checks pass; the builder refuses to overwrite an existing dataset.

## Measured publication and validation

The 660 documents cover 1,271 pages and 67,346 nonblank content lines:

| Outcome | Lines |
| --- | ---: |
| Unique complete text match | 45,129 |
| Unique punctuation-normalized match | 1,775 |
| Complete spatial group | 7,333 |
| Repeated text resolved using local anchors | 2,963 |
| Positioned total | **57,200 (84.9%)** |
| Unresolved total | **10,146 (15.1%)** |

Unresolved reasons: 4,922 lack a complete text match; 4,152 have ambiguous repeated text; 445 have competing groups; 443 are only substrings of larger regions; 128 have competing region claims; 56 contain no alphanumeric text. None of these lines or their documents was removed. This coverage does not represent a measured spatial accuracy rate.

Publication took 29.631 seconds with peak process RSS 526.1 MiB. Independent persisted-data verification checked every source/label byte, suffix, bbox union, coordinate and assignment, including a full replay with the final implementation: all passed. Pure alignment replay took 1.038 seconds; the full verification took 11.075 seconds. Later typing-only changes in the publisher are recorded separately by the verification report's implementation hashes; alignment outputs were unchanged.

Review included 48 stratified text/geometry receipts, all 12 selected repeated-text anchor neighborhoods, and summaries of 12 extreme-width/long-group cases. Three selected page rasters were inspected visually: repeated consignee/notify identity, repeated continuation email, and a table row spanning equipment, packages, mass and volume. No wrong-region assignment was found in these reviewed examples. This is sampled spatial review, distinct from the exhaustive byte/geometry checks; it does not prove every contextual assignment semantically correct.

Tests passed: 32 tests across positional alignment and Paddle integration; a subsequent 50-test run across positional alignment, training-data handling and V7 extraction also passed. Those totals overlap. Ruff and focused type checking passed. The training-data tests emitted existing multiprocessing `fork()` deprecation warnings. Tests cover repeated text, column interleaving, conflicting claims, partial regions, changed identifiers, search limits, page identity, resolution invariance and reversible suffixes.

Detailed receipts and verification script: `docs/analysis/real660-position-enrichment-20261006/`. The analysis directory remains gitignored, following repository policy.

## Training configuration and execution

The new configuration is:

`configs/training/production/t5gemma2_270m_lora.mpci_bl_real660_r16_positions_v7_e10_compact_eva_a32_r32_local_schedulefree_v1.yaml`

It retains the completed R14 compact run's optimization and generation settings: rank 32, alpha 32, EVA, ScheduleFree AdamW, gradient checkpointing, local microbatch 1 and accumulation 32, 10 epochs. Evaluation and checkpointing are at steps 95 and 190 (epochs 5 and 10). The prompt adds a short explanation of coordinate suffixes; the output schema remains unchanged. The training constraint artifact uses the trainer's required canonical JSON encoding.

The rebuilt Docker training image successfully prepared all 600 training and 60 validation records. No source truncation or target filtering occurred. Maximum source length is 10,283 tokens in train and 8,108 in validation, below the unchanged 19,200 limit. Maximum compact targets are 3,134 and 1,333 tokens including EOS. Initial preparation took 38.219 seconds; cached revalidation took 12.520 seconds. No training or model inference was launched and no paid API calls were made.

Run training from the repository root:

```bash
docker compose --profile training run --rm --build kie-trainer train \
  --config configs/training/production/t5gemma2_270m_lora.mpci_bl_real660_r16_positions_v7_e10_compact_eva_a32_r32_local_schedulefree_v1.yaml \
  --project-root /workspace
```

Important comparison caveat: the completed baseline run used R14 labels, while this variant preserves the latest R15 labels, including the intervening address-punctuation changes. A comparison against that historical run is therefore not purely positional. A controlled positional ablation should use R15 labels in both arms. Spatial suffixes also increase source token counts; GPU training throughput and peak VRAM have not been benchmarked by this preparation-only task.

To build another variant, edit a copy of the build configuration with pinned source hashes and a new destination, then run:

```bash
.venv/bin/python -m document_ocr.spatial_inputs \
  --config configs/positional_inputs/real660_r15_paddle_v1.yaml \
  --project-root .
```

Running that exact build configuration again intentionally refuses to replace the already-published R16 dataset.
