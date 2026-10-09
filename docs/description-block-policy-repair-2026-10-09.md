# Complete goods-description policy repair — 2026-10-09

Follow-up: the user subsequently clarified main-product-passage boundaries.
The [boundary-refinement repair](description-boundary-repair-2026-10-09.md) applies
that decision on top of this pass; this document retains the earlier repair receipt.

## Scope and decision

This pass applies the approved [description policy](kie-real-baseline-label-policy-2026-10-05.md#current-decision-description-block-copying-2026-10-09)
to the current 600-real + 1,500-synthetic training dataset, its 60 real validation
records, and all 200 current synthesis templates. It does not relabel historical
training runs or regenerate complete samples.

Copy complete actual goods-description wording, including attached packing,
counts, capacities, weights, product codes, qualifiers and origin. Exclude
independently owned Marks, documentary references, standalone accounting totals,
equipment/party/carrier fields and separate HS/DG code declarations. Preserve
source order and punctuation; apply uppercase and whitespace normalization only.
Physical proximity or product relevance is not sufficient to transfer Marks into
description. Conversely, a broad table heading does not establish a reference
boundary inside an otherwise unseparated product line.

The final user adjudication keeps **`1012322163DXH DEGREE 1`** in train422. The
earlier proposed omission was superseded before publication. A full 660-source
follow-up found no other equivalent purchase-order/description-header case;
explicitly captioned PO references remain excluded.

## What was wrong

Three layers contributed to the inconsistency:

1. Some real labels summarized product identity, removed numbers from attached
   packing phrases, rearranged text or imported product-like material from Marks.
2. Some template lexical regions inherited those boundaries. The numeric renderer
   could correctly vary a packing quantity while the target expression still
   emitted only the generated product wording, omitting the surrounding phrase.
3. The shared cargo-string validator rejected any `STC` substring, apart from a
   narrow containment pattern. It consequently rejected genuine descriptions such
   as `4 PALLETS STC LUBRICATING OILS NON HAZARDOUS`.

The preceding [scope audit](analysis/description-block-policy-audit-20261009/REPORT.md)
is preserved. Its flagged subset was not treated as the complete repair scope:
all 660 real inputs were read, with targeted PDF ownership review, followed by
complete deterministic replay of all 1,500 descendants.

## Dataset results

**Published and independently validated.** The live-file validation passes all
2,160 active records, all source/render projections, 660 alignments and 4,520
publication hashes in **28.05 seconds**. Exactly 673 files changed during
publication. The complete 5,196-file pre-edit backup was reverified afterward.
There are no remaining held description decisions or failed template replays.

[Post-publication validation](analysis/description-policy-repair-20261009/validation-live.json)
records the checks and deliberately rejected corruptions.

| Cohort | Reviewed/replayed | Description changes | Input changes | Records changed in either way |
|---|---:|---:|---:|---:|
| Real training | 600 | 206 | 3 | See per-record receipts |
| Real validation | 60 | 29 | 2 | See per-record receipts |
| Real total | 660 | 235 | 5 | 237 |
| Synthetic | 1,500 | 310 | 22 | 330 |
| Total | 2,160 | 545 | 27 | 567 |

Membership, IDs, row order and splits remain unchanged. No goods grouping,
package accounting, allocation, HS/DG, party, route, equipment or other target
value is changed by this pass. There is no filtering or whole-sample regeneration.

Examples:

- `RED SPLIT LENTILS ... PACKED IN 50 KG BAGS` becomes the actual source wording
  `RED SPLIT LENTILS CROP: 2022, ORIGIN: CANADA PACKED IN 3930X50 KG BAGS IN 8X20' FCL`.
- `EXTRACORPOREAL TUBING SET ADULT` retains its printed `-EGYPT` suffix.
- Glycerine label wording taken from the Marks block is replaced by the actual
  goods-column wording: `REFINED GLYCERINE 99.5% G995E (FOOD GRADE) IN 250KG METAL DRUM ON PALLETS.`
- Standalone Marks/VIN/reference text is no longer appended merely because it is
  product-related.

The previous 500 synthetic samples are covered inside the current 1,500: **110**
of those records change, including **107 description changes and five input
changes**. Historical 500-sample dataset exports and campaign artifacts remain
historical; the current combined dataset contains the repaired versions.

### Five real OCR recoveries

The complete description occurrence was missing or incomplete in the primary
OCR for these sources. Existing saved PaddleOCR supplied the exact missing words
and measured bboxes; PDF inspection established the correct column. No text was
invented from a PDF-only reading.

| Split/row | Source | Recovery |
|---|---|---|
| Train18 | `e9275395` | Actual goods-column `RATHIPON BROWN 6RL`, distinct from the Marks copy |
| Train266 | `f7e44d6c` | Lamp-holder specification, code, model and marking; existing brand retained once |
| Train446 | `2b88d69f` | Glycerine goods-column wording and packing continuation |
| Validation42 | `13ea92fa` | Central refractory-material / mould-powder occurrence |
| Validation43 | `75006aac` | Actual `SPARE PARTS`, rather than the separate Marks typo |

Eleven lines are inserted. Exact inverse replay preserves all pre-existing raw
characters and coordinates. Alignment line indices and anchor references are
updated, and the new records retain saved OCR-region provenance. These are narrow
recoveries, not a claim that all OCR omissions throughout these PDFs were repaired.

[Recovery evidence and tests](analysis/description-policy-repair-20261009/input-recovery/README.md).

## Template corrections and existing descendants

All **200** templates receive reviewed, source-hash-pinned description-block
declarations. **43** source-template description targets change. All 200 compile
and pass their applicable identity checks; all 1,500 saved descendants replay.

There are three input-changing source families:

- **`56e593ae` — eight descendants:** narrow the product region that incorrectly
  extended through a separate Marks block. Restore that block independently.
  Two old generated values had explicitly identified reference/marking tails;
  remove those exact tails with receipts, keeping their complete product prefixes.
  The other six generated product values remain intact.
- **`e9275395` — seven descendants:** insert the recovered true goods occurrence
  and bind it to the same generated product as its repeated printed copy. Do not
  use the Marks occurrence as evidence for a goods-column label.
- **`cc1f69ab` — seven descendants:** remove the fixed source-only `211615MTS`
  textile quantity from synthetic documents whose goods have changed. Authentic
  real OCR remains unchanged. No mass/unit interpretation is invented.

All **1,478 other synthetic raw inputs are byte-identical** to their saved
candidate inputs. All non-description target values remain identical.

[Template migration, exact receipts and benchmark details](analysis/description-policy-repair-20261009/template-migration.md).

## Future extraction and synthesis

- The V7 description field, cargo-map field and scoped review/correction
  instructions share the complete-copy policy. Numeric wording is distinguished
  from structured accounting, so retaining a number does not create a package
  allocation or change a mass label.
- Description validation rejects known whole-value disclaimers, but does not use
  an `STC` substring as proof that an entire product phrase is boilerplate. Other
  cargo fields and historical target models retain their existing behavior.
- A template's **mutable product region** and its **complete label-description
  region** are separate concepts. Reviewed description spans can include both
  generated product wording and host-rendered quantities.
- After rendering, the target is reconstructed from those complete approved
  regions through the exact byte edits. The compiler rejects stale source hashes,
  incomplete description coverage, overlapping ownership, cross-boundary edits,
  or product owners with no approved goods-description occurrence.
- Candidate validation independently compares rendered text, target and proof.
  Source readiness is checked before concurrent paid work begins.
- The older standalone curated path is also covered: it requires the same
  reviewed declarations and cannot bypass the policy through lexical expressions.
  Its compile/review/generate/validate/publish paths reject missing declarations.

This is a deterministic enforcement of reviewed ownership, not an automated claim
to infer arbitrary document semantics perfectly. New templates require explicit
description ownership review; missing declarations cannot silently pass.

## Geometry and preservation

Only the 22 changed synthetic inputs require coordinate rerendering. Their runs
have no rejected geometry pages. The recovered RATHIPON descriptions have
coordinates on all 45 generated description lines. For the corrected `56e593ae`
region, 43 generated description lines remain explicitly unpositioned: its true
source anchors overlap the measured caption envelope. The former broader
Marks-plus-goods region was not valid justification for coordinates. Collision
checks were not weakened and no text was discarded to obtain coverage.

Current synthetic totals are **154,488 nonempty content lines, 129,041 with
coordinates**, versus 154,424 / 128,977 before repair. Individual unknown-position
markers are retained as ` ||`.

## Validation, cost and runtime

- 660 explicit source decisions, verified hashes and exact ordered-span replay.
- 200 template compiles and 1,500 full saved-scenario replays, with no unresolved
  source decisions or replay failures.
- **564 relevant production tests passed in 29.29 seconds** in the final combined suite, including
  the actual mocked PydanticAI extraction path and the standalone synthesis guard.
- Independent publication validation checks exact edit scope, target equality
  outside descriptions, every real alignment, split integrity, mirrors and saved
  synthetic proofs. Negative mutations test dropped words, altered numbers,
  imported Marks, unrelated facts and unrelated coordinates.
- The complete independent staged pass checks **4,520 publication hashes**, all
  660 alignment files and all 2,160 active records under strict current V7 in
  approximately **33 seconds**. All five deliberate corruption probes are
  rejected; its own 12 focused tests pass.
- Recovery tests: 30 passed. Template-migration helper tests: 13 passed.
- Main renderer benchmark: **1.331 → 1.392 ms/document**, approximately 3.1 KiB
  additional traced peak allocation. Standalone renderer: **0.402 → 0.430 ms**,
  approximately 2 KiB additional peak allocation. No extra generation calls.
- Description-only validator on the same 660 original values:
  **0.769 → 0.112 microseconds/value**; traced peak 1,190 → 1,296 bytes.
- Final 200-template / 1,500-descendant staging replay: approximately **41 seconds**.
- **External model/OCR API cost: $0.** Saved generation, OCR and geometry were reused.

The complete pre-edit copy contains **5,196 verified files**:
`data/curated/backups/description-policy-20261009/`.
Per-document before/after values, source spans and input-recovery references are
in [dataset receipts](analysis/description-policy-repair-20261009/dataset-receipts.json).

## Experiment interpretation

The new descriptions intentionally differ from historical gold. Twenty-nine
validation descriptions and two validation inputs change. Do not attribute an
old-score/new-score difference solely to model learning without accounting for
these changes. The backup preserves the exact previous evaluation contract/data.

Future-run contracts/configuration use a new description-policy identity while
retaining the existing category vocabularies and training settings. Historical
configs, checkpoints, prediction files and run metrics are not rewritten.
No training run is started by this repair.

### Ready-to-use configuration and preflight

[Full readiness receipt and reproducible commands](analysis/description-policy-repair-20261009/FUTURE-RUN-READINESS.md).

- [Training configuration](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic1500_descblocks_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml):
  same rank-48 experiment settings, with the repaired dataset and a fresh schema
  contract. CLI configuration validation passes; dataset inspection accepts
  all 2,100 training and 60 validation records in 1.78 seconds.
- [Next synthesis recipe](../configs/synthesis/mpci_bl_curated_v7_description_blocks_next.yaml):
  all 200 registered sources pass production source-readiness checks in
  8.09 seconds. This is a preflight only; no descendants were generated or charged.
- An offline check using the previous run's saved tokenizer and production
  tokenization path finds **zero input or training-target overflows** across
  all 2,160 records. Maximum input lengths are 8,066 training / 5,698 validation;
  maximum target lengths including EOS are 3,139 / 1,356. Tokenization took
  0.862 seconds, with no network access or model/GPU loading.
- The unchanged inference cap of 3,072 tokens covers every validation gold
  target, but three training-only targets are longer (3,139, 3,139 and 3,094).
  Training retains those targets in full. For inference over similarly long
  documents, increase the generation cap; it was not silently changed in this
  controlled-comparison recipe.

### Historical source-mirror compatibility

The r14 source mirror predates the required nullable negotiability/notify
defaults introduced later. Its 660 records pass its preserved frozen schema;
467 are not directly compatible with today's complete V7 schema. This is
unchanged from the backup, not a regression or an active-training-set defect.
The repair validates its changed goods against today's goods model, checks its
full historical contract and proves its unrelated fields and current-schema
error sets unchanged. **All 2,160 active r16/combined records pass today's full
strict schema.** This pass does not silently migrate historical party defaults.
