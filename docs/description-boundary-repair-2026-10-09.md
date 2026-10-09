# Main-product-passage description repair — 2026-10-09

## Outcome

The approved boundary refinement is applied in place to the current mixed dataset,
its real-data mirrors, and the 200-template synthesis catalog. Production field
descriptions, reviewer instructions and generation validation now reflect the
same policy. No training or paid generation/review calls were started.

| Cohort | Reviewed/replayed | Changed description labels | Unchanged |
|---|---:|---:|---:|
| Real training | 600 | 149 | 451 |
| Real validation | 60 | 14 | 46 |
| Synthetic training | 1,500 | 249 | 1,251 |
| Total unique documents | 2,160 | 412 | 1,748 |

All 200 templates compile and pass source-identity replay and fresh lexical
replacement tests. **33 template description contracts/targets change.** Two of
these also need narrower mutable product owners. Changed synthetic labels span
38 source families: source-level membership changes and generated-text-only
issues are not identical populations.

Dataset membership remains **2,100 training / 60 validation**. There are zero
changes to OCR, positioned inputs, coordinate values, party targets, equipment,
package counts/types, weights, HS/DG fields, goods grouping or container placements.

The current data is
[`mpci-bl-real600-synthetic1500-v7-positions-v1`](../data/curated/mpci-bl-real600-synthetic1500-v7-positions-v1/).
Its train SHA-256 is
`51e55c2a3ee859f8b3da474b96f727e42328c8ef24baae99c5c9ed1215bd88af`;
validation is
`e3295832abfdf5e0e202cb37e3a0cf9ca60479f9020a5a502ed38314614b0a4a`.

## Policy and what the user's clarification adds

The policy remains complete, faithful transcription of description-owned wording,
not a summary or removal of inconvenient numbers. The refinement establishes
**main passage first**, rather than accumulating all product-related facts from
elsewhere on the document.

1. Identify the main product passage. Start after headings and generic loading
   declarations; stop before independent accounting, tracking or documentary text.
2. Keep its complete product wording, including attached capacities, models,
   qualifiers and packing specifications. Preserve OCR punctuation; uppercase and
   turn line breaks into spaces.
3. Exclude separately owned Marks, references, invoice fields, independent totals,
   detached batch/serial tables, wood declarations and auxiliary packing equations.
4. Include another passage when it genuinely continues the product description:
   a later page, another product-list segment, or a description column interrupted
   in flattened OCR by other columns. A gap alone is not an error.
5. Do not alternate between optional inclusion/exclusion of attached qualifiers.
   The adopted convention is to keep them within the main passage, and avoid
   reaching out for detached auxiliary content.

This is documented in the
[central labeling policy](kie-real-baseline-label-policy-2026-10-05.md#boundary-clarification-main-product-passage-2026-10-09).
It is an annotation policy informed by the earlier MPCI/CUSCAR audit, not a claim
that an official standard dictates every OCR substring boundary.

After repair, **611/660 real descriptions have one contiguous selected OCR
passage**. The other 49 have justified continuations. The earlier receipt inventory
contained 515 single-fragment records and 145 multi-fragment records; some of the
latter merely separated whitespace-adjacent lines. Therefore that numerical
reduction includes both semantic boundary cleanup and harmless span coalescing.

## Diagnosis and repair mechanisms

### Real and validation sources

The previous broad description policy and its applied source selections sometimes
included loading introductions, detached packing equations or auxiliary facts.
Other examples already had correct, genuinely interrupted product descriptions.
Treating every gap or number as an error would damage the latter.

The repair inspected the full 660-description inventory, not just the original
screening flags, using its source selections, intervening OCR, previous layout
adjudications and the user's explicit decisions. All changes have source-hash-pinned
exact selections. This was not 660 new PDF reviews or an LLM blanket rewrite.

Examples:

| Real train row | Before / problem | Applied description treatment |
|---|---|---|
| 237 | `1 PACKAGE(S) 01 UNIT NEW RANGE ROVER AUTOBIOGRAPHY` | Start at `01 UNIT`; retain the product-unit wording |
| 240 and related cord documents | Product plus detached pallet/carton equation | Keep the main `AC POWER CORD` passage; do not append the equation |
| 278 | Detached `STAREX` included; product-block mass omitted | Copy `ABS INJECTION ... 16,000 KG`; omit detached `STAREX` |
| 363 | Two repeated product copies, packing declarations and grand total | `T850QVN04.2 85" ASSY OPEN CELL`, once |
| 368 | Roll-loading introduction and detached declarations | Keep `25,925 MTCHAMBRIL ... HEINZEL 1077669` intact |
| 422 | Potentially ambiguous numeric product prefix | Preserve `1012322163DXH DEGREE 1 ...`, as explicitly adjudicated |
| 423 | Leading loading count and detached serial/batch table | Keep the complete main milk-powder passage, including its embedded `TOTAL 1000 BAGS , 25 KG EACH ...` wording |

Product specifications remain protected: `3930X50 KG`, the blueberry
`13.61 KGS` capacity, model numbers, and true multi-page/column continuations are
not removed merely because they contain numbers or require multiple selections.

The complete changed-real review, with OCR/PDF links and before/after values, is
[real-changes.md](analysis/description-boundary-repair-20261009/real-changes.md).

### Templates and existing synthetic descendants

Source description membership is separate from mutable generation ownership.
Changing only a product string would not repair the definition of the target.
The approved source membership is instead projected through exact saved render
edits, preserving both generated wording and any retained host-owned facts.

Two mutable owners also crossed the newly approved boundary:

- Source row 236: product capacity and wood-treatment certification shared one
  generated region. The product owner is narrowed to `ORAFTI GR SACO 25 KG(1T)`;
  the certification remains outside its description target.
- Source row 333: the product owner included `LOT OF`. Both its curated owner and
  historical anchor now start at `USED MACHINES WITH ACCESSORIES`. The source
  introduction remains printed outside the product target.

Fourteen descendants of those two families needed explicit generated-region
membership review rather than cutting proportionally through old composite edits.
Seven additional generated descriptions contained standalone accounting wording:
four `TOTAL:` statements, two trailing bare carton-count lines and one prefixed
box-count introduction. Their primary product wording is retained; the separate
accounting/tracking tail or introduction is excluded. No OCR rewriting is needed.

Those **21 explicit descendant decisions** complement deterministic projection
for the other 1,479 descendants. All 1,500 receive exact rendered-span checks.
The complete changed-synthetic review is
[synthetic-changes.md](analysis/description-boundary-repair-20261009/synthetic-changes.md).

## Production changes and recurrence prevention

- `GoodsItemDetailsV7.description` defines the main passage, genuine continuations,
  excluded loading/detached passages and preservation of attached qualifiers.
  `CargoProduct.description` reuses that definition, so the cargo mapper agrees.
- Cargo reviewer scope and the direct reviewer/corrector prompts use the same
  boundary distinction. They retain the existing typed unresolved/ambiguous route
  with an explanation and recommended review action. No extra editing round or
  agent loop was added.
- Curated source-contract meanings, source review and rendered review now expressly
  include unresolved boundary questions. Such a finding is not a pass and must be
  adjudicated before publication. Existing hash-bound review/adjudication gating
  remains in force.
- The wording prompt begins with product wording and keeps loading/accounting
  statements outside generated product regions. A targeted generation guard now
  catches the seven observed accounting patterns, including `TOTAL: 491 CARTONS`
  and bare `688 CARTONS`, which the old `TOTAL PACKAGES <number>` check missed.
  It raises a validation error, entering the existing bounded repair/hold path;
  it does not silently delete generated text.
- The guard is not used to clean source labels and is not a number blacklist.
  It permits embedded capacities, numeric product codes, chemical properties,
  and product-internal uses of `TOTAL`. Semantic ambiguity remains review work.
- Source block declarations, target files, contract target pins, catalog hashes,
  dataset manifests, current schema artifacts and the **unrun** next-run recipes
  were refreshed together. Historical run recipes and outputs were not rewritten.

The prepared training recipe is
[the description-block rank-48 configuration](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic1500_descblocks_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml).
The prepared future synthesis recipe is
[description_blocks_next](../configs/synthesis/mpci_bl_curated_v7_description_blocks_next.yaml).
Neither was launched.

## Validation and falsification

The pass separates semantic adjudication from exact mechanical checks; substring
presence by itself is never treated as proof of correct field ownership.

| Check | Result |
|---|---|
| Full live mixed dataset: schema, membership/order, selected text and before/after scope | 2,160/2,160 pass |
| Real standalone label mirrors | 1,320 match their respective dataset targets |
| Template compilation + identity replay | 200/200 pass |
| Fresh deterministic product replacements through the actual renderer | 200/200 pass |
| Saved synthetic render projection / explicit reviewed selections | 1,500/1,500 pass |
| Actual synthesis constructor, source readiness and two scenario preflights per template | 200 sources / 400 scenarios pass |
| Actual training task/schema binding and data inspection | 2,100 train + 60 validation pass |
| Unstaged files checked against backup, including inputs/alignments | 3,464 unchanged |
| Deliberate invented/truncated descriptions, raw/positioned input edits and unrelated target mutation | All 5 rejected |
| Targeted production/extraction/synthesis test suite | 351 passed, 27.26 seconds |

The historical R14 mirror retains its frozen schema rather than having unrelated
modern fields added. Its old/new targets were validated against that frozen schema,
and the edited goods against the current goods model. The active R16/mixed data
passes the current full schema.

Staging safeguards caught two accidental truncated continuation selections and a
test-helper variable reuse that would have copied auxiliary dates into two catalog
targets. They were corrected **before publication**. The published independent
comparison confirms no such extra changes escaped. Publication is hash-preflighted,
uses atomic file replacement, and is followed by a separate live-file audit.

Evidence:

- [Live validation receipt](analysis/description-boundary-repair-20261009/validation-live.json)
- [Live training/synthesis startup probe](analysis/description-boundary-repair-20261009/runtime-live.json)
- [All source adjudications](analysis/description-boundary-repair-20261009/source-decisions.json)
- [All synthetic receipts](analysis/description-boundary-repair-20261009/synthetic-receipts.json)
- [Publication receipt](analysis/description-boundary-repair-20261009/publication-receipt.json)

## Performance, cost, preservation and remaining scope

The paired CPU benchmark used **1,985 actual saved generated regions**, seven
repetitions per version, through `validate_wording`. Median total time was
**91.1 ms before / 107.0 ms after**. The expanded guard adds about 8 microseconds
per region while rejecting the seven previously missed accounting cases.
Peak traced validator allocation was 22,129 / 22,337 bytes. Whole benchmark-process
peak RSS was approximately 821 MiB, including imports and loaded audit data.
This is a negligible per-request CPU cost, not a claim of zero validator overhead
or a measured end-to-end API speedup. No GPU benchmark was needed or run.

The template compilation/identity/fresh-replacement stage completed in about
20 seconds; independent live dataset validation took about 16 seconds. Live
training-data inspection plus 400 synthesis scenario preflights took about
18 seconds. **External API cost: $0.**

The exact pre-edit snapshot covers 5,196 files under
[`data/curated/backups/description-boundaries-20261009/`](../data/curated/backups/description-boundaries-20261009/).
The publication inventory binds before/after bytes for 1,737 checked files;
409 files actually change. Unchanged JSON artifacts retain their original bytes.
Input/coordinate and non-description checks are exact, not approximate similarity.

This description-boundary repair is complete. The **inner-versus-outer packaging
policy audit is a separate next pass**: no choice of package level, structured
quantity or placement was changed here. New training can measure the effect of
this label-only description change without conflating it with a packaging-policy
change, new synthetic data, different inputs or different hyperparameters.
