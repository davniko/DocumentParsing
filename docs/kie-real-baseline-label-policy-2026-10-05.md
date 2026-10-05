# Real baseline: goods identity and field-scope policy

Recorded 2026-10-05 from the user's dataset-cleansing decisions. This is a
policy/reference note, not an instruction to relabel historical datasets or launch
another labeling campaign. The current selected dataset is R10 (474 records), documented in the
[cleansing and repetition report](analysis/real-v7-starting-dataset-2026-10-04/R10_CLEANSING_2026-10-05.md).
The preceding [R7 audit](analysis/real-v7-starting-dataset-2026-10-04/R7_FIELD_AUDIT_2026-10-05.md)
is preserved as historical evidence. Subsequent R10 exclusions do not change the
goods or address target policy. Independently printed telephone numbers remain
extractable; standalone abbreviated suffixes/extensions are not separate numbers
and must not be expanded by guessing missing prefixes.

## Goods identity: accounting, not number of product names

The baseline currently retains documents with one labeled goods entry. A long
description, several product names, several models or several HS codes does not
by itself establish multiple goods entries. Multiple HS codes are allowed.

For later labeling, examine how the document accounts for the cargo:

- Distinct product-owned package quantities, gross weights and/or volumes are
  evidence for independently accounted goods. Read the complete document and
  attachments before deciding ownership.
- Different container rows can carry portions of the same goods. Their local
  quantities/weights/volumes are container splits, not automatically new goods.
- A single shared package total, gross weight and volume, with several product
  names described together and no separate product accounting elsewhere, supports
  one goods entry with a complete description.
- A grand-total line alone does not erase independently quantified product rows
  elsewhere. Conversely, separately counted outer and inner packaging levels do
  not create different goods identities.
- Repeated descriptions on multiple pages or container rows do not add goods.
  Ambiguous accounting is reviewed; do not split or merge on keyword counts.

Examples of the distinction:

| Source pattern | Interpretation |
|---|---|
| Product A/B/C listed together; one shared 35-package total, with 9/16/10 packages assigned to three containers | One jointly accounted goods entry; three placements, subject to the complete source context |
| Same product in two containers, 100 packages and a local weight printed for each | One goods entry with two quantified placements, not two goods because two weights print |
| Product A explicitly has 551 cartons and product B explicitly has 280 cartons | Evidence for separate goods accounting; a shipment grand total does not make these independently quantified rows disappear |
| Several products/HS codes, but no product-specific package, mass or volume allocation | Multiple names/codes alone are insufficient to split goods |

These are semantic examples, not automatic rewrite rules. No new multi-goods
adjudication was performed in the R7 filtering pass.

## Product wording and additional information

Product identity, brand, model/article codes, composition, specifications,
condition, lot qualifiers, proper shipping names and explicitly printed package
capacity belong in `goodsItemDetails[].description`, in source order. Structured
shipment quantities, HS/DG codes and unrelated references are separate facts.

There is no agreed independent purpose for a cargo product-overflow field.
`additionalInformation` / `additionalGoods` are absent from the V7 model and the
current 500 targets. Do not reintroduce them to shorten long product descriptions.
Goods-specific handling instructions have a separate, defined purpose; they are
not a renamed overflow field for product wording.

## Approved reduced field scope

The user approved removing `transport.vesselFlagCountry`,
`goodsItemDetails[].marksAndNumbers` and `forwardingAndExportReferences` from the
R8 training targets. The subsequent top-up instruction explicitly keeps these
fields in the full annotation flow, reviewer schemas and live V7 schema. R8's
frozen reduced schema/prompt and removal receipts remain unchanged. New annotations
are retained separately in full form until the next dataset projection is approved.
`vesselImoNumber` remains in scope. No other rare fields were removed.

The subsequent R9 projection applies those same removals to batch005 and additionally
removes `parties.carrier` from both cohorts. All 500 now share one reduced contract.
Full annotation source copies remain preserved. Annotation still uses its full
optional field set; training must use the corresponding reduced frozen schema.

These are target omissions, not deletions from authentic OCR. Do not transfer the
removed marks/references into descriptions, party addresses, or other fields just
to retain their information. Genuine product wording already in descriptions is
unchanged. All removed values remain recoverable in R7 and exact R8 receipts.

### Full annotation fields retained for the top-up

Marks mean printed package/cargo identification, not everything underneath a
combined container/marks/description heading. Empty captions have no extracted
value. A destination phrase needs actual marking context before becoming a mark.

Real package markings can contain consignee/project names, invoice/order numbers,
product codes, origin, lot numbers or bag ranges. Literal overlap with description
or references is therefore a review signal, not proof of error. If marks remain
in the target, clarify whether to retain the full printed marking block or only
its identifying, non-product subset, consistently with the description policy.
The R7 audit documents examples for a future reintroduction decision. No new
semantic marks policy was invented as part of dropping the field.

Forwarding/export references mean explicit shipment/commercial/customs references.
Package/pallet numbering ranges are marks, not forwarding references. A booking
number explicitly printed under its own caption can legitimately equal the B/L
number; equality does not erase its distinct printed role.

## Grounding and target consistency

Labels are grounded in OCR. PDFs can clarify layout/ownership but cannot supply
values missing from OCR. Missing container rows or product descriptions are
input-quality questions, not authorization to invent complete labels.

The full annotation contract and reduced training projection are distinct. Do not
train R8 or R9 with the live full-field schema/prompt by accident: use its frozen reduced
contract, or perform an explicit aligned projection before the next training run.
Historical runs and datasets retain their recorded contracts. No training
configuration is redirected or training launched by this annotation campaign.

## Negotiability: actual consignee instruction

Read OCR before stripping order wording from a named consignee. TO ORDER, TO THE
ORDER (OF), THE ORDER OF, and populated CONSIGNED TO ORDER OF fields are negotiable,
whether a bank/company is named or not. For this target, that instruction takes
precedence over non-negotiable copy stamps or document titles. Conflicting source
titles remain audit/filter candidates, not grounds to silently change the rule.

Conditional if/unless/only-if captions and unselected CONSIGNEE OR ORDER alternatives
are not actual instructions. A named consignee without order wording supports
non-negotiable; a bank name alone is not an order instruction. If no consignee
instruction is available, an explicit sea-waybill/non-negotiable issuance may supply
the value; otherwise omit. Contradictory actual instructions require review.

The company's normalized name does not retain TO ORDER OF: the separate negotiability
field must retain that meaning. Four inconsistent populated-order-field decisions
were corrected in R9. Historical publications remain unchanged.

## Current published baseline: R12

`data/curated/mpci-bl-real-v7-reviewed-r12-reduced-660/` now contains600 training
and60 validation real records. It retains the same reduced field contract as R11:
carrier party, vessel flag, cargo marks and forwarding/export references are absent;
AAI is not part of V7. All660 have one goods accounting group with a description.
One group can contain several product names/HS codes and span many containers;
independently accounted product quantities/masses remain outside this initial scope.

Use the published dataset's own frozen schema and training prompt. Full annotations
for new batches006-008 and their removal/edit receipts are preserved separately in
`mpci-bl-real-v7-reviewed-batch006-full`. Eight historical-training sources were
explicitly reassigned to new validation with user approval; do not describe that
split as unseen by older models. The [R12 report](analysis/real-v7-starting-dataset-2026-10-04/r12_final_660/REPORT.md)
lists retained source limitations, and distinguishes source-supported partial
labels from completeness of the original PDF-to-OCR extraction.

### Training configuration and reduced task integration

The experiment config is
`configs/training/production/t5gemma2_270m_lora.mpci_bl_real660_reduced_v7_e5_eva_a32_r32_runpod_schedulefree_v1.yaml`.
It uses the published 600/60 split with pinned file hashes and the named training
task `bill_of_lading_extraction_v7_reduced`. This task derives its target and prompt
schemas from V7 with the four approved fields removed, and rejects predictions or
targets containing those fields. The full annotation task is unchanged. Base
schemas were compared with the frozen R12 schemas and match exactly; the injected
schema additionally binds the observed registry-validated category vocabulary,
as in the preceding training runs. All 660 labels validate against that injected
schema. Placements remain last and relation/category metrics remain enabled.

The short existing V7 prompt is reused without extra instructions. Its runtime
schema injection is now reduced; the fully rendered frozen dataset prompt cannot
be passed directly to the loader, which requires an output-schema placeholder.
The small hash-pinned runtime contract lives beside the production configs under
`contracts/mpci_bl_real660_reduced_v7/`; its companion constraints-build YAML
records the source datasets and registries needed to reproduce it.

Compared with the downloaded completed rank-48 / alpha-32 RunPod configuration,
only rank (now 32), dataset/task/prompt identities, and dataset-size-dependent
intervals change. Five epochs, alpha32, EVA (rho1, 512 train-only calibration
examples), ScheduleFree AdamW at1e-4, microbatch4, accumulation12, gradient
checkpointing and all other optimization/runtime settings remain unchanged.
There are13 updates/epoch and65 total; eval/save every6 updates gives ten
scheduled evaluations at6..60, with no extra final evaluation. Internal optimizer
warmup is4 steps, rounding the former5% policy upward; external warmup stays0.

Validation: CLI config and dataset inspection pass. Actual tokenizer/data
preparation retains600 train/60 validation, zero filtering/truncation. Maximum
source lengths are9056/6293 and target lengths3131/1339 (train/validation), within
the unchanged19200 source/5500 target limits and2048 validation-generation limit.
Preparation took50.72s uncached and23.63s on cached readback in the local CPU-only
environment; no weights or GPU training were loaded. Target canonicalization over
660 records took median88.90ms full vs88.44ms reduced over seven repetitions,
both115073 bytes peak traced Python memory: negligible overhead, not a claimed
training-speed improvement. 152 relevant tests pass, with existing multiprocessing
fork deprecation warnings. An old V5 ordering-test fixture was corrected to include
its already-required empty `packageIds` list; no V5 runtime contract was changed.
Machine-readable preflight receipts are under the now-ignored R12 analysis folder.

### Local run and order-independent evaluation

The local Docker configuration is
`configs/training/production/t5gemma2_270m_lora.mpci_bl_real660_reduced_v7_e5_eva_a32_r32_local_schedulefree_v1.yaml`.
It preserves the RunPod config's dataset, schema/prompt, rank32/alpha32, EVA,
ScheduleFree AdamW, learning rate, five epochs, gradient checkpointing, and length
limits. Train microbatch1 and accumulation32 give effective batch32, 19 updates
per epoch and95 total. Plain step-based evaluation and saving both run every9
updates: ten evaluations at9,18,...,90. Internal optimizer warmup is5 updates
(ceil5% of95); no external warmup or decay is added. Evaluation microbatch2 and
best-checkpoint selection by field F1 remain unchanged. The run/adapter names are
distinct, and MLflow uses the local Compose service `http://mlflow-server:5000`.
The RunPod config and dataset JSONLs were not modified.

The old field scorer used positional list paths. Reversing every list in all660
correct labels yielded field F1 **0.855245**, relation F1 **0.705634**, and document
exact match **0.404545**, despite identical facts. Metrics now match list items
one-to-one within their own parent. Exact entity identifiers, goods descriptions,
and party names anchor row matching when available; remaining rows maximize exact
nested leaf agreement with a deterministic rectangular assignment. Nested lists
are matched recursively. Missing and extra occurrences stay distinct, including
repeated placement quantities. Rows cannot distribute their fields among multiple
partners, and items cannot move between unrelated parent fields. Scalar values
remain exact, not fuzzy. DG category occurrences also retain their aligned row
identity instead of collapsing repeated rows with the same UN number.

The same aligned view feeds field, extraction-fact, category and cargo-relation
metrics. Canonical exact match ignores list order for schema-valid targets. Schema
validity itself is unchanged, including historical schema-specific constraints.
Original generated/reference JSON and dataset order remain unchanged; diagnostic
scoring paths use reference indices, with unmatched predictions appended after
them. This changes shared evaluation/reward/analysis scoring wherever it calls
`assess_prediction` or `structured_metrics`, not teacher-forcing loss or training
labels. Historical metrics must be rescored before a like-for-like comparison.
Stored historical prediction flags were not rewritten; analysis tools that verify
those flags can report exact-match drift until an explicit rescoring is performed.
The new matching module is included in training's loaded-source hash receipts.

Validation includes all660 labels with reordered lists through the real Trainer
metric callback in the rebuilt image: every correctness metric is1.0. Three
independent permutations on each side preserve scores when all660 B/L numbers
are deliberately wrong; another test changes all660 descriptions and641 package
quantities, again retaining identical scores over three independent permutations.
Unit checks cover nested goods/containers/packages/DG, swapped ownership, missing
goods, duplicate seals and placement rows, invalid predictions, and scalar type
distinctions. The assignment solver matches exhaustive optima on161 small square
and rectangular matrices, including a case where greedy matching fails.

The targeted host suite passes100 tests; the rebuilt CUDA training image passes
104 tests, including the Torch-dependent analysis checks (11.17s). The actual Docker
tokenizer preparation
retains600/60 with zero filtering or truncation (9.38s cached); the actual CUDA
Transformers argument builder confirms95 updates, eval/save9, warmup5. No model
weights or training were started. Benchmarks over660 records (seven repetitions,
median; traced Python memory separately) show correct reordered labels at0.3414s
old /0.3261s new, peak20.95MB /12.98MB. Reordered labels with one deliberate error
per document take0.3540s old /0.4881s new, peak21.11MB /21.74MB. The latter's extra
matching work is about0.20ms per document, approximately12ms per60-document eval;
it is CPU scoring overhead, not GPU memory or forward-pass cost. Lint/type checks
pass; existing multiprocessing fork deprecation warnings remain.

Start from the repository root (Compose also starts MLflow on localhost:5000):

```bash
docker compose --profile training run --rm --build kie-trainer train \
  --config configs/training/production/t5gemma2_270m_lora.mpci_bl_real660_reduced_v7_e5_eva_a32_r32_local_schedulefree_v1.yaml \
  --project-root /workspace
```
