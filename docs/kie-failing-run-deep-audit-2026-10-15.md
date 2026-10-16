# Failed synthetic-only run: L/R22 versus M/R23 deep audit

Date: 2026-10-15. **Read-only diagnosis. No labels, templates, production code,
predictions or training configurations were changed; no inference or paid model
calls were made.** L denotes synthetic-only R22; M denotes the subsequent run
with the same 7,000 synthetic samples plus 600 real samples. Final saved
predictions are predictions from each run's selected/exported weights, not an
assumption that epoch 10 was best.

Snapshot note: the later R26 coverage inventory below was frozen **before** the
separate1,000-description refresh. Its exact pre-refresh data is now preserved in
[the R26 backup](analysis/goods-description-refresh1000-20261015/backup-r26-before-refresh/manifest.json).
See the [completed refresh](kie-goods-description-refresh1000-2026-10-15.md) for
current description lengths. Historical L/M inputs, references and predictions
are unchanged; none of these later repairs is credited to their performance.

Follow-up, 2026-10-16: [R27 repair status and validation/real overlap audit](kie-r27-failure-followup-and-validation-overlap-2026-10-16.md)
reopens the outstanding items, measures current repaired coverage and compares
all 36,000 validation/real-training pairs. It distinguishes corrected upstream
format/policy defects from unproven model improvements, and adds historical
own-reference subgroup results without rescoring against current targets.

## 1. Conclusions

1. The synthetic-only failure is a content/generalization failure, not merely a
   JSON-validity penalty. On the **same 56 parseable documents and identical
   non-HS gold facts**, field F1 rises **0.8603 → 0.9155** after restoring the
   real shard; relation F1 rises **0.8523 → 0.9489**. Addresses and descriptions
   are included in this comparison.
2. The earlier synthetic corpus has confirmed presentation gaps: every target
   telephone starts with `+`, every container identifier is rendered
   contiguously, and its descriptions are substantially longer than real
   training/validation descriptions. R24 and R25 close the first two observed
   input/target-presentation gaps. R26 improves DG wording. **None was used to
   train M, and their model-performance benefit remains unmeasured.**
3. The remaining errors are not one uniform problem. They include party/list
   ownership, documentary-field confusion, goods-block selection, package
   specificity, unsupported allocations, missing repeated records and copying
   errors. More valid-looking synthetic data alone is not evidence that these
   particular distinctions have been taught.
4. A narrow validation-completeness concern was found: `a44bb232` labels only
   the first delivery-agent telephone in a printed four-number list, while six
   comparable real-training sources label all four. This is a **review hold**,
   not a silent validation relabel or rescoring. It does not explain the broad
   L→M recovery.

The audit is complete for the requested saved-output inventory. Its mechanical
coverage is exhaustive; the source-grounded adjudications are explicitly
identified below rather than represented as a fresh manual review of every PDF.

## 2. Reproducibility and comparison contract

- Exact run paths, artifact hashes, native metrics and all cohort counts:
  [inventory.json](analysis/failing-run-deep-audit-20261015/inventory.json).
- Reproducible CPU audit:
  [audit.py](analysis/failing-run-deep-audit-20261015/audit.py).
- Previous full field/group/facet replay, reused rather than reinvented:
  [per-field replay](analysis/training-latest-20261015/per-field-replay.py),
  [fields](analysis/training-latest-20261015/per-field-fields.csv),
  [groups](analysis/training-latest-20261015/per-field-groups.csv),
  [facets](analysis/training-latest-20261015/per-field-facets.csv),
  [documents](analysis/training-latest-20261015/per-field-documents.csv).

Each prediction is assessed against **its own saved `reference_text`**, not a
later dataset revision. All 60 source IDs, raw OCR and positioned inputs are
identical between L and M. Each has 2,576 expected scalar facts. Fourteen HS
strings in 13 documents changed presentation between references; their digits
did not change. Cross-run content comparisons therefore exclude **all 51 HS
facts and all HS predictions**, leaving 2,525 identical expected facts. Native
HS results remain separate: L 32/51, M 48/51 correct; that difference is not
presented as a clean unchanged-contract improvement.

Replay reproduces every saved native field/relation/category aggregate to
`1e-12`. Historical L task constraints no longer bind the current schema
fingerprint. The current diagnostic schema would accept 40 L outputs, while the
saved native result accepted 38. **The report retains native 38**, does not
bypass the binding guard, and does not call current schema acceptance historical
equivalence. M retains native 49. The full drift record is in the inventory.

Scoring is case-insensitive exact scalar matching with unordered, parent-scoped
list alignment. Field accuracy is intersection/union, not document accuracy.
Diagnostic punctuation/digit equivalences do not alter headline scores.

## 3. Native final results and matched content

| Own frozen references, all 60 | L: synthetic 7,000 | M: synthetic 7,000 + real 600 |
|---|---:|---:|
| Field precision | 0.860713 | 0.913288 |
| Field recall | 0.815606 | 0.883152 |
| Field F1 | **0.837552** | **0.897967** |
| Field accuracy | 0.788072 | 0.857197 |
| Relation F1 | 0.830275 | 0.933333 |
| Category F1 | 0.837736 | 0.927644 |
| Parseable JSON | 58/60 | 58/60 |
| Native schema-valid | 38/60 | 49/60 |
| Whole-target exact | 1/60 | 7/60 |

| Identical non-HS facts | L F1 / accuracy | M F1 / accuracy | Correct L→M |
|---|---:|---:|---:|
| All 60, 2,525 expected | 0.841228 / 0.791205 | 0.896538 / 0.855551 | 2,069→2,227 |
| Same 56 parseable, 2,354 expected | **0.860321 / 0.825534** | **0.915499 / 0.890541** | **2,011→2,156** |

In the matched cohort, false-positive values fall 310→200 and false-negative
values fall 343→198. The gain is not generated solely by dropping difficult
JSON failures or changing HS separators. Restoring real data is the major
experimental intervention, but also increases training examples/optimizer
updates; it does not isolate phone formatting, layout coverage, geography or
any one causal mechanism.

The exported checkpoints also differ: L selects step2,190/epoch10, whereas M
selects step1,785/epoch7.5, despite completing2,380 total steps. Thus “M simply
had more optimizer steps before prediction” is not a sufficient explanation.
The identical seeded EVA configuration selected512 calibration documents in
each run, but changing dataset membership changes the actual selection: M's
512 include41 real documents; its256 distinct forwarded calibration documents
include22 real documents, while L forwards141 synthetic documents before
convergence. The initialization implementation uses the recorded selection in
order with `shuffle=False`; these counts are observed exposure, **not evidence
that EVA caused the recovery**. Isolating that contribution needs a controlled
calibration/training-data ablation. Source hashes and exact checkpoint/context
counts are in
[initialization-checkpoint-context.json](analysis/failing-run-deep-audit-20261015/initialization-checkpoint-context.json).

Each run's own 58-parseable-document F1 is L 0.852852 and M 0.913105. These
different memberships must not be mislabeled a matched comparison.

### Correct values by field, including failures

| Field/facet | Expected | L correct | M correct | Same 56: L→M |
|---|---:|---:|---:|---:|
| Telephone numbers | 116 | 36 | 95 | 36→93 /108 |
| Address lines | 175 | 133 | 138 | 128→135 /164 |
| Party names | 182 | 147 | 154 | 142→150 /170 |
| Country strings | 256 | 216 | 228 | 209→224 /243 |
| Complete goods descriptions | 60 | 34 | 39 | 33→38 /56 |
| Container identifiers | 120 | 97 | 108 | 92→102 /106 |
| Container seals | 122 | 116 | 110 | **111→109 /114** |
| Container size | 106 | 94 | 102 | 94→96 /98 |
| Container type | 106 | 94 | 102 | 94→96 /98 |
| Declared package quantity | 59 | 51 | 54 | 49→52 /55 |
| Package category | 55 | 47 | 46 | **46→45 /52** |
| Placement equipment identifiers | 120 | 96 | 109 | 91→102 /106 |
| Placement quantities, scalar | 100 | 90 | 95 | 90→94 /98 |
| Dates | 86 | 81 | 83 | 79→80 /81 |
| DG scalars | 3 | 1 | 3 | 1→3 /3 |

The gain is broad, but seals and package categories do not improve. Whole
description exactness remains only 39/60. Telephone recovery accounts for
57/145 additional correct values in the matched comparison, about 39%; this is
a contribution accounting, not proof that a single format repair causes it.

### Relations and categories are anchored facts, not isolated leaves

All 60 contain 220 expected relation facts. L retrieves 181 with 35 extras;
M retrieves 203 with 12 extras. Container-use facts improve 96→109/120;
correct quantity-with-container facts improve 85→94/100. A correct scalar
quantity attached to the wrong identifier is not a correct relation.
Exact relation graphs improve 40→50/60; among the 54 documents with supported
relations, 40→47 are exact. Three additional M exact graphs are empty negatives.

For the same 56 parseable documents, relation F1 is 0.852300→0.948905
(176→195 correct of 204), and category F1 is 0.871795→0.936759
(221→237 correct of 250). All-60 category facts improve 222→250/269,
but package categories alone fall 47→46/55. Category graphs are exact in
36→41/60 documents.

## 4. Every-error inventory and its limits

The audit records **944 aligned scalar discrepancies**: L 565 and M 379.

| Native aligned discrepancy | L | M |
|---|---:|---:|
| Wrong value at a matched path | 250 | 138 |
| Missing expected value | 225 | 163 |
| Extra gold-absent value | 90 | 78 |
| Total | **565** | **379** |

A wrong value contributes both a false positive and a false negative to scoring
but one row here. JSON failures account for 90 of L's missing leaves and 84 of
M's. Consequently these counts are not 944 independently adjudicated semantic
mistakes. Nor does “gold absent” automatically mean fabricated.

- [All scalar errors, JSON](analysis/failing-run-deep-audit-20261015/all-leaf-errors.json)
  and [CSV](analysis/failing-run-deep-audit-20261015/all-leaf-errors.csv): complete
  values, aligned paths, source IDs, OCR context hits, training-value support,
  and mutually exclusive mechanical taxonomy.
- [All 228 relation/category missing/extra facts](analysis/failing-run-deep-audit-20261015/fact-errors.json).
- [245 role-scoped phone pair records](analysis/failing-run-deep-audit-20261015/phone-errors.json),
  including correct pairs, residual errors and missing/extra values.
- [16 source-grounded adjudications](analysis/failing-run-deep-audit-20261015/manual-adjudications.json),
  preserving actual OCR line references, conclusions and review holds.

These inventories overlap intentionally. A wrong equipment identifier may
produce scalar, category-anchor and relation errors; those are not three
independent causes. OCR literal screens ignore case and fold whitespace but do
not establish party/block ownership. A number printed elsewhere may be a tax
identifier, carrier telephone, vessel number or another party's contact.

### JSON failures

L fails `45ed05eb` (37 expected leaves) and `891555b9` (53). M fails `13ea92fa`
(60) and `61d1d525` (24). These disjoint failures explain the 56-document
intersection. M's seven-container `13ea92fa` loops through nested goods/freight,
omits later containers, and ends inside a string. Its length is consistent
with the aggregate 3,072-token/missing-EOS event, but per-document generated
token metadata was not saved, so cap attribution is not independently proven.
The longest gold target is only 1,334 tokens. `61d1d525` instead closes its
object prematurely and has trailing fields; it is not a long-target problem.
Constrained decoding may prevent these structural losses but does not resolve
wrong roles, omissions or copy errors in otherwise valid JSON.

## 5. Source-grounded failure mechanisms

### 5.1 Telephones: presentation was a major gap; ownership is also real

Role-scoped pairing first retains native exact pairs, then pairs identical full
digit strings, then uses one-to-one minimum digit-edit distance within the
same aligned party role. Residual pairing is diagnostic, not a claim of semantic
identity or a new metric.

| Phone diagnostic, all 60 | L | M |
|---|---:|---:|
| Exact | 36 | 95 |
| Same complete digits, different presentation | 21 | 0 |
| Changed/incomplete digits | 21 | 4 |
| Email/nonphone payload | 1 | 0 |
| Documentary annotation included | 1 | 1 |
| Missing | 36 | 16 |
| Extra | 6 | 7 |

Removing any expected telephone whose last seven digits occur anywhere in
digit-compacted real-training OCR leaves 75 expected values: L gets 19 right,
M 56. On the matched cohort, 68 remain, with 19→55 correct. This conservative
control supports generalization beyond exact phone memorization; it is not
proof of independence from all shared company/layout information.

Concrete ownership/list failures, recorded with full IDs in the adjudications:

- `684f442c`: two notify parties; M copies the second party's contacts into both
  objects and leaves the first party's US number inside its address.
- `bde3a722`: M emits BROOK TRADERS twice instead of PL STONE plus BROOK.
  The wrong phone values are real numbers from the other notify party.
- `d8363a4d`: `PHONE&FAX002` and `02 2575 2002` wrap onto separate lines.
  L emits an unsupported `+92` prefix; M correctly retains `002` for consignee
  but omits it for notify. This changes digits, not just punctuation.
- `40e523d2`: M includes `(4LINES)` in the telephone value. The annotation is
  printed but is not a separate dialing component.
- `aa9ed483`: M extracts `92244000K` from the explicitly printed `RUT` tax field
  as a telephone. This is a role error, not randomly invented digits.
- `66feaac3`: M borrows consignee contacts from the goods-area continuation,
  contrary to the agreed direct-party-block policy. Its shipper source really
  does print `SOUTH KOREA` in continuation; that gold value is not phone-based
  geographic inference.

R24 addresses missing presentations, literal-prefix ownership and repeated
bindings upstream. It does not by itself prove repair of two-party ownership,
telephone-list segmentation, captions or tax-versus-phone classification.

### 5.2 Validation phone-list completeness hold

[Validation `a44bb232`, OCR line 55](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_a44bb232a1bbaa7dccfe7ca709e0a3d1f03988fb03ed947b218375b4a1c3e972/ocr.txt#L55)
prints `TEL.: (+203 4865088 /4865082/ 4865080/ 4863088` for the delivery
agent. Frozen gold retains only `(+203 4865088`. M merges the complete list
into one scalar. Both models also borrow `+39.0187.5371` from the TARROS carrier
footer into a private party; that distinct ownership error is confirmed.

Six real-training sources retain the corresponding four numbers as four phone
entries, not just the first number:

| Real source | OCR occurrence | R23 real row |
|---|---|---:|
| `cbd4c105` | [line 24](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_cbd4c10514b4f35a26fee9a99ab053023ace7ff91370af4c202d58343099d11a/ocr.txt#L24) | 100 |
| `e18486c6` | [line 21](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_e18486c69179f0a9a6cb73e85c76385f07d13e1fd3237a250cc3d4d74eab9566/ocr.txt#L21) | 283 |
| `68d08294` | [line 29](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_68d08294f302764335935c9f2d43b9270ae174a5612cec9f8b1bb4b6da742b10/ocr.txt#L29) | 333 |
| `7750e3d9` | [line 22](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_7750e3d96684f1f97a45fb42be766b700a1316caadb001b5ac47d6a420882f42/ocr.txt#L22) | 336 |
| `a0f7d986` | [line 25 and repeated line 127](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_a0f7d986fd766a3ba88d1d9206ba32db2d6a0eb480efff23c36fb33967c66010/ocr.txt#L25) | 372 |
| `7b12b905` | [line 17](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_7b12b9055a542c5d1fe3968ed5e078914df02aad42d9372783092da79a294aa9/ocr.txt#L17) | 594 |

Rows refer to the frozen [R23 real shard](../data/curated/mpci-bl-real600-synthetic7000-hs-preserved-r23-positions-v1/real-train.jsonl).
The `a0f7d986` envelope has a previously identified unmatched-parenthesis concern,
which does not remove the existence of the four printed identities. This
comparison warrants a bounded policy/completeness adjudication, not an automatic
claim that every list extra is wrong or permission to modify frozen evaluation.
Earlier literal-presence audits checked support of existing values, not the
completeness of every omitted value.

### 5.3 Equipment, quantities and negative examples

All eleven validation container identifiers without a contiguous literal OCR
copy are wrong in L; M gets nine right. The remaining two keep hyphens in
`5d6b477c`, contrary to canonical identifier targets. R25 directly supplies
fragmented/spaced/hyphenated presentations and checks every owned occurrence.

Other equipment errors are semantic, not separator normalization:

- `53e77979`: a CAT loader serial/VIN becomes equipment/placement identity.
- `0961dfa2`: a used unpacked vehicle is classified as packaged vehicle; L also
  treats its chassis as a placement identity.
- `5d6b477c`: L invents allocation `4` for four equipment rows; M invents total
  package quantity `69`. Neither is fixed merely by joining identifier pieces.
- `6fa0b967`: shipment total `7080` is assigned where the visible container row
  declares `4015`.
- `ff146232`: total `48 PALLETS` has no per-container split. L assigns `48`;
  M copies the `1` from each `1x40HC` inventory into allocations. Neither is a
  supported cargo split.
- `aa9ed483`: ten equipment rows each say `12 Package(s)` and the continuation
  says `120 BAGS`. M selects `10`, the container count, as the cargo total;
  L's `188` is not the declared package total either.

These distinctions require role/level restraint and legitimate missing-quantity
examples, not arithmetic plausibility constraints on bag weights or payloads.

### 5.4 Descriptions and package categories

L has 24 parsed description mismatches and two JSON failures; M has 19 and two.
On the same 56, exact descriptions improve 33→38. The existing manual inventories
cover [all 23 paired L mismatches](kie-real-synthetic-format-audit-2026-10-15.md)
and [all 19 parsed M mismatches](analysis/training-latest-20261015/SOURCE_REVIEW.md).
M's 19 comprise eight omissions, four boundary extensions, three wrong/foreign
source passages, three copying errors and one repetition. None is merely a
casing mismatch. Examples include lost product/list/page continuations,
PETROBEL postal text absorbed into steel-casing wording, and vessel/liability
wording replacing fabric composition. R26's DG wording work is not a general
solution to these ordinary-cargo boundary mistakes.

All four validation free-text package fallbacks have a package error in both
outputs, including BALES, RACKS and PLASTIC PALLET. Generic versus more specific
categories also remain problematic where training examples already exist:
`4f981d62` uses generic PACKAGE rather than IBC; `ce3f0739` uses generic DRUM
rather than STEEL DRUM; `0961dfa2` misses UNPACKED SINGLE UNIT. The corpus already
has respectively 105, 56 and 40 synthetic examples of those specific categories.
Thus missing-class coverage is not a sufficient explanation.

The sole validation DG document `d9be84a2` improves from one to three correct
DG scalar values. L's `3077 III` corrupts the UN anchor, making otherwise correct
packing-group text an incorrect anchored fact. M still selects fibreboard boxes
from the regulatory sub-block rather than the approved primary carton total.
Three validation DG facts cannot support a broad conclusion about all DG fields.

## 6. Training representativeness versus validity

[coverage.json](analysis/failing-run-deep-audit-20261015/coverage.json) streams
the frozen L/M synthetic shard, real/validation, and later R24–R26 synthetic
shards. It measures presentation support, not semantic certification.

| Trait | Original synthetic 7,000 | Real 600 | Validation 60 |
|---|---:|---:|---:|
| Telephone targets | 12,663 | 1,110 | 116 |
| Plain leading `+` | 12,663 | 469 | 42 |
| Parenthesized/braced leading `+` | 0 | 39 | 5 |
| Plain leading `00` | 0 | 168 | 26 |
| Plain national leading `0`, not `00` | 0 | 154 | 19 |
| Other/bare phone envelope | 0 | 280 | 24 |
| Width-independent phone punctuation masks | 12 | 99 | 23 |
| Descriptions under 10 whitespace words | 751 (10.73%) | 469 (78.17%) | 43 (71.67%) |
| Free-text package fallbacks | 25/7,000 | 10/600 | 4/59 |
| Documents without container entries | 128 (1.83%) | 29 (4.83%) | 6 (10%) |
| Placement quantities present | 12,758/14,472 | 1,152/1,251 | 100/120 |
| Positioned content-line coverage | 83.79% | 85.05% | 83.79% |

Prefix categories are literal after whitespace trimming, without stripping an
outer parenthesis: validation has **42 plain `+`**, or **47 including its five
parenthesized-plus values**. Masks replace each digit run with `D` but preserve
spaces/punctuation; they do not require exact digit widths.

The original synthetic masks cover 4/23 validation masks, containing 13/116
expected telephone values. R24 covers 17/23, containing 110/116; the real shard
also covers 17/23. This establishes improved representation, not that the six
unseen masks are necessarily unlearnable.

The 7,000 contain 617 genuinely mixed-equipment documents versus 9/600 real
training and none in validation. That intentional coverage is not itself a
defect. There are already 1,714 placements without quantities, so restraint
examples are not wholly absent; their OCR contexts must distinguish inventory
counts, cargo totals and ownership rather than just remove random labels.

Position coverage alone is not evidence of geometry failure: the exact same
synthetic input shard was used by L and M, and all 60 validation positioned
inputs are identical. R24/R25 keep line counts/geometry; R26 changes only DG
wording/reflow. No coordinate-corruption explanation was established here.

Country wording is a separate documented gap. See the
[country-presentation audit](kie-country-presentations-audit-2026-10-15.md):
**48 confirmed current-policy violations across 36 documents**, plus **nine
related exporter/registration-country holds**, not 57 confirmed violations.
The current synthetic target vocabulary also misses several printed validation
abbreviations/name forms. R24–R26 do not repair country presentation.

## 7. What the later repairs do—and do not establish

| Work | Confirmed relevant gap closed | Not established by that repair |
|---|---|---|
| [R24 phone presentations](kie-phone-presentation-repair-2026-10-15.md) | Source-conditioned envelopes, repeated-owner/literal-prefix safety; 17/23 validation masks now represented | Measured F1 gain; party-row ownership, tax classification, complete phone-list annotation consistency |
| [R25 container ID presentations](kie-container-id-presentation-repair-2026-10-15.md) | 1,297/14,472 IDs now require joining noncontiguous source-style text; all 413 native source probes pass | VIN versus equipment discrimination; correct allocation level; measured F1 gain |
| [R26 DG LLM wording](kie-dg-llm-wording-repair-2026-10-15.md) | 261 current training records get reviewed regenerated wording; sampled host DG facts remain controlled | Broad ordinary-cargo description-boundary solution; coverage of all DG combinations; measured F1 gain |

R25's fragmented-ID rate is 8.96%, close to real 10.55% and validation 9.17%.
R26 increases DG description words from 1,640 to 2,926 over 452 DG samples
(mean 3.63→6.47, versus real DG mean 6.47). This is useful presentation
alignment, not proof that every description should have equal length. The
overall corpus still substantially underrepresents short descriptions.

The cited repair reports preserve their own tests, native synthesis probes,
independent migration checks, geometry invariants and publication hashes. This
audit verified the published corpus counts; it did not rerun paid generation
or claim a new trained-model evaluation of those datasets.

## 8. Bounded next priorities

1. Retain real supervision for the next controlled comparison unless testing its
   removal is the explicit experiment. Apply already validated R24–R26 together
   only with a clear experiment label; do not attribute its result to one repair.
2. Review the single identified validation four-number-list completeness hold
   separately and preserve historical metrics. This can alter interpretation of
   individual residual phone errors, not retroactively explain 145 recovered
   unchanged facts.
3. Before more blanket scaling, target short ordinary descriptions, product
   continuations, documentary-boundary negatives, multiple notify identities,
   tax-versus-contact distinctions, free-text package fallbacks, and specific
   package levels where the source prints both broad and narrow wording.
4. Treat zero-equipment/VIN contexts, inventory counts versus cargo counts, and
   shipment totals versus explicit per-container quantities as separate
   supervision cases. A categorical/format sampler cannot substitute for those
   ownership examples.
5. Resolve the confirmed country presentation/target mismatches at their proper
   layers in the separate country task; do not silently normalize all country
   aliases as if their geographic interpretation were unambiguous.

These are evidence-backed priorities, not authorization in this audit to alter
the dataset or launch another run. No finding here claims that R24–R26 have
already delivered a model-performance improvement.

## 9. Execution and evidence bounds

Reproduce from repository root:

```bash
.venv/bin/python docs/analysis/failing-run-deep-audit-20261015/audit.py
```

The final replay completed in **18.99 seconds**, peak RSS **830,472 KiB**.
Ruff formatting/check passes. Assertions verify own-native aggregates, exact
60-document input equality, unchanged non-HS reference facts, relation/category
set accounting and protected-file hashes before/after. The script writes only
its audit inventories; hashes pin the saved predictions, references, constraints,
datasets, manual-adjudication inventory and reusable replay helper.

All aligned errors and fact differences are inventoried. Existing manual
description reviews were explicitly reused; 16 concrete mechanism cases are
source-grounded in the new adjudication inventory. This is not a claim of
fresh visual inspection of every PDF, universal causal identification, or a
new annotation certification pass. The completed deliverable is an exhaustive
saved-output discrepancy audit plus bounded, source-supported explanations and
repair-coverage conclusions.
