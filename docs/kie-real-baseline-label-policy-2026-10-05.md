# Real baseline: goods identity and field-scope policy

Recorded 2026-10-05 from the user's dataset-cleansing decisions. This is a
policy/reference note, not an instruction to relabel historical datasets or launch
another labeling campaign. At its original recording, the selected dataset was
R10 (474 records), documented in the
[cleansing and repetition report](analysis/real-v7-starting-dataset-2026-10-04/R10_CLEANSING_2026-10-05.md).
The preceding [R7 audit](analysis/real-v7-starting-dataset-2026-10-04/R7_FIELD_AUDIT_2026-10-05.md)
is preserved as historical evidence. Subsequent R10 exclusions do not change the
goods or address target policy. Dated decisions below supersede earlier wording
only within their stated scope. Independently printed telephone numbers remain
extractable; standalone abbreviated suffixes/extensions are not separate numbers
and must not be expanded by guessing missing prefixes.

## Telephone presentation and synthesis (2026-10-15)

Telephone targets preserve the document's printed dialing prefix and meaningful
punctuation; they are not normalized to E.164. Physical line wraps may be joined
without changing the number. Do not add `+` or a country calling code, remove a
printed `00`/national zero, or repair digits from country/company knowledge.
Attached extension text belongs to its complete number when retained by the
reviewed source; an abbreviated extension/range suffix is not an independently
complete phone target.

Synthesis separates a sampled numeric identity from its presentation. Source
styles and explicitly configured presentation families may vary. Repeated
occurrences share the identity and preserve their own owned presentation. The
generated phone label follows the reviewed source target's presentation, with
an owned printed occurrence supporting it. A `+` in the unchanged source frame
must not be emitted a second time inside its replacement. Input presentation
changes and dependent labels are published together; real OCR is never rewritten
to accommodate an incorrect model prediction.

Party ownership is independent of literal presence elsewhere: a goods-area
phone is not a consignee contact merely because that number occurs in the
document. An explicitly delivery-only party is not duplicated into notify.
This clarification does not authorize unrelated punctuation cleanup or a new
telephone-normalization target contract.

## Container identifier presentation and synthesis (2026-10-15)

Container identifier targets and their placement references retain canonical
uppercase ISO 6346 identities, including the check digit. Remove presentation
spaces/separators from these targets only; do not change identifier characters
or infer an identity from adjacent seals or equipment codes.

Synthetic OCR preserves each verified source occurrence's identifier format:
spaces, dots, slashes, hyphens and physical line boundaries. Repeated occurrences
of one identity share its sampled canonical value but retain their independently
owned formats. This is an automatic rendering step, not an additional identity
sample. It must not modify neighboring text or any target value. A reviewed
character-level OCR alias is a separate explicit binding, not a separator mask.
Unsupported or ambiguous ownership must fail validation rather than trigger a
global replacement or reconstruction of the surrounding source block. See the
[implementation and dataset repair report](kie-container-id-presentation-repair-2026-10-15.md).

## Related party identities and postal ownership (2026-10-10)

The user explicitly selected **the primary printed company A** for an `A ON
BEHALF OF B` / `A O/B B` relationship in a shipper or consignee block. Keep only
A's identity in `name`, with A-owned postal and contact values. Record B and the
relationship outside the extraction target in audit metadata. A relationship is
not one long company name, and does not by itself establish a forwarding-agent
or NVOCC role. Separately printed independent roles remain separately extractable.

`A VIA B` follows the same one-entity rule when A and B are visibly separate
organizations: keep primary A in `name`, and record B as relationship context.
This differs from a trading alias for the same entity. A sole undifferentiated
postal/contact block after both names does not establish A's ownership; omit the
unresolved address, country and contacts from A's target. Do not infer ownership
from either company's name or apparent country. For synthesis, a certified
organization component generates one primary organization; its newly generated
postal/contact values remain attached to that single emitted identity.

Care-of (`C/O`, `CARE OF`) names describe postal routing: put the primary party
in `name` and the printed care-of recipient with its postal block in
`addressLine`. Do not erase a genuine named recipient as ordinary name pollution.
An account (`A/C`) organization or named represented principal is relationship
context, not part of the primary party's legal name. Capacity captions such as
`AS AGENT`, `AS PRINCIPAL` and `NVOCC` are not legal-name suffixes.

An address explicitly attached to B cannot become A's address. An undifferentiated
A/O-B postal block does not establish which entity owns the address: preserve
the OCR and archive the postal context with ownership unresolved, while omitting
that uncertain component from A's target. This does not hold back a resolved A
identity or independently established contacts. Do not assign ownership from
real-world headquarters, country, language, or company-name knowledge.

This is an explicit training-target policy. The MPCI business specification
separately defines the B/L shipper and forwarding agent; it does not make `O/B`
an automatic forwarding-agent classification. Its ultimate-party filing
requirements must not be confused with a legal-name concatenation in this KIE
target. See the [MPCI business specification](https://naic.icp.gov.ae/portal/assets/mpci-guidelines/UAE%20MPCI%20Business%20Specification%20Document%20V1.0.pdf),
table10 and appendixC.

## Goods identity: accounting, not number of product names

### Declared SET and unspecified BALE categories (2026-10-16)

When `SET(S)` is the declared shipment/counting unit, use `PACKAGE_SET` and
retain that declared count and same-level allocations. A set does not establish
the number of constituent pieces, a particular inner/outer level, a vehicle
count or the narrower meaning of a kit.

When `BALE(S)` is the declared unit and compression is not stated, use generic
`PACKAGE_BALE`. Do not infer compressed or non-compressed from the commodity,
usual practice or absence of a compression modifier. Explicitly stated
non-compressed bales retain `PACKAGE_BALE_NON_COMPRESSED`; a supplied more
specific category requires its supported printed distinction. This category
change does not resolve a separately inconsistent or absent count.

These are semantic extraction categories in the supplied vocabulary, not claims
that the saved MPCI application accepts a new submission code. Its authentic
405-code export remains unchanged. The reviewed extraction vocabulary records
the absence of an exact application mapping explicitly; downstream projection
must surface that boundary rather than substitute Unit/Package or invent a
compression state. In particular, `UN` plus `SETS` is not the adopted policy.
See the [packaging investigation and scheduled R28 redesign](kie-r27-package-vocabulary-audit-2026-10-16.md#r28-scheduled-redesign-and-bounded-r27-coverage-run--2026-10-16).

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

## Shipment packaging and allocations (2026-10-09)

Use the **declared shipment accounting unit**, established by package columns,
container package rows and shipment declarations in the complete document.
Quantity and category form one fact. Neither the innermost contents nor outermost
handling unit is automatically the target. Container allocation quantities use
the same unit as the goods package total.

- `100 PALLETS`, five container rows of `20 PALLETS`, and product packing of
  `4000 PAPER BAGS` means 100 pallets with allocations20/20/20/20/20. The bags
  are contained packing, not an alternative target total.
- An explicit declared count of drums/bags/cartons remains that unit when
  pallets merely describe supporting packing.
- `1x40'HC CONTAINER S.T.C. 38 PACKAGE(S)` establishes an allocation of38 to
  the sole identified container carrying that goods item. A separate tabular
  quantity or repetition beside the identifier is not required. This rule does
  not apply just because only one identifier survived OCR from multiple containers.
- Explicit per-container counts are allocations. Preserve supported partial
  allocations; do not balance them by equal splitting, mass/capacity division,
  or assigning a whole-shipment total to one surviving container.
- Disjoint shipping units can form additive package rows. Nested levels cannot.
  An explicit mixed-unit aggregate (`19 pallets +3 loose cartons =22 packages`)
  can remain22 generic packages; it is not22 pallets or731 contained cartons.
- Conflicting declarations or unresolved ownership require review. PDF layout
  helps establish ownership, but PDF-only missing counts cannot become OCR labels.

Description boundaries remain governed by the main-product-passage policy below.
Embedded packing in that passage can remain descriptive text without becoming
the structured shipment accounting unit. Separate loading/package declarations
do not become product descriptions.

For synthesis, bind all repeated declared counts (digits and words) and their
package nouns. A reviewed contained count is private context, scaled through an
exact source-supported ratio. Non-integral content counts fail before generation;
they are not rounded. Physical/load scaling uses the same declared-unit baseline.

Evidence: [audit](package-accounting-policy-audit-2026-10-09.md) and
[repair and validation](package-accounting-repair-2026-10-09.md).

### Category vocabulary versus genuine fallback (2026-10-16)

Resolve the declared package unit against the pinned extraction vocabulary
before using `typeOfPackages`. Keep its semantic categories distinct from exact
MPCI submission-code projection. A category need not repeat every printed modifier:
`39 PALLETS` with `PACKING TYPE: PLASTIC PALLET` supports generic
`PACKAGE_PALLET` when the vocabulary's plastic-pallet categories require SRS
dimensions that are not printed. Do not invent those dimensions or another
unsupported subtype. Preserve the source wording in OCR.

An exact MPCI category missing only from the task's admitted subset is a
vocabulary-coverage gap, not a genuinely unknown package type. `PARCEL`/`PARCELS`
and `RACK`/`RACKS` resolve to `PACKAGE_PARCEL` (PC) and `PACKAGE_RACK` (RK).
Admit such proven categories explicitly with their source and rendering
contracts; never manufacture category tokens from arbitrary text.

Avoiding fallback does not authorize guessing or erasing the declared unit.
Bare `BALES` resolves to semantic `PACKAGE_BALE`, without asserting compressed
BL or non-compressed BN. `SETS` resolves to `PACKAGE_SET`, without inferring a
component-piece count or substituting UNIT. Their exact application-code
projection remains unresolved as specified above. Generic `BULKS` or `Total
Items` still does not establish a supported bulk subtype or physical packaging.
Retain those two free-text types for review unless the complete owned OCR
provides the missing distinction. Neither those holds nor the unmapped semantic
extensions are claims of a complete valid MPCI submission: the saved form's
submission contract additionally requires an accepted package code.

See [the vocabulary audit and exact reviewed inventory](kie-r27-package-vocabulary-audit-2026-10-16.md).

## Product wording and additional information

### Current decision: description-block copying (2026-10-09)

The user approved this policy for current-label repair and subsequent labeling,
template compilation and synthesis:

> Copy the complete goods-description wording and its continuations in printed
> order, preserving embedded packing, quantities, capacities and qualifiers.
> Exclude separately owned Marks and other documentary fields. Apply only the
> agreed casing and line-break normalization.

Block ownership comes first. Within the actual goods-description block, retain
product identity, brand, model/article codes, composition, specifications,
condition, lot qualifiers, proper shipping names, origin/`MADE IN` wording and
complete packing/count/capacity/weight phrases. Do not summarize, rearrange,
correct spelling, or surgically remove numbers from a descriptive phrase.

#### Boundary clarification: main product passage (2026-10-09)

The user further approved a **main-block-first** interpretation, not collection
of every product-related fact anywhere on the document. Identify the first and
last product-description wording; copy that bounded passage. Start after headings
and generic loading/package declarations (`SAID TO CONTAIN`, `STC`, a standalone
shipment count or its `... PACKAGES OF` introduction). End before distinct
accounting, tracking or documentary passages. A physical line is not necessarily
a semantic boundary: a heading can share a line with the first product words.

Detached batch/SL tables, invoices, shipping/wood-treatment declarations and
auxiliary packing equations are not reasons to extend the description. Conversely,
attached qualifiers inside the main product passage stay intact; do not surgically
delete a capacity, model number or incidental qualifier from the middle. This is
a consistent ownership rule, not optional inclusion decided afresh per sample.

Use another block only for an actual product continuation: a continued product
list on another page, a continued sentence, or a description column interrupted
in flattened OCR by another column. A Marks/reference/HS interruption does not
erase genuine subsequent product wording. Record why each non-contiguous portion
is a continuation rather than detached enrichment. Do not force a single OCR
substring when doing so would include Marks or discard a real continuation.

The latest explicit examples are:

- Train 363: `T850QVN04.2 85" ASSY OPEN CELL`, once; omit packing equations,
  declarations and repeated copies.
- Train 368: retain `25,925 MTCHAMBRIL ... HEINZEL 1077669`; omit the preceding
  `38 ROLLS WITH` and detached origin/wood declaration. Preserve the OCR spelling.
- Train 422: preserve `1012322163DXH DEGREE 1`; its broad table header does not
  establish a separate identifier boundary.
- Train 423: preserve the attached milk-powder/25-kg packing passage, including
  its internal wording `TOTAL 1000 BAGS`; omit the leading loading count and
  the detached serial/batch table. `TOTAL` is not a global deletion keyword.

For future extraction, the existing section reviewer returns `unresolved` with
an `ambiguous` finding, competing interpretations and a recommended review action
when context/layout does not settle a boundary. For synthesis, an unresolved
boundary is a nonempty rendered-review finding and blocks publication until
adjudicated. No additional iterative agent round is introduced. Exact source-span
checks certify transcription and edit ownership; they do not replace the semantic
boundary review. An uncertain source contract is not admitted by guessing.

This clarification changes description membership only. It does **not** decide
the separate inner/outer packaging policy, or authorize changes to package counts,
types, cargo grouping, masses or container placements.

Exclude separately identifiable fields/sub-blocks:

- Marks and Numbers, even when containing product specifications, lots or VINs.
- Import/export/customs references and invoice/commercial reference fields.
- Separately stated net/gross weight, volume and package totals.
- Container/equipment/seal fields, party information and freight/carrier clauses.
- Separately identified HS-code and DG-code declarations; product names and
  chemical descriptions themselves remain descriptive wording.

“Separate” means a distinct field or statement, not necessarily a different
physical column or line. A combined cargo-panel caption does not by itself
establish the boundary. A locally captioned Marks sub-block is excluded even
inside a broad description column; a lot or package-seal phrase in the actual
description remains. Do not infer ownership from a token blacklist or the
nearest flattened OCR heading.

A broad table heading such as `PURCHASE ORDER DESCRIPTION` does not make a
digit/alphanumeric prefix inside an unseparated product line a purchase-order
reference. Preserve that complete line unless an independently identifiable
reference field establishes a different owner. In the reviewed source, this
means retaining `1012322163DXH DEGREE 1` above the elbow/reducer/socket wording;
the table headings themselves are not description content.

Examples:

| Printed source | Description treatment |
|---|---|
| Separate package field `8 PALLETS`; goods `Walnuts packed in 15kg bags` | `WALNUTS PACKED IN 15KG BAGS`; eight pallets remain structured packaging |
| `PACKED IN 3930X50 KG BAGS` | Retain the entire phrase, including 3930 |
| `80 DRUMS OF 210 KGS NET POLYSORBATE 60` | Retain intact; do not remove embedded numbers |
| Product followed by a distinct `NET WEIGHT: 12,000 KG` field | Keep product; exclude the separate net-weight field |
| `EXTRACORPOREAL TUBING SET ADULT -EGYPT` | Preserve the attached suffix |
| Part number printed only in a separate Marks column | Do not append it to description |

Targets use the current consistent uppercase policy. Join physical line breaks
with spaces, preserving printed punctuation and fragment order. Select one
representative occurrence of an entirely repeated description block; preserve
distinct continuations. No global word/sub-string deduplication is authorized.
PDFs can establish ownership/layout but cannot introduce words missing from OCR.

Structured package, mass, HS, DG and placement meanings are unchanged. A printed
number may legitimately occur in both a description phrase and its structured
field; copying it does not create a new goods item, quantity or allocation.
In synthesis, host-sampled quantities remain host-controlled. The final target
copies both generated wording and host-rendered facts inside the approved
description block, after rendering. It does not grant the wording model authority
to invent shipment accounting.

The [scope audit](analysis/description-block-policy-audit-20261009/REPORT.md)
records why old product-only target expressions and Marks enrichment must change.
Historical metrics and receipts retain their historical policy; this decision
does not retroactively certify or overwrite those results.

There is no agreed independent purpose for a cargo product-overflow field.
`additionalInformation` / `additionalGoods` are absent from the V7 model and its
current targets. Do not reintroduce them to shorten long product descriptions.
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

## Date normalization: source evidence, otherwise DD-MM (2026-10-10)

Every structured date field follows the same rule. First identify the date's
document role: an issue date, shipped/on-board date, invoice date and departure
date are not interchangeable.

- A four-digit-year-first numeric date uses year-month-day. A written month or
  explicitly stated format establishes the printed convention.
- Otherwise use unambiguous numeric dates only when they establish the same
  source convention for the date being interpreted. Do not infer a convention
  from country, issuer, carrier or language.
- If both day/month orders remain valid and the source does not establish an
  order, assume **DD-MM**: the first number is the day, the second is the month.
  `03/06/2025` therefore becomes `2025-06-03`. Order ambiguity alone must not
  remove a supported date label.
- An impossible, incomplete or genuinely conflicting date is still a review
  issue; this default does not invent a year or resolve conflicting printed
  values. Preserve the distinction between missing role evidence and ambiguous
  numeric presentation.

This supersedes both the earlier month-first default and the later erroneous
instruction to omit dates when both orders are possible. It applies to real,
validation and synthetic targets, extraction/review descriptions, and template
date rendering. A synthetic target must agree with its final printed date under
this rule; a hidden source format or previous ISO label cannot establish an order
that the rendered OCR does not supply. Raw date-bearing reference text remains
source text unless the target actually defines a normalized date field.

## Route-role occurrence ownership (2026-10-10)

A location target requires the location occurrence retained in OCR for that
transport role, or an explicit declaration assigning one occurrence to multiple
roles. Identical spelling in a discharge, payment, issue-place or party-address
field is not evidence of a separate delivery role. PDF layout may establish the
role of text that OCR retained despite a missing/detached heading; it must not
restore a wholly omitted value occurrence using another role's equal string.
This distinction applies to every route role, not only place of delivery.

Templates must bind a location to its reviewed role occurrence. A shared sampled
destination node makes existing discharge/delivery values agree; it does not
authorize adding a role absent from the source target and its evidence. Retain
supported identical discharge and delivery fields rather than deduplicating
different roles.

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

## Source-owner repair clarifications — 2026-10-12

A clearly marked postal continuation belongs in that party's complete address,
even when OCR reading order places it later. Establish ownership from its actual
marker/footnote and source layout, not from an equal country occurring under
another role. Country wording already present in OCR may populate the country
target and complete the address. This does not transfer an unrelated contact
or weaken the direct-consignee-block contact policy.

The pre-training re-audit establishes two concrete certificates: source
`0d495914`'s `CN>` country continuation and `23e145de`'s matched asterisk country
footnote. The latter's PDF was inspected only to resolve ownership; `EGYPT`
already occurs in its OCR. Its notify phone remains notify-owned. No PDF-only
postal text is introduced.

Incidental per-unit fill/capacity remains permissible product/packing detail.
The extraction objective does not require multiplying it into a shipment total
or validating universal container payload limits. Whole-shipment package counts,
net/gross totals, equipment and carrying settings remain separate authoritative
shipment facts. A generated explicit total cannot silently appear without its
supported target; incompatible physical-state wording still requires correction.

Reviewed source exclusions are compilation requirements, not merely a preferred
dataset selection. A source-owned `required_target_absences` declaration rejects
an old source authority with a populated excluded target before paid generation.
It does not delete the source OCR occurrence or infer a new owner by proximity.

See [the full repair-lineage inventory](kie-synthesis-repair-lineage-reaudit-2026-10-12.md)
for exact scope, source promotion and validation evidence.

## Synthetic private contacts — 2026-10-13

Public carrier/issuer companies can remain fixed context; their named employees,
signatories and personal mailboxes are separate private identities. Generate
source-only individuals without inventing extraction-party roles. Own every
printed repeat of that individual and keep separate CPF/phone facts outside
the name region.

For generated party mailboxes, supply the current company and current contact
names. Private source recipients are not style examples: replace the complete
recipient/account identity, including separator-normalized variants. Generic
department wording can remain. Private original business domains must change;
public shared mail-provider domains can remain. Source-only public carrier
domains can remain while their personal recipient changes. Email/website
domains are independent and are not required to match.

Every source target mailbox requires explicit recipient and domain classification
before paid synthesis. No metadata omission grants a private-recipient or
private-domain exemption. Source-only phones/faxes use their declared owner and
country. Distinct fax draws must be fixed-line-capable; explicitly identical
TEL/FAX and local/international source variants share one generated subscriber.
Opaque individual identifiers preserve source format, without a requirement to
implement national checksum rules.

These are synthetic anonymization rules, not instructions to relabel real OCR
with fictional values. See [the private-contact repair report](kie-synthesis-private-contacts-repair-2026-10-13.md).

## Synthetic fiscal/commercial references and decoder contract — 2026-10-13

Generate private tax/registration/customs and invoice/order/booking/customer
references deterministically through exact source owners. Preserve formatting
classes and separators, not original identities. Repeated-page and differently
punctuated aliases share one generated identity. No national-checksum validity
is required. Preserve public carrier registrations, SCACs, form/legal constants,
and independently owned dates, quantities, weights and postal text.

Every source-only identifier requires an explicit private/public/dependent
classification before paid synthesis. Private source owners require their
generator; future target-owned identifiers can use the existing target alias
recipe. An unclassified new source is rejected, never silently exempted.

Decoder/training targets are direct extraction objects: the contents formerly
inside documentPatch, with no schemaVersion or outer documentPatch. Keep the
same facts, explicit instruction nulls, category vocabulary, dropped-field
policy and relation semantics. Versioned source/review artifacts remain
provenance, not decoder targets.

See [the repair and direct-target report](kie-synthesis-private-identifiers-and-direct-targets-2026-10-13.md).

## Full printed HS identifiers and DG properties — 2026-10-14

The commodity identity sampled from the HS registry remains its genuine six-digit
HS6 code. Pass that identity, not invented national suffixes, to the wording model.
The source owns the printed customs identifier's digit width and separators.
For synthesis, retain all source-shaped code slots, extend sampled HS6 to each
source width with deterministic fictional digits, and preserve source dots,
spaces, commas, ASCII hyphens or colons. Identifiers contain 6–20 ASCII digits; the
extraction target retains every digit, leading zero and printed separator.
Different national extensions are different identifiers even if their first six
digits coincide. Repeated presentations of the same complete digit sequence use
the first owned presentation in the label. Never collapse a complete identifier
to HS6 or discard another extension because its parent classification matches.
These suffixes teach extraction, not valid national tariff classification.

This replaces the historical digits-only/6–18-digit AI-adapter restriction in the
extraction contract. The saved canonical MPCI form supports printable ASCII.
Any downstream adapter retaining the older normalization needs separate alignment;
it is not evidence that the form itself forbids source punctuation.

The 2026-10-15 real-data audit also confirms an owned `HS CODE: 5205:22`
presentation. ASCII colons are retained, not removed. The form accepts printable
ASCII, while the extraction contract deliberately restricts separators to the
observed set above. A slash between separate codes is not an internal separator;
Unicode lookalike punctuation is not accepted as ASCII. Historical datasets and
run references remain frozen; corrected real labels are published in the new
7,600-record training configuration's dataset revision.

DG packing groups and subsidiary hazards are sampled with their exact eligible
HMT identity. Preserve the source's independently owned field presentations;
do not add a structured value when no corresponding declaration is rendered.
A flash-bearing template requires a pinned property registry. The initial
registry joins CDC/NIOSH scalar closed-cup flash points to exact CAS and the same
DOT/UN identity through the existing ECICS/HMT links. Qualified identities,
mixtures/NOS entries, ranges, open-cup properties and conflicting identity links
are not inferred. Convert Fahrenheit/Celsius explicitly and reconcile the target
to the value actually printed at the source's precision. The property remains
host-controlled; the wording model cannot invent or change it.

See [the full repair, coverage and validation report](kie-hs-dg-format-repair-2026-10-14.md).

## DG commercial wording in synthesis (2026-10-15)

Dangerous-goods descriptions are LLM-generated, like ordinary product wording:
sample the registry identity and shipment facts first, then generate compatible
commercial wording at the source's descriptive depth and information pattern.
This applies to chemicals and DG vehicles. A registry name is context, not a
mandatory verbatim description. Short identity-only examples may naturally
remain short; grades, intrinsic specifications and source-like detail can vary.

The sampled substance/formulation or vehicle propulsion and its DG declarations
remain consistent. Independently printed proper shipping names and technical-name
aliases use the sampled identity, never the complete generated commercial prose.
Preserve which optional regulatory fields the template actually prints. The LLM
does not add or resample UN/class/packing-group/subsidiary/flash-point declarations.
Project the description label from the reviewed rendered product block, retaining
the established casing, boundaries and line-break normalization. Commercial repeats
outside the block may repeat generated wording without expanding the target boundary.

The earlier host-owned DG-description shortcut was an implementation defect, not
an approved policy. See [the correction and validation report](kie-dg-llm-wording-repair-2026-10-15.md).

## Country identity and printed country presentation — 2026-10-16

Party and locality country targets now share one contract: establish the country
or territory from an explicitly owned OCR country name, code or unambiguous
country adjective, then emit the pinned registry's **preferred English name in
uppercase**. This replaces the former party rule that copied the printed
country alias. It applies to party countries, route countries, issue country and
freight-payment country. These are extraction identity targets, not downstream
MPCI ISO-code fields.

The separately pinned preferred-name table, keyed by ISO alpha-2 identity, is the
normalization authority. Country names and codes printed in OCR may vary; their
identity target does not. Keep ISO identity, accepted input aliases and sampled
printed presentation as separate data. Do not assume an ISO registry's formal
name is necessarily the preferred English label. The country-presentation
registry and its preferred-name table are versioned together under
`artifacts/registries/compiled/country-presentations-iso-cldr-48.2.3-v1/`.

This normalization does **not** rewrite addresses, localities, company names,
goods-origin text or raw/positioned OCR. Their existing literal spelling,
punctuation, casing and line-break policies remain in force. For example, a
party's address may retain `USA` while its country target is `UNITED STATES`.
Changing only a synthetic country's printed alias does not change its identity.

Identity must be supported by that party's or location's own country evidence or
an explicitly reviewed continuation/shared-role ownership link. City names,
telephone prefixes, company knowledge, carrier nationality and unrelated party
or route countries do not supply it. An exporter declaration can support a
shipper only when the source explicitly establishes that they are the same
entity; the label alone is not a universal identity equivalence. Equal city
text in different role boxes is not an automatic shared-country declaration.
When both role/place occurrences remain in OCR, a reviewed document-local
co-reference may establish their common country from an explicit qualified
occurrence of that same place. For example, retained loading `MUNDRA` and
`PLACE ISSUED: MUNDRA, India` identify the same document-local place; this does
not reconstruct a missing loading occurrence or geocode an unfamiliar city.
Likewise, an issue place and the same place's country-qualified issuing stamp
within one issuance block can share country evidence. Record the linked
occurrences and exclude competing same-name places; do not generalize this to
unrelated party addresses, different localities or carrier headquarters.

Ambiguous identities and unsupported optional country fields are omitted and
flagged for review in new extraction. Existing source/data holds require an
explicit adjudication before removal or migration; no bulk cleanup silently
drops them. The explicitly approved business alias **`KOREA` maps to
`SOUTH KOREA`** without requiring another KR/South/Republic occurrence in the
same role. This is a pinned alias decision, not city/telephone inference; an
explicit conflicting North Korea/KP declaration still requires adjudication.
The documented dotted initialisms **`A.R.E` and `A.R.E.` map to `EGYPT`**:
Egypt's [official EGPC model agreement](https://eug.petroleum.gov.eg/dp/jsp/assets/docs/2023%20EGPC%20Model%20Agreement%20%28English%29.pdf)
explicitly defines A.R.E. as Arab Republic of Egypt, and the
[government postal example](https://www.fineart.gov.eg/Eng/About/connect.htm)
uses the dotted spelling. These are input aliases, not newly sampled country
presentations. Preserve punctuation in lookup: undotted **`ARE` remains ISO3
for UAE**, not an automatic Egyptian alias. A source pairing undotted ARE with
an Egyptian target requires a document-specific adjudication. Source `c7060e22`
was explicitly approved on 2026-10-16: retain its notify `EGYPT` target and its
printed `ARE` unchanged; this does not expand the global alias registry. See the
[edge-case inventory](kie-dataset-edge-case-inventory.md). Typos can
receive a document-local reviewed decision with its own evidence; that decision
does not create a global fuzzy alias or modify source OCR. Hong Kong and China
remain distinct country/territory identities, not interchangeable spellings.

Historical training references and metrics remain frozen. The country migration
is a separately validated dataset/source-authority revision; it is not a
retroactive rescore of earlier models. Source-authority proposals and unresolved
identity/ownership cases are recorded before the coordinated registry contract
and index refresh.

## Surgical real-training anonymization — 2026-10-16

For the user-authorized manual R27 real600 pass, replace private identities,
contacts, specific premises and identifiers in OCR and their dependent existing
targets. Preserve raw punctuation, whitespace, line/page boundaries, casing
style, phone dialing conventions and ID presentation masks. Preserve every
date, vessel/voyage, route/locality/country, cargo specification, quantity,
category and allocation; only private identifying substrings embedded in cargo
wording (VIN/serial/lot etc.) may change with their existing target copy.

**Unresolved nonzero address numbers are anonymized**, retaining width and
format; do not infer that they are geographic postcodes from country alone.
Clearly identified geographic postcodes and obvious all-zero nonidentifying
placeholders remain. Public carrier/issuer identity and its general corporate
directory may remain with recorded document-role evidence; personal employees,
signatories and mailboxes do not inherit that exemption. Being a publicly
searchable trader or a transport-sector consignee alone is not sufficient.

Original PDFs and historical sources remain protected. This manual training-copy
operation is not a synthesis campaign, not new ownership labeling, and not a
claim that retained shipment facts prevent re-identification. See
[the scoped report](kie-r27-real-training-anonymization-2026-10-16.md).
