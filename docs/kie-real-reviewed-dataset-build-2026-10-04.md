# Real V7 dataset build — reviewed batches 001–005

## Current revision — R10, 2026-10-05

**474 records: 427 train / 47 validation** at
`data/curated/mpci-bl-real-v7-reviewed-r10-reduced-474/`.
Removed 24 records missing both primary parties, the older valve-shipment duplicate
`002/015-9be27592` (keeping `005/001-0750aae7`), and dummy wrapper `005/014-fe260ee7`.
Two standalone abbreviated phone suffixes were removed from one retained record.
The user's previously excluded `004/022-eec05b59` reference was also corrected
without reintroducing it. R9 is the complete pre-edit backup; OCR and unrelated
labels remain unchanged. Repeated-party/goods groups and partial-page/container
records were analyzed, not automatically filtered. See the
[R10 report](analysis/real-v7-starting-dataset-2026-10-04/R10_CLEANSING_2026-10-05.md).
Use R10's frozen reduced schema and prompt; no training config was redirected.

## Previous combined revision — R9, 2026-10-05

**500 records: 450 train / 50 validation** at
`data/curated/mpci-bl-real-v7-reviewed-r9-reduced-500/`.
The cohorts below are now projected and combined: carrier party removed from both;
flags, marks and forwarding/export references removed from batch005 as already
done for R8. Four negotiability labels corrected from populated order-consignment
instructions. No further records filtered. OCR and all unrelated labels preserved;
the original cohorts remain immutable backups.

Use R9's frozen reduced schema/prompt together. The annotation flow remains full
scope; negotiability descriptions and the diagnostic gate now apply the clarified
consignee-wording rule. See the [R9 report and outlier inventory](analysis/real-v7-starting-dataset-2026-10-04/R9_REDUCED_500_2026-10-05.md).
Sections below describe historical states, not outstanding top-up work. No training
config was redirected or training launched.

## Completed top-up — 2026-10-05

Batch005 adds **167 reviewed records: 154 train / 13 validation**, at
`data/curated/mpci-bl-real-v7-reviewed-batch005-full/`. Together with the unchanged
333-record R8 baseline this is **500 records: 450 train / 50 validation** before
the next filters. Full annotation fields are retained in this batch and restored
to the active annotation flow as explicitly requested; R8 remains a frozen
reduced-field projection. The cohorts are not yet concatenated across contracts.

All 167 were manually adjudicated: 105 changed, 62 unchanged. Four duplicate/test-
like shipments were replaced by fresh records. Estimated total labeling cost was
$3.196966, including replacements. See the
[complete top-up report](analysis/real-v7-starting-dataset-2026-10-04/BATCH005_TOPUP_2026-10-05.md)
for validation, source-quality candidates and the next filtering discussion.

## Preserved R8 cohort — reduced-field baseline, 2026-10-05

**333 documents: 296 train / 37 validation** at
`data/curated/mpci-bl-real-v7-reviewed-r8-reduced-fields/`.
The user approved exclusion of only the two product-description OCR omissions,
`004/023-5fe4e926` and `004/106-d7cd79d3`. The five partial-container OCR candidates
remain included. Vessel flag, cargo marks/numbers and forwarding/export references
are removed from both splits and R8's frozen extraction/training schema snapshots.
The subsequent top-up instruction restores them to the full annotation flow;
the retained R8 files are unchanged.
Vessel IMO numbers remain. No removed text was reassigned to another label or
deleted from OCR. All other facts, original split assignments and ordering remain.

R7 is preserved in full, with its historical schemas and adjudication receipts.
R8 has current schema/prompt snapshots and exact removal receipts. Do not mix the
older unprojected V7 targets with the narrowed working contract. No historical
training configs were changed. See the
[R8 publication report](analysis/real-v7-starting-dataset-2026-10-04/R8_REDUCED_FIELDS_2026-10-05.md).

To restore the original 450-document baseline (400 train / 50 validation), add
**117 accepted documents: 104 train / 13 validation**. No top-up processing has
been launched. These are accepted additions needed, not a prediction of the
number of raw candidates required before quality filtering.

## Previous working cohort — input-quality filtered baseline, 2026-10-05

**335 documents: 298 train / 37 validation** at
`data/curated/mpci-bl-real-v7-reviewed-r7-input-filtered/`.
The approved 24 mixed/no-container input-quality exclusions plus the explicit
trailer exclusion `003/068-df5d72cd` are removed. R6 and the original 450-document
source backup remain unchanged. Retained OCR, labels, row order and splits are
byte-preserved. Every retained record has one described goods entry; 99 have
multiple labeled containers. No rare fields were removed and no training config
was redirected.

The [R7 field audit](analysis/real-v7-starting-dataset-2026-10-04/R7_FIELD_AUDIT_2026-10-05.md)
contains the full field census, marks/reference review, rare-field evidence and
new input-quality candidates awaiting a separate decision. The
[goods policy note](kie-real-baseline-label-policy-2026-10-05.md) records the user's
accounting-based distinction between shared goods and independent goods items.

## Previous working cohort — single-labeled-goods baseline, 2026-10-05

**360 documents: 319 train / 41 validation** at
`data/curated/mpci-bl-real-v7-reviewed-r6-single-goods/`.
From R5, the user requested exclusion of nine description-empty cargo records,
multi-goods-labeled records (42), and ten redundant shipment copies after selecting
the best-supported copy. Retained OCR and labels are unchanged. This is a target
entry-count restriction, not a new semantic goods-grouping policy.

The chosen duplicate representative is `001/02-44a6df44`. R5 and the complete
source snapshot remain unchanged. Fifty-six no-container records remain; their
source patterns and input-quality findings are documented in the
[baseline refinement report](analysis/real-v7-starting-dataset-2026-10-04/BASELINE_REFINEMENT_2026-10-05.md).
No additional container-based filtering or training was performed.

## Previous working cohort — first filter, 2026-10-05

The first filtered dataset has **421 documents: 373 train / 48 validation** at
`data/curated/mpci-bl-real-v7-reviewed-r5-filtered/`. User-approved filtering removed
21 PDFs over five pages, five scope/test/demo candidates and three confirmed
cargo-OCR omission cases. Retained OCR, labels and split assignments are unchanged.

The complete 450-document source backup is
`data/curated/mpci-bl-real-v7-reviewed-r4_source_20261005/`; the original R4 publication
also remains unchanged. See the [filtering report](analysis/real-v7-starting-dataset-2026-10-04/FILTERING_2026-10-05.md)
for exclusions, integrity checks and remaining review cohorts. Historical batch
counts below describe their original publication, not filtered membership.

## Historical publication: batch 004 after manual adjudication

**450 manually adjudicated documents are published at
`data/curated/mpci-bl-real-v7-reviewed-r4/`: 400 train / 50 validation.**
Batch004 adds all **200 requested documents (180 train / 20 validation)**.
Every new document received source review and a final manual decision, including
the automatic passes. **No new document is unresolved or excluded.** Original OCR,
PDFs and split assignments are unchanged, and all earlier 250 records are preserved.

The flow now permits **one editing wave**, followed by non-editing final review.
Batch004 processing cost an estimated **$4.03219512**; manual adjudication and
publication incurred no additional API charges. The closure section below contains
the counts, decisions, checks and limitations. Earlier sections are historical
execution records, not the current dataset state. No training or synthesis was launched.

## Scope and publication boundary

The user authorized manual adjudication/publication of the existing 50-document
pilot, followed by the complete pipeline on 100 new real documents. This is not a
new synthesis or training run. Previous datasets, original OCR, PDFs and raw agent
outputs are preserved. No Codex subagents were used.

The initial 50-document dataset snapshot lives at (current 450-record revision
is the `-r4` path above):

`data/curated/mpci-bl-real-v7-reviewed/`

It contains `train.jsonl`, `validation.jsonl`, a schema snapshot, per-document
`samples/<documentId>/ocr.txt` and `labels.json`, and batch-specific manifests and
validation receipts. Each training row contains `documentId`, `joinedRawText`,
`joinedRawTextSha256` and `target`; labels use the described V7 contract.

Automated candidates from the next batch are deliberately separate. Passing the
pipeline does not automatically append a document to this manually adjudicated
dataset. This is the requested batch/review/publication workflow, not another
whole-dataset repair campaign.

## Batch 001: published

All **50 documents** are included: **40 training / 10 validation**, preserving the
original source assignments. None was regenerated or dropped. The earlier pilot
had 46 issue-closed documents and four source-review documents; the latter are
now explicitly adjudicated under the existing extraction contract.

### Source decisions

| Pilot document | Decision and reason |
|---|---|
| `16-1d51aca4` | OCR gives both `0MRFKE1MA` and `OMRFKE1WA` for the single voyage. Remove the chosen voyage rather than select by majority. Keep the vessel. Retain the two distinct VAT strings and printed telephone strings in their list fields: those are literal extractions, not a claim that both are externally correct. |
| `36-b2617901` | Retain both explicit ACID reference strings (`565169637202302022`, `5651696372023020022`) in source order. This free-text reference list does not select an authoritative customs identifier. PDF inspection confirms the two blocks belong to the shipment; it does not license silently repairing OCR. The OCR does not contain the PDF's local 500/430 bag counts, so those are not introduced. |
| `49-da651f2f` | Keep `FINNISH WHITEWOOD` as the complete product description; remove inferred `origin.name=Finland`, because the goods-origin field requires an explicit origin/manufacture declaration. The source/PDF prints `KOS`, not an established canonical mass unit, so gross weight remains absent. The 148 packages and 465.332 cubic metres remain. |
| `50-e815e426` | PDF inspection confirms the gross/net contradiction exists in the document itself. Do not swap the local headings or select whichever total looks physically convenient. Keep both mass fields absent under the existing undecidable-value rule, retaining 1,400 bags and the two explicitly printed 700-bag placements. |

These decisions resolve **what the extraction target must contain**. They do not
claim that an internally contradictory source shipment has become factually
consistent. The exact conflicts remain in audit receipts, while original OCR is
unchanged. None is a silent conversion of an unreviewed model disagreement into
an absent label.

### Additional final label corrections

Manual examination of all 50 final target summaries and party objects identified
five labels with repeated postal components. Their source party blocks were
checked before editing:

- `15-f0c8de4d`: `PORT SAID(PORT SAID-EGYPT)` becomes `PORT SAID-EGYPT`.
- `17-9c68fe94`: `BAS/EGYPT Egypt` becomes `BAS/EGYPT`; postcode `31714` remains.
- `18-8239b894` and `43-150e9dad`: remove the repeated `TW` after `TAIWAN, R.O.C`;
  retain the complete country expression in the postal line. Document 43's
  separate country becomes the printed name `TAIWAN`, consistent with document 18.
- `39-7ebdff63`: remove the second `NASR CITY`; block G, lots `(7,10,11)` and
  postcode `11816` remain untouched.

Earlier independently inspected manual decisions are preserved too: explicit
notify-party restoration in document 13; source country/postcode ordering in 36;
comma-only address formatting in 40; goods-origin caption cleanup in 43; and
rejection of PDF-only values in 22/26. Receipts distinguish earlier edits from
this publication pass's edits.

In total, **10 documents have manual label changes** across these stages.
This final publication pass edits seven documents; the remaining decisions
confirm an already-correct target without inventing a change.

### Verification

`scripts/labeling/publish_real_v7_batch1.py` applies finite, explicitly reviewed
edits with exact preconditions. They are not source-specific rules in the
production extraction code. It stages the complete dataset before publication.

- 50/50 V7 targets validate and canonicalize.
- 50/50 original OCR hashes, original PDF hashes and split assignments verify.
- 50/50 exact edit receipts replay to the published target, with no unrecorded
  changes outside the reviewed paths.
- **145/145 source-fixed checks pass**, recomputed on the published targets.
  These include the earlier 140 controls and five final adjudication checks.
- Independent readback checks all 50 JSONL records against per-document copies,
  source hashes and edit receipts, including placement-last target ordering.
- Integrity falsification rejects 50 changed-OCR mutants, 50 unrecorded-label
  mutants and 50 split-switch mutants. These test publication integrity; they
  are not a numerical estimate of semantic field accuracy.
- Relevant maintained suites: **570 tests passed in 23.20 seconds**.
- Publication took **1.70 seconds**; this pass made **zero paid model calls**.

The relevant receipts are under the dataset's `batches/001/` directory.
The earlier 50-document experimentation cost is not charged again or hidden:
the previous R5 report records an estimated **$1.52810170** across its fresh and
bounded follow-up phases.

## Batch 002: selection and execution

Selected **100 new documents: 90 train / 10 validation**, seed `2026100402`.
The selection excludes all 70 prior direct-pilot IDs and all prior OCR hashes.
The remaining eligible pool at selection was 1,086 records. Existing labels are
not passed to any agent, and are not copied into the agent input folders.
Each source PDF is resolved through its provenance record and verified by hash.

Output directory:

`artifacts/kie-labeling/direct-real-batch002-20261004/`

The existing maintained pipeline is unchanged:

1. Complete plain OCR → fresh full V7 extraction.
2. Five section reviews plus candidate-blind cargo source mapping.
3. Full original PDF for cargo mapping and goods/container relationship review.
4. Up to two scoped correction waves and affected-section re-review.
5. Explicit passing-candidate versus adjudication status; no automatic gold export.

Configuration: Luna, high reasoning, strict native structured output, concurrency
16. The runner captures request schemas, OCR hashes and complete-PDF hashes,
freezes implementation/configuration, records all attempts and token usage, and
stops admissions on provider credit/authentication/schema failures. There is no
implicit request retry or silent overwrite.

`scripts/labeling/run_real_v7_batch.py` reuses the maintained `DirectLabelingFlow`;
it does not introduce a second extraction implementation. Its request proof uses
the native output format's **name**, because the wire schema may omit its root
title. This verifies that cargo requests actually include the complete PDF.

### Completed execution

All **100 documents completed** with application-valid targets. There were no
failed documents, stopped admissions, unknown-billing calls or additional bulk
reruns. The final automated result is:

| Original split | Passing candidates | Adjudication queue | Total |
|---|---:|---:|---:|
| Train | 67 | 23 | 90 |
| Validation | 6 | 4 | 10 |
| **Total** | **73** | **27** | **100** |

The complete extraction/review/correction run cost an estimated **$2.25842053**
(**$0.02258 per document**, including all attempts). It took **1,695.89 seconds /
28.26 minutes**, with peak concurrency **16** and peak RSS **718,976 KiB /
702.13 MiB**. These measurements include the live pipeline, not preceding manual
work or subsequent local validation. They are not a controlled performance
comparison against the different 50-document pilot.

All **1,345 calls** are accounted for: 100 extraction calls, 830 section reviews,
100 cargo maps, 155 relationship reviews and 160 scoped correction calls.
Usage: 12,302,337 input tokens (3,634,628 cache-read; 8,663,674 cache-write),
2,277,423 output tokens, including 1,896,716 reasoning tokens. Reasoning is not
charged twice. The estimate uses the recorded per-million-token rates: $0.10
ordinary input, $0.01 cache-read, $0.125 cache-write and $0.50 output. This is an
estimate from response receipts, not a provider invoice.

### Request and result validation

`scripts/labeling/validate_real_v7_batches.py` independently read back the actual
request/response artifacts and the published first batch:

- **100/100 final targets** validate against the application schema.
- **1,345/1,345 responses** validate against their captured strict native wire
  schema; zero wire-schema errors.
- **1,345/1,345 requests** contain the exact selected OCR and use high reasoning.
- **255/255 cargo-map / relationship requests** contain the complete selected
  PDF with the correct hash.
- Frozen source, PDF, implementation and configuration hashes remain unchanged.
- All-attempt costs and call counts reconcile with the runner's final ledger;
  zero unknown-billing calls.
- The first 50 published records still pass independent readback, edit replay,
  source integrity and split checks after the second batch finishes.

Four calls failed **application-level** validation despite native-schema-valid
JSON: two initial extractions, one cargo map and one correction. These are not
transport failures or silently accepted labels. The two initial drafts enter the
explicit review path and their documents finish in the adjudication queue; the
failed correction likewise remains held. The failed cargo map in `014-6181bac5`
was explicitly replaced by the relationship reviewer; the saved replacement
passes independent `CargoSourceMap` and OCR-grounding validation. Its unit-less
printed weight remains absent. Both the failure and replacement receipts remain.

Schema/integrity checks establish those specific properties; the **73 passing
candidates are not silently promoted to manually adjudicated gold**. This keeps
the requested batch-review boundary intact. The published dataset currently
contains the first **50** documents, not 150.

### Next manual review: a finite 27-document queue

`adjudication-queue.json` records each document's OCR/PDF/target paths, final
section findings, explanations and proposed corrections. Section counts overlap
for one document:

| Section | Documents with remaining findings | Examples / work to adjudicate |
|---|---:|---|
| Cargo | 9 | Reference text inside description; product text inside marks; unsupported HS/count values; cargo-map or correction validation failures. |
| Parties | 8 | Carrier-role evidence; contact ownership; uncaptioned personal names; duplicate/boundary formatting; unsupported postal country. |
| Metadata / freight | 7 | Missing year-first dates; issue/on-board date ownership; payment-place ownership or conflicting instructions; reference completeness. |
| Route / transport | 4 | Competing voyage strings; unsupported vessel correction; omitted discharge role or printed port country. |
| Equipment | 0 | No outstanding equipment findings in the final queue. |

These are **review findings, not 27 proven irreparable sources or a count of
confirmed errors**. Some are a blocked correction or a disputed interpretation,
which must be checked against OCR and layout. For example:

- `048-ffa8239a`: voyage identifiers differ by `1` versus `I`, a single-value
  conflict of the kind already adjudicated in batch 001.
- `007-30116fcc`: the reviewer identifies omitted explicit year-first issue and
  on-board dates; this requires checking the saved final candidate and source.
- `064-5facbbbd`: the HS string `40139090` fails literal OCR grounding.
- `024-353f18ac` / `035-2f4d1db9`: cargo-map numeric evidence fails its validator;
  inspect the numeric statement rather than regenerate the document.
- `098-1891d90d`: a map-only finding attempted to change target fields and was
  rejected by the correction-scope guard.

The next batch-review step is to adjudicate those finite findings, verify the
candidate outputs against the agreed extraction policies, record exact edits,
and append the approved batch atomically. No further model requests, new batch,
dataset-wide processing or training were launched after this run.

Artifacts: `timing.json` contains the all-attempt ledger; `validation.json`
contains the reconciled request checks and per-document findings;
`adjudication-queue.json` contains the 27 held cases. Complete original and revised
targets, section decisions, PDF requests and response receipts remain under
`runs/<batch-document>/`.

## Batch 002: manual adjudication and publication completed

All 100 OCR/target pairs were reviewed, not just the 27 automated holds.
PDF layout was inspected where needed; 28 rendered page images remain in the
manual-review directory. PDF-only values were not introduced into labels.
All 27 holds are resolved under the agreed extraction policies. No document was
dropped and no user decision remains outstanding for this batch.

### Changes and examples

85 documents received label changes; 15 required no change. These are document
counts, not an estimated error rate: corrections range from formatting and
canonical equipment categories to missing/incorrect semantic assignments.
There are 204 finite reviewed edit operations plus 117 equipment-category edit
operations (39 complete printed equipment phrases, each replacing one raw field
with the two established categories). All 321 operations have exact before/add/
remove receipts; compound edits such as a whole goods list count as one operation.

| Label section | Documents changed | Examples and policy applied |
|---|---:|---|
| Goods | 42 | Restore selected handling/free-time instructions; remove shipment boilerplate and identifiers belonging elsewhere; preserve product codes in description; correct inner package totals and goods/container ownership. |
| Parties | 31 | Restore OCR-present contact/address continuations; deduplicate postal country/locality; retain postcodes/site numbers; distinguish carrier, destination agent, and uncaptioned identity names. |
| References | 21 | Restore explicitly captioned shipment/tax identifiers; remove package marks and product codes incorrectly duplicated as forwarding references. |
| Freight | 19 | Remove inferred or conflicting payment locations and placeholder ORIGIN/DESTINATION values; correct selected payment arrangement. |
| Containers | 12 | Normalize complete printed equipment phrases; correct seal ownership; preserve incomplete printed type information without inventing dimensions. |
| Route | 12 | Remove copied discharge-to-delivery labels; restore explicit role/country wording and keep meaningful OLD/EAST distinctions. |
| Issue place | 5 | Remove inferred country; restore OCR-present named issue place using layout context. |
| Transport | 4 | Voyage/vessel ownership, caption cleanup, and omission of an unresolved conflicting voyage. |
| On-board / issue dates | 4 / 3 | Recover clearly printed year-first dates; leave undecidable or PDF-only values absent. |
| B/L number / negotiability | 1 / 1 | Remove an unrelated stamp serial; apply explicit consignee order wording. |

Counts overlap. Concrete decisions include:

- `035-2f4d1db9`: the DG item prints four inner plastic receptacles inside an outer
  fibreboard box. The package target is four plastic receptacles, preserving the
  0.5 L capacity in description and the printed EMS handling instruction.
- Document `037` in the batch receipt: two printed
  800-bag portions of the same goods become a 1,600-bag total, not a duplicated
  product. The exact document ID is authoritative in the manifest.
- `044` and `082`: seals were attached to the wrong container after page/row
  flattening. Ownership was checked against the source layout; an orphan seal
  does not get attached to the next identified container.
- `062`: a printed 20,000 KG product statement is not a labeled net weight.
  Keep the source-supported description; do not import the PDF-only net caption.
- `019`, `066`, `067`: package-label invoice/order/PO identifiers stay in cargo
  marks, not duplicated as forwarding references; product part/GPC codes stay in
  description. This follows the existing field descriptions, not a new policy.
- `077`, `083`, `090`: related HMM document variants are all in training. Their
  local OCR omissions remain distinct; one variant's fields are not copied into
  another. No equal allocation is inferred from a shipment total.
- `092`: the source's decimal-comma volume is 189.0 cubic metres; the complete
  1,938-carton total is retained instead of one 646-carton portion. PDF layout can
  disambiguate printed separators but cannot introduce missing container IDs.
- `096`: 3,065 and 4,015 printed packages of the same goods sum to 7,080; the
  explicitly supported 4,015-container placement remains, without inventing the
  missing other assignment.

The source can contain conflicting values without making its extraction target
unresolved: for example, `048` has conflicting voyage strings and `073` has an
ambiguous numeric date. Their singular fields remain absent by a documented
decision, while all extractable information is retained. Similarly, lack of a
per-product package breakdown does not authorize an equal split or residual guess.

The original automated statuses are preserved: 60 of the 73 passing candidates
received manual changes; 13 did not. Of the 27 held candidates, 25 received changes
and two were accepted unchanged after source review. This is direct evidence
that automated passage alone is insufficient for publication; this batch is
accepted on the completed manual review plus validation, not on its earlier
73/27 status split. The same reviewed-publication boundary applies to batch003.

### Publication and validation evidence

`scripts/labeling/publish_real_v7_batch2.py` applies the finite decision inventory.
It fails on stale values, additions that overwrite fields, missing decisions,
schema inconsistencies, changed OCR/PDF, split changes or receipt mismatches.
The final executed script is preserved with its manifest-matching hash at
`artifacts/kie-labeling/direct-real-batch002-20261004/manual-review/publisher-executed.py`;
subsequent script edits are formatting only.

- 150/150 targets validate and canonicalize under V7; placement remains last.
- 150/150 exact edit replays reproduce the published target with no unrecorded
  label changes, and match per-document files and aggregate JSONL.
- Original source OCR/PDF hashes and original train/validation assignments verify.
- IDs and exact OCR hashes are unique across the combined dataset.
- A cross-split check also finds no shared normalized B/L number or container ID
  between its 130 training and 20 validation records.
- All 105 pre-existing non-aggregate files were preserved byte-for-byte.
- All 100 batch002 documents have explicit manual decisions and zero outstanding
  document adjudications.
- Independent readback of the actual batch002 calls reconfirmed 1,345 strict,
  high-reasoning OCR requests/responses; all 255 cargo requests included full PDF;
  no native-schema errors or unknown-billing calls.
- Actual negative controls against an isolated copy reject coordinated label
  edits in both JSONL and per-sample files, coordinated OCR/hash edits, duplicate
  and missing records, stale before-values and overwrite-through-add. Restoring
  the original copy returns all 150 passing. These are integrity tests, not a
  manufactured percentage of semantic accuracy.
- Relevant maintained tests this turn: **567 passed** (76 in 18.95 s; 491 in
  13.24 s). Ruff passes for the three changed/new publication/validation tools.
- Publication staging/validation: **8.76 s**, peak RSS **172,936 KiB**. The isolated
  negative-control run took **6.95 s**. Production extraction was unchanged, so no
  extraction performance regression is introduced by these offline tools.

The filesystem rejected renaming the existing dataset directory on the Windows
DrvFS mount. No old dataset was modified. Publication instead uses the explicitly
named `-r2` revision and a fresh-directory rename after staging; the original
50 remains an intact snapshot. This is why the current dataset path differs.
The 150-record revision is the one to use going forward. Its batch002 manifest
contains current aggregate hashes; historical batch001 aggregate hashes still
describe the original 50-record snapshot.

Three temporary staging copies created during publication checks were removed
after confirming their OCR was preserved in the published revision. No source,
paid model result, original dataset or final reviewed label was deleted.

No paid model calls were needed for this manual review/publication: **$0 additional
API cost**. This does not erase the original batch002 run's estimated $2.25842053.

## Batch 003: fresh 100-document run

Selection: 90 train / 10 validation, seed **2026100403**, 986 eligible sources at
selection, excluding 170 prior pilot/batch IDs and prior OCR hashes. An additional
explicit check establishes zero ID/OCR overlap with all 150 published records.
Original splits remain unchanged. The agent receives fresh OCR, not old labels.

Output: `artifacts/kie-labeling/direct-real-batch003-20261004/`.
Configuration and flow are unchanged: Luna/high, native strict PydanticAI output,
concurrency 16, five section reviews, full-PDF cargo mapping/relation review and
bounded scoped correction. Implementation and input snapshots are frozen.
### Completed execution

All 100 selected documents completed the bounded pipeline. There were no
document-level transport failures, stopped admissions, missing selected records
or unknown-billing calls. Final automated statuses:

| Original split | Passing candidates | Adjudication cases | Total |
|---|---:|---:|---:|
| Train | 61 | 29 | 90 |
| Validation | 8 | 2 | 10 |
| **Total** | **69** | **31** | **100** |

The run took **1,756.93 seconds / 29.28 minutes**, concurrency 16, peak RSS
**736,800 KiB / 719.53 MiB**. Estimated all-attempt API cost:
**$2.22876718**, or **$0.02229 per selected document**. These are receipt-derived
estimates under the recorded rates, not a provider invoice. No bulk rerun was
launched. The run made **1,318 calls**: 100 extractors, 801 section reviews,
100 cargo maps, 156 relationship reviews, 161 scoped corrections.

**98 records have application-valid final targets.** The remaining two,
`003-cd075e92` and `078-4704110c`, are explicit adjudication records, not accepted
labels. Their original strict-schema drafts, validation errors, reviews and
rejected proposals are retained. In 003 the draft contains physical line breaks
inside semantic goods values and a proposed numeric repair failed OCR grounding;
in 078 an invalid printed container identifier and an unsupported replacement
prevented a valid final target. Do not silently normalize these into accepted
labels without the manual source/ownership review.

Three extraction calls failed application validation despite native-schema-valid
JSON: the two above plus `008-11c21de2` (invalid IMO check digit). One correction
call for 008 also failed application validation. Its final target is schema-valid
but the source/identifier decision remains held. These failures remain in the
all-attempt cost and review records, rather than being erased by a rerun.

### Completed request/result validation

Independent readback validates all **1,318 responses against their captured
native strict schemas**, with zero wire-schema errors. All 1,318 requests contain
the exact selected OCR and use high reasoning. All **256 cargo-map/relationship
requests** include the complete selected PDF with its matching hash. Frozen
source, PDF, configuration and implementation hashes verify. All 98 available
final targets validate against the application schema; the two absent targets
remain explicitly held. The 150 published records separately pass exact replay,
schema, OCR/PDF provenance and per-record/aggregate readback again after the run.

All-attempt usage reconciles to the runner: 12,180,926 input tokens (3,593,728
cache-read, 8,583,244 cache-write), 2,239,058 output tokens including 1,855,231
reasoning tokens. Reasoning tokens are not added again to output cost. Zero
unknown-billing calls. The validation receipt and complete ledger are in
`validation.json` and `timing.json` in the batch003 directory.

### Original pre-adjudication queue (closed by the review below)

`adjudication-queue.json` records the 31 held documents with OCR, PDF and available
target paths, reviewer explanations and proposed corrections. Section counts
overlap:

| Section | Held documents with findings | Main review work |
|---|---:|---|
| Cargo | 21 | Numeric/source-map interpretation, exact sums, product/marks ownership, package levels and unsupported correction proposals. |
| Route / transport | 5 | Source-literal voyage/IMO values, missing or conflicting role evidence. |
| Metadata / freight | 4 | Negotiability, reference identity/format duplication and selected freight roles. |
| Parties | 4 | Missing OCR-supported contacts/continuations, carrier-versus-agent role and PDF-only identity information. |
| Equipment | 2 | Invalid/extra-character source IDs and attempted unsupported identifier substitutions. |

These are review findings, not 31 confirmed irreparable documents. No new user
policy decision has been established as necessary at this stage. The next action
was manual review of batch003, with the flagged 31 and checks of the other
69 before publication—consistent with what the batch002 audit demonstrated.
At that pre-adjudication checkpoint no batch003 record had been appended.

## Batch 003: completed source adjudication and publication

**All 100 records are adjudicated and accepted under the agreed OCR-grounded
label policy. The combined dataset is now 250 documents: 220 train / 30
validation. No batch003 document remains held, and no user policy decision is
needed to close this batch.**

Current revision: `data/curated/mpci-bl-real-v7-reviewed-r3/`.
The preceding `-r2` revision remains intact. No new batch, training run or
synthesis run was launched in this pass.

### Review scope and acceptance basis

I reviewed the complete OCR and candidate labels for **every selected document**,
including all 69 automatic passes and all 31 held cases. Where reading order,
column assignment or party ownership needed clarification, I consulted the
source PDF. The paid extraction/review artifacts remain unchanged. The PDF is
layout/context evidence, not permission to import absent words, numeric values,
units or separate field occurrences into OCR-trained labels.

The accepted new records contain 414 party objects (316 address lines and 27
explicit same-as links), 126 goods items, 166 identified containers, 174 goods
placements, 96 HS-code entries and seven DG declarations. Every extraction
section was included in the source review; this was not an address-only or
held-cases-only inspection.

81 records changed and 19 remained unchanged. **55 of the 69 automatic passes
needed at least one change.** This includes formatting/normalization changes as
well as substantive errors; it is not a claim that all 55 had equally serious
defects. Conversely, five held candidates needed no label edits after their
uncertainties were resolved. Automatic review status is therefore not the
certification criterion. The current batch is accepted on its completed manual
source review and recorded finite adjudications. The observed error rate does
not support skipping that review in later batches.

### Main resolutions

| Family | Representative decisions |
|---|---|
| Product identity and allocation | 045's three identical product entries became one goods item with three exact container portions, 42 pallets and the complete gross/volume totals. 007's unallocated yarn assortment remains joint rather than acquiring PDF-only 472-carton splits. 036's commercial film-chemical assortment is not counted again on top of its DG subdeclarations; the complete commercial total and all three distinct DG classes are retained together, without inventing a PART A/B/C-to-DG mapping. |
| Inner packaging and quantities | 082 retains the one pallet actually present in OCR, not the 13 inner cartons visible only in PDF. 093 retains 13 independently quantified products with their inner drums, not a second eight-pallet aggregate. 090 recovers the omitted 70 SETS and sole-container allocation. 099 uses 20 printed racks, with five per container, rather than treating “5 Racks” as a product variant. |
| Product/marks/reference boundaries | Product codes, lot qualifiers and capacities move to description; package-label PO/invoice numbers remain marks. Administrative/customs references remain outside both. Source mark order and distinct printed spelling variants were checked explicitly in 020/041. |
| Missing supported facts | 061 recovers all four ten-digit HS codes. Other corrections restore owned phones, contact continuations, party registration/tax references and explicitly selected free-time instructions. No inferred registry HS codes or seals were added. |
| Party identity and postal text | Postal components and postcodes remain complete, with commas at component boundaries. Contact/tax/company-status text is excluded from addressLine. 056 retains the clearly primary company postal address rather than concatenating a second contact address. 026's owned contact continuation is recovered without adding its PDF-only company identity. 100's branch identity moves out of the postal address. |
| Country and route ownership | 034's exporter registration country is not borrowed into the shipper address; the separately owned notify-country continuation is included. 034/082 do not copy a discharge occurrence into a PDF-only delivery field. Distinguishing locality words such as OLD PORT remain, while generic facility descriptors are removed. |
| Freight and negotiability | Selected values are distinguished from empty form captions and stock legal text. 071's empty Prepaid/Collect columns do not authorize prepaid. 055's filled prepaid term and actual named consignee govern over generic captions. “As arranged” and a named payment locality alone do not automatically become payable_elsewhere. |
| Numeric interpretation | Complete printed component sums are allowed. 091's `22863 500KGS` is 22863.500 kg: the PDF confirms the decimal position using the same OCR digits. Unitless quantities are not silently assigned kg/CBM, package capacity is not shipment mass, and cubic feet are not relabelled cubic metres. |
| Invalid identifiers and source omissions | 050's malformed five-letter container token is not “repaired” by guessing a deleted letter. 078 retains its valid container and supported goods but omits an invalid second identifier and unsupported foreign key. 008's invalid IMO is not corrected by guessing its check digit. Original OCR and raw values remain available. |
| Boilerplate | Generic all-vehicle damage clauses and general tariff/liability text are excluded from product descriptions/handling. Explicitly selected cargo instructions are retained. |

These are finite data adjudications, not new source-specific production
heuristics. The full explanations for all 100 documents—including records left
unchanged—are in `batches/003/manual-decisions.json` and the manifest receipts.
Unrepresentable or unallocated facts remain in the original OCR and audit; they
are not claimed as facts of an arbitrary goods item. In particular, 084's
“some coils” rust/strapping remarks cannot establish which of six grades is
affected, and 098's shipment mass cannot be guessed across its two products.
These optional-field decisions are resolved policy outcomes, not silently
pending document reviews.

### Implementation and validation

- `scripts/labeling/inspect_real_v7_batch3.py` provides original line numbers and
  full candidate labels. The two missing final targets are explicitly identified
  as reviewed-draft inputs, not silently accepted or replaced.
- `scripts/labeling/adjudicate_real_v7_batch3.py` records the finite manual
  decisions. Every selected index must have a decision and rationale.
- `scripts/labeling/publish_real_v7_batch3.py` applies those decisions, records
  exact before/add/remove edits, validates the result, stages a fresh revision
  and publishes it without altering the prior dataset. Arrays are replaced
  atomically where changed; 233 receipt operations are **not** 233 independent
  semantic errors. Canonical removal of explicit nulls in the two rejected drafts
  is included in the receipts.
- The publication probe now accepts explicit dataset/output paths. It tests the
  actual readback validator on an isolated copy, not on production data.

All **250 final records** pass application-schema validation, canonical target
comparison, exact edit replay from the original paid target/draft, OCR/PDF/source
hash checks, unique IDs/OCR hashes, original split preservation, JSONL/sample-file
agreement and goods-placement-last ordering. The preceding 150 JSONL records are
byte-identical prefixes of the new files; all **310 historical files** copied
into the new revision retain their hashes. The original input text was not edited.

Additional readback checks cover all 126 new goods items: no placement names an
absent container, no duplicate container identifiers, no allocation quantity
exceeds its fully specified package total, and no comparable net mass exceeds
gross mass. A lexical absence screen over descriptions, addresses, names, marks,
handling, contact numbers/emails, seals, container IDs and HS codes found no
unsupported alphanumeric tokens after presentation normalization. This last
screen is only an absence check: ownership/completeness were established by
source review, not inferred from whole-document token presence.

Six deliberately corrupted publication cases were rejected: jointly changing
JSONL and sample labels without a receipt; jointly changing OCR and its local
hash; duplicate record; missing record; stale edit-before value; and an add edit
that tries to overwrite an existing key. The restored positive control passes
all 250 again. These controls establish integrity behavior, not a fabricated
semantic-accuracy percentage.

The original 1,318 agent responses were revalidated against their captured native
schemas, with zero wire-schema errors; exact OCR and all 256 full-PDF cargo
requests were rechecked. The original automated statuses remain untouched in
those artifacts; the new manual receipts supersede them for dataset acceptance.

Maintained extraction/review, training-v7, source-projection and agent tests:
**486 passed in 20.70 seconds**; underlying B/L and MPCI schema tests:
**65 passed in 1.21 seconds** (**551 tests total**). Ruff passes on the four new/changed review and
publication tools (long literal source/adjudication strings are deliberately not
reflowed). Publication staging/validation took **15.74 seconds**, peak RSS
**310,144 KiB**; the six isolated negative controls took **11.93 seconds**.
This is an offline data publication change: production extraction prompts,
models, schemas and inference performance were not changed.

**Additional API cost: $0.** No paid re-extraction or model review was used in
this adjudication. The earlier batch003 pipeline cost remains **$2.22876718**
estimated, all attempts included; it has not been reset or excluded.

One generated dry-run staging copy was removed after verifying that all 250 OCR
copies were preserved in the published revision. No source, paid result, prior
dataset revision or final reviewed label was deleted.

Review and validation artifacts:

- `data/curated/mpci-bl-real-v7-reviewed-r3/batches/003/manifest.json`
- `data/curated/mpci-bl-real-v7-reviewed-r3/batches/003/manual-decisions.json`
- `data/curated/mpci-bl-real-v7-reviewed-r3/batches/003/validation.json`
- `data/curated/mpci-bl-real-v7-reviewed-r3/batches/003/combined-readback-validation.json`
- `artifacts/kie-labeling/direct-real-batch003-20261004/manual-review/publication-negative-controls.json`

Current combined-dataset readback:

```bash
UV_CACHE_DIR=/tmp/documentparsing-uv-cache uv run --no-sync --with jsonschema \
  python scripts/labeling/validate_real_v7_batches.py \
  --dataset data/curated/mpci-bl-real-v7-reviewed-r3
```

## Reproduction

Batch preparation/run (already launched; do not repeat into the same directory):

```bash
UV_CACHE_DIR=/tmp/documentparsing-uv-cache uv run --no-sync --with jsonschema \
  python scripts/labeling/run_real_v7_batch.py prepare \
  --output artifacts/kie-labeling/direct-real-batch003-20261004 --seed 2026100403
UV_CACHE_DIR=/tmp/documentparsing-uv-cache uv run --no-sync --with jsonschema \
  python scripts/labeling/run_real_v7_batch.py run \
  --output artifacts/kie-labeling/direct-real-batch003-20261004
```

Readback/actual-request verification:

```bash
UV_CACHE_DIR=/tmp/documentparsing-uv-cache uv run --no-sync --with jsonschema \
  python scripts/labeling/validate_real_v7_batches.py \
  --dataset data/curated/mpci-bl-real-v7-reviewed-r2 \
  --batch artifacts/kie-labeling/direct-real-batch003-20261004
```

No active training configuration, model, original real labels or synthesis
templates were changed. This batch's validation documents are part of a reviewed
development dataset, not a newly untouched benchmark.


## User adjudication: batch002 document 021 (2026-10-04)

The user directed removal of `AMRELGAMMAL257@GMAIL.COM` from the consignee
label because it is printed in the goods area, not directly in the consignee
block. Original OCR/PDFs, other labels and the historical r2 snapshot remain
unchanged. The earlier interpretation that CN> authorized this email is
superseded. Future processing must respect the requested party-block boundary;
this narrow amendment does not perform a broader dataset sweep or prompt change.

The original automated final target already omitted the email; the prior claim
that its removal was an agent error is withdrawn. Current batch002 counts are
**84 changed / 16 unchanged**, including **59 of 73** automatic passes changed.
Earlier report counts describe the pre-amendment publication. Dataset size remains
**250 (220 train / 30 validation)**. The exact prior receipt and removal are saved
in `data/curated/mpci-bl-real-v7-reviewed-r3/batches/002/user-adjudication-021.json`.

Post-amendment validation passed for all 250 schemas and exact edit replays, with
750 integrity controls rejected as expected. The original OCR, all unrelated
files and every other historical training row remain byte-identical. Target
field ordering is preserved. Validation took 5.641 seconds; no paid calls.

## Batch 004: final 200-document batch — completed

### Bounded flow changes

The user's instruction was small, general clarifications and proven validator
fixes, removal of the second editing round, then a fresh 200-document batch and
manual resolution/publication. No new agent layer or bulk-repair loop was added.

- `direct.py`: one correction wave only. Changed/dependent sections are still
  reviewed afterward, but that final review cannot authorize another edit. Its
  remaining findings go to manual adjudication. Removed obsolete second-wave
  retry/fingerprint bookkeeping. Existing tests now exercise a new finding and a
  proposed reversal without allowing a second correction request.
- `direct_grounding.py`: permit presentation hyphens in HS codes while preserving
  all digits and rejecting prefixes/suffixes of longer continuous/dotted/hyphenated
  codes. The 250-record preflight changed only the four previously known false
  absence flags in one record. It does not widen ownership inference.
- V7 field descriptions and section priorities: consignee information must be in
  its own printed block; year-first dates have an explicit interpretation; freight
  payment place is separate from selection of payment terms; a generic country
  mentioned in carrier conditions is not a route country; observed cargo condition
  and explicit per-package capacity are distinct from generic damage clauses and
  repeated package counts. Updated CLI/help and the batch runner to one-wave wording.

The earlier user decision excluding batch002 document021's goods-area consignee
email remains applied. Shipper/notify/delivery-agent continuations were not silently
forbidden by the narrower consignee policy.

### Selection, execution and spend

Seed `2026100404` selects **180 train / 20 validation** sources. Both IDs and OCR
hashes of the previous 270 pilot/batch documents were excluded; the eligible pool
was 886. Existing labels were not supplied to the extractor. All original source
PDF/OCR hashes and split assignments are recorded in `selection.json`.

Luna/high, native strict structured output, PydanticAI, concurrency16:

| Execution measure | Result |
|---|---:|
| Completed documents | 200 |
| Automatic passing candidates | 104 |
| Automatic adjudication cases | 96 |
| Application-valid final automated targets | 196 |
| Captured requests / native-schema-valid responses | 2,392 / 2,392 |
| Cargo requests with the complete source PDF | 488 |
| Unknown-billing calls | 0 |
| Estimated API cost, all attempts | $4.03219512 |
| Cost per document | $0.020161 |
| Pipeline wall time | 2,910.293 seconds / 48.50 minutes |
| Maximum simultaneous requests | 16 |
| Peak process RSS | 914,140 KiB / 892.71 MiB |

Four documents (005,042,195,199) needed manual repair of preserved strict-schema
drafts before application-level validation. None was dropped or silently replaced.
All 2,392 captured responses pass their actual submitted native schemas; this does
not mean every value was semantically correct. There were no unknown-cost calls,
no admission stop, and no second automated editing wave. Model-run receipts are
preserved unchanged alongside the later manual decisions.

### Manual review and decisions

All 200 OCRs and final candidate labels were individually reviewed, with PDF
inspection for layout/ownership where needed. Reviews cover the complete target,
not merely the fields flagged by agents. The final labels differ from the automated
candidate in **163 documents**; **37 are unchanged**. Of the 104 automatic passes,
**73 have final label changes**. These counts include formatting, normalization,
omissions, semantic corrections and canonical serialization; they are **not** a
document error rate or a field-accuracy measurement. Automatic passing remains a
candidate status, not permission for unattended gold publication.

Net changed documents by section (overlapping, not additive):

| Section | Documents | Main decisions |
|---|---:|---|
| Goods | 98 | Product identity versus repeated portions; inner packaging; unknown mixed-container quantities; product/lot/GPC text; masses/units; removal of generic conditions. |
| Parties | 73 | Full postal wording, comma boundaries, numeric retention, identity versus address, country aliases, contact ownership and permitted continuations. |
| Containers | 30 | Complete carrier-equipment categories versus partial wording, seal boundaries, correct temperature ownership. |
| Route | 26 | Locality/country normalization, source-owned roles, removal of PDF-only/unjustified locations. |
| References | 21 | Distinct captioned references, duplicate administrative identifiers, product codes moved into product descriptions. |
| Freight | 19 | Explicit prepaid/collect terms versus payer/payment locality; exclude empty captions. |
| Place of issue | 10 | Correct field ownership and printed country normalization. |
| Main transport | 6 | Vessel/voyage boundaries and exclusion of unrelated legs. |
| Primary B/L number | 5 | Actual document identifier, not an unrelated/unsupported number. |
| Issue date / onboard date | 3 / 3 | Role-aware dates; ambiguous or OCR-absent values left absent. |

Examples of the adopted decisions:

- 027/069/072/094 retain a shared `A ON BEHALF OF B` identity where the source
  provides one shared postal block. This is different from combining two competing
  companies with different addresses. A cross-document consistency check corrected
  overly narrow initial manual names before final delivery.
- 053's linked email belongs to the notify party, not the consignee. 171's
  goods-area consignee email remains excluded. 167's matching continuation markers
  reconnect a delivery-agent email without putting those markers in the email.
- 086 retains independently accounted `LENOL 30 TP 30` across its separate and
  mixed container portions. The mixed container's shared 24 IBC is not assigned
  independently to each product. An initially proposed regrouping was reversed in
  an explicit amendment after checking the existing goods-identity policy.
- 067 retains source-distinct 472/467 carton package rows. 177's two quantified
  kraft grades do not each receive all eight container memberships merely because
  the containers occur elsewhere in the same document. 200's first-container
  20-pallet count is not relabeled as the full shipment count.
- 153 preserves `0016781/SIF1900` as a compound seal identifier alongside the
  separate carrier seal, consistent with 162's analogous printed compound seal.
- 191 prints inconsistent product/part-code combinations. The target preserves
  the three combinations actually printed and their supported portions; it does
  not invent a corrected code or distribute the first container's shared quantity.
- 173/195 exclude masses without the required OCR-owned mass meaning/unit. PDF
  headings absent from OCR cannot authorize those labels. 199 does not invent
  digits to turn an invalid container-like token or four-digit HS fragment into
  a schema-valid identifier.
- Final lexical readback caught the PDF-only `PORT SAID WEST` discharge label in
  049; the whole discharge value/country was removed, not merely its `WEST` token.
  A source mentions `PORT SAID` in another role, which does not supply discharge
  evidence. This demonstrates why the final readback was retained.
- The same established carrier-equipment grammar used by preceding batches is
  applied consistently to complete `DC20`, `20 DRY VAN`, `40'OT`, `40RH` phrases.
  It does not infer types from bare lengths, NOR or arbitrary thermal/ISO tokens.

Optional facts that cannot be supported are deliberately absent with an explicit
source decision. This resolves their extraction targets without pretending to
resolve the factual contradictions of the original documents. No source text was
edited to make a label look grounded.

### Final validation and publication

The final dataset is `data/curated/mpci-bl-real-v7-reviewed-r4/`, **450 records:
400 train / 50 validation**. Every record passes schema/canonicalization,
exact edit replay from its immutable automated target/draft, original OCR/PDF
hashes, original split assignment, unique document/OCR hashes, JSONL-versus-sample
agreement and placement-last ordering. The old 250 rows remain byte-identical
prefixes and all **515 historical files** retain their hashes.

The new 200 contain **260 goods items, 313 containers and 337 placement entries**.
Readback finds no dangling placement, duplicate container ID, allocation quantity
greater than a fully known same-level package total, or net mass above gross mass
when both have the same unit. No printed B/L identifier collides across the train
and validation splits. That last check is not an assertion that every shipment in
the corpus is unique: differently extracted copies can lack a target B/L number.

The lexical diagnostic initially flagged 19 values; 049's unsupported location was
corrected. The remaining **18** are individually explained joins/wraps/Unicode
presentation cases, such as `MUA`/`NG`, `SMSHIN@T`/`WSC.CO.KR`, and street/phone
tokens joined in OCR. They are documented with the final label and its hash, not
treated as an automatic proof of ownership. There are no unresolved diagnostic
findings in this batch. Semantic acceptance rests on the source review plus these
checks, not on a claim that a regex establishes semantic completeness.

Six isolated destructive-input controls are rejected by the actual publication
validator: changing both label copies without a receipt; changing OCR and its
local hash; duplicate record; missing record; stale edit-before value; and an add
operation overwriting a value. Both the initial and restored 450-record positive
controls pass. The per-record integrity comparisons also reject 1,350 mutations.
Explicit review amendments preserve previous receipts, replay correctly and reject
implicit overwrites/no-op amendments.

Maintained direct-labeling, V7 training, source-projection, labeling-agent and
underlying B/L/MPCI schema suites: **551 passed in 20.62 seconds**. Ruff and diff
whitespace checks pass. An initial test command named nonexistent test files and
ran no tests; the corrected command ran the complete stated suite successfully.
Preflight direct/V7 tests were 17.79s before /17.90s after. Local 250-record HS gate
benchmark, seven repetitions: **46.923ms before /46.885ms after**, traced peak
allocations **492,322 /488,754 bytes**. These are local validation measures, not
an assertion of identical hosted-model latency or semantic regression immunity.
Final staging/publication validation took **26.728 seconds**, peak RSS
**165,768 KiB**. Final content audit took **2.385 seconds**; it verifies that every
document used at most one automated editing wave.

The initial readback snapshot is retained under batch004's
`manual-review/pre-final-readback-r4/` for exact history; it is superseded, not a
training dataset. `instructions025.json` and `instructions026.json` explicitly
record the final consistency/readback amendments. No paid artifacts, previous
dataset revision, source OCR or PDF was deleted.

**Batch004 is complete: 200/200 reviewed, resolved and included; no outstanding
user decisions.** This authorizes these reviewed labels for the requested dataset,
not unattended acceptance of future agent passes. The paid flow still produces
manual work; no measured zero-false-pass claim is made. Known numeric/HS punctuation
diagnostic false holds were adjudicated locally rather than introducing speculative
production validators during this bounded task. No further processing, training
or synthetic generation was launched.

### Batch004 artifacts

- Final dataset: `data/curated/mpci-bl-real-v7-reviewed-r4/`
- Exact publication receipts: `batches/004/{manifest,manual-decisions,source-notes,validation}.json`
- Independent combined readback: `batches/004/combined-readback-validation.json`
- Pipeline and wire-schema validation: `artifacts/kie-labeling/direct-real-batch004-20261004/validation.json`
- Source/edit review: batch004 `manual-review/decisions.json` and `instructions*.json`
- Final content checks and net change counts: batch004 `manual-review/final-content-audit.json`
- Negative controls: batch004 `manual-review/final-publication-negative-controls.json`

The publication receipt's `changedDocuments` counts documents with edit history,
including reversals. The **163 net changed /37 net unchanged** counts above compare
final targets directly with the immutable automated candidates and are the proper
figures when discussing the delivered labels.

## Final top-up: R12, 600 train / 60 validation (2026-10-05)

The final real-only baseline is now
`data/curated/mpci-bl-real-v7-reviewed-r12-reduced-660/`.
See the [complete publication and analysis report](analysis/real-v7-starting-dataset-2026-10-04/r12_final_660/REPORT.md)
for selection, source-level review, diversity, exclusions, validation, cost and
retained input-quality considerations.

The approved missing-page filter removed `005/060-188cb27f` and
`005/130-e352e3f6`, preserving R10 and producing R11's 425 train /47 validation.
Three processing groups (006, 007, 008) then processed 261 candidates and retained
188: **175 train /13 validation**. All 188 have manually adjudicated full labels;
107 differ from automated candidates, 81 are unchanged. The other 73 are explicitly
excluded with source evidence. Original OCR/PDF bytes were not changed.

R12 preserves the 472 baseline records exactly, applies the same dropped-field
projection to the 188 additions, and freezes the matching reduced schema/prompt.
Full new annotations and exact edit receipts are in the neighboring
`mpci-bl-real-v7-reviewed-batch006-full/` (covering all three processing groups).

The new split is document-disjoint. Eight historical-training sources are in new
validation, as authorized; one fresh historical-validation source is assigned new
training. Membership changes are recorded explicitly. The validation set is not
claimed to be unseen by historical models, or disjoint in parties/products/templates.

Final checks: all660 reduced schemas/readbacks;188 full schema/edit replays;
original source hashes; split separation;564 integrity mutations and six isolated
negative controls; **587 maintained tests passed**. Independent published-file
audit took41.818s with175,692KiB peakRSS. Known API cost is approximately**$5.09**,
including excluded candidates and retries;17 interrupted/error requests from007
have uncertain usage, recorded separately. No training was launched.

Six older records still have documented partial container allocations, and other
R10 source-quality recommendations remain visible. Only the two newly approved
missing-page exclusions were applied to the old baseline. The report distinguishes
these inherited considerations from the completed new-source adjudications; it
does not claim previously identified OCR defects have disappeared.
