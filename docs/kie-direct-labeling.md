# Direct Bill-of-Lading labeling baseline

Implemented 2026-10-02. Two **20-document Luna high extraction pilots are complete**:
the first had 17 application-valid drafts; the field-clarification rerun has 18,
with all 20 responses conforming to the actual wire schema. Each batch cost about
$0.060. See the [first pilot](kie-direct-labeling-pilot-2026-10-02.md) and
[clarification rerun, improvements and regressions](kie-direct-labeling-pilot-r2-2026-10-02.md).
The same-panel **review/correction/re-review pilot is also complete**: 172 calls,
237 seconds, estimated $0.191. It repaired important defects but also exposed
false passes and incorrect corrections; see the [review experiment and independent
adjudication](kie-direct-review-pilot-2026-10-02.md). Existing datasets remain unchanged.
The [2026-10-03 improvements and paired reruns](kie-direct-review-improvements-2026-10-03.md)
are complete: targeted reviewer context, field-ownership clarifications, change-aware
re-review and a native equipment-pair constraint. Known-defect detection improved
from 5/9 to 8/9, but semantic false passes remain; the report separates them from
workflow pass counts and records the full $0.4315 experimental spend.
The [field-policy and explicit-adjudication follow-up](kie-direct-review-policy-and-adjudication-2026-10-03.md)
adds comma-separated postal components, locality-only names, audited joint
cross-section corrections, and a matched high/xhigh comparison. Final focused
checks pass for product-code transfers, route formatting and fax exclusion;
the report records the wider panel's remaining disagreements and $0.5038 total cost.
The [complete flow on 50 new sources](kie-direct-full-flow-50-new-2026-10-03.md)
also finished: 50 application-valid final targets, 37 agent-passing candidates,
13 held, and three independently confirmed false passes. It cost $0.568 across
427 requests; the report separates successful repairs from those residual defects.

The [same-panel implementation rerun](kie-direct-full-flow-50-new-2026-10-03.md#implementation-rerun-r2)
is also complete: 37 agent-passing, 11 held and two failed refinement operations,
at $0.789. It did **not** demonstrate an overall quality/performance improvement:
local literal gates worked, but cargo row ownership still falsely passed, and the
extra cargo reviewer strayed outside its scope. A narrowly corrected product-code
description regression passed a separate $0.012 replay. This is an experimental
direct-labeling flow, **not approved for bulk gold publication**. The report keeps
the frozen batch result separate from that later correction.

The [whole-PDF cargo R3 implementation and rerun](kie-direct-cargo-fullpdf-r3-2026-10-03.md)
completed with 50 structurally valid final targets, 39 agent-passing candidates,
11 held and no failed document operations. The full batch cost $0.954; all bounded
probes plus the batch cost $1.114. It improves known PDF-leak, route and product-code
controls but is slower and more expensive than R2. Source-map numeric parsing and
some semantic ownership disagreements remain; the report distinguishes a correct
xhigh cargo proposal from the automatic flow's still-held verdict.

The [R4 repairable-accounting rerun and independent audit](kie-direct-corrections-r4-2026-10-04.md)
is complete: 50 application-valid candidates, 43 agent-passing and seven held.
Original source controls improved from 37/42 to 41/42, and expanded controls from
3/10 to 9/10. Independent review nevertheless found five passing documents with
remaining errors or normalization defects. The full batch cost an estimated
$1.112; pilots and mutation probes bring this experiment to $1.393. Bulk gold
publication remains gated on the specific acceptance/correction issues in that
report; the repaired cookware accounting and other successful controls are not
discarded or conflated with those remaining issues.

## The maintained entry points

- [Extraction models](../src/document_ocr/label_schemas/bill_of_lading_v7.py)
  define the V7 target: 20 described models and 85 described properties.
- [Section/review models](../src/document_ocr/labeling_agents/direct_models.py)
  derive section views from the extraction model rather than duplicating field
  definitions. Review findings and layout requests have their own small models.
- [Flow](../src/document_ocr/labeling_agents/direct.py) implements extraction,
  parallel section review, typed corrections, dependency checks and re-review.
- [Configuration](../configs/labeling_agents/mpci_bl_direct.yaml) selects
  `gpt-6-luna`, high reasoning, a 32,768-token output/reasoning limit, a
  300-second request timeout, and at most sixteen concurrent calls.
  The ceiling was raised after a complete-PDF cargo mapping request exhausted
  16,384 tokens without producing an answer; it is not a requested output length.
- [Extractor](../prompts/labeling_agents/direct_extractor.md),
  [reviewer](../prompts/labeling_agents/direct_reviewer.md), and
  [corrector](../prompts/labeling_agents/direct_corrector.md) prompts define each
  stage's responsibility. Field-specific extraction policy lives on the Pydantic
  fields, not in a second prompt rulebook.

The commands are integrated into the existing `document-kie-label-agents` CLI.
The new flow does not call the historical span-repair or completion scripts.
Those scripts, historical workflows and their receipts have not been deleted or
silently redefined. Pre-existing uncommitted changes were preserved.

## What the extractor sees and returns

Input is the complete **literal OCR text**, with its original newlines and page
markers, as a text message part. There is no OCR-as-JSON line inventory. The
separate system instructions identify the annotation task and tell the model to
use the schema's field semantics. Native structured output supplies the described
Pydantic schema. OCR and PDF contents are treated as data, not agent instructions.

Output is the actual target:

```json
{
  "schemaVersion": "7.0.0",
  "documentPatch": {
    "parties": {
      "shipper": {
        "name": "EXPORTER LTD",
        "addressLine": "UNIT 4, NEWBERG OR 97132, USA",
        "country": "USA"
      }
    }
  }
}
```

This is an illustrative shape, not a claim that these values belong to an actual
document. The extractor produces no evidence inventory, coordinates, per-scalar
citations or rationale. Optional fields can be null in the constrained response;
the saved target omits nulls. Populated lists must contain entries, but missing
lists stay absent: minimum list lengths do not require inventing source facts.

Core agreed semantics are recorded directly on the models:

- Parties: full postal `addressLine` including printed city, postcode and country;
  separate `country`, no city target or legacy stripped `address`. Preserve source
  wording/order with comma-space separators between postal components and spaces
  at ordinary word boundaries; wraps inside words/identifiers are rejoined. Names, tax IDs and
  contacts remain outside postal text. Owned continuations can follow nonpostal
  lines. Do not concatenate alternative addresses. Explicit contacts are distinct
  from uncaptioned identity names; multiple contact names use `; `.
- Roles: distinguish carrier, signing/forwarding/delivery agents and consolidator.
  Bare `TO ORDER` supplies no named consignee. A named bank/entity in `TO ORDER OF`
  can supply that identity; negotiability is a separate field. A named consignee
  without order wording is non-negotiable; absent consignee evidence remains unknown
  unless explicit issuance wording establishes it. A copy stamp alone
  does not determine the original instrument's negotiability. Shared-context
  `A ON BEHALF OF B` wording is retained; competing identities/addresses are reviewed.
- Goods: product wording, specifications, condition, lot qualifiers and printed
  package capacity belong in `description`; there is no `additionalInformation`.
  Different independently quantified products remain separate. Shared product
  portions across containers use placements. Multiple HS codes alone do not split
  goods. Marks, administrative references and carrier boilerplate have distinct
  ownership and must not spill into product descriptions.
- Packages/placements: inner package level; outer levels are not a second target
  package level. Exact totals require complete, nonduplicated same-level portions.
  Unknown per-container quantities remain absent. Membership is not inferred from
  mere co-occurrence. Placements remain last in each goods item. Repeated placement
  IDs are still permitted for separately supported package allocations; dangling
  IDs are rejected.
- Measurements: preserve printed magnitude and canonical source units, including
  metric tonnes; do not turn package capacity into shipment mass or cargo mass
  into VGM. Carrying temperature and DG flash point are separate.
- Locations (clarified 2026-10-03): extract the locality name, excluding separate
  country wording and generic facility descriptors. Preserve distinguishing actual
  place-name words. Normalize explicit location-owned country names/codes/adjectives
  to uppercase conventional English short country names; never geocode from a city.
  This is a policy change from retaining the complete printed port phrase.
- Identifiers, dates and rare fields: preserve printed identifiers under explicit
  separator rules; no invented check digits or HS extensions. Dates are ISO when
  unambiguous; ambiguous dates require review. Master/original B/L references,
  vessel flag, consolidator, goods origin and payment alternatives need their own
  role evidence. DG categories are normalized from printed declarations, not
  enriched from a registry lookup.

This remains an **extraction target**, not a customs-form submission validator:
missing source values do not become required merely because the form needs them.
The field contract builds on the approved rules in
[the real-data repair record](kie-real-data-repair-2026-10-01.md#agreed-annotation-contract)
and the local MPCI category registry.

## Review and correction

```text
plain OCR → one extraction → draft target
                              ↓ (explicit refine command)
          five section reviews + candidate-blind cargo source map
                              → focused cargo association comparison
                              → scoped corrections → affected-section re-review
                              → one further actionable correction wave at most
```

The five review scopes are parties; route/transport; metadata/freight/references;
equipment; and cargo facts. Each receives the full OCR, its field
definitions and its candidate section. Cargo/equipment also receive each other's
candidate fields as explicitly read-only association context. Parties and metadata
receive only their own candidate fields. Empty
sections are reviewed too, so omitted facts can be discovered.
Each scope has a short responsibility checklist. Invalid candidate sections also
carry precise application-schema diagnostics, without implying a replacement
value. This makes Python-only constraints visible to the reviewer without
silently repairing malformed source identifiers.

Reviews report `pass`, `corrections_needed`, or `unresolved`, with a concise verdict
explanation and actionable findings. Findings identify a field/entity, issue kind,
explanation and optional suggested correction/short OCR quote. A `wrong_owner`
finding can specify `reassignTo` for a cross-section move. Correct fields require
no evidence inventory.
Reviewers check both output support **and source completeness**; stylistic edits
and manufactured findings are excluded by instruction.

The cargo source mapper and relationship reviewer **always receive the complete
original PDF** as a native file, alongside the complete plain OCR. The source
mapper sees no candidate labels. Its small typed product/portion map retains
outer packing and shared totals as internal review context, not training labels.
The comparator receives that map and only candidate grouping/package/placement
fields. The map is a fallible working interpretation, not authoritative evidence.
The relationship finding schema excludes marks, handling and unrelated fact edits;
the general cargo reviewer has the complementary fact-only finding schema.
Cargo corrections likewise receive the complete PDF and the source map. The map
is reused within that document's correction waves, not recomputed from the labels.
`refine --pdf` is therefore required. Route review receives page 1 upfront and can
request other pages. The initial extractor remains OCR-only.

Other reviewers/correctors can request at most three specific PDF pages for a layout
question. The flow renders those pages locally, capped at 150 DPI/2,500 pixels on
the long side, then makes one assisted follow-up. PDF-only values are forbidden by
both prompts. Missing PDF, invalid page requests, or a second layout request become
explicit unresolved decisions. Images are not supplied to the initial extractor.
Requested page numbers persist per section so correction and re-review receive
the established layout context too. Joint correction scopes share their requested
pages; equipment/cargo also share their requested pages. Unrelated sections do not inherit those images. PDF layout can establish a
field role even if its OCR heading is missing, but the value must exist in OCR.
Literal-fidelity gates independently flag absent vessel/voyage/IMO text, altered
HS digit sequences, OCR-absent equipment/seal IDs and absent cargo amounts. Exact
complete same-level sums can explain unprinted goods totals; unknown/shared
portions do not license partial totals or guessed local allocations. Numeric
presence alone does not establish ownership, units, boundaries or completeness.
The gates cannot be overridden by a model's pass. Newly introduced unsupported
values block the affected correction commit, preserving the previous target and
an explicit hold. They do not guarantee freedom from semantic PDF leakage.

Corrections return one explicit accept/reject/revise/unresolved decision and a short
source-based explanation per supplied finding ID, followed by complete typed
section values. Missing/duplicate decisions are held, not silently accepted.
Cross-section transfers join the source and destination into one correction scope;
both ends commit together. These are not JSON patches or replacement OCR.
They may reject a mistaken reviewer suggestion. Mixed actionable/ambiguous findings
can retain supported corrections without resolving an unrelated ambiguity.
The flow allows at most two correction waves, skipping unchanged candidate/finding
pairs, with edits limited to the adjudicated findings, including within a corrected
text field; unrelated new suspicions belong in subsequent review. It
validates dependent scopes together, validates the full target and re-reviews
affected scopes. A bad equipment/cargo change cannot discard a valid independent
party correction. Genuine unresolved issues remain visible, with passing siblings
preserved. Provider failures are explicit failures, not clean reviews; parallel
review calls drain and persist completed siblings before reporting an exception.
Malformed reviewer/corrector structured output becomes an explicit component
hold; it cannot count as a completed semantic review or discard independently
completed sections. Provider/network failures still surface as failures.
Raw V7 drafts with application-level defects are reviewable too. Independent good
corrections survive a bad untouched dependency group, but a final invalid document
has only `reviewed-draft.json`, an explicit validation error, and no `target.json`.
Re-review receives the exact before/after changes in its assigned section, alongside
the complete current section and OCR. Changed lists are shown whole because item
grouping/order may have changed. Relevant correction decisions and their findings
are supplied as untrusted context, not authority; the task is to check the current
labels, including possible correction regressions. Verdicts explain the outcome
even when passing. Decision explanations are audit metadata, never training labels.

`audit_reasoning_effort` optionally overrides reasoning for correction and final
re-review only; extraction and initial review keep `reasoning_effort`. Both accept
`xhigh`. No escalation is automatic. The default remains high throughout.

States have deliberately narrow meanings:

| State | Meaning |
|---|---|
| `draft` | Direct extraction passed structural/category checks; not reviewed. |
| `reviewed_candidate` | All section reviews pass on the resulting target. |
| `needs_adjudication` | At least one section still needs correction or a source decision. |
| failure | Provider, schema or execution error; no successful stage is implied. |

No state automatically promotes a target to gold or publishes training data.
`goldApproved` is explicitly false in refinement results. The selected-document
experiment will measure whether these simpler agents actually produce correct,
complete annotations; structural tests alone do not establish that.

## Bounded execution and receipts

- `extract` makes exactly one request. Native-output validation and SDK transport
  retries are disabled; failures cannot silently trigger repeated paid calls.
- `refine` starts with five concurrent reviews, then one focused cargo-association
  review. The latter checks local product/container portions and can request
  continuation pages; equal grand totals cannot establish correct allocation.
  Correction is restricted to flagged sections and at most two waves; affected
  dependency groups are re-reviewed (including their focused cargo check).
  Each review/correction call allows at most one additional PDF-assisted request. There is
  no autonomous retry loop, bulk launch or stronger-model fallback.
- Every run requires a fresh output directory. OCR, config, schema, prompts,
  request identities, responses, usage, timings and errors are saved. Existing
  targets are never overwritten. Missing provider usage is unknown billing, not
  automatically zero spending.
- Wholly fact-free optional objects are explicitly normalized to null before
  strict JSON validation. Raw provider text and normalization receipts remain
  separate. This never repairs identifiers, deletes populated facts or invents
  missing values. Detailed exception causes accompany rejected responses.
- The registry is hash-checked before calls. All 405 package categories remain
  available, including rare categories. Tokens already spell their display names,
  so the schema does not redundantly repeat the same 405 names as a glossary. A
  future non-self-describing token receives its extra display meaning.
- Native output also represents the container's canonical size/type pair versus
  printed-wording alternative with nested `anyOf`. Python still validates the
  complete target, including constraints that are not provider-expressible.

## Commands

From the repository root with labeling dependencies installed (`uv sync --locked
--extra labeling`), inspect the schema without credentials or a paid request:

```bash
.venv/bin/document-kie-label-agents schema \
  --config configs/labeling_agents/mpci_bl_direct.yaml \
  --project-root . \
  --output /tmp/mpci-direct-schema.json
```

The next, explicitly launched single-document extraction is:

```bash
.venv/bin/document-kie-label-agents extract \
  --config configs/labeling_agents/mpci_bl_direct.yaml \
  --project-root . \
  --ocr /absolute/path/to/source.txt \
  --output artifacts/kie-labeling/direct-pilot/document-001
```

The command reads `OPENAI_API_KEY` from the environment or repository `.env` without
logging it. It writes `target.json` and `status.json` plus call receipts. Replace
the input path and use a new output directory for each document/attempt.

Optional subsequent refinement accepts that target, or an untrusted V7 candidate
with the correct envelope/known sections. It does not require repeating extraction:

```bash
.venv/bin/document-kie-label-agents refine \
  --config configs/labeling_agents/mpci_bl_direct.yaml \
  --project-root . \
  --ocr /absolute/path/to/source.txt \
  --candidate artifacts/kie-labeling/direct-pilot/document-001/target.json \
  --pdf /absolute/path/to/source.pdf \
  --output artifacts/kie-labeling/direct-review/document-001
```

The complete `--pdf` is required for cargo layout review. Old V6 labels cannot be silently treated as V7: their
split addresses/overflow cargo need actual semantic extraction, not a version edit.

## Verification and measured impact

The [R5 batch-readiness report](kie-direct-batch-readiness-r5-2026-10-04.md)
documents the latest field-policy clarifications, changed-field correction guard,
immediately preceding-wave re-review, and frozen fresh 50-document replay. Auditor
decisions retain explanations and exact changed paths; independent re-review receives
the changes without the previous auditor's verdict anchoring its judgment. Known
unresolved components go to an explicit review queue after at most two waves.

R5 finished with 50 application-valid candidates, 43 automated passes/seven holds,
and 140/140 named source checks. Independent inspection and finite manual pilot
adjudication leave 46 documents with the tested issues resolved and four source
reviews outstanding; two formatting/boundary false passes were corrected explicitly.
The report recommends a supervised 100-document batch with flagged-case review and
a small passing-case spot check, not unattended gold publication. The full fresh run
plus all bounded continuations cost an estimated $1.5281. Production datasets remain
unchanged. Current focused verification: **570 tests passed**.

Earlier review-flow verification: **520 tests passed in 15.65 seconds**, Ruff and
mypy passed. New behavioral tests cover exact review scope, a shared multi-document
request limit, initially invalid drafts with independent corrections, final export
guards, and persistent PDF layout context through correction/re-review. The live
experiment's semantic findings and costs are reported separately above.

Original implementation measurements:

- **487 targeted tests passed**: direct agent flow, existing labeling agents,
  V7 training/extraction integration, relation constraints, and historical B/L
  schema suites. Tests include actual PydanticAI model execution and an OpenAI
  Responses SDK request through an in-process HTTP mock—not a hand-built payload.
- The HTTP test verifies plain OCR, native strict output, field descriptions,
  optional-null fields, reasoning settings, provider usage and response receipts.
  It exercises real request serialization without a network call.
- Failure tests reject unknown package categories, orphan placements, legacy
  party fields and empty targets; preserve successful sibling reviews; exercise
  PDF page rendering and the bounded assistance path; retain independent good
  corrections when another dependency group fails.
- Ruff passed; mypy passed for the extraction models and three new flow modules.
  The actual installed CLI successfully exported the schema without an API call.
- All **291 targets** in the historical V7 reviewed snapshot `138b4c825a9c…`
  validated and serialized unchanged. This checks schema compatibility, **not**
  renewal of their historical semantic acceptance decisions. Five full replays
  had a median of **0.07769 s**, about **3,746 documents/s**, with **121,496 bytes**
  peak traced allocation. The probe process peaked at **172.88 MiB RSS**, including
  imports and tokenization tools; this is not model-serving memory.
- Within implementation, replacing dictionary-schema expansion with native
  reusable Pydantic definitions and removing the redundant category glossary
  reduced the pre-provider schema from **91,957 to 43,315 UTF-8 bytes** and from
  **22,168 to 10,559 `o200k_base` tokens** (~52% fewer). No field/category was
  removed. Tokenizer counts are an input-size proxy, not a billed cost claim.
- API spend for this implementation: **$0**. No live provider latency, annotation
  quality, or per-document cost is claimed before the requested pilot.

V1–V6 model files were not changed. V7 has the same value shape, but descriptions
and model-definition names change its schema fingerprint. Existing V7 task
constraint sidecars must be rebuilt for a future training dataset, not silently
reused with stale hashes. The training schema builder was updated to reference
the new V7 definitions; the short training prompt still strips description/title
metadata and does not inherit the annotator's full instructional schema.

Implementation references: [PydanticAI native output](https://pydantic.dev/docs/ai/core-concepts/output/),
[request capabilities/hooks](https://pydantic.dev/docs/ai/capabilities/overview/),
and [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
The request hook binds the pinned package vocabulary; the output-validation hook
canonicalizes empty optional objects, with receipts. Normal Pydantic validation
and provider strict-schema preparation remain on the actual runtime path.
