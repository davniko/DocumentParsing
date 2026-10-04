# Real-only extraction reference repair

Started 2026-10-01. This is the execution record for the real-only repair approved
after `kie-real-data-baseline-audit-2026-10-01.md`. Work is in progress; existence
of this document or a candidate file is not a dataset quality sign-off.

Latest execution: [Component-scoped completion](#component-scoped-completion-2026-10-02).
The 1,057-source training population is now being processed, not held behind the
three unresolved validation-party decisions. Earlier snapshots below remain
historical evidence; the completion manifest will distinguish accepted components,
whole-document eligibility and specific unresolved fields.

Latest continuation: [Acceptance replay and transport recovery](#acceptance-replay-and-transport-recovery).

Current continuation: [Funded repair wave and acceptance falsification](#funded-repair-wave-and-acceptance-falsification).

## Scope and immutable inputs

- Inventory: all 1,157 historical IDs (1,057 training, 100 validation), including
  the explicitly held drum/pail source. Synthetic records are out of scope.
- Starting labels: the frozen last-trained v6 real subset. Historical labels,
  current GROUND-015 candidates and exact repair receipts are comparison evidence,
  not competing silently selected sources of truth.
- OCR is immutable. The frozen v6 inputs match historical OCR for every retained
  real record. The two later improved training OCR derivatives remain comparison
  evidence, not silently mixed into this label-only baseline. No new
  OCR, PDF-only labels, training, synthesis or template modification is authorized
  by this pass.
- Every candidate, decision, provider receipt and publication is keyed to input
  and policy hashes. No existing dataset is overwritten.

## Agreed annotation contract

1. Party targets contain `addressLine` and `country`, **not separate city or legacy
   address fields**. Names, contacts and `sameAs` retain their distinct roles.
   `addressLine` contains the complete printed postal address, including city and
   country where printed. Exclude names, tax identifiers, contacts and unrelated
   administrative tokens. Postal continuations may follow nonpostal lines. Fold
   physical whitespace to single spaces; preserve words, numbers, punctuation and
   printed order. Do not geocode, correct spelling, concatenate old split labels,
   or infer geographic information absent from OCR. Genuine wrap-split words need
   explicit evidence and a recorded normalization, not silent rewriting.
2. Product identity, brand, model, composition, specification and condition belong
   in `description`. The new model target has no `additionalInformation`; old
   contents remain archived. Move product-owned text, do not indiscriminately
   concatenate administrative, duplicate or unrelated AAI contents.
3. A goods item is an OCR-supported cargo/accounting unit. Multiple products or
   HS codes do not themselves force a split. Distinct independently quantified
   product rows remain distinct. A shared goods item may have several container
   placements. Do not duplicate aggregate and portion facts or invent per-product
   quantities, weights or allocations. Preserve source ordering.
4. Keep inner package levels in the extraction target; archive outer-level facts
   in review evidence. Printed product/package-capacity wording can remain in its
   description. Never treat a per-package capacity as shipment mass.
5. Retain marks only as goods-owned marks, distinct from product descriptions,
   equipment identifiers, references and carrier boilerplate.
6. Preserve explicitly printed measurement units using the existing canonical
   unit vocabulary. Exact totals are allowed only with complete, source-owned
   components and an explicit derivation. Never invent missing unit labels.
7. Include only OCR-supported values and roles or an explicitly allowed derived
   or normalized representation. Country aliases/demonyms require unambiguous
   country meaning in context. Preserve arbitrary identifiers as printed under
   the approved separator/normalization policy. No guessed container check digits.
8. Missing data is not the same as an unresolved annotation. Unresolved decisions
   remain recorded and block whole-document publication until adjudicated.
9. Retain sparse labels: no fabricated fields to satisfy a schema, no empty lists
   or null placeholders in published targets. `sameAs` requires an explicit OCR
   reference, not merely similar-looking party text.

## Execution and acceptance

Start with all 100 validation documents, then the training population. Each
document receives a full-OCR, section-level annotation review and an independent
verification pass, not a whole-text presence-only scan. Reviews check support,
ownership, missing facts and cross-field consistency. Prior candidate decisions
are reused only when they satisfy this contract on the pinned OCR.

Local checks validate exact evidence, address assembly, strict schema, references,
input immutability, edit preconditions and population accounting. They do not
prove semantic ownership by themselves. Independently disputed semantic decisions
require adjudication. Record actual costs including retries and unsuccessful
requests; do not extrapolate from an unrelated synthetic-address pilot.

Before publication, exercise the actual training canonicalizer, prompt schema,
category constraints and metrics. Legacy city stripping must not run on the new
target. Historical schemas remain intact for reproducible prior-run comparisons.

## Progress

**Status: target-contract implementation complete; semantic dataset repair not
complete. No new training-ready dataset has been published.** The attempted broad
automatic annotation/review process did not meet the acceptance bar. It is not
approved for the 1,057-document training population. This is a quality blocker,
not a budget exhaustion or a request to change the agreed labeling rules.

### Exact population and work accounting

The [per-document inventory](../artifacts/kie-training/analysis/real-data-repair-20261001/inventory/6ec5922970e51aeed018e7de6b43192914c612359d5b7ca68e9beba052dabb0d.json)
records all 1,157 IDs, their original split, OCR hash, candidate/review request IDs,
review-to-candidate hash match, unresolved findings and manual decision paths.
It explicitly marks every document `not_approved`.

| Item | Count | Meaning |
|---|---:|---|
| Original real population | 1,157 | 1,057 train + 100 validation |
| Pinned original OCR verified | 1,157 | Packet text matches inventory hash |
| Frozen last-trained real targets available | 1,156 | One historical drum/pail source lacks this frozen target |
| Validation documents with structurally valid annotation candidates | 89 | Not semantic acceptance |
| Validation documents with an independent review response | 38 | All 38 latest reviews match their annotation target hash |
| Documents with automated adjudication candidates | 12 | Still subject to source adjudication, not approved |
| Documents with recorded manual partial decisions | 8 | Includes rejection of incorrect proposed edits and retention of sound labels |
| Whole-document approvals / published repaired records | **0 / 0** | Do not confuse draft coverage with repaired-dataset coverage |
| Training documents processed by the new annotation model | **0** | The training split was not sent through an unreliable bulk process |

Eleven validation documents still lack a mechanically valid annotation candidate;
62 lack a separate review response. These counts overlap and are not additional
populations. All 100 require completion of document-level semantic adjudication.
The source hashes of all three audit input views were rechecked unchanged.

### Implemented target and training-path changes

- [New explicit v7 target](../src/document_ocr/label_schemas/bill_of_lading_v7.py):
  `addressLine` + `country`, no `address` or `city`; no cargo
  `additionalInformation`. Existing historical schemas remain unchanged to
  reproduce old runs. This is not an automatic semantic migration of old values.
- [Task registration/canonicalization](../src/document_ocr/training/tasks.py):
  new `bill_of_lading_extraction_v7` task, sparse strict target validation,
  aligned injected schema, goods-local placement last.
- [Training config](../src/document_ocr/training/config.py),
  [category-constraint construction](../src/document_ocr/training/relation_constraints.py)
  and [metrics](../src/document_ocr/training/metrics.py) support the new task.
  Fact, category and relation metrics continue to use the goods-local contract.
- [Concise prompt](../prompts/training/mpci_bl_extraction_v7.txt) describes the
  approved extraction rules and injects the actual new schema. Original OCR is
  passed literally. No legacy city-stripping transformation runs here.
- Multiple HS codes and multiple separately supported placements to the same
  container remain allowed. A uniqueness constraint on placement container IDs
  was rejected during testing because it would discard valid repeated package
  allocations. Dangling container references are rejected.
- No training config was activated and no training, synthesis, template rebuild
  or OCR replacement was launched.

### What the validation work actually uncovered

The failures below are **candidate annotation failures**, not newly introduced
defects in the preserved dataset. They explain why schema-valid output or an
empty reviewer finding list cannot be used as the sole acceptance criterion.

| Source / evidence | Observed problem | Source-based disposition |
|---|---|---|
| `0600699f`, OCR lines 19–22 | An address edit retained `POSTAL CODE:` inside the postal value. A separate reviewer focused on phones and missed the caption. | Keep `62815`, omit the caption; retain city/country and the other postal words. |
| `14f66f…`, lines 6–8 and consignee block | Proposed postal fragments included tax-office text, tax ID, phone, or VAT despite an explanation claiming these were excluded. Postal continuation after a phone is real and must be retained. | Reject the contaminated rewrite; the existing candidate's clean address is supported. |
| `0961dfa2` | An edit's explanation said to replace the goods list, but its JSON value was `null`, deleting the entire cargo section. | Reject deletion; retain the truck, chassis, package and HS facts; record supported condition and mass corrections. |
| `1891d90d`, adjudication request 174 | Explanation rejected boilerplate `WEIGH & MEASURE`, while the proposed target inserted it as handling instructions. | Reject that proposed addition. Do not equate explanatory text with the actual target edit. |
| `1c402851`, adjudication request 176 | Cargo description absorbed outer pallet/bag hierarchy and repeated lot wording despite the agreed division of product text and structured facts. | Retain product description, inner bag count and lot marks in their appropriate fields; do not copy the whole cargo box into description. |
| `99adb051` | An annotation proposed deleting the container because the source's joined separator typo appeared to make the ID too long. | Reuse the prior explicit source adjudication: `TEMU9523456` / `CM08875356`; the intervening `7` is the approved mistaken separator, not part of either identifier. New draft not accepted. |

The one-time script's exact quote checks catch fabricated substrings and stale
coordinates, but not whether a real substring belongs to the requested role.
Likewise, a strict schema cannot determine that `WEIGH & MEASURE` is boilerplate.
Automated adjudication also produced decision enums inconsistent with their
reasons and changes. More explanation from the same model is not an independent
proof of the edited label.

An additional local screen inspected **101 candidate versions** (89 annotation
and 12 adjudication versions). It raised 10 postal-caption flags, two removed
section flags, two name/address overlap flags and two ordered-text-support flags.
These are overlapping review triggers, not 16 proven defective documents and not
a complete semantic defect census. No flag is a blanket repair instruction.

### Bounded follow-up diagnostics

1. Four small output-mode/reasoning probes compared native versus prompted JSON
   and medium versus high reasoning on a known answer. All four returned the
   expected values. A local telemetry accessor error then marked their requests
   failed; the original answers and charged usage remain recorded. The accessor
   was corrected without repeating paid work. A toy pass does not establish
   reliability on real documents.
2. OpenRouter generation telemetry confirmed substantial reasoning for two of
   the faulty real adjudications: 2,771 and 3,823 reasoning tokens. These errors
   cannot simply be attributed to reasoning having been disabled.
3. A narrower party-only probe cannot edit cargo, routes, dates or references.
   Three requests to the selected Fireworks route were rate-limited. A separately
   identified DeepInfra route produced one response and two more rate limits.
   The successful response correctly removed the postal-code caption but omitted
   the printed `TW` suffix from `TAIWAN, R.O.C, TW` and combined two separately
   labeled phones into one array entry. It is not accepted as a complete repair.
   This diagnostic used the same requested GLM model with explicit high reasoning
   and a recorded provider; it was not a silent fallback in production.

The provider's structured-output contract concerns JSON/schema shape, not semantic
correctness; see [OpenRouter structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs).
The local experiments confirm that distinction directly.

### Source-based decisions already saved

[Manual candidate script](../artifacts/kie-training/analysis/real-data-repair-20261001/manual_validation.py)
records exact quotes, starting-target hash, resulting-target hash and rationale:

- `025a144d`: two distinct containers support 56,000 kg and 190 CBM by exact sum;
  repeated pages are not extra containers. Omit ambiguous port wording as marks.
- `0600699f`: restore printed 3,710 kg; keep the postal code without its caption;
  country value `TAIWAN`, while the literal address retains its printed aliases.
- `06f8c293`: add the explicit arrival/release delivery agent; exclude its `ADD:`
  caption. Remove unsupported negotiability. Do not guess a decimal point in
  `22863 500KGS` or infer goods origin from exporter country.
- `0961dfa2`: preserve cargo deleted by the bad proposal; add printed 7,500 kg and
  source-declared used/damage condition, without generic carrier liability text.
- `14f66f…`: reject contaminated address rewrites, demurrage-as-handling, and
  unwarranted removal of canonical 40 HC equipment meaning.
- `1891d90d`: keep the supported package count and both printed contact numbers;
  do not invent a convention for fitting two different contact names into one
  `contactName` value. That completeness decision remains open.
- `1c402851`: retain product/HS/weights/lot facts separately and the inner bag
  level. Pre-carriage and agent-role coverage still need explicit disposition.
- `217e911a`: retain `WOVEN FABRICS`, the one-container placement and 129-bale
  allocation; exclude ACID/payment wording from description and consignment
  preambles from the bank's company name.

These are **partial decisions on eight documents, not eight approved documents**.
Four resulting candidate targets change the seed; four retain the seed while
rejecting a bad proposed change. Unexamined fields have not been silently certified.
All old proposals are preserved, including a superseded contact-name proposal.

### Cost, validation and performance

- Recorded settled provider charges: **$0.354434615**.
- Conservative reservations for requests without settled cost: **$0.4414596**.
  Reservations are not confirmed spending; these include unsuccessful/rate-limited
  calls. They remain counted against the local cap until reconciled.
- The first three requests have stale `running` ledger statuses from an initial
  receipt-serialization failure; no corresponding process is running. Their
  $0.02687055 reservation remains intact. This is not unfinished background work.
- **37 targeted tests passed**, including actual prompt/schema injection,
  strict rejection of removed fields, source-edit replay, source immutability,
  dangling and repeated placements, categorical constraint construction, metrics
  and historical v6 projection.
- Ruff passed for changed production files, tests and the one-time scripts.
  Mypy passed on all five changed/new production modules. `git diff --check` passed.
- Measured canonicalization across 1,156 matched v6/v7 records: median **139.81 µs**
  versus **141.85 µs** per record across three repetitions, about **+2.04 µs / 1.46%**.
  Command-level peak RSS **118.7 MiB**. This is negligible schema-validation
  overhead, not a training-speed benchmark or semantic certification.

### Remaining work and the acceptance blocker

The next necessary work is semantic source adjudication, not another blind bulk
rewrite. Existing paid proposals and prior GROUND decisions are reusable evidence,
but their changed values must be compared with their reasons and source roles.

1. Finish all 100 validation documents against the pinned OCR. Resolve the known
   bad proposals above, carry forward prior explicit source adjudications, and
   account for every omitted AAI item and cargo grouping change. Report any
   genuinely ambiguous OCR as such; do not silently convert an unknown decision
   into absent gold.
2. Use field-limited edits for the agreed repair families. Separate postal-label
   extraction from contacts and unrelated party roles; separate cargo description
   ownership from equipment/date/freight auditing. Broad mutation permission was
   unnecessary and allowed unrelated regressions.
3. Assess this source-adjudicated reference before processing 1,057 training IDs.
   The demonstrated annotation/reviewer combination is **not** an automatic gold
   label producer. A stronger independent reviewer would still need the same
   held-out source checks; no claim of guaranteed correctness or reliable cost
   projection is justified by the current trials.
4. Only then publish a complete approved real-only derivative with exact split
   and provenance accounting, consistent prompt/schema, and unchanged OCR.
   Regrading saved predictions is possible only on comparable fields and is not
   evidence of improved retraining performance. No training is part of this pass.

The rule change itself is implemented and verified. The blocker is the quality
of automatic semantic adjudication, not inability to express `addressLine`.
There is no need to ask the user to re-approve the same labeling policy. The open
execution choice is whether to invest in direct source adjudication of the
validation reference or qualify a different independent reviewer first. Do not
resume the existing bulk process as though the failed checks had passed.

## Bounded API / interface comparison — 2026-10-01

This follow-up answers whether a different API, model, reasoning effort or task
decomposition helps. It does **not** publish repairs or certify whole documents.
No synthetic records, OCR, production labels or training settings were changed.
The user subsequently requested cheap models for the repair workflow; the Sol
diagnostic below had already completed. No further expensive-model use is planned
without a cost estimate first.

### Documentation and implementation findings

The OpenAI-docs skill was used to verify Responses API model/settings and pricing,
not to infer data quality from a model's marketing description.

- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
  explicitly distinguishes schema compliance from semantic correctness and
  suggests simplifying/splitting tasks when errors persist.
- [PydanticAI output documentation](https://pydantic.dev/docs/ai/core-concepts/output/)
  describes native/tool/prompted output and custom output validators with bounded
  `ModelRetry`. These enforce whatever checks we implement; they do not provide
  an automatic semantic truth checker. A model-generated explanation is not a
  substitute for validating the returned value.
- [PydanticAI OpenAI integration](https://pydantic.dev/docs/ai/models/openai/)
  provides `OpenAIResponsesModel` and explicit reasoning settings. The probes use
  Responses API, `openai_store=False`, and SDK retries disabled. This is an API
  setting, not a claim about all provider data-retention policies.
- The Jev service found is **TypeSafe Jev through OpenRouter's Decisions API**,
  not an identified OpenAI Decisions endpoint. Its finite-choice decisions can
  support routing; they do not generate arbitrary postal strings or cargo labels.
  [OpenRouter's Jev comparison](https://openrouter.ai/blog/tutorials/jev-vs-llm-when-to-use-each/)
  and [TypeSafe API documentation](https://docs.typesafe.ai/api) describe that
  distinction. Confidence is not a domain-certified correctness probability.

Two local interface defects were confirmed:

1. The earlier whole-document annotation schema accepts generic
   `edits(path, value: JsonValue, evidence, reason)`. It can represent deleting
   unrelated cargo with `null` while explaining a replacement in prose. Strict
   JSON validation cannot reconcile that contradiction. Narrow typed outputs
   remove that authority: postal selection has no cargo edit field.
2. The full target JSON schema does not expose several Python-only cross-field
   rules. `ContainerInformationV6` requires size/type together and forbids its
   raw type description beside categories; package category and raw package
   description are mutually exclusive. The generated JSON schema has neither
   those conditional constraints nor explanatory field descriptions. V7 reuses
   these models. Three full-output probes failed exactly these validations.
   This is not evidence that those constraints should be weakened. The annotation
   interface must express them, or extract literal evidence into a simpler typed
   intermediate representation and let deterministic conversion enforce them.

### Experiment design and reproducibility

All files are under
[`real-data-repair-20261001/model-comparison/`](../artifacts/kie-training/analysis/real-data-repair-20261001/model-comparison/).
The [summary](../artifacts/kie-training/analysis/real-data-repair-20261001/model-comparison/summary.json)
contains per-arm results, failures, latency, tokens, costs and integrity checks.
Each receipt retains exact input, prompt, output schema, model settings, raw
response, parsed answer, source hash, request ID where available and charges.

- Initial panel: **16 checks across 12 documents**, eight postal and eight cargo.
  It includes previous failures, deterministic newly selected documents and two
  goods-topology challenges. Source-adjudicated expected spans were frozen before
  calls. Old labels and expected answers were not sent to the models.
- Transfer panel: **nine checks across four additional documents**, seven postal
  and two cargo. Documents were selected by deterministic hashing in two length /
  split strata; roles deliberately cover continuations, repeated pages, named
  sites, postcodes and an additional notify party. References were frozen before
  those documents were sent to the compared models. These are new documents, not
  necessarily unseen carrier/layout families.
- In total: **25 checks across 16 documents**, not 25 whole-document repairs or
  a random accuracy estimate for all 1,157 records.
- Every section call receives the **whole OCR with line IDs**. This avoids giving
  a model a manually preselected perfect crop. Postal roles are supplied, so role
  discovery/completeness itself is not tested. Cargo probes test descriptions
  and item grouping, not every quantity, HS code, unit, DG fact or placement.
- Section output is exact source fragments plus `unresolved`. Deterministic
  assembly checks quotes, ordering, overlap and source hash. It cannot prove
  whether a real quote belongs to the requested role or whether content is missing.
- Prompt v2 adds generic rules for repeated party occurrences, all field captions,
  leading consignment quantities/masses and attached condition statements. It has
  no source IDs, carrier-specific cases or reference answers.
- OpenAI arms use native structured output. The original GLM arm uses Fireworks
  native output. A separately identified GLM arm uses Z.AI prompted JSON because
  Fireworks was mostly unavailable. The latter comparison changes both provider
  and output mode; it is not a clean model-only or output-mode-only ablation.

### Results

Exact reference matches, with technical failures included in the denominator:

| Arm | Initial 16 | Additional 9 | Scope / caveat |
|---|---:|---:|---|
| GPT-6 Luna, medium, section prompt v1 | 14/16 | 6/9 | 20/25 overall |
| GPT-6 Luna, high, section prompt v1 | 12/16 | Not run | More reasoning was not better on this panel |
| GPT-6 Luna, medium, section prompt v2 | 14/16 | 8/9 | 22/25 overall; one initial invalid exact quote |
| GLM-5.3-Flash, high, Fireworks/native | 0/16 | Not run | 14 upstream 429s, one invalid quote response, one content error; unsuitable for quality ranking |
| GLM-5.3-Flash, high, Z.AI/prompted | 8/16 | Not run | Two empty responses; 14 mechanically parsed responses |
| GPT-6 Sol, medium, section prompt v1 | 15/16 | 9/9 | 24/25; diagnostic only, not selected for bulk use |

These are small paired diagnostic results, not statistically established general
model rankings. Unresolved entries are review signals, not absent fields. One
GLM response includes such a signal and must not be silently published.

Important observed error mechanisms:

- Luna v1 selected an entire postal block twice from repeated pages (`0600`,
  `99629`) and from repeated delivery-agent declarations (`e004a`). The generic
  v2 occurrence rule fixed those particular repetitions, including the two
  transfer documents. It did not make all postal decisions correct.
- Both Luna v2 and GLM/Z.AI selected an extra `MERTER` from a tax-office phrase
  in `14f6`, producing `...NO:17 MERTER MERTER / GUNGOREN / ISTANBUL`. Each quote
  exists in OCR, so exact-source validation alone accepts this semantic mistake.
- Luna v1/high included consignment mass `60,000KG` in the ferro-molybdenum
  description. GLM additionally included `60 PALLETS ( 60 BAGS )`. The v2 Luna
  instruction correctly separated the product text on that case.
- On the new steel-balls source `aa9ed`, both Luna prompts pulled `PACKED IN
  120 BAGS` into description; v2 also repeated the product phrase. Our reference
  retains `FORGED STEEL GRINDING BALLS STEEL BALLS 5,0 NG DIAMETER`, with the
  shipment/package count handled separately. The source contains ten container
  rows plus a final attachment specification, so completeness and duplication
  are competing requirements, not merely JSON formatting.
- Sol's remaining reference mismatch omitted printed `IN DRUM` from the WANNATE
  product line. Its greater capability did not eliminate all boundary decisions.
- One Luna v2 postal answer rewrote a source line while quoting it. The strict
  span check rejected it. A separate rerun with an output validator and one
  allowed retry returned the correct spans on its **first** response. This proves
  that request path works, but does not prove a retry caused the improvement.

#### Whole-document versus scoped output

Four paired Luna-medium diagnostics used the full typed V7 output instead of
postal/cargo selections. Three failed unrelated equipment/package validators;
the fourth retained the unwanted consignment mass. None met both whole-output
validation and the selected-field reference check. The scoped counterparts met
three of four selected-field references.

On those exact four cases, full-output calls cost an estimated **$0.0065794**
versus **$0.0016663** for scoped calls; median request latency was **26.67 s**
versus **4.42 s**. Full output also does substantially more work, so this is a
diagnosis of avoidable task coupling, not a like-for-like throughput benchmark
for a complete annotation system. The full control did not include the older
annotator's separate category registry or old draft labels; do not attribute all
differences to decomposition alone.

#### Can Jev or agreement be the approval checker?

Jev received 12 source-adjudicated correct candidates and 12 deliberately corrupted
versions, with full OCR and the extraction policy. It matched **22/24** expected
decisions. It accepted all 12 correct controls but also accepted two corrupt ones:

1. An address containing the `POSTAL CODE:` caption as part of its postal value.
2. A goods list omitting the separately printed forklift entirely.

Those false accepts had `correct` choice probabilities 0.69 and 0.56, with returned
confidence 0.55 and 0.34. A threshold would hold these examples, but choosing a
threshold after observing failures is not evidence of calibrated safety on new
documents. Jev is a possible review-priority signal, **not a publication gate**.

Independent blind Luna-v1 and Sol extraction agreed on 19/25 values, all correct
on this panel, and disagreed on six. However, the cheaper Luna-v2 / GLM comparison
provides a direct counterexample to treating agreement as proof: they agreed on
the wrong duplicated `MERTER` address. Therefore neither model consensus nor a
checker confidence score authorizes an automatic semantic sign-off by itself.

### Costs, limitations and validation

- This session made 128 annotation request attempts across all arms and 24 Jev
  controls. OpenAI costs are calculated from returned tokens at the published
  [Luna](https://developers.openai.com/api/docs/models/gpt-6-luna) and
  [Sol](https://developers.openai.com/api/docs/models/gpt-6-sol) rates; OpenRouter
  successful-response costs are provider-reported. These are not a reconciled
  OpenAI billing invoice.
- Known / token-estimated total: **$0.206042042**, including **$0.166916** for the
  completed Sol comparison and **$0.002193492** for Jev. The original repair
  ledger is separate and is not reset or folded into this total.
- **$0.48 remains conservatively reserved**, not confirmed spending: fourteen
  Fireworks 429s and two empty Z.AI responses with no inline billing. Read-only
  generation lookups for the two available IDs returned 404; receipts preserve
  that result rather than calling those requests free. Known plus reserved
  liability is approximately **$0.686**, below the $1 probe ceiling.
- Initial 16-check cost: Luna medium **$0.0055279**, high **$0.0058684**. V2's
  complete 25-check cost was **$0.0088703**. These are narrow-check costs, not a
  defensible quote for complete real-document annotation and review.
- The first Fireworks batch unnecessarily continued after upstream rate limits.
  Admissions now stop on authentication, credit or rate-limit errors; active
  calls drain. SDK retries are disabled, provider routes are explicit and prior
  paid receipts are reused. No automatic provider or model fallback was added.
- **16 targeted tests passed** across the existing repair checks and the new
  comparison checks. Tests cover source fidelity, ordering, overlap, nonexistent
  evidence and section-authority boundaries. Deliberate counterexamples show
  that lexical checks alone cannot certify ownership or completeness. Ruff and
  `git diff --check` passed.
- All **three pinned input files** and **1,157 OCR packet hashes** still match
  the existing manifest. No original document was repaired or published here.
- Offline summary/integrity replay: **8.60 s**, peak RSS **112.5 MiB**. Exact
  source projection averaged **6.71 microseconds per selection** on the panel.
  No production hot path changed; this is not a training benchmark.

### Recommended continuation: cheap typed drafts, explicit source adjudication

The experiment supports a narrower process, not resuming the original generic
whole-document mutation loop or replacing its reviewer with Jev.

1. Use **Luna medium** as the current inexpensive candidate extractor. Keep GLM
   available only on an explicitly working route; the measured route/mode did
   not outperform Luna. Do not escalate reasoning indiscriminately. Any expensive
   model escalation first needs a priced, bounded proposal.
2. Split by coherent section, not one request per scalar: party identity/postal
   information; goods descriptions/grouping; cargo facts and placements; route /
   transport / references. Keep full OCR accessible for continuations. Use typed
   final-section values/evidence, not arbitrary JSON-path edits. Description
   grouping and quantities must be reconciled before goods are assembled.
3. Explicitly surface cross-field constraints. Prefer deterministic conversion
   for known category aliases, unit names, exact sums and references, with quoted
   operands and no guessed interpretation. A schema-only rejection should lead
   to one bounded correction of that section, not rewriting the whole document.
4. Use source checks to reject fabricated or moved text and preserve immutable
   OCR. Track candidate additions, omissions, duplicated occurrences and changed
   grouping separately. A mechanically valid section is a **candidate**, not
   an accepted reference. Display disagreements with the actual source and exact
   differing spans instead of requesting repeated free-form whole-document
   reviewer explanations.
5. Finish the **100-document validation reference with direct source adjudication**.
   Reuse the frozen section candidates and prior explicit decisions. Check both
   included facts and source facts omitted by every candidate; model agreement
   cannot catch the latter reliably. This is a finite reference-building task,
   not a claim that a classifier makes semantic review unnecessary.
6. Apply the same fixed policy to the 1,057 training IDs, keeping explicit review
   decisions and issue-grouped queues. Publish only fully adjudicated documents
   and verify section composition, unchanged OCR, schema, relations and split
   accounting. Do not call a sample approved merely because its address passed.

Remaining unqualified areas include party-role discovery, separate country
normalization, contacts, all cargo quantities/units and allocations, route roles,
dates, DG and omissions outside the probed fields. Their existing evidence is
reusable; this comparison has not magically certified them. The user need not
re-approve the already agreed labeling policy. The next implementation should
replace the unsafe annotation interface, not quietly relax the quality bar.

## Continuation: documented API check and 100-document party-section pass

### OpenAI Decisions API: documentation result, not a guessed endpoint

On 2026-10-01, checked the official
[DevDay 2026 announcement](https://learn.chatgpt.com/docs/whats-new/devday-2026),
[API reference](https://developers.openai.com/api/reference/overview),
[changelog](https://developers.openai.com/api/docs/changelog), and
[documentation index](https://developers.openai.com/api/docs/llms.txt).
The DevDay page is dated September 29, 2026. **No public documented Decisions
endpoint, request contract or pricing could be verified in these sources.** This
is not a claim that no private/preview product exists. No guessed endpoint was
called, and ordinary Responses classification was not renamed “Decisions.”
Consequently there is no defensible Decisions-versus-Jev comparison to report.
The earlier Jev results remain the actual measured classifier evidence.

The OpenAI Docs skill was used for that check and to confirm the ordinary
[structured-output interface](https://developers.openai.com/api/docs/guides/structured-outputs)
and [Luna model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna).
Continuation used GPT-6 Luna **medium**, through PydanticAI/Responses. No Sol calls,
new GLM calls, expensive-model escalation, training, or synthesis were launched.

### Implemented change: section evidence, not whole-document edit authority

The new one-time runner is
[`party_sections.py`](../artifacts/kie-training/analysis/real-data-repair-20261001/party_sections.py).
It handles **all supported party roles together**, including additional notify
parties on attachments. It sees numbered full OCR, not the old target: therefore
an omitted party in the seed cannot constrain its role roster. It does not receive
authority to alter cargo, routes, dates, equipment or other sections.

The typed result contains role evidence and exact source fragments for name,
complete `addressLine`, country, contacts, and explicit `sameAs` references.
Non-target party-like blocks and unresolved questions are retained separately.
Deterministic code joins selected fragments, checks source order and overlap,
validates party/reference constraints and composes only the party section into
the V7 candidate. Country is retained in `addressLine`; there is no city target.
The current experiment preserves printed country names/codes rather than guessing
countries from cities or forcing a new geography canonicalization policy.

This is a complete section-drafting component, **not** a complete annotation
pipeline or a semantic approval engine. In particular:

- Exact quotes prove that characters were printed, not that they belong to a
  particular party or field.
- An empty `unresolved` array is not approval.
- A missing field in both output and evidence still needs source review.
- An exclusion explanation can itself be wrong.
- Existing targets remain comparison material, not trusted gold.

### Probe, falsification and general guard

Seven difficult validation documents were tried before the full 100-document
pass. All seven were structurally renderable, but one (`06f8c293`) **stripped the
country from three addresses while still emitting each separate country**.
This contradicted the requested contract despite the explicit instruction.

The fix is an exact source-coordinate invariant: each selected country fragment
must also be covered by that party's selected postal fragments. The validator
rejects the candidate and permits one bounded correction request; it does not
silently append a country or restore seed content. The instruction was clarified
to state that the separate country field is additional extraction, not a reason
to strip the address. This catches the observed failure. It does not pretend to
detect a country omitted from both fields.

The shared prototype span selector also now supports an **explicit occurrence
index** when the same substring appears twice on one OCR line. Without an index,
such a quote still fails as nonunique. This was needed for a repeated carrier URL;
it avoids including caption/punctuation merely to make an otherwise correct quote
unique. Country containment uses those same coordinates, so a matching country
word elsewhere on the line cannot satisfy the postal-coverage check.

Transport/accounting controls remain bounded: concurrency four, SDK retries zero,
one output-validation retry, no provider/model fallback, immutable request and
response receipts, and a separate **$0.50 ceiling for this phase**. Existing repair
and model-comparison ledgers were not reset. An interface change under the same
policy cannot silently rerun already paid source requests.

### Measured scope and current state

Authoritative machine inventory:
[`party-sections/final-summary.json`](../artifacts/kie-training/analysis/real-data-repair-20261001/party-sections/final-summary.json).

| Measure | Result |
|---|---:|
| Validation IDs attempted in the main pass | 100 / 100 |
| Mechanically valid returned party sections | 99 |
| Failed extraction recovered offline from exact source evidence | 1 |
| Party records in the resulting candidates | 398 |
| Documents with direct full-OCR party review decisions | 17 |
| Party sections source-adjudicated after corrections | 11 |
| Reviewed party sections with named open questions | 6 |
| Other party sections awaiting complete adjudication | 83 |
| Additional field-only correction within those 83 | 1 |
| Whole documents approved across every section | **0** |

The 398 proposed party records versus 391 seed records are **not proof of seven
correct recovered parties**. Both additions and omissions have a source-level
review inventory. Thirty-seven model outputs contain unresolved notes; this is
not a count of defective documents. Some notes simply explain blank captions.

The 17 reviewed documents consist of the seven development sources and ten
additional validation sources chosen by sorting
`sha256("party-section-transfer-v1" + documentId)` after excluding those seven.
Their complete OCR, not only selected address snippets, was read. Source checks,
exact draft corrections and open questions are recorded in
[`source-decisions.json`](../artifacts/kie-training/analysis/real-data-repair-20261001/party-sections/source-decisions.json).
The second caption case was inspected separately and has only a field-scoped
approval. These are deliberately distinct scopes, not a hand-picked model
accuracy estimate.

### Actual corrections and preserved earlier work

Four draft corrections were applied during the 17-document adjudication:

| Source | Before | Corrected result |
|---|---|---|
| `0600699f` | `... EGYPT POSTAL CODE: 62815` in address | `... EGYPT 62815`; postal digits retained, caption excluded |
| `14f66f23` | `... NO:17 MERTER MERTER / GUNGOREN / ISTANBUL` | Remove the first tax-office `MERTER`; retain the later genuine locality continuation |
| `75471c13` | Phone `+20 2 2673 4000` | Restore printed `(EXT. 5065)` within the phone value |
| `70cfaed9` | Only E-Business URL selected | Also retain the distinct printed carrier home-page URL, once |

A whole-candidate caption screen found a second case, `150e9dad`. After inspecting
the complete source, its consignee label was corrected from
`... SUEF, EGYPT POSTAL CODE:62815` to `... SUEF, EGYPT 62815`. Exact comparison proves
no other party value changed. This does **not** approve its other party decisions.
Both observed postal-caption flags are cleared in the recorded corrected views;
that screen is not claimed to cover every possible caption or semantic error.

The sole failed request, `7d485217`, twice quoted the excluded importer ID as
`37707623` instead of the printed `377077623`. Its existing answer was recovered
offline by fixing that negative-evidence quote; no target value changed and no
extra API call was made. The failed paid receipt remains immutable. Carrier-role
semantics in this document still require review.

The adjudicated references also correct the fallible seed where evidence supports
it. Examples include:

- `06f8c293`: a previously missing delivery agent, including its full postal line,
  phone and email, is recovered from the explicit arrival/release block. The
  separately named carrier is not merged with it.
- `48c177a3`: restore the printed shipper postal line `BEQAA LEBANON`; exclude share
  capital from the carrier name and retain its printed contact website.
- `29264000`: remove the unsupported consignee `EG` country/`EG.` address suffix;
  a repeated page shows it joined to an email-like token, not a clear postal
  country declaration. No missing `@` is invented.
- `0600699f` and `99629272`: `TAIWAN` is the separate country, while the complete
  printed `TAIWAN, R.O.C, TW` wording remains in each address.
- `75471c13`: retain the explicitly printed carrier locality `Hamburg` without
  inventing Germany.

Eleven complete party sections now have replayable source decisions in
[`party-sections/adjudicated/`](../artifacts/kie-training/analysis/real-data-repair-20261001/party-sections/adjudicated/).
The six open reviews are stored there too, with `sectionStatus: review_open` and
`documentApproved: false`; those entries are not accepted labels. Earlier manual
non-party corrections are retained during composition. Where two historical
manual versions differed only in party contact wording, their unaffected fields
were required to agree exactly before composition. The superseded contact-name
proposal was not treated as a settled labeling policy.

### Remaining six reviewed questions: bounded and explicit

| Source | Remaining decision |
|---|---|
| `025a144d` | An uncaptioned personal name immediately follows the consignee company. Determine the consistent company-name/contact-name boundary rather than accepting the model's merged name. |
| `53e77979` | Same uncaptioned-person boundary in consignee and notify blocks, plus printed `EGYI` without an established country normalization decision. |
| `1891d90d` | Two named people and their phones, but one `contactName` field. Earlier semicolon formatting was a superseded proposal, not an approved general rule. |
| `d9be84a2` | Rider logistics contacts require ownership adjudication; matching an email domain alone must not silently attach them to shipper. There are also two contact people. |
| `aa9ed483` | Delivery-agent postal/contact continuation is interrupted by route headings; registered-office carrier identification also needs consistent treatment. |
| `c7ff7761` | `SHIPPING AGENT REFERENCES` identifies an agent but does not explicitly distinguish forwarding versus delivery. Egyptian locality alone is not a role rule. |

These are **not irreparable documents** and not a reason to repeat whole-document
generation. Their supported postal fragments, identities and other evidence are
saved. The next work should settle these issue groups once against the extraction
contract and source context, then apply that policy to related candidates.

### Cost, latency and validation

Only Luna medium was used for paid work in this phase:

- Seven initial requests plus 100 main requests: **107 paid document requests**,
  **123 provider responses including validation retries**.
- Main 100-document pass: **$0.14195394** token-estimated cost.
- Entire new phase: **$0.15190114** token-estimated cost, with **$0 unsettled
  reservations** in this phase. Earlier ledgers and their unresolved reservations
  remain separate and unchanged.
- All-attempt usage: **483,078 input tokens**, **217,874 output tokens**; **59,374
  cached-input tokens are a subset of input**, not extra tokens. Reasoning is
  included in billed output usage. Cost uses the documented Luna rates used in
  the preceding experiment; it is not an account invoice reconciliation.
- Main-pass median document latency **14.89 s**, maximum **48.91 s**, concurrency
  four. Sixteen main requests needed their one allowed validation retry.
- A simple linear estimate for **party drafting alone** across 1,157 documents is
  approximately **$1.64** at this workload mix. This is not a quote for semantic
  adjudication, other sections, or the complete repaired dataset.

Validation completed:

- **32 targeted tests passed** in 6.53 s; Ruff passed. Tests cover exact text,
  ordering/overlap, nonexistent quotes, explicit occurrence indexes, source-local
  country containment, duplicate roles/contacts, reference constraints, section
  authority and unchanged unrelated fields. Deliberate omission cases remain in
  the tests to demonstrate why mechanical validity is not semantic approval.
- **All three pinned source files and all 1,157 OCR hashes verified unchanged**.
- All 100 final mechanical sections (including the offline quote recovery), 17
  adjudication receipts and the additional field correction replayed exactly.
- Full offline accounting/integrity replay: **9.20 s**, **147.1 MiB peak RSS**, no
  swap. No production training/synthesis hot path was modified.
- Paired span-projection benchmark on the same 26 existing selections, eight
  alternating runs: old median **6.842 μs**, new **6.828 μs**, effectively parity
  (**−0.21%**, within measurement noise). Explicit repeated-occurrence support
  did not impose a material throughput cost.
- Of seven previously source-adjudicated postal checks belonging to validation,
  the raw new drafts matched **5/7**. The two misses are the caption/tax-office
  cases corrected above. **99% mechanical success is not 99% semantic accuracy.**

### What to continue next

No new training-ready real dataset has been published. The concrete progress is
cheap full-party drafts for the complete validation split, eleven adjudicated
party sections, exact applied corrections and a reusable typed evidence path.
The next finite tasks are:

1. Complete source adjudication for the remaining 83 party sections, reusing
   existing paid drafts; the field-only correction does not reduce that count.
2. Resolve the six reviewed issue groups consistently, without another generic
   actor/critic rewrite cycle or treating a classifier score as authority.
3. Use separate typed outputs for goods grouping, cargo quantities/placements
   and other document facts, reconciling their dependencies before publication.
   Existing valid manual cargo corrections remain available; this pass did not
   overwrite them or declare those other sections complete.
4. Finish the 100-document whole-reference adjudication, then use the frozen
   rules/evidence interfaces on the 1,057 real training IDs. Publication must
   validate all sections, relations, split coverage and unchanged raw OCR.

The response to this experiment is **not** a more expensive model by default.
Cost of drafting is already small. The remaining measured problem is semantic
ownership/completeness and a small set of annotation-contract decisions. Those
need explicit source decisions; hiding them behind a “pass” would repeat the
previous failure.

## Follow-up audit: full validation-party review and explicit corrections

### Outcome and launch boundary

The follow-up reviewed **all 100 validation party candidates**, then applied
source-adjudicated corrections to **25 documents**, through 35 explicit decisions
changing 32 target paths (an array/whole-party change is one path). These numbers
include the newly approved naming convention, not only previously erroneous facts.
This is **not a clean bill of health for the complete validation set**. The audit
also falsified using a model review's empty finding list as an acceptance gate.
Accordingly the conditional launch of the **1,057 training-source annotations has
not happened**. No production dataset, source OCR, training configuration, or
synthetic template was modified in this phase.

The exact working artifacts are:

- [Verified replay/accounting summary](../artifacts/kie-training/analysis/real-data-repair-20261001/party-audit/verified-summary.json).
- [Every reviewer finding and its disposition](../artifacts/kie-training/analysis/real-data-repair-20261001/party-audit/finding-dispositions.json).
- [Applied field-scoped derivatives and exact receipts](../artifacts/kie-training/analysis/real-data-repair-20261001/party-audit/field-corrections/).
- [Independent review runner](../artifacts/kie-training/analysis/real-data-repair-20261001/audit_party_sections.py).
- [Source-decision replay](../artifacts/kie-training/analysis/real-data-repair-20261001/apply_party_audit_decisions.py).
- [Versioned evidence compiler](../artifacts/kie-training/analysis/real-data-repair-20261001/party_contract_v2.py).

The original paid drafts, previous manual decisions and prior ledgers are immutable.
The new derivative files contain the full candidate target plus exact party evidence,
before hashes, changed paths and reasons; `wholeSectionApproved` and
`documentApproved` remain false for these field-scoped repairs. That explicitly
prevents an approved phone correction from certifying an unreviewed address or cargo.

### Naming decisions explicitly approved by the user

1. Keep an uncaptioned personal name below a company's identity in `name`, unless
   the OCR establishes it as a contact. Do not delete that printed identity text.
2. Retain multiple explicitly named contacts in the existing `contactName` string,
   separated by `; `. The evidence groups each person separately, so a two-line
   single person's name is joined with spaces, not misinterpreted as two people.

Applied examples:

- `53e77979`: `EL HEKMA FOR GENERAL CONTRUSTION CORPORATION` now includes
  `HASSAN ABD ELNASSER MOHAMED` in both consignee and notify identities, removing
  its unsupported duplicate contact assignment. Its separate `EGYI` country
  question is not silently resolved by this naming correction.
- `1891d90d`, `4112ef7e`: `Mr. Amr Rabbany Mr. Shady Farid` becomes
  `Mr. Amr Rabbany; Mr. Shady Farid`.
- `d9be84a2`: `HUANG JINLONG/ PANNI` becomes `HUANG JINLONG; PANNI`.
  Ownership uses the repeated shipper telephone as well as the company-domain
  context; domain matching alone was not made a general ownership rule.

### What the independent review actually established

GPT-6 Luna medium reviewed numbered complete OCR and the party candidate, without
the seed/old labels as authority. It inventoried party blocks and returned exact
source-quoted findings. It had no permission to edit labels. No Codex subagents or
expensive-model escalation were used.

| Measure | Count |
|---|---:|
| Validation sources reviewed | 100 |
| Valid model review outputs | 99 |
| Review recovered offline without another API request | 1 |
| Sources with reviewer flags | 48 |
| Reviewer findings | 74 |
| Findings corrected as proposed | 30 |
| Findings corrected under a different party owner | 3 |
| Finding accepted only in part | 1 |
| Rejected findings | 14 |
| Nondefect explanatory review notes | 5 |
| Findings still needing source/policy adjudication | 21 across 15 documents |

The remaining 52 sources received no model finding after the offline recovery.
**That is not 52 newly certified party sections.** There are confirmed counterexamples
to treating this as acceptance: the reviewer missed a missing shipper country in
`5222fa98` despite `LUXEMBOURG` already appearing in its postal address, and accepted
`TO ORDER` as the consignee identity in `51820556`. Both were corrected in the
source-decision pass. Earlier eleven whole-party source adjudications remain
separate from these new field-scoped decisions; this pass does not relabel all
unflagged candidates as gold.

The failed review (`c931959d`) used an occurrence index across repeated pages
instead of within one cited line. Its final already-paid response had all the
quoted text on the stated lines, uniquely. The offline recovery corrected only
those occurrence indexes, checked every quote, and preserved the findings.
This repair costs zero additional tokens and does not imply semantic approval.

### Falsification controls: reviewer versus deterministic invariants

Seven deliberately corrupted copies of previously reviewed candidates were passed
through the same review, without revealing that they were controls. Original
datasets were not changed.

| Deliberate corruption | Reviewer detected it? |
|---|---|
| Delete consignee postcode `62815` | Yes |
| Add `POSTAL CODE:` to the address | **No** |
| Insert tax-office `MERTER` into the street address | Yes |
| Delete telephone extension `5065` | Yes |
| Delete `LEBANON` from both address and country | Yes |
| Remove the complete supported delivery agent | Yes |
| Delete postcode `511300` after the telephone line | Yes |

The caption control is particularly useful: the reviewer sometimes requested
restoring this caption even in already correctly repaired candidates. A typed
postal-caption rejection now catches this known failure; all seven known control
errors are detectable by the combined mechanisms, but this is a regression set,
**not a measured universal semantic-accuracy rate**. The reviewer also recommended
incorrect changes such as merging carrier share capital into its name and
promoting an explicitly named signing agent to contractual carrier.

### Substantive source-backed corrections

| Source(s) | Correction and ownership evidence |
|---|---|
| `13ea92fa` | Added NUN Overland GmbH forwarding agent from the attachment's forwarding order, matching B/L and forwarding conditions; restored printed address, two phones, two emails, website and named contact. Did not infer Austria from the address. |
| `8475b366` | Added the named principal exporter block at the top of the B/L as shipper, with exact postal text and phone. This is not a customs-ID-only declaration. Country remains absent. Other interleaved delivery details are not certified by this fix. |
| `66feaac3` | Restored `BLOCK NO.: 13019` and `PIECE NO.: 1,2` in the consignee postal line using the matching `***` continuation. The reviewer also flagged a phone that was already present; that recommendation was rejected. |
| `d87589f1` | Restored `#15093 SIHEUNG-SI SOUTH KOREA` from the shipper/exporter-contact continuation. |
| `5222fa98` | Restored separate shipper country `LUXEMBOURG`, using the second occurrence on the postal line; the complete address was already present. |
| `45ed05eb` | Restored `MOSTAFA METWALLY` and `002-01159888371` to consignee and notify. Their `**` continuations establish ownership. The reviewer favored shipper based on neighboring exporter declarations; that assignment was rejected. |
| `c82b915f` | Restored `0114741143` to consignee and first notify from the matching `**` continuation; shipper's continuation uses `*`. |
| `64513c56` | Restored omitted notify telephone `(202) 22677003`, preserving the mobile and other fields. |
| `62472e22` | Restored `/ EXT 112` to its phone; kept `91-755-2460107 / 108` as printed abbreviated notation, without inventing a prefix or treating `108` as a complete independent number. Its broken email remains a separate unresolved issue. |
| `c570eef4` | Restored `4863542` under the joint TEL/FAX caption; did not invent a prefix. |
| `5e03e6b2`, `f02bbfd5`, `5fa29caa` | Removed only telephone list separators/trailing comma; selected the exact TEL occurrence rather than a coincidentally matching FAX occurrence. |
| `51820556`, `e42f749e`, `62472e22` | Removed `TO ORDER`/`TO THE ORDER` as a consignee identity where no named party or other consignee details were printed. OCR and non-party facts are unchanged. |
| `e815e426` | Removed unsupported forwarding-agent role from generic `SHIPPING AGENT REFERENCES`. The block remains in OCR and negative-evidence receipts, not forced into delivery or another role. |
| `57371038` | Restored `Société Anonyme` in carrier identity, excluding the following share-capital amount. |
| `9d52b70f` | Restored `FOR IMPORT AND TRADING USED SPARE PARTS FOR CARS` as business-identity continuation, stopping before the postal words on the same line. |
| `4a68e8cb`, `5cabcd4b` | Retained printed carrier-owned website consistently. |
| `765685ca` | Restored the postal terminal period immediately before `TEL` in consignee and notify; no invented formatting elsewhere. |

These are explicit source annotations, not production rules keyed on document IDs.
The reusable mechanisms are exact source selection, occurrence-aware ownership
evidence, separate contact-person grouping, caption rejection, schema validation,
party-only edit authority, and immutable replay. They **do not** claim to solve
unknown semantic ownership through string presence.

### What still prevents the conditional scale-up

The 21 unresolved reviewer findings are enumerated, rather than an unspecified
long tail. Their 15 source prefixes are:

`150e9dad`, `2a22d65b`, `39a66fab`, `62472e22`, `765685ca`, `839005dd`,
`aa9ed483`, `bde3a722`, `c570eef4`, `d1fdcd92`, `e42f749e`, `e47eaf30`,
`ed2d1b59`, `f0cd1799`, `ff29c608`.

They cluster into:

- **Issuer versus contractual carrier:** letterhead/imprint alone, generic carrier
  terms and separate signing entities must not be interpreted differently per
  document. `ed2d1b59` has both a `Carrier / dcsa` entry and a different named issuer.
- **Interrupted postal/contact ownership:** route headings interrupt an agent's
  apparent address/phone; continuation text may contain a tax ID and an unrelated
  country. `aa9ed483`, `150e9dad`, `ff29c608`, `bde3a722`, `e47eaf30` illustrate this.
- **Multiple identities/addresses within one target role:** `f0cd1799` names a
  Brazilian shipper and a UK `ON BEHALF OF` continuation; `d1fdcd92` has differing
  notify-address wording. Concatenating both can create a fictitious postal address.
- **Remaining field-boundary policies:** a business division versus postal
  department, malformed country wording, abbreviated/broken email continuation,
  and whether a repeated alias adds a distinct fact.

This is the inventory of **open reviewer findings**, not an assertion that exactly
15 documents contain every remaining party error. The independent checks already
found genuine reviewer misses. A source-to-label completeness pass and explicit
source decisions must accompany the frozen validation reference; a second empty
model verdict is not a substitute. Other document sections are outside this pass
and remain part of the real-only rebuilding task.

The earlier questions not captured by those 21 findings also remain visible:
`53e77979`'s `EGYI` country decision is not settled by its approved name correction;
`7d485217`'s letterhead-carrier attribution was not certified merely because the
new reviewer raised no objection. Therefore the known open source/policy list is
at least **17 distinct documents**, and the unflagged candidates still require
completeness adjudication. This is not a claim that all other 83 documents are
already fully approved. Previously open naming questions are now governed by the
user's explicit two-rule decision, not by a superseded semicolon proposal.

Recommended next sequence:

1. Finish these bounded source/policy decisions, retaining unresolved evidence
   rather than forcing a label. Do not regenerate all 100 drafts.
2. Complete the validation reference's party completeness/ownership inventory and
   replay the approved corrections and adversarial cases. The new evidence compiler
   is tested on the 25 corrected sources, but its revised generation instructions
   have **not** yet been demonstrated on an independent inference batch.
3. Only then launch remaining training-source drafting/review under the frozen
   rules. Alternatively, training-source candidate drafting could be run earlier
   as explicitly unapproved work, but that does not satisfy the user's requested
   “validate first, then proceed” condition and was not selected here.

### Cost, tests, measured execution and side effects

- **New phase cost: $0.18236156 token-estimated**, including all seven controls
  and retries; **$0 unsettled reservation**. Separate $0.75 phase cap preserved.
  The preceding $0.15190114 party-draft ledger was not reset or overwritten.
- 107 requests, **121 provider responses including retries**; 491,440 input tokens
  and 278,714 output tokens. 68,216 cached-input tokens are a subset, not additive.
  Reasoning is included in output accounting. This is not invoice reconciliation.
- **39 targeted tests passed**, 6.38 seconds. Tests exercise grouped/split contact
  names, contact-only parties, caption rejection while preserving postal digits,
  placeholder identity rejection, exact quote/occurrence validation, schema and
  section authority, alongside the previous evidence tests.
- Ruff passed for executable review, compiler, repair/replay and test scripts.
  The narrative finding-disposition file was checked with only the line-length
  rule excluded; it stores long human adjudication explanations.
- All **three original dataset hashes and 1,157 OCR hashes/split identities**
  verified unchanged. All 25 correction receipts recompiled exactly, and the
  correction script replayed idempotently. No unrelated target value changed;
  previous manual non-party corrections were preserved.
- Full offline verification: **4.09 s**, **158.75 MiB peak RSS**. No production hot
  path was modified. Paired compilation on the same 25 corrected sources, seven
  rounds, measured **0.282 ms** old versus **0.415 ms** V2 per document: **+0.133 ms**
  for explicit contact grouping and extra validation. This is negligible compared
  with model-call latency, but is recorded rather than described as a speedup.

## 2026-10-02 continuation: representation research and completed category review

### What is finished, and what is not

The interrupted category-review batch has finished for all **100 validation
documents**. In this continuation, **30 additional source-backed field decisions
were applied to 21 documents**, in two immutable derivative layers (14 decisions
in seven documents, then 16 decisions in 14 other documents). Together with the
previous 25-document correction pass, **37 distinct validation documents** now
have recorded field-scoped corrections. The layers overlap; 25 + 7 + 14 is not a
distinct-document count.

Every current candidate compiles through the current evidence contract: **100/100**,
including candidates not touched in earlier correction passes. This found and
removed another `TO ORDER` identity that had still been present in an older draft.
All corrected spans are reproducible from original OCR, and unrelated target
sections are unchanged. **This is mechanical/source-fidelity evidence, not 100
independent semantic approvals.** There are **20 explicit review-held documents**
after adjudicating this review, detailed below. The remaining 80 are not thereby
declared fully gold. No new whole-party or whole-document approvals were issued.

The 1,057 training sources have **not** been bulk relabelled. The launch condition
was to validate the party-label process first, not to silently turn a review's
empty issue list into training-ready acceptance. This continuation does not claim
that the complete real-only repair task is finished. The known blockers are source
ownership and singular-target ambiguity, not API cost or inability to edit JSON.

No production dataset, real OCR, historical labels, training configuration, or
synthetic template was changed. The corrections are in the explicitly separated
real-only repair workspace.

### User question: what does “on behalf of” mean, and where does it fit?

I inspected the mapped MPCI form at `artifacts/mpci-ai-schema/` (there is no current
equivalent `analysis/mpci-ai-schema/` directory). The inspected platform snapshot
is revision `72f9aff735c2581f2ccff74e7fde34c2aea9b265`, not a claim that the remote
platform was refreshed today.

- Its [Party definition](../artifacts/mpci-ai-schema/mpci-cuscar-canonical.schema.json)
  has a party-function role, multiple name/address lines, contacts, and **one**
  country/city pair per party. There is **no principal/represented-party foreign
  key or dedicated `onBehalfOf` field**. The name/address arrays are not a typed
  relationship between two separately domiciled companies.
- [DCSA's shipper definition](https://models.dcsa.org/2024Q3/EARoot/EA3/EA1/EA1/EA542.htm)
  explicitly accommodates contracting/delivery by a person, in that person's name,
  or on that person's behalf. [Maersk's carriage terms](https://terms.maersk.com/carriage)
  likewise distinguish the contracting merchant's authority to act for the owner
  and other entitled parties. Representation wording is not intrinsically an
  annotation error or proof of a different document class.
- The [DCSA B/L introduction](https://dcsa.org/standards/bill-of-lading/documentation-bill-of-lading-3/bill-of-lading-3-introduction)
  distinguishes B/L and sea-waybill characteristics through title/negotiability
  and delivery mechanics. An `ON BEHALF OF` phrase alone does not make a document
  a house B/L, forwarding instruction, or non-B/L.
- **Application to our extraction contract:** preserve an explicit `A ON BEHALF
  OF B` identity in the source's party role when there is one coherent postal
  context. Do not invent a forwarding-agent role for A or silently replace A with
  B. This is our source-faithful extraction/mapping choice, not an assertion that
  every customs authority requires a combined legal name.
- Distinguish identity wording from **signature agency**: `signed for/on behalf
  of the carrier X, by agent Y` establishes X as the carrier; it does not make Y
  part of X's company name. A vessel's Master is also a signing capacity, not a
  named contracting carrier. Care-of wording alone does not establish a new
  forwarding role.
- Where the source gives two companies **and two separate postal addresses**, do
  not concatenate them into an invented single address. The user explicitly
  authorized review holds for cases that cannot be represented correctly. Those
  cases remain held; there was no permission to choose a primary company/address
  and silently discard the other.

Concrete application:

| Source | Evidence and action |
|---|---|
| `150e9dad` | `AUO CORPORATION` plus the `++` continuation's `ON / BEHALF OF SAMSUNG / ELECTRONICS TAIWAN CO.,LTD` becomes one complete shipper name. Taiwan postal address remains owned by that shipper block. The unrelated trailing `84899112 Egypt` fragment in the notify block remains separately held. |
| `ff29c608` | Restored the shipper's explicitly linked `ON BEHALF OF SAMSUNG ELECTRONICS TAIWAN CO., LTD`. Consignee `+++` owns the separately printed postcode and telephone; reviewer advice to reassign them to second notify was rejected. |
| `c9c96562` | `CONAIR ITALY S.R.L RASMI KACHLAN` becomes `CONAIR ITALY S.R.L ON BEHALF OF RASMI KACHLAN`. One printed Italian postal context, no invented additional role. |
| `5222fa98`, `75471c13`, `da651f2f` | Existing explicit representation wording is retained. It is not stripped merely because the string contains two entities. In `da651f2f`, the separate unsupported carrier `The Master of Oslo Wave 3` was removed. |
| `f0cd1799` | MARFRIG has a Brazilian address; the shipper continuation names WESTON IMPORTERS and a care-of UK address. This cannot be collapsed into one postal address/country without a policy decision. Both blocks remain in audit evidence, and the document remains held. |

The read-only full-source representation census covers **all 1,157 OCR inputs**:
534 sources contain a representation phrase in OCR or a seed party name (493 train,
41 validation), with 784 OCR phrase occurrences. Most are boilerplate/signatures,
not defects. **33 seed records** have the phrase in party-name fields. These are
screening counts, not a claim that 534 sources need identity repair. The inventory
preserves each line and explicitly handles the one packet with no seed target.

### Review hardening and what the probes actually showed

The review now asks eight separate questions: role coverage, identity, postal
completeness, postal contamination, country, contacts, explicit references, and
repeated/alternative parties. It shares the extraction contract instead of using a
shorter contradictory summary. Separate country and website inventories are
compared against the proposed labels independently of the model's overall verdict.
Every supplied website cue must be classified; every proposed party must receive
a country decision. The reviewer cannot silently skip an inventory item.

The first review version drifted into nonexistent fields such as `signingAgent`
and confused a country already inside addressLine with the separate country field.
The shared role contract and independent inventories address these specific
interface defects. They do not turn model interpretation into an oracle.

All **10 intentionally corrupted controls** were detected under the revised
contract, including deleting separate country, inserting `TO ORDER` identity,
omitting carrier homepage, deleting a postcode/extension/party, and inserting
postal or tax-office captions. This is a measured regression set, **not** an
estimate of universal semantic recall.

An eight-source fresh-extraction probe independently showed why wholesale
redrafting is not the default fix: four drafts matched the corrected candidate
exactly; one correctly removed malformed `EGYI`; three omitted a supported phone,
carrier locality, or website. None replaced the existing candidate automatically.

Review validation was also separated from edit compilation. A review argument may
cite source lines out of order; an actual compiled field still requires ordered,
non-overlapping, occurrence-aware exact spans. Already-paid reviews rejected only
for argument-order/evidence-shape reasons were recovered offline without changing
their findings. Two final review responses needed explicit citation transcription
adjudication: split `ON` across its actual source lines, and correct a Turkish
caption quote. Their conclusions and proposed labels were not changed.

### Confirmed corrections versus false review alarms

Additional applied changes include:

- Keep malformed `EGYI` in addressLine, but remove it as a separately recognized
  country; remove `***@mlh-shipping.com`, whose complete email local part is absent.
- Restore India through the explicit `#` shipper continuation—not through unrelated
  foreign-exporter metadata. Preserve the consignee's `+++`-linked postcode.
- Apply the approved Linda/Mark naming rule; retain the full printed former-name
  carrier identity in `bde3a722`.
- Restore attributed carrier homepages in five sources, plus the separately printed
  COSCO homepage alongside its existing e-business site.
- Prefer the full printed carrier legal form over its signature abbreviation in
  two sources, excluding share capital.
- Remove linked continuation-marker characters and one repeated SIHEUNG-SI locality
  from each of two OHYOUNG postal labels, preserving postcode, numbers, district and
  both printed country wordings. Remove one analogous linked `*` from another
  shipper address. **Real OCR is unchanged.**
- Restore branch routing text and a postal punctuation boundary; remove the
  otherwise empty `TO ORDER` consignee and the unsupported vessel-Master carrier.

Source review rejected the model's recommendations to include Russia-policy/help/
clause links as contact websites; replace country with a whole street/city line;
infer Austria from `A-8430`; normalize malformed country tokens; merge a signing
person into a carrier name; or transfer explicitly marker-linked contacts to a
different party merely because its block is nearer. These rejected recommendations
are retained, not hidden or counted as additional corrected defects.

The final 100-document **review disposition**, distinct from semantic approval:

| Disposition | Documents |
|---|---:|
| No finding in this reviewer run | 43 |
| Findings rejected after source/policy review | 14 |
| Omission supported by source/target policy | 10 |
| Corrected, with no remaining finding from this review | 13 |
| Corrected but another ambiguity remains | 1 |
| Other explicit review holds | 19 |

### Exact remaining held scope and next work

The 20 held documents fall into these overlapping intervention families:

| Family | Sources | Required next action |
|---|---|---|
| Multiple postal addresses / represented party in another country | `66d5d7f6`, `d1fdcd92`, `f0cd1799` | Source-layout/party relationship adjudication; keep both alternatives in evidence. Do not choose or concatenate implicitly. |
| Interleaved or unowned postal/contact continuation | `150e9dad`, `ff29c608`, `7a02e126`, `aa9ed483`, `bde3a722`, `e47eaf30` | Resolve the exact fragment owner or document its omission as unsupported. No regeneration or new OCR wording. |
| Destination-office to delivery-role equivalence | `661601d9`, `7f6df7bd`, `c9739182`, `bde3a722`, `e47eaf30` | Establish one general target-role convention from actual form/transport semantics; do not let different model calls choose differently. Existing guesses are not certified. |
| Issuer/brand/signature or competing identity | `298806fe`, `2a22d65b`, `839005dd`, `aa9ed483`, `ed2d1b59`, `f67e893f` | Adjudicate source context; no lookup-derived carrier identity absent from OCR. Preserve supported company text without merging unrelated entities. |
| Uncaptioned party-like blocks | `67e40df5`, `765685ca` | Establish a defined role or explicitly exclude with evidence, not geographic proximity. |
| Department versus postal-routing wording | `e42f749e` | Set the consistent treatment of printed internal department `POLYESTER EXPORT BUSINESS`; the owned India continuation is already repaired. |

Previously noted `7d485217` remains a separate completeness-adjudication item even
though this reviewer raised no finding; its full BORCHARD issuer letterhead is
stronger evidence than a bare logo, but this report does not invent a new approval.
Accordingly **20 review holds is not the total amount of semantic work remaining**.
The old eleven explicit whole-party adjudications remain separate historical
receipts; all other samples still need an accountable final completeness decision.

This is a concrete decision queue, not permission for another blind full redraft.
Next work should settle these source/role conventions once, apply exact affected
field changes, and finish the validation-party reference before the conditional
1,057-source annotation launch. Sources that do not fit can remain explicit holds
under the user's instruction. Cargo/other sections remain separate subsequent work.

### Artifacts, reproduction, accounting and measured validation

All current files are under
`artifacts/kie-training/analysis/real-data-repair-20261001/`:

- `party_reconciliation.py`: source-first extraction and category review; exact
  request/candidate hashes, response accounting, explicit interruption recovery.
- `resolve_party_sources.py`: first seven source-backed correction receipts.
- `finalize_party_reconciliation.py`: second 14-document repair and complete
  source-adjudicated finding dispositions. Its `final_candidates()` function loads
  the latest view with exact predecessor, OCR, evidence and target checks. The
  pre-review view remains separate so historical paid reviews still reproduce.
- `party-reconciliation/adjudication.json`: all dispositions and 100 compiler proofs.
- `party-reconciliation/source-corrections/` and `review-corrections/`: immutable
  derivative targets, before/after evidence, reasons and changed paths.
- `party-reconciliation/representation-inventory.json`: full real-source census.
- `party-reconciliation/completed-review-summary.json`: paid-review completion and
  accounting before the two explicit citation adjudications.
- `party-reconciliation/citation-adjudications/`: the two exact offline citation
  corrections, with the original responses preserved.
- `party-reconciliation/final-replay-benchmark.json`: measured latest-view replay.

Cost for this reconciliation phase is **$0.32902154 token-estimated** from saved
responses, including probes, controls and retries. The plugin interruption left
**eight requests with unknown actual usage**, conservatively retained as up to
**$0.212301 additional liability**. Thus accounted phase cost is $0.3290, with a
conservative ceiling of $0.5414—not falsely reported as zero-cost interrupted work.
136 historical request receipts, 158 saved provider responses; 793,502 input and
559,404 output tokens, with 333,674 cached input tokens included in the input total.
No unresolved in-flight requests remain. The $1 phase ceiling was not reset.

The preceding party-draft and party-audit costs ($0.15190114 and $0.18236156) remain
separate and unchanged. Across these three party phases, known token-estimated
spend is **$0.66328424**, plus the same interrupted-request uncertainty. This is
not a total for every earlier real-label research experiment or an invoice total.
The final source adjudications used **zero additional API calls**.

Validation:

- **46 targeted tests passed**, 6.58 seconds, including evidence compilation,
  section authority, grouped contacts, omitted independent inventories, and strict
  distinction between review citations and actual edit coordinates.
- Ruff passed on the touched scripts; only narrative line length is exempted.
- Idempotent repair replay: **7.38 s**, **107.23 MiB peak RSS**. It reproduces all
  14 final derivative files and the disposition artifact byte-for-byte.
- Latest 100-candidate load plus provenance verification: **1.582 s**; compilation
  median **0.359 ms/document** across seven rounds; peak **105.98 MiB RSS**.
  No production hot path changed. This workload differs from the earlier
  25-document benchmark, so these figures are not claimed as a paired speedup.
- Three original dataset files and **all 1,157 OCR hashes/splits** verified unchanged.
  Earlier 25 correction receipts still recompile; all current 100 candidates pass
  the revised compiler. Zero OCR edits and zero non-party target edits.

The audit therefore delivered real, validated field corrections and a concrete
inventory. It did **not** establish that an automatically passing party candidate
is correct. Declaring the remaining real sources ready on this evidence would
repeat the failure the user explicitly asked this audit to prevent.

## PDF ownership adjudication, 2026-10-02

### Completed work and authority boundary

The 20-source ownership/representation queue above has now been checked against
original PDFs as well as pinned OCR. **17 holds resolved; 3 remain.** Eleven exact
field decisions were applied in six documents. The four correction layers together
contain **76 decisions / 73 changed-path events across 40 distinct validation
documents**. Counts include sequential edits to the same field and policy-alignment
changes, not 76 independent original defects.

The PDF inspection covered **23 sources / 26 pages**, including three additional
negative/completeness controls described below. No OCR was rerun or replaced.
Every new target string comes from existing OCR spans. PDF boxes establish the
owner of an interleaved fragment, not permission to import a PDF-only value.
This distinction matters: otherwise a seemingly improved label would ask the
text-only model to extract information absent from its input.

These are immutable candidate derivatives under `party-reconciliation/pdf-corrections/`,
not writes to the active training dataset. All 100 current validation-party
candidates recompile exactly. The three original dataset files, all 1,157 OCR
hashes, and the 1,057/100 split identities are unchanged. No cargo, route or other
non-party target field was modified.

### Actual fixes

| Source | Before | Corrected party target / reason |
|---|---|---|
| `aa9ed483` | Delivery address only `18 ELPHARAANA`; country/phone absent; carrier absent | Delivery address becomes `18 ELPHARAANA ALEXANDRIA ALX EGYPT 21514`, country `EGYPT`, phone `+2034868092`. All fragments are in OCR lines 48/53–55 and the PDF release-contact box. Add issuer `Danmar Lines Ltd`, postal `P.O. Box 2680, 4002 Basel (Switzerland)`, country `Switzerland`, all from OCR line 5. Exclude the separately captioned fax. |
| `bde3a722`, `e47eaf30` | PAN MARINE destination-office phone absent because OCR placed it after route headings | Add `020 6 23597220` from each source's own OCR occurrence. PDF confirms the phone is inside the destination-office box. The printed SOKHNA/EGYPT row also belongs to that box, not to a neighboring route value. |
| `ed2d1b59` | Carrier `dcsa`, no address | Carrier becomes `HappySun Logistics Ltd.`, address `Dongcheng Qu 332, 200000, Shanghai City, China`, country `China`. The PDF places this typed entity inside the Carrier box; `dcsa` is the form logo. OCR contains both the actual company/address and its carrier-clauses signature context. No signatory is added as a contact. |
| `e42f749e` | Shipper address skipped `POLYESTER EXPORT BUSINESS` | Restore this internal departmental routing line between `RELIANCE CORPORATE PARK` and `8B, 1ST FLOOR...`. It is postal routing inside the shipper block, not a cargo description from another section. Earlier explicitly linked India/phone correction remains. |
| `f67e893f` | Carrier's printed homepage absent | Add `www.turkon.com` from its own tariff line. Explicit carrier identity establishes ownership. Retain the complete printed English legal name; do not append its Turkish translation as a second company. |

The registered-office issuer decision is stronger than guessing a company from a
bare logo or domain. In the Danmar source the full company/address is the document
imprint; its original signature confirms carrier identity. The carrier's own
[published terms](https://www.dhl.com/content/dam/dhl/local/at/dhl-global-forwarding/documents/pdf/at-global-forwarding-danmar.pdf)
also distinguish Danmar from its agent. No address or identifier from those public
terms was inserted into our labels.

### Resolved without inventing labels

- **Destination office:** `661601d9`, `7f6df7bd`, `c9739182` retain their existing
  delivery-agent assignments. The named carrier destination-office block is the
  shipment's destination release contact. The mapping also applies to the two
  PAN MARINE cases above. This is a documented extraction-role equivalence, not a
  rule that any company near a destination belongs in deliveryAgent. DCSA's
  [express-release definition](https://models.dcsa.org/2024Q4/EARoot/EA3/EA1/EA1/EA474.htm)
  describes the destination office's shipment-release role. Neither DCSA nor the
  MPCI snapshot supplies a dedicated principal/agent relationship field.
- **Signature overprint:** `298806fe`'s SEA EXPERT wording is an agent stamp across
  the UEN line. Carrier TransLiner remains unchanged. Do not append the agent to
  carrier identity or label its name as a registration number.
- **Unassigned contacts:** `7a02e126`'s `***` email/phone has no matching party
  marker in the PDF either. Current omission is intentional, not a missing-phone
  repair. Domain similarity and proximity are not ownership evidence.
- **Unassigned attachment company:** `765685ca`'s LATT address has no target-role
  heading even in the PDF. Do not guess forwardingAgent versus deliveryAgent.
- **Stray country text:** `150e9dad` and `ff29c608` actually print `84899112 Egypt`
  after Taiwan notify fax details. This is not an OCR-created alternate Egyptian
  postal address. Retain the complete Taiwan postal context and its country;
  preserve the inconsistent trailing source text in OCR/audit, without training
  it as a second postal country. Existing `ON BEHALF OF` name and `+++` postcode
  repairs remain intact.
- **PDF-only data:** `2a22d65b`, `839005dd`, `67e40df5` have carrier or other party
  details in the PDF that OCR omitted. These do not justify adding unseen values
  to labels. In `67e40df5`, the shipper, consignee, notify reference and carrier
  blocks are absent from OCR; an empty OCR-conditioned party section is therefore
  appropriate. This does not certify that OCR extracted the document adequately.
- Separate PDF notify blocks in `e42f749e` and `f67e893f` are also absent from OCR.
  No notify party/reference was reconstructed from the PDF alone.

Three further source checks closed previously noted narrow questions without
changing targets:

1. `7d485217`: BORCHARD's full issuer imprint, address, telephone and email are in
   OCR and match the carrier block. Existing carrier-only target is supported.
   PDF shipper/consignee/notify, a new-address notice and the agent signature are
   absent from OCR; they remain absent from labels. In particular, do not replace
   the OCR's printed old address with a PDF-only newer one.
2. `87db3c18`: the PDF prints one joint `TEL/FAX:` caption over the two retained
   numbers, not separately distinguished TEL and FAX slots. Existing joint-caption
   policy permits both as phone values. The issuer's PDF letterhead is absent from
   OCR, so it cannot supply a carrier target.
3. `c7ff7761`: GULF AGENCY is captioned `SHIPPING AGENT REFERENCES`, not specifically
   forwarding or delivery; current exclusion is supported. Shipper/consignee/notify
   blocks in the PDF are missing from OCR. Keep the explicitly OCR-named WAN HAI
   carrier; do not fill the missing parties from the PDF or from geographical clues.

### The three retained holds

These are not three failures to format JSON, nor failed provider requests. They
are known source/representation questions under the current singular postal target.

| Source / original PDF | Evidence | Why held |
|---|---|---|
| [`66d5d7f6`](../data/corpora/blc-swb-remaining-max15-23947b3edb6b/files/swb/2024/2024-02-29_474e8a2a-9bc0-42c9-9d04-0d93fa27b243.pdf) | Rochem consignee and notify print `Port Tawfik Free Zone Area Suez Egypt`, then `568 El Horreya Avenue Glym Alexandria Egypt`, plus `ALEXANDRIA 55698 Egypt`. | Two actual addresses in one role, not duplicated extraction of one city. The draft currently concatenates them and is **not accepted**. No authorized primary-address choice. |
| [`d1fdcd92`](../data/corpora/blc-swb-remaining-max15-23947b3edb6b/files/blc/2024/2024-03-08_9e4ad56f-0c0c-4a86-8a8e-53b01f08094f.pdf) | Notify main block has `PIECE 1B BLOCK ELMAHAGER`; attachment has `BUILDING 54 STREET / PIECE,3 BLOCK,13027...`. | Competing postal contexts. Only the later email is explicitly marked `NOTIFY PARTY CONTINUED`. Do not silently replace or combine addresses. |
| [`f0cd1799`](../data/pilots/blc-followup-500-c0139c175677/files/blc/2024/2024-06-19_36b928dc-7316-4334-8c95-b1d474f45750.pdf) | MARFRIG Brazilian address; `CONTINUATION SHIPPER` gives `ON BEHALF OF` WESTON IMPORTERS and a care-of UK address. | Two represented identities and postal countries cannot be flattened into one address/country while preserving ownership. No invented forwarding role or discarded principal. |

They remain held as the user requested for cases that do not fit. Original and
candidate evidence is preserved. Holding them is not authorization to publish the
other 97 documents: finding disposition and whole-section completeness approval
remain distinct. The separate broader completeness review still precedes the
conditional 1,057-training-source bulk relabel. This pass has not launched it.

### Reusable policy and processing implications

Carry these decisions into the next versioned annotation/review instruction,
without rewriting the policy snapshots of already-paid requests:

- Treat explicit named destination-release offices as delivery-role contexts.
  Generic signing agents and merely nearby companies do not establish that role.
- Preserve one source's `A ON BEHALF OF B` identity wording when it has one postal
  context. Keep signatures, represented identity, and separate address ownership
  distinct. Multiple conflicting contexts are review output, not a model choice.
- Include internal postal department/routing lines when their source ownership is
  established; exclude company identity and cargo/registration text. The decision
  is semantic, not a keyword deletion rule for words such as EXPORT or BUSINESS.
- Prefer the actual named issuer/contractual carrier over a form logo. A full
  issuer imprint is stronger than an isolated brand token. Do not use public
  lookup results to fill identity/address gaps.
- Classify a printed root contact website separately from a deep clause/help URL.
  Both must have a known owner. A whole-source URL inventory remains independent
  of the reviewer's overall verdict.
- Resolve OCR-interleaved ownership through the original source only where the
  text itself exists in the model input. Track genuinely missing OCR blocks as
  input limitations. Do not convert that missing-input problem into hidden labels.
- Keep conflicting or unsupported source fragments in audit evidence. Neither an
  extraction pass nor a reviewer may invent an ownership relationship to avoid a
  hold or to populate an optional field.

The deterministic compiler proves selected values and edit scope, not universal
semantic correctness. The new mutation tests deliberately alter parent hashes,
source hashes, evidence strings, compiled labels and unrelated target fields;
each is rejected, including an unrelated change with its target hash recomputed.
That is a concrete safety property, not a replacement for semantic adjudication.

### Validation, cost and reproducibility

- **53 targeted tests passed**, 6.59 seconds. Seven added replay tests exercise
  generic provenance/scope failure modes, not configured values or particular
  source-company exceptions. Ruff passed for all four changed/new scripts.
- The repair script reran idempotently, reproducing its immutable outputs:
  **10.23 seconds / 113.46 MiB peak RSS**.
- Independent verifier rehashed 3 original datasets, 1,157 OCR inputs and 26 PDF
  page receipts from 23 sources. Current candidate loading/provenance replay took
  **1.924 seconds**. Full verifier peak RSS: **159.53 MiB**.
- Paired compiler timing over the same 100 documents, seven rounds: **0.3614 ms**
  before versus **0.3663 ms** after the field corrections (**+0.0049 ms/document**).
  This is negligible output-size/runtime variation, not a production-path change.
- **Zero additional API calls or cost** for this PDF/source-adjudication pass.
  Prior token-estimated accounting stays $0.66328424 across the three named party
  phases plus up to $0.212301 unresolved interrupted-request liability. No ledger
  was reset, and this is still not a total for all historical experiments.

Scripts and receipts under `artifacts/kie-training/analysis/real-data-repair-20261001/`:

- `inspect_party_pdfs.py`: render/re-hash only specifically selected original pages.
- `resolve_party_pdf_ownership.py`: 20 explicit source adjudications, six exact
  derivatives; `pdf_candidates()` is the latest verified candidate loader.
- `verify_party_pdf_repairs.py`: independent integrity/scope replay and paired timing.
- `party-reconciliation/pdf-adjudication.json`: the 20 dispositions and 3 holds.
- `party-reconciliation/pdf-preservation-benchmark.json`: full checks and hashes.
- `party-reconciliation/pdf-review/`: pinned original PDF references and page images.
- `party-reconciliation/pdf-corrections/`: before/after evidence, decision reasons,
  unchanged-input proofs and strict V7 targets.

No blanket semantic approval or training-ready publication was created. Next work
is the remaining whole-reference completeness decision and other section audits,
not another blind rewrite of the repaired party fields. The three singular-target
holds stay separate so they cannot silently re-enter a future training export.

## Component-scoped completion, 2026-10-02

The user explicitly requested completion, with independent acceptance of resolved
parts. The earlier blanket `semanticApproval: false` states are not an adequate
completion workflow. This continuation uses eleven acceptance components: seven
party roles, cargo, equipment, route/transport, and references/dates/freight. A hold
on one component cannot erase verified work on the others. Whole-document training
exports still require all components, including absent-but-reviewed components,
to pass. Partial component exports must never silently become negative labels for
held fields.

### Concrete acceptance gates

1. Pin all 1,157 source IDs, original splits and OCR hashes. Real OCR, historical
   labels, old training artifacts and production datasets are read-only.
2. Compile party labels from exact occurrence-aware source selections. Nonparty
   leaves each require explicit literal, identifier, numeric, date, category or
   exact-sum evidence. Classification evidence is not confused with literal value
   assembly. Relation/heading context is separate from value characters.
3. Independently review complete OCR against both present and omitted target facts.
   Reviewers receive the actual V7 schema, category/semantic rules and scoped prior
   source adjudications. Unsupported fields such as faxNumbers cannot be requested
   merely because a generic document reviewer expects them.
4. Compare independent party country and website inventories to the proposed
   targets, separately from the reviewer's verdict. Every extraction question must
   receive an explicit scoped resolution or remain held; no silent question loss.
5. Reconcile real defects only within affected components. A revision cannot alter
   an accepted sibling component. Re-run source compilation and independent review
   against the actual revised target, not the previous target hash.
6. Test gates using deliberately faulty labels and intact controls. Distinguish
   schema/evidence checks from semantic detection; record false alarms as well as
   detected defects. Do not call a control suite perfect because it detects the
   injected error while also making incorrect demands elsewhere.
7. Publish component decisions and complete-document exports separately, with
   source/target/evidence hashes, exact remaining holds, provenance, all-attempt
   costs, and training-task/schema replay. Completed components have an explicit
   accepted state, not an indefinite partial-work disclaimer.

### Failures found while exercising the actual path

- Evidence sometimes quoted headings or relationship endpoints along with the
  literal value. The compiler rejected these. The output contract now separates
  value spans from ownership context. Saved responses can be recovered offline
  only by a unique value selection inside an already cited region, with no target
  changes. Ambiguous selections stay unresolved.
- An actor changed printed `21-MAY-2025` into `2105-05-21`, altered/truncated the
  digits of `7219.90.00`, and proposed a positive value for printed `0.000 M3`.
  These were rejected by exact evidence checks, despite structural validity.
- The illustrative DCSA source prints HS `6551` and container `16515`, outside the
  present target's 6–18-digit HS and ISO-container constraints. An actor invented
  an HS suffix; it was rejected. The incompatible source facts are explicit
  questions, never fabricated schema-conforming values. Other components remain
  independently usable.
- An initial blind semantic test detected eight injected errors (country omission,
  postcode omission, wrong party identity, tax contamination, swapped ports,
  wrong mass unit, incorrect allocation quantity, omitted seal). It also produced
  false demands for nonexistent fax fields and excluded postal captions. This
  falsified using that reviewer unchanged. Actual-schema injection, schema-path
  checking and the full approved party-field rules are now part of review; a
  fresh blind replay is queued before acceptance.
- Booking-reference rejection was a false alarm. The local MPCI snapshot has free
  text reference rows and its representative example includes `BOOK-SYN-1002`.
  UNECE [reference qualifier 1153](https://service.unece.org/trade/untdid/d03b/tred/tred1153.htm)
  includes booking references; [CUSCAR](https://service.unece.org/trade/untdid/d11b/trmd/cuscar_c.htm)
  has reference-bearing booking/consignment groups. Our extraction field collects
  captioned commercial/shipment/customs references, not arbitrary identity,
  address, contact or boilerplate numbers. This is a project mapping to the local
  free-text field, not a claim that all EDIFACT RFF qualifiers are interchangeable.

### Reproducible operation and spending

New self-contained scripts are alongside the existing repair evidence:
`complete_real_labels.py`, `completion_controls.py`, `reconcile_completion.py`,
`recover_completion_evidence.py`, `replay_completion_probe.py`, `test_completion.py`.
New requests, reservations, immutable responses, metadata recoveries, review keys
and candidates are under `completion/`. No historical ledger was reset. The model
remains the inexpensive Luna model with medium reasoning. Costs include failed
requests and retries; unknown interrupted requests retain their reserved liability.
The original four failed probe requests cost $0.01665968; their stored responses
were recovered/adjudicated without rerunning the four extraction calls. Initial
eight blind control calls cost $0.033042. These are intermediate figures, not the
final cost of completing the 1,157-document repair.

Current source-preservation and publication totals will be recorded after the
full passes and reconciliation. No model training is launched by this work.

### Whole-population continuation: draft recovery and falsification

The training-source party draft pass covered all **1,057** training sources. The
100 validation-source party candidates are reused from the earlier source/PDF
review, then checked again with all other sections. Thirty-two training drafts
initially failed exact proof/schema checks. A bounded reconciliation recovered
20; the remaining **12 were recovered offline** using saved responses and direct
source review, at **$0 additional model cost**. These are actual new compiled
candidates, not approval of the unreviewed responses.

The dominant remaining party-proof error was counting a quote's occurrence across
the document instead of within its cited line. Country substrings in company names
also needed distinguishing from the same country in the selected postal address.
Recovery selects the unique occurrence **inside the already selected postal span**,
not an arbitrary matching word elsewhere. Exact receipts distinguish metadata-only
changes from semantic corrections:

- `a13e53d4`: removed an invented second `THE` at an address line boundary; kept
  the first, actually printed `THE`. Kept the complete carrier name once, without
  appending its abbreviated signature repetition.
- `5030c6d8` / `c7d9a941`: split explicitly printed phone-list entries and correctly
  joined an explicitly printed line-wrapped prefix. No missing prefixes are inferred.
- `2ac48368`: a freight **Collection Business Unit** is not an established delivery
  agent. Removed that inferred role, without changing the real OCR.
- `d670d18f`: removed overlapping double-selections of country text and the phone
  switchboard-capacity note `(10 lines)`. Missing shipper/consignee role headings
  remain explicit source questions; this cannot authorize guessed role acceptance.
- `5055721d`: two provider responses degenerated into repeated whitespace. The
  actual source has three clear headed parties; their evidence was selected locally
  rather than spending another model call. All postal code digits are retained.

The revised blind semantic controls detected all nine injected defects and did not
flag the intact product-condition control. Review of the additional findings found
a false request to expand an omitted telephone prefix despite the literal policy;
that request is not an authorized repair. Missing commercial/import reference
findings are tracked separately. A further original/master-B/L-role control has
been added: an ordinary B/L number or ORIGINAL watermark must not create a distinct
original/parent reference automatically.

Nonparty reconstruction now also recovers unambiguous dates and reference spelling
from the **already cited source**, locally. For example, `21-MAY-2025` cannot become
`2105-01-01`; a complete named-month date has a deterministic conversion. Ambiguous
numeric dates such as `03/04/2024` are not guessed by this recovery. A reference
caption can join its value only when their cited source text matches the proposed
reference's alphanumeric content; punctuation then follows the source. No new OCR
region is searched to manufacture support. Tests reject incomplete dates, unrelated
reference captions, and edits to accepted sibling components.

Every old AAI entry and every extractor question must now receive an explicit,
scoped review disposition. A passed review cannot silently ignore an unresolved
question or move it to another component. The final export replays the real V7
training canonicalizer with a frozen vocabulary of **52 package and 6 container
categories**, rather than merely asserting that the JSON parses.

### Completion continuation: concrete checker and source corrections

The population run encountered a genuine external interruption: OpenAI returned
`credit_balance_exhausted`. The user topped up the account. Requests are resumed
with explicit attempt identities; previous successful candidates and every paid
response are retained. Definite quota rejections with no generated response count
as zero generation cost; interrupted/timeout requests retain reserved liability.
A request that generated a response before a later rejected retry retains its
observed charges. The quota interruption is not a semantic label rejection and
does not authorize publishing missing sections as empty.

Additional confirmed source corrections:

- `4877f7e3`: the shipper's complete postal block already ends `SHARJAH, UAE`.
  The draft had appended the separate attachment declaration `FOREIGN EXPORTER
  COUNTRY: UNITED ARAB EMIRATES`. Removed that non-postal duplicate from the
  selected address and used the actually printed postal `UAE` country. On the
  consignee, `MR.` at the end of one physical line belongs with `AHMED` and his
  phone on the next line, not in the address. Source OCR is unchanged.
- Address-caption recovery now also covers `REGD. OFFICE:-`, preserving the
  complete following street/site/postcode text. The compiler rejects these
  captions and explicit exporter-country captions in postal targets.
- `0961dfa2`: retained the printed damaged/missing vehicle-parts condition in
  goods description; it is product-specific evidence, not generic boilerplate.
- `67b70a30`: the source separately counts corrugated paper and a forklift. The
  correct target has paper with 92 pallets (36+43+13) in three containers, and
  one forklift package in the third container. The mixed third-container mass
  cannot be allocated proportionally to the two products. This is not three
  independent paper products, nor a single paper/forklift goods row.

The checker itself was falsified on `JAN. 25,2023`: `dateutil` interpreted the
missing space after the comma as a fractional-day format and defaulted the year
to 1900. A complete printed-date parser now handles named-month dates without
defaults. It rejects incomplete dates and preserves ambiguity for `03/04/2024`.
Source HS recovery likewise takes all digits from the already selected printed
code, never pads a four-digit heading into a six-digit code. Numeric occurrence
recovery distinguishes an isolated package count from the same digits embedded
inside a container identifier; two isolated matches remain unresolved.

Measured date-parser microbenchmark: 4,000 representative printed dates took
0.1065 s through the previous two-interpretation parser versus 0.0940 s through
the complete-date parser (about 12% less time on this probe). Process peak RSS was
95.3 MiB; this is not a claim about end-to-end API latency. The targeted combined
completion/party/schema suite passed 62 tests in 5.56 s at this point.

The independent reviewer also produced false positives, so its findings are
hypotheses, not automatic edit authority. In particular:

- Generic `SAID TO CONTAIN` / `SHIPPER'S LOAD STOW AND COUNT` responsibility
  captions do not belong in descriptions or handling instructions merely because
  they print near cargo. Product-specific conditions and actual handling actions
  remain target facts.
- The existing reviewed equipment grammar recognizes unqualified commercial
  `40HC`/`40HQ` codes as high-cube general-purpose equipment; explicit reefer or
  other type wording takes precedence. This does not license guessing the type
  of an arbitrary bare dimensional statement. The distinction is consistent with
  [Hapag-Lloyd's standard high-cube equipment](https://www.hapag-lloyd.com/en/services-information/cargo-fleet/container/40-standard-high-cube.html)
  and [Maersk's dry/high-cube equipment descriptions](https://www.maersk.com/support/faqs/2024/07/01/types-and-sizes-of-containers).
- `FOR ABOVE NAMED CARRIER` can explicitly link the sole header carrier identity
  to the carrier role; this is different from treating an unrelated logo alone
  as role evidence. A separately named signing agent is not the principal.

These are explicit contract clarifications, followed by current-contract review,
not silent conversion of reviewer defects into passes. Smaller field/subtree
repairs preserve the original candidate, its evidence, and all accepted sibling
components. They are still recompiled and independently reviewed before export.

### Completion: population repair and validation-engine corrections

All 1,157 source fact-draft requests have now returned; no transport failure or
missing response is a successful empty label. The first complete inventory found
853 source-compiled fact drafts and 304 drafts needing repair. The training-party
draft inventory is complete at 1,057; the 100 validation party candidates retain
their earlier source/PDF adjudications and receive the same complete-source review.
These are **intermediate processing counts**, not the final accepted export counts.

Several concrete execution issues were isolated and tested before broad repair:

- A field patch under an explicitly `null` optional parent previously raised a
  Python error. It now constructs that object only for the explicitly requested,
  schema-authorized child patch, then runs whole-target validation. It cannot
  invent an array index or edit an out-of-scope component.
- Deleting old evidence after adding replacement evidence at the same path
  incorrectly removed the new evidence too. Proof removals now apply to the old
  proposal before replacement proofs are inserted. Removing the sole proof for
  a retained scalar still fails; a patch cannot silently publish an unsupported
  value.
- Some full-object model retries returned only the newly corrected evidence.
  `recover_completion_retries.py` reuses a paid proof only for the **same field,
  same proposed value, and one non-competing proof**. Different values or competing
  source selections are not merged. Every recovered candidate is source-compiled
  and independently reviewed.
- `recover_completion_literals.py` can locate missing unique literal support for
  an already proposed value. This is **not ownership approval**. It requires one
  exact bounded source match, does not select among repeated matches, and cannot
  match seal `96` inside `40RF96` or `ABC123` inside `ABC123-456`. The separate
  semantic audit still checks field, party, goods-row and container ownership.
  This lookup is disabled when checking a returned patch: that patch must provide
  evidence for its replacements as required by its output contract.
- Printed space-grouped numbers such as `25 123,96` are recognized without
  accepting arbitrary spaces between digits. Already cited repeated units and
  explicit paired equipment codes can supply their missing scalar proof; neither
  operation adds a new source region or changes a quantity.

The source checker now also rejects contradictions with recognized printed unit
names and the reviewed equipment-code grammar. A population replay of 887 then-
compiled fact drafts found one incorrect standard-height label for explicit
`40 DRY 9'6`; it was corrected to high cube with an exact receipt. The other 886
passed the strengthened checks. Separately, that replay exposed a grammar defect
for the printed phrase `40" REEFER HIGH CUBIC`: the existing parser interpreted
the reefer type but failed to recognize the paired HIGH CUBIC height wording.
`container_semantics.py` now recognizes that paired phrase while leaving CUBIC
alone unrelated to height. The existing equipment suite passed 50 tests. A
5,000-uncached-call probe changed from 0.05186 s to 0.05278 s (about 0.18 microseconds
per call); peak RSS was approximately 35 MiB in both measurements.

Review findings are hypotheses, not commands. The small-patch response now records
an explicit corrected/rejected/held disposition for every supplied finding,
including exact evidence. Those dispositions are passed to the next independent
audit as reasoning to verify, not as approval. This is important because a reviewer
itself proposed changing a printed 9'6 high-cube container to standard height.
The deterministic source check rejects that change. Unchanged accepted components
remain pinned to their earlier applicable reviews; an unrelated correction does
not require throwing away their approvals.

Current controls passed 11/11, covering injected defects and an intact product-
condition positive control. The separate pre-existing source-adjudicated field
panel currently matches 22/25, with three cargo cases awaiting their fact repairs
(not three failed semantic comparisons). Both panels must be rerun on the final
candidates; neither is presented as a population semantic error-rate estimate.

The first population review drained under its $15 operational admission ceiling
(which includes pessimistic in-flight reservations), leaving about $13.30 charged
plus the retained uncertain-request reserve. Completion needs more than the initial
$10–15 estimate: the next bounded population reconciliation uses a $22 total ledger
ceiling, not a fresh/reset ledger. All earlier successful and failed paid attempts
remain included. No original OCR, historical dataset or split has been changed.

### Completion continuation: tested repair engine and preserved source decisions

The user replenished OpenAI credit; requests resumed without resetting the cost
ledger or discarding accepted components. The complete population remains 1,057
training and 100 validation sources. All 1,157 packet OCR hashes equal the frozen
historical OCR hashes. This is real-only relabelling, not document regeneration.

The population party reconciliation returned 402 validated repairs and four
explicit failures, for $0.54949 including those failures. Some successful responses
reject incorrect review suggestions rather than changing the label: a city-country
line is not the separate country value, and `(8LINES)` beside a telephone is not
a phone extension. Actual multiple-address ownership conflicts remain scoped
questions. The complete candidate receives another independent source review.

Additional independently reproduced checker defects were fixed:

- Binary floating-point addition made `18169.470 + 18797.260` differ from the
  correctly labelled `36966.73`. The arithmetic gate now uses exact decimal sums;
  a deliberately changed total still fails. This is not a tolerance that accepts
  approximately correct cargo quantities.
- Complete named-month dates with printed two-digit years can be normalized under
  the existing date contract. The helper uses the fixed standard 1969–2068 window
  for this syntax rather than a moving current-year window. Missing date parts
  remain unsupported, and ambiguous numerical day/month order is not guessed.
- Equipment identifiers containing printed formatting separators are normalized
  only when all original alphanumeric characters remain identical and the ISO
  check digit validates. Invalid identifiers are not corrected by changing digits.
- Incorrect quote coordinates can be recovered only when the unchanged exact
  quote has one possible occurrence. Repeated quotes without a valid source
  location remain unresolved. This is coordinate repair, not field-ownership or
  substring-boundary approval; those remain separate checks/review obligations.

The completion, V7 training-contract and equipment tests passed **90 tests in
7.82 seconds** after these changes. Tests cover invalid totals, missing date parts,
invalid ISO digits, unrelated-component edits, incomplete review dispositions,
incorrect source quotes, lost evidence, and source-coordinate ambiguity.

Three explicit historical user decisions are now pinned to current OCR and the
original adjudication report hashes, so re-extraction cannot silently reverse them:

1. `99adb051` (validation): preserve the approved container `TEMU9523456`, seal
   `CM08875356`, and 20GP interpretation of the mistyped separator. The separate
   printed 65-versus-150 package-count conflict is **not** resolved by this decision.
2. `b25a3616`: preserve the specifically approved EGYPT country interpretation of
   `SOKHNA EGYPTIAN`; this does not permit arbitrary geographic inference elsewhere.
3. `0dbc75a0`: printed `GEN` stays in OCR but is not an equipment-type, movement,
   or product-name label. The separately printed tank description remains distinct.

These decisions live in `completion/manual-adjudications/`. The source-policy
carry-forward script proves that restoring the first equipment record changes no
other target component and no OCR. All original annotations and paid responses
remain immutable audit material.

### Additional source-wide validation, not blanket pattern rejection

All 100 prior validation-party evidence packets replayed to their saved section
exactly under the current compiler. Composition now also carries their unresolved
notes into the complete-source audit, rather than losing the notes at the old/new
processing boundary. A regression test verifies this handoff. An unsupported
optional field is normally resolved as a correct omission; the question is not
automatically an instruction to invent a value or hold the whole document.

A source-wide country-boundary screen examined 3,242 selected country fragments.
Six fragments in five sources touched adjacent alphanumeric OCR text. Direct
inspection found readable country wording joined to surrounding postal/contact
text, not country substrings cut out of unrelated names:

| Source | Printed boundary | Selected country | Disposition |
|---|---|---|---|
| `1fbc935f`, consignee and notify | `Bedi SuefArab Republic of Egypt,` | `Arab Republic of Egypt` | Country follows joined locality; preserve full postal wording, no invented separator in OCR. |
| `443a7cb1`, consignee | `EXANDRIAEGYPT` | `EGYPT` | Recognizable printed country fused to locality; no country inference from locality alone. |
| `6586346f`, shipper | `DUBAI, U.A.ELICENSE No: 18525` | `U.A.E` | Country precedes the registration caption; licence details are not postal text. |
| `9ef31623`, delivery agent | `11835, EGYPTTEL +2022614860` | `EGYPT` | Country precedes telephone caption; telephone is separate. |
| `da0fffea`, shipper | `SUNGNAM-SI,KYOUNGGI-DO,KOREA463-860` | `KOREA` | Printed country precedes postcode; retain postcode in the address. |

These dispositions concern the selected country boundary, not approval of every
field in those documents. The complete component audit still applies. No blanket
word-boundary rejection was introduced: it would reject these legitimate hard-OCR
cases. The source inventory is retained under `completion/country-boundary-screen/`.

After the extended handoff/reconciliation checks, the combined completion, party,
V7 training-contract and equipment suite passed **123 tests in 7.60 seconds**.
Source-coordinate recovery added about **0.70 ms per document** on a 100-iteration
real-candidate microbenchmark; peak process RSS was 140.2 MiB. This cost is negligible
relative to provider requests and avoids paying again for uniquely recoverable
metadata. No semantic approval is inferred from that recovery.

### Final population adjudication and repair preparation

The replenished OpenAI balance is in use. No completed work or spending was reset.
The first complete population review returned substantive findings, but also false
claims (for example, reporting an already-present importer reference as missing,
or treating formatting-only ISO identifier separators as different identities).
These claims are not instructions to edit a label. A focused high-reasoning Luna
adjudication tests each disputed claim against the current target, exact source
proofs, complete OCR and the same annotation policy. The four-document probe
resolved three false-flag cases and retained genuine missing references and HS
ownership ambiguity in the fourth. The population pass is in progress.

The repair engine was also corrected at the actual failure points, rather than
retrying the same model request indefinitely:

- A complete, exactly cited equipment code can establish both canonical size and
  type. A missing paired field is filled only when both meanings are known and
  existing fields do not contradict them. A bare dimension does not qualify.
- Count-prefixed complete codes such as `1X40 HC` use the same reviewed grammar.
- All-null optional objects are removed with explicit receipts; array rows and
  supported scalar facts are not silently deleted.
- Exact printed names/references use literal evidence, not a classification or
  identifier-normalization mode merely because the model chose the wrong mode.

The next offline replay recovered **22 additional fact sections at $0 additional
API cost**. These are source-compiled candidates; their changed components still
receive independent review. The combined completion/party/schema/equipment suite
then passed **125 tests in 7.70 seconds**.

Three narrow source-inspected country repairs are staged in
`repair_country_selections.py`, with an actual-source dry-run proving equality of
all unrelated fields: select one country rather than repeated aliases in
`7f6df7bd`; retain the complete two-line `UNITED ARAB EMIRATES` in `b204c2fe`;
retain malformed `FGYPT` in the immutable postal wording but omit that separate
malformed country target in `891555b9`, following the existing contract. No OCR
is rewritten. Competing companies' postal contexts are not handled by these edits.

Review payloads now expose the existing executable normalization/equipment grammar
explicitly. This does not certify ownership: it prevents a reviewer from inventing
a conflicting formatting contract while still requiring it to verify the actual
shipment row and any contradictory printed evidence. Positive controls supplement
the existing deliberately corrupted-label controls.

Cost update: the completion ledger exceeded its initial $10–15 estimate after the
additional full-population review/reconciliation. The ongoing projection is roughly
**$25–28 total for this completion phase**, not per document and not a synthesis
price. All paid failures, retries and uncertain interrupted-request reserves remain
in the same ledger. The final report must replace this projection with measured
settled charges and separately stated uncertain liability.

An additional arithmetic falsification probe showed that a malicious/incorrect sum
proof could cite the same source occurrence twice and double its numeric value.
The sum compiler now requires distinct, occurrence-resolved, non-overlapping source
operands. A read-only sweep of **173 saved sum proofs** found no reused source
occurrence, but three proofs had underspecified occurrence coordinates where digits
also appeared inside container/customs IDs. Those coordinates are recoverable from
the already-cited row and exact measurement/package context; no total or OCR value
is changed. Tests reject both repeated operands and ambiguous repeated measurements.

A preliminary replay through the **actual V7 training task, frozen category
constraints and prompt** accepted all **1,133 then-composable documents**, with zero
canonicalization/prompt errors. The 24 remaining documents at that snapshot had
explicit incomplete fact processing, not successful empty labels. The replay took
21.93 seconds and 148.2 MiB peak RSS. This check will be repeated over the final set.

The publisher distinguishes a completed source decision from unfinished corrective
work. `repair-required.jsonl` enumerates current concrete defects/inventory conflicts;
its presence prevents `completeProcessing=true`. Accepted sibling components remain
available, but a document with an unresolved component is not exported as a sparse
whole-document training example. Unchanged high-reasoning review decisions are not
repeated merely to obtain a different answer: a new review needs changed labels,
evidence or an explicitly clarified contract; otherwise the source question is
resolved directly or retained for an actual decision.

## Acceptance replay and transport recovery

This continuation keeps original OCR, source splits and previous request receipts
immutable. A top-up was reported, but a new single Luna request still returned
`credit_balance_exhausted` for the repository `.env` key. The inherited environment
has no competing OpenAI key, organization or project override. The user was asked
to verify the funded account/project. This is a transport/billing rejection, not
an annotation or quality decision; it incurred no charge.

Available OpenRouter credit was checked read-only. An explicit GLM-5.3-Flash probe
was made, with provider routing and request costs recorded. Fireworks initially
returned an upstream shared-pool 429 after a paid response. Z.ai then returned
empty/truncated responses despite a stop finish reason; those failed parsing and
were not admitted as reviews. Fireworks subsequently completed two high-reasoning
controls, detecting the omitted postcode and wrong-party assignment, for
$0.01085865. The larger control panel must qualify that backend before any semantic
approval depends on it. Neither cheaper pricing nor a parsed JSON object is a
substitute for that qualification. The provider pricing was checked against the
[OpenRouter model page](https://openrouter.ai/z-ai/glm-5.3-flash); routing remains
explicit, not a silent provider fallback.

### Local repairs and source checks

- Strengthened fact replay repaired 24 existing proof sets without generation:
  21 documents needed the extraction-specific equipment contract, and three
  needed unambiguous numeric operand coordinates. A bare dimension/high-cube
  statement cannot inherit the general-purpose default used internally by the
  shipment sampler. Complete commercial codes still map to canonical pairs.
- The ambiguous repeated `40` equipment quotation exposed a mismatch between the
  contextual interpretation helper and the compiler. Contextual category meaning
  now uses the compiler's source-presence rule, without pretending to establish an
  exact occurrence or ownership. Tests reject absent quotations and ensure the
  helper does not manufacture an occurrence number.
- In `aa9ed483`, shipment wording `PACKED IN 120 BAGS.` had reappeared in the goods
  description. The exact source-inspected edit removes that phrase from the
  description only. The 120-bag structured total, individual placements, product
  name and diameter specification remain unchanged. The input is unchanged.
- In `b204c2fe`, the saved draft's volume-unit context quoted `Total: 20.000`, which
  does not exist. Its actual line is `Total: 21,920.000 kgs. 20.000 cu. m.`. Recovering
  that exact context and disambiguating the numeric occurrence recovers the paid
  fact draft without regenerating it or altering OCR. This proves the quoted
  measurement, not complete semantic acceptance of every other field.
- The source-reference regression panel now matches **25/25** address/cargo
  expectations. These are specific, independently established field references,
  not 25 whole-document approvals or an estimate of population error rate.
- Targeted completion/V7/constraint/equipment tests: **105 passed in 9.03 seconds**.
  A broader invocation also exposed an unrelated historical V5 output-order test
  whose fixture lacks its schema's required `packageIds`; it fails before the
  order assertion. No historical V5 target or test was changed to conceal that.

### Request admission and preservation

The previous admission code stopped the entire wave when temporary worst-case
reservations reached the cap, even if active calls would shortly release most of
that reservation. Admission now waits for those active calls to settle. It never
releases uncertain historical liabilities, never exceeds the cap to admit work,
and wakes waiting requests on a fatal provider stop. A concurrent falsification
test covers temporary reservation pressure, genuine budget exhaustion, uncertain
costs and provider shutdown. The measured local overhead is 0.677 microseconds per
reserve/settle pair versus 0.419 previously (five repetitions of 10,000 pairs):
0.258 microseconds added, negligible beside provider latency.

A failed credit-only retry no longer hides earlier paid candidate responses from
offline recovery: all same-document, same-OCR fact receipts are searched in order.
Every recovered candidate still passes the current source compiler and requires
its independent semantic review. A stale success flag is never publication proof.

`verify_completion_snapshot.py` records the actual test output, source-reference
panel, qualified current-policy control results, original source hashes, packet
identity/splits, and replay through the actual V7 training canonicalizer and
prompt. `publish_completion.py` exports independently accepted components and only
fully accepted documents; unresolved records remain explicitly inventoried. This
separates durable completed work from the remaining repairs rather than presenting
all work as either globally accepted or globally worthless.

### First durable component-accepted snapshot

Published to
[`mpci-bl-real-v7-reviewed/138b4c82…`](../artifacts/kie-training/datasets/mpci-bl-real-v7-reviewed/138b4c825a9cb9477303455a388133fc42fa6f6cbc4683fa795b837ce67b3e1f/manifest.json):

- **291 whole documents accepted:** 269 train and 22 validation. These are actual
  whole-target acceptance decisions under the stated checks, not merely parsed
  candidates. The original 1,057/100 membership is preserved, not re-randomized.
- **10,557 component decisions accepted**, including source-supported absence.
  Nonempty accepted party groups include 933 shippers, 889 consignees, 912 notify
  groups, 750 carriers, 78 forwarding agents, 351 delivery agents and two
  consolidators. Absence approvals are reported separately in the inventory;
  they are not represented as thousands of newly labelled addresses.
- **791 documents held** for scoped semantic/reconciliation issues and **75 with
  incomplete processing/review**. These are not labelled irreparable. There are
  1,843 open repair/inventory findings across 743 documents; overlapping findings
  are not 1,843 distinct samples or necessarily 1,843 confirmed label defects.
- All 1,157 IDs remain in `documents.jsonl`. Only fully accepted documents enter
  the two whole-document JSONL files. No partial labels become negative examples.
- Export replay: **66.80 seconds**, **216.42 MiB peak RSS**. The associated current
  verification replay compiled **1,134** candidates, explicitly recorded the 23
  remaining fact-processing failures, and took **44.43 seconds**, **152.34 MiB**.

This snapshot is intentionally separate from existing training datasets and has
no training config pointing at it. Subsequent work targets the remaining findings
and processing failures while preserving already accepted components.

One additional orchestration defect was corrected: a local worker exception used
to terminate `asyncio.gather` and close the provider client while unrelated paid
requests were still active. All component stages now drain their siblings, save
per-document exceptions, and then report the failed stage. A test deliberately
throws from one worker and verifies that its paid sibling finishes. This prevents
losing in-flight work merely because a different document has a bad source proof.

## Funded repair wave and acceptance falsification

Direct OpenAI still returned `credit_balance_exhausted` after the reported top-up.
The explicitly configured OpenRouter route to the **same GPT-6 Luna model** passed
all 14 then-current blind positive/mutation controls, and the population repair
continued through that funded route. No silent model/provider substitution is used.
The ledger retains all prior settled and uncertain charges; its cap is not reset.

The first resumed fact-patch wave returned 687 source/schema-valid patches and 41
failed attempts. Source/schema validity is an intermediate executable condition;
the independent component review follows. Party reconciliation then completed its
requests and the full independent review is in progress. A read-only replay of
1,152 current fact candidates passed the tightened numeric/equipment compiler.
Five remaining paid drafts have source-inspected, zero-cost recoveries prepared in
`resolve_completion_final_drafts.py`, including an explicitly held malformed weight.

### Additional checker corrections

- `20TANK CONTAINER` previously hid the tank token behind its joined length and
  fell through to the general-purpose CONTAINER noun. The grammar now recognizes
  that boundary and retains generic tank wording without inventing pressure type.
  Physical sampler envelope selection remains private and cannot enrich labels.
- A review helper exposed partial proof fragments as if they described the whole
  source. It now supplies only positive complete-code interpretations. A quoted
  `40` fragment does not imply that its actual `40 DRY 9'6` source row lacks type.
- Local normalization could alter an accepted equipment sibling during a reference
  repair. The patch boundary now restricts both value normalization and evidence
  updates to authorized components. A previously failed paid patch replays against
  its original parent successfully; explicit out-of-scope model edits still fail.
- Numeric evidence now recognizes `.24` and word counts such as `Ninety Nine`, but
  cannot arbitrarily delete decimal punctuation. `193.25` cannot become 19325, nor
  can sums exploit that conversion. Malformed `1.150.0` remains unresolved.
  Repeated-token caching is bounded and immutable: measured lookup 0.0243 microseconds
  versus 1.3398 for the earlier parser (five repetitions of 30,000 calls). Uncached
  stricter parsing was 1.8952 microseconds; correctness is not traded for speed.
- Combined targeted completion, V7, constraints and equipment tests: **111 passed
  in 7.99 seconds**, before the additional grouping-control extension below.

### Manual acceptance falsification: shared cargo

Manual source inspection deliberately selected complex already-accepted records,
rather than only reviewing failures. It found a semantic contract weakness in the
prior review: separately quantified **container portions** were being treated as
independent products solely because each portion had a count/weight. The agreed
description pilot already stated that repeated portions of the same shared product
belong in one goods item. The generic accounting-unit wording left that distinction
too loose in the full-document reviewer.

- `0701d457`: shared yarn description with two 680-carton container portions should
  be one goods item, 1,360 cartons, two placements, 51,800 kg and 110 m3 from complete
  printed portions—not two copies of the same product.
- `2aa1d48b`: six PAPER REELS container portions should be one shared goods item
  with six placements and exact totals—not six product identities created solely
  to retain portion-level weights.
- `f0f120de`: three drum portions of grape concentrate were already represented as
  one goods item with three placements. Its complete party/postal/contact fields,
  three temperature settings, seals, weights and references matched the inspected OCR.

The two incorrect cargo approvals in historical snapshot `138b4c82...` are explicitly
withdrawn in the dataset README; unchanged accepted siblings are not withdrawn.
All **52** current repeated/contained-description candidates receive the clarified
grouping question. They are not automatically merged: distinct product specifications,
lot identities and separately quantified different products must stay separate.
Four additional blind controls test both incorrect split versions and correct
merged versions of the two inspected sources. This is a targeted correction and
population replay, not an assertion that repeated model approval proves semantics.
