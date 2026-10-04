# Full direct-labeling flow on 50 new real documents

Completed 2026-10-03. This is the requested out-of-panel experiment following the
[field-policy and adjudication improvements](kie-direct-review-policy-and-adjudication-2026-10-03.md).

## Executive result

**All 50 documents completed the full bounded flow.** The run made 427 requests,
took 905.07 seconds (15 minutes 5 seconds), and cost an estimated **$0.56785**,
approximately **1.14 US cents per document** including extraction and all review,
correction and layout-assisted calls.

- Fresh extraction: **47/50 application-valid**, three native-schema-valid but
  application-invalid drafts explicitly passed to review.
- After refinement: **50/50 application-valid**.
- Agent outcomes: **37 reviewed candidates; 13 needing adjudication**.
- Final section verdicts: **236 pass, nine correction-needed, five unresolved**.
- Independent source checks found **three false-passing documents among the 37**:
  a significant cargo-placement alignment error, a description-boundary error,
  and three overlong HS codes in one document.
- One PDF-only-value regression was correctly caught by final review and remains
  held. At least one held finding is itself an incorrect reviewer suggestion.

Thus the full flow operates successfully on new sources, repairs concrete
defects, and is cheap at this scope. **37/50 is a workflow pass rate, not an
accuracy estimate or gold-label certification.** The 34 other passing documents
are not implied to be exhaustively independently certified by subtraction.
No production labels, OCR, PDFs or training configurations were modified.

## Selection and frozen execution

- Population: 1,156 real records from the previously trained V6 snapshot in the
  real-data baseline audit, with original OCR and train/validation membership.
- Excluded all 20 IDs and OCR hashes previously used in the direct-labeling pilot.
  Remaining eligible population: 1,136.
- Seed `2026100350`: 40 training and 10 validation documents, sampled without
  replacement. All 50 have distinct IDs and OCR hashes and verified source PDFs.
- OCR length: 1,332–26,659 characters. Historical labels indicate zero to ten
  containers and one to nine goods entries; these labels are not gold. The panel
  includes two labeled DG documents and one with container temperature setpoints.
  An additional source has refrigerated goods handling without a printed
  container identifier, so the old-label temperature feature is not an exhaustive
  count of thermal text.
- This is a seeded new-document panel, not a population-weighted metric sample
  or a claim that none of these sources were ever inspected in older repair work.

Every document received a **fresh OCR-only extraction**; previous labels were
saved for inspection but never supplied to agents. The extraction then underwent
the maintained five section reviews, at most one correction wave, dependency
checks, and re-review of affected sections. Original PDFs were provided only on
request for layout. Nineteen documents received layout assistance.

All stages used `gpt-6-luna`, reasoning `high`, native strict JSON Schema through
PydanticAI, a 16,384-token output/reasoning limit, and at most eight globally
concurrent requests. There was no xhigh comparison, prompt adjustment during the
run, automatic retry, hidden repair round or selective substitution of a better
previous answer. Code, schemas, prompts, registry and selection were hash-frozen
before the first paid request.

Application-invalid responses were not silently accepted: each was saved as an
invalid draft, checked against the actual native schema, and sent through the
existing raw-draft review entry point. Their rejected extraction calls remain in
the usage totals. All three were application-valid after refinement.

## Results by stage and section

Twenty-one documents passed all initial reviews. Twenty-seven documents received
audited corrections and changed labels; two required source adjudication without
an applied correction. The resulting 37/13 final split includes new defects
detected during re-review and unresolved source ambiguity.

| Section | Initial pass / correct / unresolved | Final pass / correct / unresolved |
|---|---:|---:|
| Parties | 40 / 7 / 3 | 44 / 4 / 2 |
| Route and transport | 46 / 4 / 0 | 49 / 1 / 0 |
| Metadata, freight and references | 39 / 11 / 0 | 50 / 0 / 0 |
| Equipment | 47 / 1 / 2 | 48 / 0 / 2 |
| Cargo, packages and placements | 35 / 14 / 1 | 45 / 4 / 1 |
| **Total section verdicts** | **207 / 37 / 6** | **236 / 9 / 5** |

Initial reviewers produced 52 findings; final reviews retain 14. Auditors made
47 explicit finding decisions: **44 accept, one reject, two revise**. A reduction
in findings measures the workflow, not independent correctness: final review
missed the three defects documented below and generated a false-positive URL
finding in another case.

## Concrete successes observed in sources and outputs

- **Joined container ID/tare recovery:** source 13 prints
  `CAXU93098433830` and `TGHU89736093870`. The PDF table separates the IDs from
  tare weights 3830 and 3870. Equipment and placement identifiers were corrected
  together to `CAXU9309843` and `TGHU8973609`, repairing the application's
  identifier failures without inventing characters. The separate allocation
  error in this document remains; successful ID repair does not certify its cargo.
- **Duplicate-reference recovery:** source 07's fresh response repeated
  `PO 21126099`, violating the application uniqueness constraint. Review removed
  the duplicate and restored structural validity. A missing package category was
  subsequently flagged, so the document remains held for that further finding.
- **Description/package separation:** source 28 put `5 PALLETS STC 54 BOXES`
  in description. Correction removed the non-product description and retained
  the inner 54-box package information, repairing the application rejection.
  Conflicting consignee headings remain a distinct unresolved issue. This does
  not independently settle the product-name/marks ownership in that source.
- **Shared cargo:** source 05 keeps 209 cartons in one goods item, allocated
  56/46/53/54 across four containers. Source 14 keeps 8,960 bags allocated across
  nine containers. These inspected quantity/placement controls remained correct.
- **Refrigeration:** source 47 yields one shared juice-concentrate item, 240 drums,
  80 per container, and −18°C on all three reefer containers. Historical labels
  had three goods entries for the repeated product. The source supports the shared
  representation under the current policy.
- **DG and product codes:** sources 03 and 44 retain their printed UN/class
  declarations; source 44 also retains packing group III, both product codes in
  description, and excludes the separately labeled fax numbers from phones.
- **Units and multi-HS goods:** source 21 preserves 25.344/24.750 metric tonnes;
  source 06 preserves 45,000 metric tonnes. Source 15 remains one shared goods
  item with four HS codes instead of splitting solely because codes differ.
- **Addresses:** source 31 retains both `10TH OF RAMADAN CITY` and the distinct
  `P.O.BOX NO:10TH OF RAMADAN` text. It does not delete a legitimate postal
  component merely because the locality wording appears twice.
- **Adjudication rather than rubber-stamping:** in source 32, the auditor revised
  two proposals: remove wrongly owned `045W` and `LEADERS`, but do not invent an
  equipment role for them. In source 48, it rejected removal of a legitimate
  carrier homepage independently printed outside a disallowed deep link.

## Independent audit: confirmed remaining defects

The offline screen covers all 50 final outputs for normalized textual absence
of container IDs, seals, HS codes, phones, emails, B/L IDs, vessel/IMO/voyage
values; it also screens postal numeric retention against OCR and obvious
nonpostal captions. These are review screens, not proof of ownership or
completeness. Targeted source controls inspect 17 facts/relationships across 12
documents; 14 pass after refinement versus 13 before. All 13 initially passing
controls remain passing, one identifier control is repaired, and three fail.

OCR, changed values, agent explanations and selected PDF layouts were inspected
directly. This is not an independent field-by-field gold annotation of all 50.

### 1. False pass: cookware product/container alignment (`13-15b8d670`)

The identifier correction succeeded, but **six of nine container-to-product
assignments remain shifted**. Both PDF pages establish quantity lines followed
by their associated product descriptions; page 1's last 438-box row continues
with the first description on page 2. The model instead aligns several products
with the following quantity/container row and assigns the final product back
to the first page's otherwise unmatched container.

Examples:

| Container | Source-supported product | Final model product |
|---|---|---|
| TGHU8973609 | 16-piece granite cookware | 34–36 diameter steel pots |
| TRKU4445041 | 12-piece steel cookware | 16-piece granite cookware |
| TRKU4455883 | 21-piece steel cookware | 12-piece steel cookware |
| TRKU4458711 | 16-piece granite cookware | 21-piece steel cookware |
| TRKU4469953 | 41 diameter steel pots | 16-piece granite cookware |
| WFHU5147586 | 34–36 diameter steel pots | 41 diameter steel pots |

The total is still **4,540 boxes**. Correct grand totals and valid foreign keys
therefore conceal incorrect product grouping, per-product totals, masses and
placements. This is the most substantive independently confirmed failure.

Evidence: [OCR](../artifacts/kie-labeling/direct-full-20261003-luna-high50/inputs/13-15b8d670/ocr.txt),
[page 1 layout](../artifacts/kie-labeling/direct-full-20261003-luna-high50/manual-layout/13-page-1.png),
[page 2 layout](../artifacts/kie-labeling/direct-full-20261003-luna-high50/manual-layout/13-page-2.png),
[comparison](../artifacts/kie-labeling/direct-full-20261003-luna-high50/comparisons/13-15b8d670.md).

### 2. False pass: description contains a shipment total (`14-b956bed6`)

The final description is `PVC RESIN H-73 QUANTITY : 224 MT`. Under the current
field policy it should retain `PVC RESIN H-73`, excluding the shipment-total
phrase. This is a boundary-policy miss, not a fabricated product or broken
container allocation. Both initial and final cargo verdicts passed it.

Evidence: [comparison and OCR](../artifacts/kie-labeling/direct-full-20261003-luna-high50/comparisons/14-b956bed6.md).

### 3. False pass: three padded HS codes (`40-4799921c`)

| Printed OCR | Required separator removal | Model output |
|---|---|---|
| `3907.99.80.00.00` | `390799800000` | `39079980000000` |
| `3815.90.90.00.00` | `381590900000` | `38159090000000` |
| `3909.50.90.00.00` | `390950900000` | `39095090000000` |

The model added two zeros to each code. The codes remain schema-valid because
their lengths fall within the schema's supported range. Both cargo reviews
missed the error; the normalized source-presence screen caught all three. This
is a clear case for a narrow deterministic digit-fidelity gate, not additional
linguistic adjudication.

Evidence: [comparison and OCR](../artifacts/kie-labeling/direct-full-20261003-luna-high50/comparisons/40-4799921c.md).

### 4. Correction regression caught and held: PDF-only transport (`22-0cd26d76`)

The initial extraction correctly omitted `EF OLIVIA` / `ONVC3S1MA`: the values
appear in PDF but not OCR. Layout-assisted review requested them; correction
inserted them. Final review then correctly rejected them as PDF-only. The held
draft still contains these values because this experiment has one correction
wave, not an autonomous loop. It is not an accepted training record.

Evidence: [OCR and full comparison](../artifacts/kie-labeling/direct-full-20261003-luna-high50/comparisons/22-0cd26d76.md),
[PDF layout](../artifacts/kie-labeling/direct-full-20261003-luna-high50/manual-layout/22-page-1.png).

### 5. Reviewer false positive: deep website link (`04-836b99c1`)

The final party reviewer asks to add
`www.arkasline.com/TRANSPORT/MERCHANT/FINANCING REQUIREMENTS`. The field explicitly
excludes links to specific legal/help/clause material. This hold should not be
counted as proof that the extraction's absent website is wrong. It illustrates
why “13 held” does not mean 13 equally certain label defects.

## Complete inventory of agent-held documents

These are final agent findings, not all independently proven errors. The original
findings, explanations and proposed corrections remain in `analysis.json` and the
per-document comparisons.

| Document | Final outstanding finding(s) |
|---|---|
| 02 `44a6df44` | HMM Netherlands company/address has no clearly established supported party role. |
| 04 `836b99c1` | Carrier deep-link request; independently identified reviewer false positive above. |
| 07 `92da45e4` | Missing generic package category despite printed PACKAGE(S). |
| 17 `9c68fe94` | Seal/NOSEAL ownership ambiguity; street number/name split incorrectly into phone/contact fields. |
| 22 `0cd26d76` | PDF-only vessel/voyage inserted by correction; final review catches it. |
| 24 `71569af6` | Requested preservation of a space at a wrapped street-number component; formatting-level issue. |
| 28 `e9275395` | Two printed consignee headings conflict with the singular consignee target. |
| 31 `8b9d3e8f` | Bonded-warehouse transit instruction absent from handling instructions. |
| 32 `6700b15c` | `***045W` notation lacks a clear equipment/seal role. |
| 38 `a51616d3` | `BAGS` assigned to marks rather than goods/package wording. |
| 43 `150e9dad` | Product-label invoice/order/PO and origin text missing from marks; proposed ownership requires checking rather than accepting solely from the review. |
| 48 `7a02e126` | Missing comma between DOVER GARDENS and AUSTRALIA. |
| 49 `da651f2f` | Printed mass has unit `KOS`, outside current allowed unit normalization. |

The three independently confirmed false passes are **additional** to these 13.
No post-hoc manual edit was applied to make the pilot appear cleaner; model outputs
and the independent findings remain separate.

## Cost, execution and validation

| Stage | Requests | Estimated USD |
|---|---:|---:|
| Fresh extraction | 50 | 0.147547 |
| Initial reviews, including PDF follow-ups | 271 | 0.273542 |
| Audited corrections, including assistance | 36 | 0.053611 |
| Final reviews, including assistance | 70 | 0.093149 |
| **Total** | **427** | **0.567850** |

Usage: 2,894,739 input tokens (990,151 cache-read; 1,903,307 cache-write) and
639,814 output tokens, including 534,886 reasoning tokens. Cache counts are
subsets of input; reasoning is a subset of output, not an extra charge.

The OpenAI Docs check verified Luna's supported high-reasoning setting and the
[official model pricing](https://developers.openai.com/api/docs/models/gpt-6-luna)
used for the estimate: USD/M input 0.10, cache read 0.01, cache write 0.125,
output 0.50. All attempts have recorded usage; no unknown-billing request was
treated as zero. This is estimated usage cost, not invoice reconciliation.

Runtime: 905.07 seconds, observed peak eight concurrent requests, median
individual request 10.02 seconds. A spot measurement during the run showed about
274 MiB process RSS; no peak-RSS claim is made. This panel differs from the
previous challenge panel, so pass rates/costs are not a controlled before/after
accuracy or speed benchmark.

Validation performed:

- **41 targeted direct-flow/V7 tests passed** before launch, in 13.02 seconds.
- **427/427 raw responses satisfy the actual strict native wire schemas**,
  including the three application-invalid extraction drafts.
- Application validation passes for **all 50 final targets**.
- The full OCR appears unchanged as literal text in every captured request;
  all requests use high reasoning and native strict JSON Schema.
- Frozen source, input, previous-label, PDF, registry and implementation hashes
  match after completion. No overlap with the old pilot's document IDs/OCR hashes.
- Offline source screens, before/after comparisons, 17 targeted controls, and
  direct layout checks distinguish schema validity from semantic correctness.

## Deliverables and next diagnostic target

Artifact root:
[direct-full-20261003-luna-high50](../artifacts/kie-labeling/direct-full-20261003-luna-high50/).

- [Selection/provenance](../artifacts/kie-labeling/direct-full-20261003-luna-high50/selection.json)
- [Per-document/stage analysis](../artifacts/kie-labeling/direct-full-20261003-luna-high50/analysis.json)
- [Independent findings](../artifacts/kie-labeling/direct-full-20261003-luna-high50/audit-findings.json)
- [Source-based diagnostic controls](../artifacts/kie-labeling/direct-full-20261003-luna-high50/source-checks.json)
- [OCR, fresh/final labels and review comparisons](../artifacts/kie-labeling/direct-full-20261003-luna-high50/comparisons/)
- Self-contained `run.py`, `analyze.py`, `source_checks.py`; frozen implementation
  snapshot and all request/response/usage records. Analysis scripts make no API calls.

The highest-value next improvement is targeted, not another broad blind rerun:

1. **Digit fidelity:** validate normalized HS/identifier values against the OCR
   before accepting them. This directly catches source 40's added zeros.
2. **OCR/PDF boundary:** apply that principle to added literal transport values,
   so the source-22 regression is blocked before committing the correction.
3. **Cargo row ownership:** review quantity-before-description and cross-page
   continuation alignment explicitly, using all relevant pages where the OCR
   separates columns. Source 13 is now a concrete falsification case: arithmetic
   totals and valid references alone cannot establish the correct product graph.
4. Treat URL/spacing disagreements as lower priority than those substantive
   errors; do not spend an unrestricted repair loop on every stylistic objection.

These are recommendations from the experiment, not claims that new guards or
label repairs were implemented in this turn. Production flow and prompts stayed
fixed throughout the run so the 50-document result remains reproducible.

## Implementation rerun R2

Completed 2026-10-03, after the user approved implementing the analysis recommendations.
The original experiment above is unchanged. R2 artifacts are in
[direct-full-20261003-luna-high50-r2](../artifacts/kie-labeling/direct-full-20261003-luna-high50-r2/).

**Conclusion: useful individual fixes, but not an overall improvement sufficient
for scaling.** The same number of documents received an agent pass; two refinement
operations failed application validation; a difficult cargo graph still falsely
passed. The new review stage increased cost and latency and sometimes reviewed
fields outside its assigned purpose. Do not interpret this as a bulk-gold approval.

### What was implemented

- V7 field definitions distinguish product wording from shipment totals,
  administrative references, destinations and column placement; clarify package
  markings and shipment-specific handling; rejoin wrapped postal identifiers
  without inserting a space into the identifier.
- Reviewer/corrector instructions require a decisive source fact and field rule,
  distinguish correct omission from actual unresolved target facts, and keep PDF
  assistance limited to ownership/layout with OCR-grounded values.
- A separate cargo-association reviewer checks local product/container portions,
  package levels and cross-page continuations. It shares requested PDF pages with
  equipment review. Its verdict cannot be silently overridden by a general pass.
- Narrow deterministic gates flag absent HS digit sequences and vessel/voyage/IMO
  text under permitted separator normalization. They never rewrite a label or
  certify ownership/completeness. A model pass cannot override a gate failure.
- At most two correction waves: final-review findings can now be adjudicated;
  unchanged candidate/finding pairs are not repeatedly submitted. Each wave keeps
  its target, findings and decisions. Mixed actionable/ambiguous sections can
  preserve supported fixes while retaining only undecidable facts as holds.
- Updated CLI help and maintained flow documentation. No target-schema shape,
  production dataset, source OCR/PDF, training config, or active training changed.

Implementation lives in `labeling_agents/direct.py`, `direct_models.py`, the new
`direct_grounding.py`, the direct prompts, and the V7 field descriptions. These
remain experimental labeling entry points, not an automatic publication service.

### Matched experiment and accounting

All 50 document IDs, OCR hashes and PDF hashes match the frozen first run. Each
received a fresh extraction; previous labels and diagnostic answers were never
given to agents. Luna/high, strict native JSON Schema, full plain-text OCR and
global concurrency eight were retained. There were no automatic provider retries
or reasoning/model escalations. Code/schema/prompts were frozen for the entire
batch; the subsequent product-code replay below is explicitly separate.

| Measure | Baseline | R2 |
|---|---:|---:|
| Application-valid fresh extractions | 47/50 | 50/50 |
| Application-valid final targets | 50/50 | 48/50 |
| Agent-passing candidates | 37 | 37 |
| Completed, held candidates | 13 | 11 |
| Failed refinement operations | 0 | 2 |
| Requests, including rejected responses | 427 | 558 |
| Estimated cost at recorded run rates | $0.567850 | $0.788749 |
| Wall time | 905.07 s | 1,261.79 s |

R2 cost about **1.58 US cents/document**, 38.9% above baseline; wall time rose
39.4%. Peak process RSS was **369.94 MiB**; the old run did not record a comparable
RSS measurement. Request latency and provider caching differ between runs, so this
is an observed end-to-end comparison, not a controlled local CPU benchmark.

R2 usage: 4,067,205 input tokens, including 1,120,547 cache-read and 2,944,984
cache-write tokens; 818,506 output tokens, including 683,913 reasoning tokens.
Cache and reasoning numbers are subsets, not extra billable token totals.
All 558 requests have usage receipts; none has unknown billing. All returned
responses conformed to their actual wire JSON Schema. Application validators
nevertheless rejected two responses, as described below. Cost is an estimate using
the captured experiment's configured rates, not a reconciled account statement.

The added association reviewer made **70 requests / $0.134979**, including layout
assistance and rechecks. It did not earn an overall quality/efficiency claim in this
test. The two-wave flow made 62 correction requests, versus 36 previously.
Ten documents used a nonempty second correction wave. Some second waves resolved
findings; others repeated interpretation disagreements. More calls are not evidence
of better labels.

Transitions: 28 previous passes still passed; nine previously held documents now
passed; eight previous passes became held and one failed. Three previously held
documents remained held and one failed. These are workflow outcomes, not accuracy.

### Independent checks: what improved and what did not

The same **17 source-adjudicated controls** scored 14/17 on the baseline's final
targets and **15/17 on R2 final targets**. R2's fresh extractions scored 16/17:
review introduced one regression in this control set. The expanded set has 25
checks with available final targets: **20 passed and five failed**, concentrated
in three documents. These limited controls do not imply field-level accuracy or
certify whole documents. Failed refinement cases remain outside final-target checks.

Confirmed improvements/preserved behavior:

- `14-b956bed6`: description is `PVC RESIN H-73`, without the shipment quantity
  `224 MT`; all nine container allocations and 8,960 bags remain intact.
- `40-4799921c`: all three HS values preserve the printed 12 digits; no padded zeros.
- `22-0cd26d76`: a PDF-assisted reviewer again proposed PDF-only vessel/voyage
  values. The auditor **rejected** those suggestions; final transport values remain
  absent. The offline gate independently rejects the old defective values too.
- `24-71569af6`: the wrapped postal identifier is correctly joined as
  `8-2-293/A/A1`; no unnecessary space is inserted.
- `28-e9275395`: `RATHIPON BROWN 6RL` is retained in description.
- `31-8b9d3e8f`: bonded-warehouse handling is retained, as are both the city and the
  distinct PO-box wording.
- `48-7a02e126`: destination wording is absent from product description.
- Matched DG declarations, fax exclusion, printed tonne units, package capacities,
  multi-HS goods, shared cargo and all three −18°C setpoints remained correct on
  their checked examples.

Confirmed semantic failures despite agent passes:

1. **`13-15b8d670`, cargo row ownership.** Independently inspecting both PDF pages
   confirms quantity-before-description blocks and a page-spanning final block.
   Correct 16-PCS granite totals are **1,716 boxes / 30,354 kg**; R2 emits
   **2,152 / 32,514**. The 12-PCS product belongs to TRKU4445041 with 761 boxes,
   not TRKU4455883 with 357. Both general and focused reviewers misread associations,
   including after the focused reviewer requested page 1 and received both pages.
   Correction ultimately produced six groups instead of the five supported groups,
   an undescribed TGHU8973609 portion, and an unjustified unknown-count extra
   WFHU5147586 placement. Matching grand totals did not establish the correct graph.
   The focused reviewer therefore **failed its principal falsification case**.
2. **`17-9c68fe94`, misleading OCR heading.** The OCR places Damietta below
   `PORT OF LOADING`; the PDF establishes port of discharge. Route review passed
   loading without requesting that layout. The phone/street boundary repair in this
   document succeeded, but that does not validate its route. This needs a deliberate
   policy/context path for incorrect OCR headings, not more confidence in a text-only
   pass. Values are present in OCR; field ownership is the problem.
3. **`44-d9be84a2`, product-code deletion.** Fresh extraction correctly retained
   codes `0402620044` and `0402620045`. The reviewer/corrector removed them as
   references and final review passed. The revised description text had used an
   over-broad reference exclusion. This regression was corrected and replayed
   separately below; it remains a failure in the frozen R2 batch statistics.

The simple whole-text absence screen flags **zero** remaining literal-absence
candidates in R2's final targets. The defects above demonstrate why that result is
not a semantic quality certificate.

### Held and failed cases

| Source | Outcome and diagnosis |
|---|---|
| `07-92da45e4` | Held because the focused association reviewer requested ACID as a cargo mark, outside its intended scope; the general cargo reviewer correctly called it administrative. |
| `16-1d51aca4` | Conflicting voyage text across OCR copies: `0MRFKE1MA` versus `OMRFKE1WA`; a real source adjudication candidate. |
| `20-2865b21b` | Correction rejected: it emitted placement ID `OCU7146195`, which fails the application ISO-identifier validator. Follow-up tracing also establishes that this value is absent from OCR: the earlier failure was PDF-to-OCR value leakage, not merely an incomplete identifier. No invalid final target was exported. |
| `23-0b03d5a2` | Reviewer/auditor disagreement over deriving a named payment place from `DESTINATION` and the discharge port. Relative payment terms need a clearer representation rule. |
| `25-bbf194b6` | Cargo correction introduced a placement without an emitted container. The dependent-group validator rejected it. Follow-up tracing confirms the proposed ID is PDF-only; adding equipment would be the wrong repair. The existing target correctly omits that identifier. |
| `26-78866b48` | Repeated review disagreements involving marks and PDF-only portion counts, weight and volume. Follow-up tracing confirms that 916/929, 21446.130 and 135.332 are absent from OCR; their removal was correct, and their proposed restoration was not. |
| `30-9ea52ed1` | Disagreement over canonicalizing printed `40' HIGHCUBE` versus retaining typeDescription; clarify code/wording equivalence consistently. |
| `32-6700b15c` | `***045W` continuation ownership remains disputed. Column position alone is inadequate; matching continuation markers need review. |
| `38-a51616d3` | Reviewer returned `unresolved` with a `missing` finding and no remedy because OCR lacks the required container ID. This is native-schema-valid but violates the application's verdict/finding coherence rule. Completed sibling receipts are preserved; no final target was exported. |
| `41-ff7f0853` | Ambiguous numeric date is held. Separate inspection also finds unsupported `prepaid` inferred from an empty FREIGHT ADVANCE caption; the hold reason is not an exhaustive defect inventory. |
| `45-2809b1ea` | Disagreement over the printed CargoX platform identifier's reference-field scope. |
| `49-da651f2f` | Printed `KOS` mass unit versus kilogram normalization remains held under the current unit contract. |
| `50-e815e426` | Explicit aggregate gross/net labels and per-container labels are inverted relative to each other. Requires an explicit source-total precedence rule or adjudication. |

The two application failures are not failures of native constrained JSON syntax.
ISO validation and cross-field verdict invariants are Python checks not fully
expressed by that JSON Schema. Distinguishing these layers remains necessary.

### Targeted product-code regression correction

After freezing and analyzing R2, the general description field was narrowed to
exclude **shipment/administrative references**, and explicitly include
**model/product/article codes**. This uses no document-specific numbers or strings.

A separate full refinement replay started from R2's **defective final** source-44
target, rather than feeding a desired correction. The reviewer detected both
missing codes, the auditor restored their associations with carton capacities,
and re-review passed. All non-description target fields are identical by
JSON-value comparison. DG, quantities, HS code, marks, parties and container placement
were unchanged. Ten requests cost **$0.012118**. Total new experimental spend is
therefore **$0.800867**, excluding the previously completed baseline.

The replay, exact schema, source-code snapshot, provenance and all wire/usage
records are in
[product-code-regression-replay](../artifacts/kie-labeling/direct-full-20261003-luna-high50-r2/product-code-regression-replay/).
It does not overwrite R2 or retroactively change its pass count or controls.

### Local validation and release decision

- **49 targeted tests pass**; final invocation took 13.90 seconds. Tests cover
  native structured requests, bounded waves, unchanged-finding suppression,
  mixed ambiguous/actionable corrections, dependency preservation, literal gates
  overriding false passes, and shared equipment/cargo layout context.
- Ruff and `git diff --check` pass. Maintained CLI schema export succeeds without
  a paid request; V7 training prompt/schema compatibility remains tested.
- On the old 50 final targets, literal gates identify exactly five known defective
  values: three padded HS codes and two PDF-only transport values. All **56/56**
  deliberate HS-padding counterexamples are flagged.
- Gate scan: 0.0362 seconds cold with allocation tracing, 166,137 peak traced bytes;
  median 0.00336 seconds per 50 documents over 20 warm repetitions. This establishes
  negligible local overhead, not semantic completeness.
- No source texts, PDFs, historical targets, datasets or training configurations
  were changed. Results remain isolated pilot artifacts.

**Do not scale this combined R2 flow as a proven gold-label producer.** The source
fidelity gates and the targeted product-code correction are supported by concrete
checks. The added association reviewer and additional adjudication wave are
implemented and tested as mechanics, but the full live quality/efficiency gate was
not met. Their mere existence is not a successful resolution of cargo ownership.

The next bounded diagnostic should replace, rather than add to, broad cargo review:
use a small source-first local-association representation and compare it with the
candidate graph, test the known shifted-row case and sound controls, restrict its
output contract to relationships, and omit irrelevant cargo-schema/package-registry
text from that focused context. Separately, coordinate new equipment references
with cargo corrections and represent unsupported/absent identifiers without losing
independent completed reviews. Those changes require validation; they are not
claimed as implemented by this experiment.

## Follow-up diagnosis: cargo semantics, association graphs and reviewer failures

Read-only analysis requested after R2; no new paid requests, label edits, prompt
changes or production changes. This section uses the recorded public verdicts and
decision explanations, not hidden model reasoning. Both cookware PDF pages were
visually rechecked, and candidate edges were compared locally against the
source-adjudicated mapping. Findings below distinguish observed mechanisms from
unproven design proposals.

### Package hierarchy is already working on several informative controls

The actual instructions already distinguish inner packages, product capacities,
complete totals, local portions and unknown allocations. The following observed
successes argue against diagnosing a general inability to understand packaging:

| Source | Source structure | Correct retained behavior |
|---|---|---|
| `02-44a6df44`, `27-0ef73471` | 60 pallets / 60 bags; one product in three containers | 60 bags total; three memberships, no invented 20-bag splits. These are two records, not independent semantic patterns. |
| `28-e9275395` | 5 pallets containing 54 boxes | 54 boxes, placed in the sole container; not 59 packages. |
| `31-8b9d3e8f` | 5 drums packed in 2 pallets | 5 drums and the single supported placement. |
| `35-b5b7d7a7` | 72 pallets, no declared inner shipment-package count | Pallets remain the package level. |
| `39-7ebdff63` | Same goods repeated in three containers, 21 pallets each | One goods item, 63 pallets total, three 21-pallet placements. |
| `43-150e9dad` | Two panel products share a 17-pallet / 112-carton container; 13 pallets belong to one product and 4 to the other | Both memberships retained; no proportional carton allocation; other known 119/32-carton portions retained; incomplete product totals stay absent. |

The mixed-panel case is especially important: 13/4 pallets cannot allocate 112
cartons without a supported conversion. A graph must retain that shared carton
fact at its shared scope, not distribute it among product nodes.

Useful concise guidance would organize the existing rules by **product identity,
quantity role, package level and scope**. A cookware specification such as 16 PCS
describes the set; 408 BOXES counts shipment packages; 9 containers counts
equipment. A declared inner shipment-package level takes precedence over outer
pallets, but pallets are legitimate when no inner level is established. Inner
retail capacity is not automatically a shipment-package count. This is the
project's extraction policy, not a claim that every customs system selects the
same level.

### The cookware failure is a shifted association, not a bag/pallet confusion

The document's first page establishes count/mass followed by product wording.
Its final 438-box/7,540-kg block continues into the top of page 2. The later bold
count/mass lines likewise precede their product wording. Page 2's separate
container column makes a different visual grouping tempting.

The general reviewer used the preceding product instead. Its proposed 12-PCS
assignment was 357 boxes in TRKU4455883; the source-adjudicated assignment is
761 in TRKU4445041. The focused reviewer initially identified TRKU4445041 for
12-PCS but called its quantity unknown. The auditor explicitly rejected that
membership and accepted the incorrect 357/TRKU4455883 interpretation. The final
two reviewers then endorsed the shifted interpretation.

This is not evidence that reviewers lacked the pages: the focused reviewer
requested page 1 after page 2 was supplied and subsequently had both. Nor did
all calls initially agree: the auditor adjudicated conflicting proposals the
wrong way.

Local comparison: **3 of 9 quantified product/container associations match**;
six have the wrong or missing product owner. All nine container quantities can
still sum to **4,540**, the correct shipment total. There is an additional
unknown-count 34–36-Ø placement in the final output. Matching aggregate arithmetic
therefore cannot detect this error. Some descriptions are individually correct;
the product/portion link is not.

### Why the focused reviewer was not actually narrow

`direct.py` passes `_section_context(target, "cargo")` to it. This includes the
general cargo checklist, the full goods schema, candidate descriptions, marks,
handling, and the package vocabulary. In the source-13 request inspected, field
definitions alone occupied **25,916 characters**, versus **6,376 characters** of
OCR; the complete context block was 29,878 characters. Those are character counts,
not token estimates. The recorded request had 13,073 input tokens including its
other content.

The output is the unrestricted `SectionReview`, whose field is free text. A prompt
saying "other fields have a separate reviewer" does not prevent it issuing edits
for those fields. Across all saved focused-review responses there were 18 findings;
**five clearly concerned marks or handling rather than grouping/package levels/
placements**, including a repeated ACID-marking dispute. Four more concerned
weight/volume totals. This is observable scope drift, not merely a speculation
about prompt length.

In source 07, the general reviewer correctly excluded ACID as administrative.
The focused reviewer requested it as a mark because of column position. The
auditor added it, later removed it, and the focused reviewer requested it again.
An extra correction wave did not resolve the contradictory interpretations.

### PDF-only values explain more failures than the initial summary indicated

Exact and alphanumeric-normalized searches over the supplied OCR found:

- `20-2865b21b`: `OCU7146195` absent. The focused reviewer supplied it in an
  `ocrExcerpt` after viewing layout; the corrector emitted it and application
  validation rejected its format. Proper behavior is to keep supported goods
  facts and omit the unsupported ID/placement, not repair the ID from the PDF.
- `25-bbf194b6`: `FYCU1056536` absent. General cargo review explicitly recognized
  the third identifier as PDF-only. The focused reviewer nevertheless claimed an
  OCR excerpt containing it, and the auditor accepted the addition. The missing
  equipment validation error is downstream of that false evidence claim. A joint
  equipment edit would propagate the error rather than solve it.
- `26-78866b48`: local counts 916/929, gross total 21446.130 and volume 135.332
  absent. OCR supports 1,845 cartons and both container IDs, so memberships with
  unknown local counts are correct. Review added the PDF values, re-review
  correctly removed them, then focused review requested them again.

The narrow gate currently covers HS and transport values, not these container IDs
or cargo measures. `ocrExcerpt` is an optional explanation string, not verified
source provenance. A fabricated excerpt can therefore become an accepted premise.
Extending exact checks for new IDs is straightforward. Numeric checks must retain
scope/unit and distinguish printed amounts from permitted exact derivations;
whole-document number presence alone cannot certify ownership.

### Other failures need different interventions

- **Route 17:** the recorded reviewer cites the OCR's loading heading; the PDF
  puts Damietta under discharge. The route call never requests layout. An agent's
  confidence cannot reliably trigger access to evidence it does not know is
  missing. Consider providing the route-bearing page upfront for the pilot, and
  record when ownership is recoverable only from layout. This also distinguishes
  teacher annotation from what an OCR-only student can learn.
- **Product codes 44:** confirmed over-broad reference exclusion; already narrowed
  and separately replayed successfully. Not a package or graph failure.
- **Source 38:** facts about an unidentified container do not fit an ID-required
  output object. This is different from two competing source values. The current
  review status model rejects an unresolved/missing combination; a legitimate
  omission or explicit non-target diagnostic should not crash refinement.
- **Payment-place 23, equipment 30, CargoX 45:** target-policy boundary disputes.
  Resolve whether relative terms may become named localities, which equivalent
  equipment wording supports category pairs, and which platform references are
  in scope. Repeated general reviews cannot settle an unstated policy.
- **Voyage 16, numeric date 41, unit 49, conflicting masses 50:** source/policy
  uncertainty is real under the current contract. Preserve known components and
  hold the disputed component rather than guess. Source 41 additionally has an
  unsupported prepaid value; an existing hold is not an exhaustive defect list.
- **Continuation 32:** a matching `***` marker near the vessel makes the remote
  `045W` value an ownership question, not automatically a container because of
  where it appears. Source-wide continuation semantics matter.

### Proposed graph: internal working representation, not a new training schema

Use a compact **cargo-portion table** as the readable representation of a typed
graph. Each source cargo occurrence records product/group identity, supported
container membership, package quantity/type/level, and optionally its mass. A
quantity records whether it belongs to one product portion, a shared cargo group,
or the whole consignment. Explicit outer packaging is retained internally to avoid
mixing it with target inner counts. Repeated printed copies do not add occurrences.

For example, the corrected 12-PCS cookware portion is simply:

```json
{
  "product": "12 PCS ELANDALOS BRAND S.STEEL COOKWARE SET",
  "container": "TRKU4445041",
  "packages": {"quantity": 761, "type": "BOX"},
  "grossWeight": {"value": 10900, "unit": "kilogram"}
}
```

Record all local portions **before** grouping repeated products and summing
complete portions. Then emit the existing description, package facts and
splitGoodsPlacement fields. A shared quantity across multiple products remains
a shared fact; it does not become a quantity on every product edge. No equal or
proportional splits, no cross-level arithmetic, and no Cartesian-product inference
from a list of goods and a list of containers. Source quantities and relationships
must be established independently of the candidate under review.

This does not require a graph database, graph neural network, scalar-level span
receipts, or another dataset-wide schema migration. It also does not guarantee
correct reading: a model can emit the same wrong edges in a tidy graph. Where two
descriptions differ, semantic identity matching remains an explicit task rather
than a presumed deterministic string match.

### Smallest useful next experiment

1. Replace the broad focused-review contract, not add another reviewer. Give it
   only grouping/quantity/membership definitions and a narrow output model; avoid
   the full package registry and marks/handling fields in that call.
2. Compare source-first local portions with candidate associations. For difficult
   layout, contrast the genuinely plausible alternatives (preceding/following/
   cross-page/unknown) rather than accepting a general "matches the table" verdict.
   Do not hard-code a universal before/after direction.
3. Reject proposed PDF-only IDs before the auditor acts. Separately check typed
   printed/derived numeric quantities; arithmetic correctness is necessary but
   cannot prove their scope or product ownership.
4. Test on roughly 10–12 selected hard and sound controls before another 50-run:
   cookware; mixed panel products; bags/pallets; drums/pallets; shared goods;
   unknown per-container counts; PDF-only IDs/measures; route misheading; and an
   actual source conflict. Preserve all applicable good outputs.
5. Include deliberately corrupted candidates: swap product owners while preserving
   totals; substitute pallets for cartons; repeat totals as allocations; invent
   equal splits; add a PDF-only ID. Grade exact association correctness, false
   passes, false holds, unchanged-good-label rate, cost and latency—not pass count.
6. If the compact source mapping still fails the cookware pages, inspect rendered
   crops with adjacent rows/page continuation or compare a stronger model on that
   case only. A graph whose edges are still wrong has falsified this proposal;
   neither increasing retries nor automatically widening the run is justified.

These are proposals, not implemented or empirically proven improvements. The
OpenAI Docs check supports scoped evaluation and comparison against explicit
criteria rather than relying on a general model verdict:
[evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices).
Its [Structured Outputs guidance](https://developers.openai.com/api/docs/guides/structured-outputs)
also distinguishes format compliance from semantic correctness. These sources
inform the test design; the diagnoses above come from this repository's traces.

Primary receipts: [analysis](../artifacts/kie-labeling/direct-full-20261003-luna-high50-r2/analysis.json),
[paired comparisons](../artifacts/kie-labeling/direct-full-20261003-luna-high50-r2/paired-comparison.json),
[source controls](../artifacts/kie-labeling/direct-full-20261003-luna-high50-r2/source-checks.json),
[offline counterexamples](../artifacts/kie-labeling/direct-full-20261003-luna-high50-r2/offline-checks.json),
and [timing](../artifacts/kie-labeling/direct-full-20261003-luna-high50-r2/timing.json).
