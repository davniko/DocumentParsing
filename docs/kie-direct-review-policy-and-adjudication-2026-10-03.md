# Direct-label review: field policy, explicit adjudication and reasoning comparison

Date: 2026-10-03. Scope: the existing 20-document real-source pilot, not bulk
relabeling or training. This pass follows the
[previous reviewer experiment](kie-direct-review-improvements-2026-10-03.md).

## Outcome

The requested policy and orchestration changes are implemented and tested. The
latest focused replays correctly transfer the product code, retain the other
references, exclude fax-only numbers, and produce the requested locality/country
representation at both high and xhigh reasoning. These are observed corrections,
not merely new instructions.

The wider panel still contains semantic disagreements and source ambiguities.
In particular, a following-page country continuation remains missing, and a
combined alternative address received a false pass at high reasoning. The xhigh
audit detected that address problem, but introduced other erroneous findings.
Consequently xhigh is now an available **stage-specific option**, not a silently
enabled default or a general claim of gold-label quality.

All original OCR, PDFs, silver candidates and previous outputs are unchanged.
Production datasets were not edited. The experiments cost an estimated
**$0.50382936 in total across 345 requests**, including the matched comparison and
final focused replays. This is usage-based accounting, not an invoice.

## 1. Diagnosis: the instructions and responsibilities mattered

There is no controlled evidence here that using the same model for review and
correction caused the observed errors. The previous caution about shared-model
agreement was a limitation of the evidence, not a root-cause diagnosis. We found
more concrete causes to address first:

1. **Address punctuation:** the old field definition requested joining physical
   lines with spaces. It did not request semantic postal components separated by
   commas. Space-only outputs were therefore partly following our contract.
2. **Port wording:** the previous definition explicitly favored retaining the
   complete printed port phrase. The user's desired `BUSAN`/`SOUTH KOREA` is a
   deliberate policy change, not evidence that the old output violated its old
   definition.
3. **Cross-section transfers:** a references reviewer could remove a misplaced
   product code, while a separately scoped cargo correction had no atomic
   obligation to preserve it in the description. Independent valid section
   outputs could therefore lose a supported fact between them.
4. **Adjudication visibility:** reviewers explained findings, but the corrector
   returned section values without a structured accept/reject/revise record for
   each finding. A final pass also lacked a mandatory verdict explanation.
5. **Over-broad correction:** during the new pilot, a corrector performing the
   legitimate product-code transfer also appended unrelated `EGYPT` to cargo.
   This demonstrated a need to constrain corrections to the supplied findings,
   rather than invite a new extraction of everything in a section.
6. **Telephone wording:** “joint TEL/FAX qualifies” could be misread as allowing
   the separate fax number whenever TEL and FAX occurred on the same line.
   A live xhigh review made that mistake. The definition now distinguishes
   number-level shared labels from physical-line proximity.

## 2. Field policies now supplied to the agents

### Complete postal address with component separators

`ExtractionPartyV7.addressLine` now requests:

> Write one line with comma-space separators between postal components.

A line wrap within a single component becomes a space; component wording,
numbers, order and internal punctuation are retained. The field still includes
the printed city/postcode/country and excludes company names, contacts, tax IDs,
captions and formatting markers. It does not concatenate competing addresses.
This is semantic formatting, not a rule to insert a comma at every OCR newline.

Observed example in the pilot:

```text
Before: KOLNER STR. 10 65760 ESCHBORN, GERMANY
After:  KOLNER STR. 10, 65760 ESCHBORN, GERMANY
```

The party's separate country field retains the existing source-form policy.
It was not silently changed to a global country-normalization scheme.

### Locality-only names and explicitly supported location countries

`LocationV7.name` preserves the locality's spelling and distinguishing name
words, while omitting separate country wording and generic facility descriptors
such as seaport, airport or terminal. Actual place-name words remain: this is
not a regex that deletes every occurrence of “port.”

`LocationV7.country` accepts the location's explicit country name/code or an
unambiguous national adjective, normalized to an uppercase conventional English
short country name. It does not infer a country from the city or a different
party. The model is shared by route, issue and payment locations, so those
locations use the same definition.

The `b25a3616` source now produces, in both reasoning conditions:

```json
{
  "placeOfReceipt": {"name": "BUSAN", "country": "SOUTH KOREA"},
  "portOfLoading": {"name": "BUSAN", "country": "SOUTH KOREA"},
  "portOfDischarge": {"name": "SOKHNA", "country": "EGYPT"},
  "placeOfDelivery": {"name": "SOKHNA", "country": "EGYPT"}
}
```

The respective receipt/delivery roles were checked using the page layout;
their values occur in OCR. The printed phrases contain `SOUTH KOREAN SEAPORT`
and `EGYPTIAN SEAPORT`. Other observed changes include `ZHAPU PORT` to `ZHAPU`
and `NHAVA SHEVA PORT` to `NHAVA SHEVA`.

The training prompt was aligned with these definitions. Existing training
datasets were **not migrated** by this experiment; a future dataset must follow
the same policy before training against the revised prompt.

### Telephone versus fax

A number explicitly labeled for both telephone and fax qualifies. A number
labeled only FAX does not, even beside a different TEL number on the same line.
Uncaptioned party-owned telephone continuations retain their existing policy.
This is a general ownership rule, not a sample-specific exception.

## 3. Review, adjudication and correction contracts

```text
Original silver target + literal complete OCR
  -> five scoped initial reviews
  -> joint correction scopes for explicit cross-section transfers
  -> auditor/corrector: decisions plus typed corrected section values
  -> dependency/schema checks
  -> affected-section re-review of current values and before/after changes
```

### Reviewer

Each section review now has a required short `explanation`, including passes.
Findings retain the field, issue category, explanation, proposed remedy and
optional short OCR excerpt. A supported fact assigned to another section uses
`wrong_owner` plus `reassignTo`; an unsupported fact is removed, not relocated.
These are decision justifications, not requests for hidden chain-of-thought or
per-scalar evidence inventories.

### Auditor/corrector

This is a separate PydanticAI call from the reviewer. It adjudicates the findings
and returns complete typed values for precisely its assigned scope. Every
supplied finding ID requires one of:

| Disposition | Meaning |
|---|---|
| `accept` | The defect and proposed remedy are appropriate. |
| `reject` | The original is correct or the proposed change is unsupported. |
| `revise` | There is a defect, but the suggested remedy needs changing. |

Every decision has a short source/policy explanation. Missing, duplicate or
foreign finding IDs cause an explicit hold, not a partial silent commit. An
undecidable conflict can return a correction hold.

One observed rejection concerned `DELHI` versus `DADRI, UP`: PDF layout placed
the latter under receipt and the former under freight payment. The auditor
rejected the reviewer's suggested replacement and retained the correct payment
location. This is the intended reviewer-can-be-wrong behavior.

Explicit cross-section transfers join their source and destination in one
correction response and dependency group. The product code `300398-120180`
can therefore be removed from references and retained in goods description in
the same atomic operation. Joint scopes share only relevant requested PDF pages.

The final prompt refinement forbids unsolicited edits outside the adjudicated
findings, including unrelated additions inside an otherwise corrected text
field. New suspected defects belong in the subsequent review. This preserves
unaffected facts without adding an autonomous repeated-repair loop.

### Final re-review

The auditor receives its current section, literal full OCR, field definitions,
before/after changes, and relevant correction decisions as **untrusted context**.
It checks the result rather than accepting the previous agent's authority. Its
verdict also includes an explanation. Decision records are saved separately in
`correction-decisions.json` and never inserted into extraction labels.

The existing bounded policy remains: one correction wave, with specific PDF
requests allowed for layout. PDF can resolve ownership of a value found in OCR;
it cannot authorize a PDF-only value. Provider errors are not passes. These
structural controls do not automatically prove a semantic decision correct.

## 4. Experiment design and results

### A. Full high-reasoning panel

The same 20 original silver candidates were used, with full OCR and PDFs only
when requested. Global request concurrency was eight. No prior corrected target
or final review was substituted for the original candidate.

### B. Matched xhigh auditor comparison

The experiment reused the **same original silver candidates and the same initial
high reviews**, including already requested layout pages. Only correction and
final re-review changed to xhigh. Thus the two conditions did not start from
different initial reviewer proposals. Two documents required no new correction
calls. This was not twenty fresh full extraction runs.

The [official Luna model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna)
lists xhigh support; live requests confirmed this path. The optional
`audit_reasoning_effort` config applies only to these later stages. The default
remains high throughout.

| Measure | Full high cycle | Matched xhigh audit cycle |
|---|---:|---:|
| Documents completed | 20 | 20 |
| New paid requests | 212 | 103, initial reviews reused |
| Application-valid targets | 18 | 18 |
| All-section `reviewed_candidate` statuses | 11 | 9 |
| `needs_adjudication` statuses | 9 | 11 |
| Final section pass / corrections / unresolved | 89 / 6 / 5 | 86 / 8 / 6 |
| Previously correct control checks preserved | 8/8 | 8/8 |
| Raw wire-schema failures | 0 | 0 |
| Newly incurred estimated cost | $0.270529 | $0.183450 |
| Elapsed seconds | 469.09 | 457.66, cached initial reviews |

These are workflow statuses, **not semantic accuracy scores**. The lower xhigh
pass count includes useful detections and incorrect objections; fewer passes is
not itself better or worse.

Checks retained from the earlier report include both carrier identities without
capital boilerplate, postcode value preservation, tax-reference preservation,
five shared-cargo memberships, and voyage/leg separation. The old “retain full
port wording” check is obsolete under the user's new locality-only policy and
must not be scored as a regression. Its historical result remains visible in the
JSON for traceability. The grouping marker currently applies to all document-10
historical checks; its membership and partial-total checks remain relevant.

For changed, surviving address labels, numeric-token comparison found zero lost
old numeric tokens: 38 labels in high and 39 in xhigh. This excludes a deliberately
nulled ambiguous alternative address. It checks numeric retention, not semantic
ownership or whether all omitted facts were recovered.

### C. Findings from the paired panel

| Case | Observed result and interpretation |
|---|---|
| Product code in references, `ab905fd4` | Both conditions moved it to description. Both also appended unrelated `EGYPT` during correction. The xhigh final review caught that contamination; high did not. |
| Locality wording, `b25a3616` | Both produced BUSAN/SOUTH KOREA and SOKHNA/EGYPT in the correct route roles. |
| Alternative addresses, `c0f529a4` | The address was left unset and the primary-address ambiguity remained explicit rather than concatenated. |
| Alternative addresses, `66d5d7f6` | High accepted concatenated Suez/Alexandria text. Xhigh's final review caught it; the one-wave flow left it held rather than claiming a completed correction. |
| Fax-only numbers, `b25a3616` | The xhigh panel wrongly requested fax numbers as phones because TEL occurred nearby. This motivated the number-level field clarification. |
| Following-page country, `ab905fd4` | Delivery-agent EGYPT remained omitted at both efforts. No sample-specific continuation rule was added. |
| Equipment, `24370697` | Final reviewers found missing reefer wording after initial equipment review had passed. A new final-review finding is not automatically repaired in this one-wave experiment. |
| Source identifiers, `99adb051` and `298806fe` | Unresolved malformed container identifiers still fail application validation. The system did not invent valid-looking replacements. |

Other disagreements remain recorded in per-document comparisons: whether a
thermal statement additionally belongs in goods handling, reference-caption
deduplication, ownership of a warehouse/delivery locality, and misplaced country
inference for an issue location. The high review's proposed Sweden addition to
an issue location was not an independently validated fact. These are not all
confirmed label defects, and they are not fixed by increasing reasoning alone.

### D. Final focused replays after the two additional clarifications

After observing the unwanted EGYPT cargo edit and fax request, we added the
scoped-edit and number-level TEL/FAX instructions described above. We replayed
documents 10 and 13 at **both** audit efforts from the same original targets and
initial reviews: four audit cycles, not four new source documents.

All explicit focused checks passed in all four cycles:

- Correct route locality/country values and field roles.
- Fax-only numbers neither inserted nor requested by the final party reviewer.
- Product code retained in description and removed from references.
- Printed marine-pollutant content retained in goods.
- No unsolicited EGYPT added to goods.
- Other reference values preserved.

Both xhigh focused cycles had passing final reviews. The high cycles retained
additional findings: comma placement in one repeated address, a shipment-quantity
phrase inside description, and missing place of issue. Those further findings
were not silently applied. The following-page delivery-agent country remains
missing even in the focused xhigh target, so passing reviews are not a claim
that the entire document is gold.

The final two narrow wording changes were tested on these focused cases and the
unit suite; **there was no subsequent full-20 rerun with that final revision**.

## 5. Cost, latency and validation

### All attempts in this task

| Experiment | Requests | Estimated USD |
|---|---:|---:|
| Full high panel | 212 | 0.270528775 |
| Matched xhigh correction/re-review | 103 | 0.183450095 |
| Final focused high/xhigh replays | 30 | 0.049850490 |
| **Total** | **345** | **0.503829360** |

Pricing used by the recorded scripts, USD per million tokens: ordinary input
0.10, cache read 0.01, cache write 0.125, output 0.50. Reasoning tokens are included
in output cost, not counted twice. All requests completed with usage; no unknown
failed-request charge was presented as zero.

The shared initial high reviews cost $0.137527405 over 114 requests. The comparable
audit-stage costs are $0.133001370 for high (98 requests) versus $0.183450095 for
xhigh (103), approximately 38% more for xhigh in this panel. Including the shared
initial stage gives about $0.321 for an equivalent xhigh-audited 20-document cycle,
versus $0.271 high. These are pilot costs, not guaranteed bulk projections.

The preceding R3 panel took 277.70 seconds plus a 62.39-second one-document schema
replay, versus this high panel's 469.09 seconds. The new field policy generated
additional address work, and adjudication now returns explicit decisions.
**There is no end-to-end speedup claim.** This is measured overhead for expanded
functionality; the xhigh choice adds cost and latency and is not enabled globally.

A local context-construction probe (100 cargo contexts, same current field
definitions for both implementations) measured 1.56989 seconds for previous
orchestration versus 1.57781 seconds for the current implementation: approximately
0.5% difference. Context length was identical at 27,713 characters. Peak traced
Python allocations were 1,332,390 and 940,519 bytes, respectively; this is not a
process-RSS or provider-memory benchmark and does not establish a durable memory
gain. The relevant overhead is the additional/model reasoning work, not local
context assembly.

Validation:

- **176 targeted tests passed** in 15.59 seconds across direct labeling, V7
  extraction, relation constraints, equipment semantics and inherited schemas.
- Changed Python files pass Ruff; direct flow/models pass mypy.
- **345/345 raw provider outputs conform to their actual native wire schemas.**
  Application identifier constraints remain a separate gate, as the two malformed
  source cases demonstrate.
- Tests cover joint transfers with accepted, rejected and revised suggestions;
  missing, duplicate and foreign decision IDs; holds; transitive scope/dependency
  groups; independent surviving changes; and audit context supplied to re-review.
- Frozen source/candidate/PDF and previous-output hashes were checked before
  processing and after analysis. No production dataset or original source edit.

## 6. Implementation map and experiment artifacts

Maintained code:

- [V7 field definitions](../src/document_ocr/label_schemas/bill_of_lading_v7.py):
  address formatting, locality/country policy and telephone/fax distinction.
- [Review and decision models](../src/document_ocr/labeling_agents/direct_models.py):
  required explanations, explicit transfers, dispositions and audit effort.
- [Direct flow](../src/document_ocr/labeling_agents/direct.py): typed joint scopes,
  exact decision coverage, dependency-safe commits, layout continuity and re-review.
- [Reviewer prompt](../prompts/labeling_agents/direct_reviewer.md) and
  [auditor/corrector prompt](../prompts/labeling_agents/direct_corrector.md):
  role responsibilities, narrow edits and concise decision explanations.
- [Training prompt](../prompts/training/mpci_bl_extraction_v7.txt): aligned policy.
- [Direct tests](../tests/test_direct_labeling.py): workflow/contract regressions.

Self-contained runners, frozen implementation snapshots, request/response records,
usage, final labels and per-document comparisons are in:

- [High full-panel artifacts](../artifacts/kie-labeling/direct-review-20261003-luna-high20-r4/)
  and [analysis](../artifacts/kie-labeling/direct-review-20261003-luna-high20-r4/analysis.json).
- [Matched xhigh artifacts](../artifacts/kie-labeling/direct-review-20261003-luna-xhighaudit20-r4/)
  and [analysis](../artifacts/kie-labeling/direct-review-20261003-luna-xhighaudit20-r4/analysis.json).
- [Final focused replay artifacts](../artifacts/kie-labeling/direct-review-20261003-luna-high20-r4/focused/)
  and [explicit checks](../artifacts/kie-labeling/direct-review-20261003-luna-high20-r4/focused/analysis.json).

## 7. Decision after this experiment

Keep the clarified field definitions and explicit adjudication/transfer contracts.
The specific product-loss, route-format and fax-interpretation failures have
observed fixes. Preserve high as the default and make targeted xhigh escalation
an explicit choice: it caught important errors, but its full-panel results do not
establish uniform superiority. Do not treat its two focused passing reviews as
a complete independent semantic certification.

The country-continuation case was intentionally not turned into a bespoke rule,
consistent with the user's lower priority for it. Competing-address ownership and
newly discovered final-review findings remain named, inspectable cases rather
than being erased by a global “passed” claim. No bulk relabeling, synthesis or
training launch was part of this pass.
