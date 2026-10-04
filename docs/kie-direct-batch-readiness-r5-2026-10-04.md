# Direct labeling R5: clear field policies and bounded, reviewable batches

## Scope and decision standard

This pass implements the user's clarified extraction/review expectations and reruns
the **same 50 OCR/PDF pairs from fresh extraction**. It does not relabel the real
dataset, modify source OCR/PDFs, change synthesis templates or start training.
Luna/high, strict PydanticAI native structured responses, concurrency 16, at most
two correction waves. Prior labels and diagnostic expected answers are not inputs.

The deployment target is human-reviewed batches of 100, not an unattended claim of
perfect gold annotation. Assess actual source errors in agent-passing outputs,
unnecessary holds, genuine ambiguities, schema/fidelity gates, runtime and cost.
Finite diagnostics do not prove that a never-seen source cannot fail.

## Confirmed failures and changes

- **Unexplained collateral changes.** R4's second corrector could erase a fact
  outside the intended remedy while returning an otherwise valid complete section.
  Each correction decision now lists its changed value paths; local code computes
  the exact before/after difference and rejects undeclared or rejection-authorized
  changes. Allowed paths are typed schema choices, not free-form spellings. Findings
  may share a path when several defects affect one list, and an authorized value can
  remain unchanged after adjudication. This is a scope/receipt check, not semantic proof.
  Lists are atomic because regrouping can change entity identity and order; source
  reviewers still check their contents. Decisions retain concise explanations.
- **Wrong comparison baseline.** Re-review previously compared to the initial
  extraction. A value added in wave one and erased in wave two could therefore
  disappear from the diff. Re-review now compares to the immediately preceding
  candidate. It receives the exact changes, without the previous auditor's verdict
  and justification anchoring its independent source decision.
- **Negotiability.** Positive order-consignment evidence must be the actual
  consignee instruction, not generic contract terms or an explanatory form caption.
  A named consignee without order wording is **non-negotiable**, as requested; explicit
  non-negotiable/waybill issuance is also supported. Missing consignee evidence is
  unknown, not a negative default. Conflicting shipment-specific declarations go to
  review. No regex over the entire document assigns the label. The metadata review
  checklist explicitly checks this field even when omitted from the candidate.
- **Address normalization.** Retain complete postal information with comma-space
  separation, include a repeated postal component once, and keep distinct postal
  digits/qualifiers even when attached to a repeated country token. This does not
  authorize deleting district, building or post-box facts.
- **Dates.** Numeric day/month order requires source-format evidence, not assumptions
  from issuer geography or language. Policy-defined absence resolves the target
  decision and does not itself require an endless hold.
- **References and marks.** A vessel IMO/Lloyds identifier cannot become a generic
  shipment reference because its checksum is invalid. Customs identifiers remain
  customs references even under a combined marks/container heading. Ambiguous
  ownership should trigger a PDF layout request; PDF-only values remain excluded.
- **Cargo accounting.** A sole goods identity covering the complete shipment owns
  its shipment mass/volume totals. Competing same-scope printed totals/complete
  portion sums require a review finding unless source evidence resolves them.
  An isolated second package word does not establish containment or a missing count.
- **Product capacity versus shipment mass.** The fact review's earlier responsibility
  checklist said *all masses* belong to accounting, despite the description field
  requiring printed per-package capacity. It now explicitly owns specifications and
  capacity wording; accounting owns shipment counts/masses/volumes. Source 38 now
  retains `CHEESE, 19KG (APPROX)` without mislabeling it as shipment mass.
- **Identity continuations.** Follow linked party-name continuations as well as
  postal continuations. The shipper's `ON BEHALF OF SAMSUNG ELECTRONICS TAIWAN...`
  continuation in source 18 was recovered by a blind new party review.
- **Identifier format.** Both container and placement identifiers now expose their
  existing canonical format in native JSON Schema. An attempted placement containing
  presentation spaces/hyphens previously satisfied the wire schema but failed Python
  validation. Checksum validation still runs locally; no digits are invented.
- **Map-only corrections.** A finding solely about the internal cargo source map
  cannot mutate target facts. This blocks the unrelated ACID-to-marks edit observed
  during replay without banning legitimate target corrections with their own findings.

### Freight diagnosis correction

Source PDFs 41 and 49 both have an **empty, preprinted FREIGHT ADVANCE receipt box**.
It does not establish prepaid freight. The earlier concern about silently erasing
the field exposed a real orchestration weakness, but the previously added prepaid
value was not a reliable semantic control. This pass explicitly corrects that
expectation instead of preserving a wrong value for apparent stability.

The original PDFs were inspected. BIMCO's [sample GRAINCONBILL](https://www.bimco.org/media/0cwklgrd/sample-copy-grainconbill.pdf)
also shows the unpopulated receipt field. BIMCO's [NUBALTWOOD terms](https://www.bimco.org/media/2aqnk3y5/sample-copy-nubaltwood.pdf)
distinguish a freight advance from the remainder payable on discharge. Therefore
neither a blank caption nor an advance automatically establishes the complete
shipment payment arrangement. The general field rule requires a selected instruction.

## Validation design

- Actual PydanticAI request-path tests, including deliberate undeclared removal,
  overlapping authorized list changes, rejection-authorized changes and two-wave reversals.
- Replay exact R4 revisions against the new change contract; falsify with omitted
  change declarations. Measure local overhead separately from provider latency.
- Preserve the original 42 controls, ten expanded controls and 21 supplemental
  source checks. Add explicitly named readiness checks after inspecting previously
  untested sources and the old failures. These are properties, not whole-document F1.
- Inspect all flagged outputs and before/after changes; distinguish a checker false
  flag from a genuine source error. Inspect PDF layout where ownership is disputed.
- Save wire schemas, responses, all-attempt usage, source/code hashes, final targets,
  per-section findings and a document review queue. Never silently call a held
  candidate accepted merely because its target JSON validates.

The OpenAI Docs skill informed preserving native structured outputs while testing
semantic correctness independently. [Official guidance](https://developers.openai.com/api/docs/guides/structured-outputs)
separates schema conformance from mistakes in extracted content.

## Offline results

Relevant suites: **570 passed in 22.46 seconds**; Ruff and focused mypy checks pass.
An additional older V5 ordering test fails because its fixture omits required
`packageIds`; neither that fixture nor the V5 schema was changed in this pass. This
is reported separately, not disguised as an all-repository test pass. On 48 recorded R4
revision pairs, all accurately declared deltas pass and all 48 omitted-declaration
mutants fail. Median local cost per revision: 5.07 microseconds for the existing
diff versus 5.77 microseconds for diff plus declaration validation; traced peak
allocations 1,696 versus 2,172 bytes. The sub-microsecond delta is negligible beside
network/model inference. This benchmark does not claim end-to-end speed improvement.

## Live rerun

Frozen inputs, implementation and full receipts:
`artifacts/kie-labeling/direct-full-20261004-luna-high50-r5/`.
### Fresh run versus bounded development continuations

The full **50-document fresh-extraction run** completed with 50 application-valid
candidates, 27 agent-passing and 23 held, no failed documents. It made 555 calls,
cost an estimated **$0.93466936**, took **726.44 seconds (12.11 minutes)**, and peaked
at 438,124 KiB RSS with concurrency 16. All 555 native responses validated against
their captured wire schema; no calls had unknown billing status.

The high hold count exposed an overly strict change-declaration interface introduced
in this pass: agents named parent paths instead of leaf paths, and multiple findings
legitimately touched the same atomic list. This was an implementation defect, not
evidence that those source documents were unlabelable. It was fixed using typed
path choices and union-of-authorized-scope validation, without weakening the guard
against undeclared changes.

Existing paid extractions/reviews were then reused in finite continuations. These
are **not represented as a second fresh 50-document run**:

| Stage | Scope | Calls | Estimated USD | Elapsed |
|---|---|---:|---:|---:|
| Fresh complete flow | 50 documents | 555 | 0.93466936 | 726.44 s |
| Correction replay | 23 held cases + 2 blind party controls | 187 | 0.30366569 | 328.27 s |
| Targeted final checks | 2 technical holds + 5 blind section controls | 36 | 0.06542398 | 199.17 s |
| Clarified negotiability policy | Fresh metadata review on all 50 | 171 | 0.20582063 | 242.91 s |
| Negotiability confirmation | 3 omissions + 4 order/unknown controls | 19 | 0.01852204 | 69.35 s |
| **Entire R5 experiment** | Preserved all attempts | **968** | **1.52810170** | **26.10 min summed provider-run time** |

These times exclude local investigation/manual review and gaps between stages; they
are not total task wall time. Later stages peaked at 182,884–362,148 KiB RSS. The
fresh run was faster than R4's 960.56 seconds, but its unnecessary holds make that
an unsuitable claim of end-to-end quality-equivalent throughput improvement. The
extra development replays are overhead, not the planned production workflow.

One earlier replay launcher failed locally before HTTP dispatch: it indexed a
missing native-schema `title` rather than the enclosing response-format `name`.
There were zero dispatched requests and zero model responses; it was fixed before
the recorded replay. Its failed receipts remain in `correction-replay/`, not counted
as accepted/reviewed work or claimed as a provider outage.

All **968 paid responses** conform to their captured native wire schemas, with
zero unknown-billing calls. All prices above are receipt-based estimates using the experiment's recorded
input/cache/output rates, not a provider invoice. Reasoning tokens are part of
output usage, not charged twice. `all-request-validation.json` records every phase.

### Final automated output and independent checks

The combined final automated output has **43 agent-passing documents and seven
held documents**, with all 50 application-valid. The **140 named source checks
pass 140/140**:

- 90 accounting, wording, grounding, continuation and earlier-regression checks;
- 50 separately inspected consignee-policy outcomes: four order consignments,
  forty named/non-negotiable cases, six genuinely missing-evidence cases.

The checks cover all 50 documents but are **not** complete gold labels, field F1 or
document accuracy. The earlier numeric-date, duplicate-country/postcode, invalid
IMO-as-reference, ACID-as-marks and generic-order-boilerplate failures are corrected
on their tested sources. Cookware's five identities/nine placements and the shared
panel portions remain correct under their source controls.

The final negotiability confirmation was necessary because three reviewers initially
missed an omitted field despite the clarified field description. Adding that field
to the reviewer responsibility checklist recovered all three without turning the
order/unknown controls into false non-negotiable labels. No expected answers were
provided to these calls.

### Manual closure of the pilot, not another agent loop

Independent inspection still found two normalization issues among agent-passing
outputs: source 40 lacked postal-component commas, and source 43 retained the
`MADE IN` caption inside `origin.name`. Source 36, already held for conflicting
ACIDs, also reordered its country/postcode. These are disclosed false passes for
format/boundary policy, not evidence of omitted postal digits or fabricated cargo.

`adjudicate_pilot.py` records the finite manual decisions in a **separate experimental
copy**, preserving every automated target. It changes four documents:

| Source | Decision | Proof/constraint |
|---|---|---|
| 13 `15b8d670` | Add the missing explicit notify party, matching the consignee's OCR-supported values | PDF page 1 independently inspected; separate Notify block exists. No PDF-only value and no inferred `sameAs` |
| 36 `b2617901` | Restore `CANADA L4V 1V6` order | Exact token multiset retained; country/postcode sequence confirmed in OCR. ACID conflict remains held |
| 40 `4799921c` | Add commas to three party addresses | All non-comma bytes, postal digits, word order and existing punctuation unchanged |
| 43 `150e9dad` | `origin.name: MADE IN TAIWAN → TAIWAN` for both goods | Keep the full declaration in marks; all grouping/counts/placements and other facts unchanged |

Three agent holds are settled: 13's missing role is fixed as above; without target
changes, 22 correctly omits a rejected PDF-only vessel/voyage proposal and 26 correctly
extracts the supplied OCR without guessing an unavailable attachment's contents.
This certifies the **target decision**, not the unseen attachment's completeness.

The pilot now has **46 documents with its checks/findings resolved and four left
for source adjudication**. This is not an assertion that every scalar in those 46
has a separately authored gold reference. All 50 final candidate JSONs validate;
all 140 source checks still pass after manual edits; exact delta replay and source
OCR hashes pass for all 50. No unrelated label or input changes were made.

Remaining review, with no guesses made:

| Source | Remaining decision |
|---|---|
| 16 `1d51aca4` | Repeated copies disagree on consignee VAT and voyage identifiers |
| 36 `b2617901` | Two differently printed ACID values for the same caption |
| 49 `da651f2f` | `KOS` does not license kilograms; absence is safe. Also adjudicate whether product wording `FINNISH WHITEWOOD` licenses a separate `origin.name=Finland` under the printed-origin policy |
| 50 `e815e426` | Complete local gross/net sums contradict the shipment totals |

The review files include exact OCR/PDF links, candidate labels, findings, and manual
receipts. The original automated status is never overwritten as if the model had
made the manual decision.

## Decision for the next 100-document batch

**Proceed with one supervised batch of 100, not unattended gold publication and
not another broad prompt-repair campaign.** The implemented normal flow is one
extraction, section reviews/layout assistance, at most two correction waves, then
an explicit manual queue. Correct components are retained; one unresolved source
does not block all other documents. Default concurrency now matches the tested 16.

Review every flagged document. Because the pilot still exposed normalization false
passes, also inspect ten passing documents, stratified across simple and complex
layouts, before releasing the batch. That is a sampling precaution, not a statistical
guarantee or permission to call the remaining 90 manually verified. Check all
negotiability decisions against the actual consignee instruction and pay particular
attention to address ordering/separators and origin-caption boundaries. If a new
systematic semantic defect appears, stop publication of the affected cohort and
diagnose that family; do not automatically rerun already-settled sections.

The observed fresh-run cost scales to about **$1.87 per 100** before variation and
manual work; conservatively allow approximately **$3** for the first supervised
batch. This is a planning estimate, not an asserted hard cap. The all-in development
spend above is fully retained rather than reset out of the accounting.

**No 100-document run, production dataset migration, synthesis, training, or pod
operation was launched.** All source OCR/PDF hashes are unchanged. Only the direct
flow/schema/prompts/tests/config/docs and isolated R5 experiment artifacts changed.

## Evidence index

- [Final automated audit](../artifacts/kie-labeling/direct-full-20261004-luna-high50-r5/final-audit.json)
- [Every paid request/schema/cost](../artifacts/kie-labeling/direct-full-20261004-luna-high50-r5/all-request-validation.json)
- [Manual decisions and experimental targets](../artifacts/kie-labeling/direct-full-20261004-luna-high50-r5/adjudicated-pilot/manifest.json)
- [Four remaining source reviews](../artifacts/kie-labeling/direct-full-20261004-luna-high50-r5/adjudicated-pilot/remaining-review.json)
- [Post-edit validation](../artifacts/kie-labeling/direct-full-20261004-luna-high50-r5/adjudicated-pilot/validation.json)
- [Guard falsification/benchmark](../artifacts/kie-labeling/direct-full-20261004-luna-high50-r5/guard-probe.json)

Reproduction uses the self-contained scripts beside those artifacts with
`UV_CACHE_DIR=/tmp/documentparsing-uv-cache uv run --no-sync --with jsonschema python ...`.
Paid runners deliberately refuse to overwrite their existing run directories.
Native request receipts freeze their implementation and input hashes. Final
formatting added two trailing commas only; Python AST equality is verified against
the last live snapshot when regenerating the final audit.
