# Direct labeling R4: repairable accounting and bounded adjudication

## Scope

Implement the five diagnosed corrections, replay difficult and sound cases, then
run the same frozen 50 OCR/PDF pairs and independently inspect the outputs. No
production labels, raw OCR, source PDF, synthesis templates or training runs are
changed. Prior labels and diagnostic answers are never supplied to the agents.

The bounded experiment is complete. Accounting recovery improved, but source review
found five agent-passing documents with remaining errors or normalization defects;
unattended gold publication is not ready. The starting evidence is
[R3](kie-direct-cargo-fullpdf-r3-2026-10-03.md).

## Diagnosis and changes

1. **Working-map authority mismatch.** The prompt called the map provisional, but
   code created it once and used its exact sums to veto later corrections. In
   source 13 the xhigh map parsed grouped kilogram tokens as decimals: it licensed
   30.354 rather than 30,354 kg. A correct target proposal was rejected. Accounting
   review and cargo correction can now return an explicit complete `mapCorrection`.
   Its references, numeric tokens and identifiers are validated. Corrector map and
   label edits commit together; invalid map edits hold the affected operation.
   Every proposal remains in call receipts and accepted correction maps have
   separate versioned artifacts. This is not permission for the map to invent
   facts or for agreement to replace source inspection.
2. **Duplicated row ownership.** Counts/placements and masses were reviewed by
   separate agents which could assign the same printed row differently. One
   accounting reviewer now owns grouping, counts, masses, volumes and placements.
   The fact reviewer owns product wording, classification, markings, origin and
   handling. Its output schema cannot issue mass/allocation corrections.
3. **Grouping/packaging drift.** Establish product identities across the complete
   document. Products jointly accounted throughout stay one entry; a product
   separately accounted elsewhere retains its identity in shared portions.
   Independently quantified products remain separate. Mere package-type words do
   not establish containment. The existing target count policy remains printed
   counts or complete same-level sums; serial ranges and mass/capacity division
   are not newly authorized target derivations.
4. **Meaning versus coordinates and incomplete moves.** Instructions require
   tracing headings/continuation markers and inspecting printed role captions
   when OCR misreads them. Corrections include their direct consequences: a
   supported identifier moved out of the B/L field must survive in its appropriate
   reference field, including within-section moves. Unrelated facts stay intact.
5. **Ambiguity versus safe repair.** An ambiguous finding with a concrete remedy
   can enter correction. For example, the date policy may require removing an
   assumed ISO date while its unresolved source interpretation remains recorded.
   Pure ambiguity without a safe remedy does not trigger repeated paid calls.
6. **Parent-only validation was invisible to section review.** The confirmation
   pilot exposed duplicate shipment references that semantic reviewers accepted,
   but the parent label validator rejected at commit. Reference uniqueness now
   resides on the reference field, so dynamically composed section models inherit
   it. Section-schema failures become actionable diagnostics even when the semantic
   reviewer says pass. This preserves the constraint rather than silently deduping
   or weakening it. Both initially invalid native-schema-compliant full-run drafts
   subsequently reached application-valid targets.

Measures in the internal map retain the printed numeric token as well as its
interpretation. Local validation requires that *that token*, not some unrelated
number elsewhere in OCR, supports the selected magnitude. A token such as
`7.740` remains genuinely ambiguous in isolation; table context and source review
must distinguish 7.74 from 7,740. No blanket multiply-by-1,000 rule is used.

## Validation design and launch boundary

- Offline tests cover map revision through both review and correction, invalid
  revision rejection, atomic label/map commits, exact sums, split responsibilities,
  safe date omission with retained ambiguity, and the existing request/commit path.
- A six-document paid pilot uses sources 13, 17, 26, 28, 32 and 43: four known
  difficult ownership/grouping cases and two sound cargo/PDF-grounding controls.
- The full rerun uses the same 50 frozen sources. Record every attempt, schema
  validation, provider usage, cost estimate, elapsed time and peak process RSS.
- Independent diagnostic controls are not treated as whole-document accuracy.
  Passing candidates need source-based checks for both incorrect additions and
  omissions/ownership errors. Holds are separated into pipeline/model defects and
  genuine source-policy ambiguity. No bulk processing is launched merely because
  schema validation or agent consensus succeeds.

## Documentation constraint

Native structured responses retain a root object with nullable typed nested
objects. The [official Structured Outputs documentation](https://developers.openai.com/api/docs/guides/structured-outputs)
distinguishes schema conformance from semantic correctness; local and source-aware
checks are still necessary. The installed PydanticAI native-output path and actual
wire schema are tested, rather than replacing them with prompt-only JSON output.

## Results

### Offline and bounded pilot evidence

Final relevant suites: **529 passed in 22.42 seconds**; Ruff check/format and
`git diff --check` pass. The tests include invalid map correction rejection through
both review and correction, atomic map/target commits, and safe omission despite
source ambiguity. Native-schema validity is not treated as semantic validity.

The first six-source pilot completed **83 calls**, **$0.173173195 estimated**, in
**407.29 seconds**, peak RSS **273,728 KiB**, concurrency eight. Four documents were
agent-passing and two held. Independent source controls confirm the cookware's five
identities, nine quantified placements and five package/mass totals; the toner case
retains 1,845 cartons without importing PDF-only local amounts. The consignor case
is ultimately corrected despite an intermediate reviewer mistake.

The pilot caught a regression before wider replay: an overbroad joint-entry rule
split a globally repeated panel product into exclusive and composite identities.
The revised definition establishes identities across the complete document first.
The invalid proposal was held, not published as an accepted target. Marks ownership
also remained a false-pass candidate despite a stronger field description; this is
being tested explicitly, rather than hidden behind the successful accounting checks.

A three-source confirmation pilot (17, 32, 43) completed 49 calls for
**$0.10012215 estimated**, in 332.68 seconds, peak RSS 218,220 KiB. It confirmed
the joint-machine representation and carrier seal, but still exposed the
`LEADERS` marking loss and parent-only reference validation problem. Before the
full run, the remaining contradictory column-position instruction was removed
and section validation was wired into review as described above. The final
pipeline snapshot, rather than a moving implementation, was used for all 50.

### Full rerun: measured results

| Metric | R3 | R4 |
|---|---:|---:|
| Frozen OCR/PDF documents | 50 | 50 |
| Application-valid final candidates | 50 | 50 |
| Agent-passing candidates | 39 | 43 |
| Held candidates | 11 | 7 |
| Failed document operations | 0 | 0 |
| Original source controls | 37/42 | 41/42 |
| Expanded paired controls | 3/10 | 9/10 |
| Paid calls, all attempts | 632 | 666 |
| Estimated full-run cost | $0.953700785 | $1.111740755 |
| Elapsed | 1,489.57 s | 960.56 s |
| Peak process RSS | 446,496 KiB | 456,996 KiB |
| Peak concurrent requests | 8 | 16 |

The run finished in **16.01 minutes**, 35.5% less elapsed time. This is not an
isolated code-speed benchmark: concurrency doubled and provider latency varies.
Peak process RSS increased 2.35% (10,500 KiB); calls increased 5.38%, and estimated
cost increased 16.57%. The added quality checks are not claimed to be cheaper.

All 666 recorded response outputs passed their actual native wire JSON Schema.
The two initial drafts that failed application constraints illustrate why this is
not equivalent to application validity. Final application validation passed for
all 50, including the seven correctly retained as held candidates. There were no
unknown-billing calls. Full OCR remained plain text, Luna reasoning was high, and
full original PDF bytes were checked on cargo calls. Source hashes and the saved
implementation snapshot were verified. No production labels or inputs changed.

Review/correction changed 39 candidates (216 changed scalar paths; list shifts can
affect several paths, so this is not a count of 216 independently corrected errors).

### What demonstrably improved

- **13, cookware:** all six independent controls now pass: five distinct products,
  all nine product/container allocations and quantities, the corresponding product
  totals/masses, and removal of joined tare text from equipment IDs. This was the
  principal immutable-map failure in R3. It is now agent-passing and matches the
  source-adjudicated accounting controls.
- **04, drums:** no package totals invented by dividing net mass by capacity or
  counting serial ranges. Explicit shared membership remains possible without an
  unprinted count.
- **17, machines:** one joint entry accounts for two unpacked machines, while the
  actual carrier seal survives the separate `Shipper Seal: NOSEAL` declaration.
  Its remaining country hold is distinct from this resolved cargo problem.
- **28:** RATHI is the shipper and WAFA the consignee; the PDF caption is no longer
  treated as two competing consignee blocks. The printed dye product is retained.
- **32:** marker-linked voyage `045W` and cargo mark `LEADERS` both survive. No
  PDF-only shipper identity is imported into the OCR-derived target.
- **33:** B/L `HLCUSYD250337760` is correctly selected and the separate carrier
  reference `21137888` is retained, rather than lost during the move.
- **43:** two globally distinct panel products retain their identities in shared
  container portions. Known 119/32-carton placements survive; the shared 112-carton
  portion is not proportionally divided. No guessed full product totals appear.
  The uncaptioned registration-like number `84899112` was not added as a phone.
- Review also recovered `PSL` in the steel specification, the continued CargoX
  identifier, and applicable package types; it retained continuation postcode
  `62815` and omitted PDF-only cargo measures/identifiers in the controls.

### Independent audit and falsification

The original 42 controls cover 27 documents. The additional ten paired controls
cover six documents. A further **21 source-derived checks across 14 documents**
passed **18/21**. Combined, these selected controls touch 44/50 documents; they
are not complete per-field gold labels and their ratios are not field F1 or
whole-document accuracy. Held candidates were inspected as well. The other six
documents were inspected directly, not silently assumed correct because they had
no automated control. Control answers were never sent to the agents.

The audit deliberately checked its own assumptions:

- Source 36's `9200426` is printed as a Lloyds/IMO identifier but fails the current
  IMO checksum contract. A proposed check requiring that invalid number in
  `vesselImoNumber` was rejected. The correct checks prohibit inventing a repaired
  IMO **and** prohibit disguising this vessel identifier as a shipment reference.
- Source 37's exporter-country declaration explicitly owns CHINA to the shipper.
  Under the current party-country/address policy, that is not a geocoded invention.
  Its preserved street and country pass; a postal-block-only policy would be a
  policy change, not grounds to call the current output a confirmed error.
- Source 10's on-board date is present in OCR under a displaced heading. The PDF
  is used for the role, not to add a missing value. Generic shipping-agent blocks
  in 19/50 do not automatically establish a delivery-agent role.

**Local numeric falsification:** 175/175 injected impossible interpretations of a
measure's own printed numeric token were rejected. These test numeric grounding,
not semantic ownership. All 50 benchmark maps passed the unmutated local check.

**Live semantic falsification:** six full-PDF relation-review calls covered three
clean/corrupted pairs. All three clean controls passed; all three corruptions were
flagged: fabricated equal 20-bag shares, swapped container counts with an unchanged
grand total, and five outer pallets substituted for 54 inner boxes. Cost was
**$0.008240655**. This establishes the tested mechanisms, not a universal guarantee.

**Local overhead:** median numeric-map grounding over the same 50 maps increased
from 9.13 ms to 18.65 ms, an added **0.190 ms/map** for own-token validation. Peak
traced allocation was unchanged at 31,507 bytes. The additional section-validation
pass over 50 documents took 16.58 ms (**0.332 ms/document**), peak traced allocation
23,578 bytes. These measured sub-millisecond per-document costs are negligible
beside provider calls; the broader paid-flow cost increase is reported above.

### Remaining false passes: concrete scope

These are **five observed agent-passing documents**, not a claim that only five
can contain an error. Four have semantic/policy defects; one is a normalization
defect with the postal data retained. No value was manually patched into the
frozen results to improve the reported scores.

| Source | Actual final output / reasoning | Expected behavior under current contract | Diagnosis |
|---|---|---|---|
| 07 `92da45e4` | Reviewer adds `ACID: 2025996202023050060` to cargo marks because it is under `MARKS AND NUMBERS` | Retain the standalone customs reference in references, not cargo markings | Review-introduced regression; heading precedence overrides the semantic reference rule. Original 42-control suite catches it. |
| 23 `0b03d5a2` | Address ends `..., EGYPT, EGYPT12568` | Preserve postcode 12568 and one country occurrence in the single address line | Both data fragments survive, but deduplication/component formatting is incomplete; reviewers accept literal preservation. |
| 36 `b2617901` | `LLOYDS/MO NUMBER 9200426` remains a forwarding/export reference | An invalid vessel identifier is not a commercial shipment reference; keep it in diagnostic evidence rather than invent a corrected IMO | Cross-section semantic classification miss. The field description should explicitly cover vessel identifiers as well as voyage/container identifiers. |
| 47 `f0f120de` | Initial review correctly flags `10-04-2025`; corrector rejects the finding using Spanish locale and final review accepts `2025-04-10` | Leave the ambiguous normalized dates absent without an actual source date-order anchor | Routing the safe-omission recommendation now works, but the adjudicator still overrides the policy. This is not a missing correction-call issue. |
| 50 `e815e426` | Generic `unto order or assigns` terms cause `negotiability=negotiable`; accounting reviewer also waives the documented local/total mass conflict | Generic contractual wording does not establish this shipment's negotiability. Conflicting masses require an explicit precedence decision or retained hold | Boilerplate/policy false pass. The map records the mass contradiction, but its uncertainty is not binding on acceptance. |

For 50, the printed totals really are gross 35,840 kg and net 35,000 kg; the local
rows print gross 17,500 kg twice and net 17,920 `KS` twice. The total values are
not invented. What is missing is an agreed rule allowing that conflict to be
waived. R3 held it; R4 selecting one representation is not independently proven
source reconciliation.

This means the improved 43-pass count must not be described as 43 gold documents.
Nor does subtracting these five automatically certify the other 38. The actual
completed deliverable is a frozen run with explicit tested properties and a
source-adjudicated residual inventory.

### Seven held cases and why

| Source | Final hold | What source/receipt review established |
|---|---|---|
| 09 `5242647c` | Corrector output validation fails | It emits hyphenated placement ID `MCLU507449-8` instead of canonical `MCLU5074498`. The invalid proposal is rejected and prior valid candidate preserved. Align the source-ID normalization contract across mapper, equipment, placements and reviewer; do not relax foreign-key validation. |
| 16 `1d51aca4` | Conflicting party VAT references across document copies | `298712164` versus `299712164` is a real competing source value. Voyage conflict is safely omitted. Separately, a route correction infers UNITED STATES from `CHARLOTTE, NC`; that exceeds the current explicit-country rule and must not be called independently validated just because the route reviewer passes. |
| 17 `9c68fe94` | Stray Bulgaria near consignee caption versus Egypt in postal block | Both cues genuinely appear in the PDF/OCR. Cargo/seal/route repairs are complete; the single country interpretation remains a source decision. PDF inspection confirms Leixoes also overlaps the delivery heading, separate from its dated-at occurrence. |
| 25 `bbf194b6` | Missing 18,780-kg goods mass plus address punctuation findings | One goods item accounts for the full shipment. Wave 1 adds its printed total; wave 2 removes it as “shipment-only”; final review asks to add it again. This is reproducible policy oscillation, not absent OCR. Comma-boundary reviewers also discover further formatting preferences on successive waves. |
| 38 `a51616d3` | Hypothetical bags-within-cartons relationship | The source supports 1,194 cartons and does not establish containment from the separate `BAGS` word. The reviewer still holds a possible hierarchy despite the explicit no-inferred-containment instruction. This is an unnecessary hold under the current rule. |
| 41 `ff7f0853` | Date ambiguity and missing charter-party reference | The ambiguous normalized issue date was successfully removed. Final review still describes that resolved absence as unresolved; it additionally finds a supported `CHARTER PARTY Dated 07/05/2024` reference too late for another bounded wave. Distinguish unresolved source interpretation from a valid absent target. |
| 49 `da651f2f` | Repeated B/L ownership and freight findings | PDF places MAHER under B/L No. as well as in the cargo marking; reviewer repeatedly contradicts the layout. `KOS` mass unit remains unsupported rather than normalized speculatively. Wave 1 adds prepaid freight; wave 2 clears it while correcting other facts without a matching decision. Final review catches that omission, so the document is held. |

Two further interactions matter: the freight reviewers disagree about the meaning
of `FREIGHT ADVANCE` in 41 versus 49; this needs one policy, not inference from
reviewer consensus. And 49 demonstrates **unexplained collateral deletion within
an authorized section**, not a dictionary-merge race: the second corrector's raw
response itself sets freight to null. Section-level replacement validation catches
structural and source-addition defects, not every supported fact that disappears.

### Next bounded work, before scaling

1. **Make corrections preserve established facts.** Require an explicit reason
   for every changed/deleted field outside the actual finding and its direct
   dependencies. Compare the proposed change against the latest candidate, not
   only the initial extraction. Previously accepted facts may be reconsidered,
   but cannot disappear without an explicit adjudication. Probe 49 and the
   add/remove oscillation in 25; do not increase the wave cap as a substitute.
2. **Resolve the small policy-priority set.** Typed customs/vessel identifiers
   versus generic column headings; ambiguous dates versus issuer locale; generic
   order boilerplate versus shipment declarations; one-product shipment totals;
   absence of containment evidence; postal separator/deduplication expectations.
   Keep these concise in the owning field definitions. Treat unresolved source
   interpretation with a valid omitted target separately from a faulty target.
3. **Close the acceptance gaps with counterexamples.** Replay clean/corrupt pairs
   for these policies, including a reference wrongly placed in marks, locale-only
   date normalization, boilerplate negotiability, and deletion of an unrelated
   accepted fact. The final check must adjudicate from source and contract, not
   treat the previous correction explanation as authority. Preserve genuine
   source conflicts (16/17/50) for explicit adjudication, not endless retries.

These are recommendations from the completed R4 analysis, not unimplemented work
silently claimed as finished. The approved R4 code changes, pilots, frozen rerun,
tests, mutation probes and independent review are complete. **Do not launch bulk
gold publication on the current agent-pass status.** The numerical/grouping fixes
are supported by concrete source controls; the remaining acceptance failures have
named sources and reproducible receipts rather than an unspecified new backlog.

## Cost and artifacts

| Operation | Estimated USD |
|---|---:|
| Six-document pilot | 0.173173195 |
| Three-document confirmation | 0.100122150 |
| Full 50-document rerun | 1.111740755 |
| Six semantic mutation calls | 0.008240655 |
| **Total this experiment** | **1.393276755** |

These are receipt-based estimates under the same pricing assumptions as R3:
$0.10/M ordinary input, $0.01/M cache read, $0.125/M cache write and $0.50/M output.
They are not a reconciled provider invoice. Output counts include reasoning;
reasoning/cache counters are not added a second time. Full-run usage is 5,892,255
input tokens and 1,134,539 output tokens (945,101 reasoning tokens within output).
At this deliberately difficult batch's mix, the full-flow estimate is $0.02223
per document, or approximately $25.73 for 1,157; that is an extrapolation, not a
bulk-run approval or a promise about the remaining document mix.

Implementation files: `direct.py` (map/target transaction and section validation),
`direct_cargo.py` (accounting scope and own-token checks), `direct_models.py`
(review/actionability descriptions), V7 label schema (field semantics and inherited
reference constraint), four direct-agent prompt files, and targeted flow tests.
No public target field was added or removed in this pass. Historical saved maps
are analyzed with their saved implementation; no backward-compatibility fallback
was added to the live map contract.

Primary evidence:

- [Frozen full run, selection, config and implementation snapshot](../artifacts/kie-labeling/direct-full-20261004-luna-high50-r4/)
- [Aggregate receipts and review inventory](../artifacts/kie-labeling/direct-full-20261004-luna-high50-r4/analysis.json)
- [Original/expanded paired controls](../artifacts/kie-labeling/direct-full-20261004-luna-high50-r4/paired-controls.json)
- [Additional source controls](../artifacts/kie-labeling/direct-full-20261004-luna-high50-r4/supplemental-checks.json)
- [Offline mutation/benchmark results](../artifacts/kie-labeling/direct-full-20261004-luna-high50-r4/offline-validation.json)
- [Live clean/corrupt probes](../artifacts/kie-labeling/direct-full-20261004-luna-high50-r4/semantic-mutations/summary.json)
- [Individual OCR/target/review comparisons](../artifacts/kie-labeling/direct-full-20261004-luna-high50-r4/comparisons/)
- [Timing and statuses](../artifacts/kie-labeling/direct-full-20261004-luna-high50-r4/timing.json)
