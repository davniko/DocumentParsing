# Direct labeling R3: whole-PDF cargo review and source-first accounting

## Scope and source of evidence

User-authorized implementation and paired rerun on the same 50 real OCR/PDF
pairs, plus bounded diagnostics. This is an isolated labeling experiment: no
production dataset, OCR, PDF, synthesis template, or training run is modified.
Historical targets are comparison material, not gold labels or agent input for
the fresh 50-document run. The targeted probes explicitly use frozen R2 initial
candidates to test known defects and sound controls.

The preceding diagnosis is in
[the R2 report](kie-direct-full-flow-50-new-2026-10-03.md#follow-up-diagnosis-cargo-semantics-association-graphs-and-reviewer-failures).
Public verdict explanations are analyzed, not hidden model reasoning.

## Implemented changes

1. **Whole PDF from the first cargo call.** The source mapper, relationship
   reviewer, and cargo-containing correction calls receive the complete original
   PDF as native file input. No cargo-page selection or three-page truncation.
   Request receipts record its SHA-256; the live capture verifies the decoded
   file against the original. The separate plain OCR is the sole value authority;
   PDF is explicitly for row ownership, layout and page continuation. Missing PDF
   is an explicit hold in the API; the refinement CLI requires it. Files beyond
   the provider size limit fail explicitly.
2. **Candidate-blind cargo source map.** A small typed internal product/portion
   representation separates product identity, inner shipment packages, outer
   packing, retail capacity, local portions, shared totals and shipment totals.
   It preserves unknown memberships/counts and printed measures without forcing
   inferred splits. This is review context, not a new extraction/training schema.
   It is constructed once per document and reused across correction waves.
3. **Actually narrow relationship review.** Its context contains the map and
   candidate descriptions/HS identities, packages and placements, not the full
   goods schema, marking fields, handling fields or party labels. Finding fields
   are constrained to grouping, package level/count and placement. General cargo
   review has a complementary fact-only finding schema. Both still see full OCR.
4. **PDF-leak guards.** Equipment/seal and placement IDs must have literal OCR
   support under presentation-separator rules. Cargo amounts must print or be an
   allowed exact same-level product total from the source map. A shared/unknown
   portion cannot license a partial product total. New unsupported facts prevent
   the affected correction group from committing; known good sibling sections
   remain preserved. Presence is not proof of ownership or correct boundaries.
5. **Operational isolation.** Invalid reviewer/corrector output produces an
   explicit component hold, not a fabricated pass or loss of independent completed
   reviews. Provider/network failures still surface. Route review receives page 1
   upfront because misleading OCR headings do not reliably trigger layout requests.
6. **Response budget.** A hard cargo probe exhausted the previous 16,384-token
   ceiling before producing a response. Configuration now allows 32,768 tokens
   including reasoning and a 300-second timeout. No automatic retries; actual
   tokens, not the configured ceiling, determine the estimate.

Native PDF behavior was checked against the installed PydanticAI request path and
[OpenAI file-input documentation](https://developers.openai.com/api/docs/guides/file-inputs):
the provider receives document text plus page images. Input-file payload hashes
are verified in the actual request, with base64 redacted from saved wire files.

## Bounded probes before/alongside the full rerun

Artifacts: [cargo probes](../artifacts/kie-labeling/direct-cargo-fullpdf-20261003-r3).
Luna/high except the explicitly named xhigh probe. Costs below use the same
recorded rate assumptions as R2, not a claim of reconciled provider invoices.

| Probe | Requests | Time | Estimated cost | Observation |
|---|---:|---:|---:|---|
| Twelve difficult/sound candidates, high | 23 | 164.35 s | $0.06665842 | Nine relationship passes, one correction, two holds; one hold was output exhaustion. These are agent verdicts, not twelve gold documents. |
| Cookware, high, larger response ceiling | 2 | 196.06 s | $0.01177806 | Produced a map, but retained the wrong product/row shift. Larger budget alone did not fix ownership. |
| Cookware, xhigh | 2 | 170.32 s | $0.011752475 | Correct cross-page product/count assignments; map still included adjacent tare digit in one container ID. Comparator used the proper candidate ID. Full correction checked separately. |

The first probe correctly kept PDF-only IDs/amounts out for sources 20, 25 and 26,
and retained the intended package levels for boxes/pallets and drums/pallets.
It preserved unknown shares in the mixed-panel document. It also demonstrated
limitations of the map itself: source 31 included `/40HC` in a container string;
some descriptions absorbed classification/marking wording. Descriptions and ID
field definitions were clarified generally before freezing R3, without injecting
source-specific answers. The map is deliberately not treated as authoritative.

The cheese source 38 was inspected against its PDF: it prints both `BAGS` and
`1194/19KG (APPROX) CARTONS`, without stating their containment relationship. A
hold about packing interpretation is reasonable. Inferring an inner/outer
hierarchy from the words alone would not resolve the source ambiguity.

## Offline validation

The initial relevant suite passed 107 tests. After the post-batch guard fixes,
the final suite passes **110 tests** (20.26 seconds).
It covers native structured requests, complete two-page PDF bytes, disjoint review
responsibilities, candidate-blind mapping, OCR-absent correction rejection, exact
totals versus incomplete/shared totals, unknown package counts, invalid reference
graphs, preservation of independent sections, bounded corrections and CLI behavior.
Ruff and `git diff --check` pass for the changed code.

[Fault injection and local timing](../artifacts/kie-labeling/direct-full-20261003-luna-high50-r3/offline-validation.json):
all **150/150** deliberately absent equipment IDs, placement IDs and cargo amounts
were flagged across the 50 OCR sources. This measures absence rejection only,
not semantic precision/recall or the ability to detect a value under the wrong row.

Median time for the old HS/transport gates was **2.713 ms per 50 documents**;
expanded gates including equipment and numeric checks were **14.145 ms**. The
absolute extra cost is **11.432 ms per 50**, about 0.23 ms/document. Traced peak
Python allocations rose from 3,154 to 94,046 bytes. This is negligible beside API
latency, but is reported rather than misrepresented as a speed improvement.

## Frozen full-panel rerun

[Run directory](../artifacts/kie-labeling/direct-full-20261003-luna-high50-r3).
The selection, complete source hashes, implementation snapshots, prompts, schemas,
request/response receipts, structured explanations and redacted wire bodies are
preserved. Fresh extraction uses Luna/high; review and correction use Luna/high;
global request concurrency is eight. At most two correction waves; no unchanged
finding retry.

### Completed results

| Metric | Frozen R2 | Frozen R3 |
|---|---:|---:|
| Documents | 50 | 50 |
| Application-valid final targets | 48 | 50 |
| Agent-passing candidates | 37 | 39 |
| Held final candidates | 11 | 11 |
| Failed document operations | 2 | 0 |
| Paid requests, all attempts | 558 | 632 |
| Estimated cost | $0.78874887 | $0.953700785 |
| Elapsed | 1,261.79 s | 1,489.57 s |
| Peak process RSS | 378,816 KiB | 446,496 KiB |

R3 has zero actual-wire JSON-schema violations across the recorded replies and
zero unknown-billing calls. Two initial extractions violated application-level
constraints despite complying with native JSON Schema; both were subsequently
recovered through the draft-review path. All 50 original source hashes remained
unchanged. PDF wire verification covered **50 source-map calls, 69 relationship
reviews and 19 cargo correction calls**, each with the complete original file.

This is a quality/control improvement on the tested properties, **not a speed or
cost improvement**: requests rose 13.3%, cost 20.9%, wall time 18.1%, and peak RSS
17.9%. The independent source-map stage alone cost $0.13981. These timings also
include provider latency and bounded concurrent diagnostic requests, so they are
not a controlled infrastructure benchmark. Whole-PDF context is user-requested;
the additional mapping call has not established that it pays for itself on every
simple document. The flow remains experimental, not promoted for bulk gold
publication.

### Independent controls, not a proxy for field F1

[Paired results](../artifacts/kie-labeling/direct-full-20261003-luna-high50-r3/paired-controls.json)
contain 42 named source-adjudicated checks across 27 documents. They are correlated
diagnostic properties, not 42 independent documents or an exhaustive accuracy score.

- Common available final controls: **R2 32/39; R3 34/39**.
- Three additional controls belong to the two previously failed operations; R3
  recovers all three. Across the whole panel R3 is **37/42**.
- All five R3 control failures belong to the **held cookware source 13**. None of
  the 29 tested properties across 22 agent-passing documents fails these checks.
  This does not certify the untested fields or the other passing documents.
- R3's fresh initial extraction was already 37/42 on these checks. Thus the final
  control score is **not evidence that review improved all of those properties**;
  preventing previously observed repair regressions is part of the gain.

Concrete successful outcomes:

- Sources 20/25: supported goods retained, PDF-only equipment IDs omitted; both
  now finish instead of failing refinement.
- Source 26: 1,845 cartons and two known memberships retained without importing
  PDF-only 916/929 local counts, mass or volume.
- Source 17: Damietta now occupies discharge rather than loading. Other party
  and grouping questions remain held.
- Source 44: both product codes remain in description.
- Sources 02/27/28/31/35/39: inner bags/boxes/drums, legitimate pallet-only goods,
  shared products and known/unknown per-container quantities retain their intended
  behavior. A hold in another section does not invalidate the tested cargo fact.
- Source 43: shared 112 cartons are not apportioned between the panel products.
  Its separate known 119/32-carton rows and placements are retained. The first
  version of our new checker incorrectly required every goods package count to
  be absent; the actual schema explicitly permits source-distinct package rows.
  The checker was corrected to allow those rows while rejecting a single partial
  total of 151. No target was changed to make this check pass.

### Deliberate semantic corruptions

[Paired mutation probe](../artifacts/kie-labeling/direct-cargo-fullpdf-20261003-r3/semantic-mutations/summary.json):
six calls, **$0.008461925**. The reviewer accepted all three clean controls and
rejected all three modified candidates:

1. Three invented equal 20-bag allocations from an unsplit 60-bag total.
2. Two container counts exchanged while preserving the exact shipment total.
3. Five outer pallets substituted for 54 inner boxes.

This falsification probe demonstrates those mechanisms on these controls. It
does not establish universal relationship accuracy.

## Held cases: exact remaining scope in the frozen batch

| Source | Held component and observed reason | Interpretation / next intervention |
|---|---|---|
| 04 | Per-product 80/60-drum counts derived from net mass / 25-kg capacity and corroborating drum ranges | The current target description permits printed counts or complete sums, not a general division/range derivation. Resolve that policy explicitly; do not weaken a numeric guard silently. |
| 13 | Product ownership shifted across attachment rows, then totals disagree with independent map | Known model error. High still fails; xhigh produced a correct proposal, discussed below. |
| 17 | Two distinct machines share one count of two unpacked units; reviewers disagree about grouping. Consignee also has stray `Bulgaria` near its caption versus Egypt in its postal block | Grouping consistency plus a real printed competing-country cue. PDF inspection confirms both the stray country and the two-machine block. Route is repaired. |
| 28 | Reviewer reads the top identity caption as another consignee | Reviewer/visual-reading failure, not proof of two actual consignees: source PDF identifies the upper party as consignor. RATHI should be shipper; WAFA consignee. R2 recovered this; R3 holds it. |
| 32 | `***045W` near container text treated as possible equipment | Continuation-ownership issue; earlier analysis links the marker to transport. Do not invent a second container or call this confirmed irreducible ambiguity. |
| 33 | Carrier reference `21137888` omitted after correcting the B/L identifier | Actionable completeness/transfer miss. Correct B/L is `HLCUSYD250337760`; preserve the separate carrier reference in references. It was only flagged at the final bounded review. |
| 38 | BAGS versus CARTONS interpretation; correction/re-review alternate | Source does not state containment. Inspect the actual packaging wording and choose a consistent target policy; don't infer hierarchy just from two package names. |
| 41 | `03/06/2024` date order | Genuine numeric date-order ambiguity under the current policy. |
| 47 | `10-04-2025` date order | Genuine numeric date-order ambiguity; cargo/temperature controls pass. |
| 49 | `227 910 KOS`: space-grouped amount plus unclear unit | Numeric space-group recognition was fixed after the batch; the `KOS` unit question remains distinct. No mass unit was invented. |
| 50 | Printed overall gross/net totals conflict with per-container gross/net sums | Genuine source contradiction, not resolved by arithmetic alone. |

The table separates genuine source/policy questions from correctable reviewer
mistakes. Agent pass/hold counts alone cannot make that distinction.

## Post-batch guard fixes and xhigh replay

Three local guard corrections were made and tested after the frozen batch:

- Comparison-only Unicode normalization handles `GRANİTE` versus `GRANITE` without
  modifying labels. If several product identities collide after normalization,
  no arithmetic exemption is granted.
- Conventional three-digit space/NBSP/thin-space grouping recognizes `227 910`;
  arbitrary sequences such as `14 6 25` are not concatenated. This remains an
  absence screen, not an ownership certificate.
- A previously flagged unsupported value cannot authorize a newly unsupported
  field merely by sharing the same diagnostic text. Prior/new findings are keyed
  by field and explanation.

The first xhigh refinement replay cost **$0.026021765** in 11 calls / 200.32 seconds;
the Unicode-corrected replay cost **$0.03603983** in 14 calls / 430.66 seconds. Both
reuse the paid candidate-blind xhigh map with explicit provenance; neither was
silently included in the frozen R3 metrics.

The latter corrector produced the correct **five products, all nine quantified
product/container edges, and all five product totals**. Its public decisions
explicitly reject the fact reviewer's shifted-row proposals. All six named
cookware controls pass on that proposal. It is preserved as
[verified cargo proposal](../artifacts/kie-labeling/direct-cargo-fullpdf-20261003-r3/xhigh-refinement-unicode-fix/verified-cargo-proposal.json)
with a [scoped audit](../artifacts/kie-labeling/direct-cargo-fullpdf-20261003-r3/xhigh-refinement-unicode-fix/proposal-audit.json).
Only the cargo properties named in that audit are verified; the enclosing
candidate is not a gold-document publication.

**The automated replay remains held.** Its source map parsed dot-grouped kilogram
amounts as decimals: `7.740` became 7.74, so granite masses summed to 30.354 rather
than 30,354. The corrector interpreted them correctly. After the Unicode fix,
counts now satisfy the gate but those derived mass totals still conflict with
the map. This is an explicit, reproducible map-normalization failure, not evidence
that the correct proposed cargo is unsupported by the document. Neither a blind
multiply-by-1,000 patch nor disabling the guard is an acceptable remedy.

Another unresolved design issue is semantic overlap: although the two review
schemas have disjoint fields, the fact reviewer still independently interprets
row ownership when assigning masses. It can therefore disagree with the graph
reviewer about the same row. A focused next experiment should unify row-owned
quantity/measure interpretation and test source-map numeric normalization, rather
than add another broad reviewer or another retry wave. Xhigh helped but did not
make every intermediate representation reliable automatically.

## Accounting and handoff

All paid probes plus the full rerun total **$1.11441326 estimated**. Failed/exhausted
requests with recorded usage are included. No production labels were rewritten
and no training or bulk annotation was launched. The requested implementation
and rerun are complete; unresolved semantic/policy limitations are exposed above,
not counted as successful repairs or silently accepted gold.

Primary receipts:
[batch analysis](../artifacts/kie-labeling/direct-full-20261003-luna-high50-r3/analysis.json),
[timing](../artifacts/kie-labeling/direct-full-20261003-luna-high50-r3/timing.json),
[paired controls](../artifacts/kie-labeling/direct-full-20261003-luna-high50-r3/paired-controls.json),
[individual comparisons](../artifacts/kie-labeling/direct-full-20261003-luna-high50-r3/comparisons).
