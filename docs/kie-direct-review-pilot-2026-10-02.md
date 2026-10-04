# Section-review, correction and re-review pilot — 2026-10-02

## Outcome

The requested **20-document experiment is complete**, including all five review
sections, one bounded correction wave, and re-review of affected dependency
groups. It used the exact second-pass silver outputs, including the two
application-invalid drafts, rather than selecting only successful extractions.

The reviewer stage is useful, particularly for goods grouping and allocations,
but this experiment **rejects treating reviewer agreement as an automatic gold
acceptance gate**. It both repaired important errors and introduced new ones.
This is a measured experimental conclusion, not an unfinished bulk repair.
No production dataset, historical label, training configuration or OCR was changed.

| Measurement | Result |
|---|---:|
| Documents | 20: 9 train / 11 validation; same frozen panel |
| Initial section decisions | 100 |
| Initial decisions | 77 pass / 21 corrections needed / 2 unresolved |
| Initial findings | 24 |
| Final section decisions | 89 pass / 8 corrections needed / 3 unresolved |
| Agent-level whole-document outcomes | 11 all-section passes; 9 held |
| Application-valid resulting targets | 18; the two bad identifiers remain blocked |
| Documents with changed candidate labels | 13 |
| API requests, all successful | 172 |
| Actual native-schema-conforming responses | 172/172 |
| API experiment wall time | 237.137 seconds |
| Peak concurrent requests | 8, globally across documents/stages |
| Median request latency | 7.753 seconds |
| Review-cycle estimated cost | **$0.1911885**, about **$0.00956/document** |
| Silver extraction plus this cycle | **$0.251062875** for 20, about **$0.01255/document** |

The 89 passing sections and 11 passing documents above are **the agents' votes**,
not independently measured semantic accuracy. Known defects remain within their
passing set. No field-level F1/accuracy is invented from incomplete gold labels.

## Inputs and separation of responsibility

Each request contains the complete literal OCR, not a JSON line inventory.
Candidate labels and field definitions are separate JSON context:

| Reviewer | Editable/reviewed candidate fields | Read-only companion labels |
|---|---|---|
| Parties | `parties` | None |
| Route/transport | `route`, `transport` | None |
| Metadata/freight | B/L identifiers, dates, negotiability, issue place, freight, references | None |
| Equipment | `containerInformation` | `goodsItemDetails` |
| Cargo | `goodsItemDetails`, including packages and placements | `containerInformation` |

Cargo and equipment need companion identifiers to check membership and dangling
references. They cannot replace the companion section. No reviewer received the
previous audit, known-error list, historical labels, or an expected answer.

All stages use **PydanticAI + OpenAI Responses `gpt-6-luna`, high reasoning**,
native strict structured output, 16,384 maximum output/reasoning tokens, and zero
SDK/model retries. Schemas and prompt meanings were not tuned during the live
batch. PDF access was requested by reviewers/correctors, not supplied initially.
Source PDFs were resolved for all 20 documents using original work-item metadata
and verified file hashes. Values remain OCR-grounded; PDFs only clarify ownership.

Requests break down into:

- 111 initial-review calls: 100 initial requests and 11 PDF follow-ups.
- 22 correction calls: 21 section corrections and one PDF follow-up.
- 39 re-review calls: 37 section reviews and two PDF follow-ups.

The two application-invalid drafts were deliberately reviewable without weakening
the final target schema. Invalid final documents have `reviewed-draft.json`, not
`target.json`; their independently corrected sections are preserved for inspection.

## Independent checks, rather than trusting pass votes

The previous extraction audit supplied a small source-reviewed sentinel panel:
**nine known semantic defects and eight correct controls**. These were kept out
of agent inputs. The checks assess particular facts, not whole-document goldness.

| Known issue/control | Initial reviewer result | Final candidate |
|---|---|---|
| `0600699f`: third shipper telephone missing | Detected | Restored |
| `0701d457`: yarn specs incorrectly paired one-to-one with container rows | Detected using PDF layout | Correct shared goods and placements |
| `b25a3616`: partial four-container sums treated as whole five-container totals | Detected | Incorrect totals removed; re-review wrongly asks to restore them |
| `4d49ad47`: omitted carrier seal alongside customs seal | Detected | Both retained |
| `c0f529a4`: Suez/Alexandria alternative addresses concatenated | Detected as ambiguity | Held; no arbitrary primary selection |
| `b25a3616`: truncated proper port name | Missed | Still shortened |
| `439f6102`: NASR CITY postal zone in company name | Missed, including PDF-assisted party review | Still in name rather than address |
| `ab905fd4`: delivery-agent EGYPT continuation omitted | Missed | Still omitted |
| `66d5d7f6`: same two-address problem in another document | Missed | All-section pass despite concatenation |

Thus **5/9 known semantic defects were detected; 4/9 were repaired**. The fifth
was appropriately held. The remaining four were missed. Eight previously correct
controls remained correct: postcode retention, tax references, carrier identity
without capital boilerplate, exclusion of TOTAL as a mark, five cargo-container
memberships, and the voyage/leg separation. Sentinel checks improved **8/17 →
12/17**, but do not cover newly introduced errors.

Neither equipment reviewer flagged the malformed identifiers in `99adb051` and
`298806fe`. **Python target validation blocked both documents**, even though the
latter received five agent passes. Native JSON-schema conformance does not execute
the application's Python-only identifier checks or establish source correctness.

## All 24 initial findings adjudicated

The independent review checked every initial finding against OCR and the current
field descriptions, all changed candidate fields, and nine manually viewed PDF
pages for disputed layout. The complete per-finding decisions are in
[adjudication.json](../artifacts/kie-labeling/direct-review-20261002-luna-high20/adjudication.json).

| Disposition | Findings | Meaning |
|---|---:|---|
| Supported issue and remedy | 12 | Includes phone, seals, cargo topology, unsupported units, handling, marks ownership, equipment categories and duplicate reference |
| Supported issue, imperfect proposed wording | 2 | Receipt/delivery ownership recovered, but proper names still shortened |
| Incorrect correction request | 6 | Three valid homepages removed; eBL file ref used as B/L; borrowed route country; guessed weight unit |
| Supported unresolved source conflict | 2 | Competing package counts; competing addresses |
| Requires ownership/policy resolution | 2 | Consignee/invoice text treated as marks; lone OCR Collect versus caption/selected declaration |

These judgments are explicit and reviewable; they are not a second model's vote
presented as ground truth. The two uncertain findings remain uncertain rather than
being counted as correct to improve a precision number.

### Strong cargo result: yarn shared across two containers

`0701d457` prints two 680-carton container rows followed by two yarn specifications
under the common cargo description. Neither OCR nor PDF allocates one spec to one
container. The silver draft had invented that allocation.

Before:

```text
Goods A: DTY 75D/72F …; 680 cartons; TRHU7658292 → 680
Goods B: DTY 100D/144F …; 680 cartons; DRYU9359817 → 680
```

After:

```text
One goods description containing both complete yarn specifications
Total: 1,360 cartons; 51,800 kg gross; 110 m³
Placements: TRHU7658292 → 680; DRYU9359817 → 680
```

The first HS code remains; removal of the redundant second goods row does not
remove that code from the shared goods. This is a real semantic improvement, not
merely passing structural validation.

Other useful results include preserving both `ML-IN0800111` and `BOLT50714395`,
restoring bonded-warehouse handling text, preserving inner drums rather than
outer pallets, recovering the IR shipping mark separately from a product code,
and removing unqualified-volume assumptions. Correct shared-paper allocations in
`2aa1d48b` and NAVIGO product placements in `66d5d7f6` stayed unchanged. The latter's
party defect is separate from its correctly preserved cargo.

### Six confirmed introduced regressions

1. `1c402851`: `billOfLadingNumber = bee-g` was introduced from `File details / EBL
   REF-bee-g`. PDF context confirms it is a platform file reference. The actual
   carrier B/L number is absent from OCR and cannot be supplied from the PDF.
   Metadata re-review accepted the wrong value.
2. `0701d457`: removed `www.one-line.com`.
3. `c9c96562`: removed `HTTPS://WWW.BLUEANCHORLINE.COM`.
4. `6fe428c6`: removed `www.cma-cgm.com`.
   These are owned carrier homepages, which the field description explicitly
   allows even in company terms. Reviewers incorrectly generalized the exclusion
   of unrelated legal-clause links. Correctors followed them; re-review missed all three.
5. `99adb051`: added `CHINA` to receipt location from the shipper address. Re-review
   caught the new wrong-owner value.
6. `b0da3b37`: added a kilogram unit to unitless `NET WEIGHT: 11049.500`, borrowing
   the preceding gross-weight unit. Re-review caught it.

Only **two of these six** were caught by the final reviewer. These are a confirmed
minimum of clear regressions, not a claim that every remaining boundary is settled.
None entered a production dataset.

### Re-review can also manufacture problems

- After `b25a3616` correctly lost the incomplete 83,840 kg / 163.76 m³ totals,
  the re-review requested those same partial totals again. The final candidate
  retains the correct omissions; the false flag is not applied automatically.
- `ab905fd4`'s corrector added `MARINE POLLUTANT` to the description. The final
  reviewer said it was missing **although that exact string was in its candidate
  input**. The saved request proves it saw the updated target, not stale labels.
- `99adb051` received opposing freight-payment decisions. OCR has one `Collect`
  after payment captions; the PDF has both a caption and `FREIGHT COLLECT`. This
  requires ownership adjudication, not blind oscillation between null and collect.

These cases demonstrate why an unrestricted “keep correcting until all agents
agree” loop would be a poor next step.

## Implementation changes and a discovered flow defect

Before this experiment:

- Narrowed party and metadata candidate context to only their assigned fields.
  Cargo/equipment companions are explicitly read-only.
- Added a shared request semaphore: eight **requests**, not eight documents each
  making eight calls.
- Allowed review of raw V7 drafts rejected by application validation, while
  requiring final full validation before a valid target export.
- Validate corrections by dependency group so an existing invalid container ID
  does not erase a sound party correction. Final whole-target checks still apply.
- Persist initial reviews separately from final reviews for exact comparisons.

The experiment then exposed a definite orchestration defect: requested PDF pages
were supplied to one call but lost before correction/re-review. In `b25a3616`,
the first reviewer and corrector used page 1 to establish receipt/delivery boxes;
the final reviewer no longer had the image and disputed those roles based on the
misordered OCR headings. This is distinct from the wrong partial-total judgment.

**Fixed after freezing the experiment:** requested page numbers are retained per
section, and later calls in that section receive the same layout context. Other
sections do not automatically receive those pages. Only page numbers are retained
in memory; bounded images are rendered on demand. Each call still has at most one
new layout-request follow-up. This does not send prior reviewers' verdicts to the
independent re-reviewer, and does not grant permission for PDF-only label values.

The original 172 calls and results are immutable. The exact executed code is saved
under `implementation-snapshot/`. The post-experiment layout fix passed the actual
PydanticAI offline flow test (image counts **0 → 1 → 1 → 1** across initial request,
assisted review, correction and re-review). Its semantic benefit is not included
in the baseline scores; it has not received another paid full-batch run.

## Validation and cost accounting

- Full request-body inspection verifies literal complete OCR, Luna/high,
  strict native schema and section scope. `jsonschema` validates all **172 raw
  responses against the actual provider wire schemas**, not a hand-built schema.
- All 172 calls completed with usage; no hidden retry, refusal, schema failure or
  unpriced failed request occurred.
- Frozen OCR, source PDFs/work-item provenance and silver-response hashes match.
- Every exported target passes application validation. Invalid whole drafts are
  saved distinctly; agent pass votes cannot override identifier/relation checks.
- **520 targeted/integration tests passed in 15.65 seconds; Ruff and mypy passed.**
  Tests cover request scoping, shared concurrency, invalid-draft
  preservation, dependency validation, native SDK serialization, PDF requests,
  and layout continuity, alongside broader schema/training integration suites.
- Local context-construction benchmark: 100 contexts took **0.1447 s before /
  0.1479 s after** layout continuity, with **703,035 / 648,537 bytes** peak traced
  allocation. The 3.2 ms aggregate difference is negligible relative to API
  latency; contexts are byte-identical. This benchmark does not measure network
  time or PDF rendering. Keeping pages available can increase image input on a
  later call, but avoids repeated layout-request round trips and loss of evidence.

Usage for the full live cycle:

| Usage | Tokens |
|---|---:|
| Input | 1,012,321 |
| Cache-read subset | 92,215 |
| Cache-write subset | 919,590 |
| Output, including reasoning | 150,532 |
| Reasoning subset | 138,530 |

At the same usage-estimation rates as the preceding pilot—USD per million:
ordinary input .10, cache read .01, cache write .125, output .50—reviewer calls cost
**$0.16383234**, correction calls **$0.02735616**, total **$0.1911885**. Cache subsets
are subtracted from input; reasoning is not charged twice. These are
usage-derived estimates, not an invoice. No dataset-wide price or gold-acceptance
rate is extrapolated from this small challenge-enriched panel.

## What this tells us to do next

1. Keep the narrow section architecture: it demonstrably repaired cargo topology
   and specific omissions at low cost. Keep deterministic structural validation
   separate from model verdicts; the identifier cases show why it is necessary.
2. Use the counterexamples above as a **frozen reviewer test panel**, including
   correct controls. Do not silently convert these pilot examples into training
   gold or keep rerunning the same panel until a favorable score appears.
3. Test a narrowly defined reviewer revision on the failure mechanisms:
   owned homepage versus unrelated URL; true B/L versus platform file reference;
   unitless measurements; repeated addresses and name/address boundaries;
   complete versus partial allocations. The current field descriptions already
   express several of these rules, so merely appending more prose is not enough.
4. For correction acceptance, explicitly compare the **actual proposed change**
   against its source and contract and ask whether a previously sound fact was
   removed. This would target the six observed regressions without requiring an
   evidence record for every correct scalar. It is a proposed experiment, not an
   implemented guarantee or an invitation to an unlimited review loop.
5. Give cargo reviews a compact check of product identity, quantified product
   rows, container portions and completeness of sums. PDF layout is useful here;
   it must remain available through re-review. Do not introduce a new training
   relation schema based solely on this pilot.

No extra full batch, bulk relabeling, synthesis or training has been launched.

## Reproducible artifacts

Everything is under
[`artifacts/kie-labeling/direct-review-20261002-luna-high20/`](../artifacts/kie-labeling/direct-review-20261002-luna-high20):

- `selection.json`: exact panel, input/provenance hashes and executed-code hashes.
- `inputs/`: unchanged silver answers, including the two rejected drafts.
- `runs/`: OCR, schemas, prompts, raw requests/responses, usage, reviews and candidates.
- `wire/` within each run: actual provider request bodies; credentials never logged;
  image data replaced by hashes in these debug receipts.
- `analysis.json`, `adjudication.json`: measured results and independent decisions.
- `comparisons/`: per-document OCR, before/after labels, all reviews and field diffs.
- `manual-pages/`: independently inspected PDF renderings, used only for layout.
- `implementation-snapshot/`: exact code used for the measured live baseline.
- `run.py`, `analyze.py`, `adjudicate.py`, `benchmark.py`: bounded experiment and audit.

The API documentation check informed the strict-output boundary: constrained JSON
does not guarantee correct ownership, completeness or custom Python invariants.
See [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
and [PydanticAI output](https://pydantic.dev/docs/ai/core-concepts/output/).
