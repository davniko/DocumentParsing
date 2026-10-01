# Party-address dataset repair: evidence, failure history, and acceptance audit

Date: 2026-10-01. Scope: the GROUND-015 working corpus and its address-repair
derivatives. This is the visible audit base for continuation, not a dataset
quality certificate or permission to train. Historical reports remain intact.

Latest execution: [section 14](#14-controlled-dataset-execution-after-the-150-document-probe)
records **782 applied party-address corrections in 735 documents**, a pinned
archive, full-corpus screening and exact replay validation. This is an actual
working-dataset update, not another in-memory-only pilot. It is **not** a claim
that every remaining address or document has been semantically signed off.

## 1. Executive assessment

The task is to make each party's input postal text and extraction label agree
with a complete, correctly owned postal address, without damaging anything
else. The intended target is `addressLine`, including printed city/country,
with separate city and country targets retained when supported. The operation
does not require geocoding, deliverable streets, or postcode-to-city inference.

Failures occurred at three distinct levels:

1. Historical label projection removed city/country from `address`, including
   components inside an address followed by a postcode. This was an explicit
   older labeling convention, not proof that the renderer failed to generate
   them. It conflicts with the newly agreed full-address target.
2. Generation/rendering could put a full generated address into a mutable
   address surface while also printing separately bound city/country and
   retaining fixed/source-only postal fragments. Some inputs therefore contain
   duplicate, misplaced, interleaved, or stale address material.
3. Repair acceptance checked selected spans, edit replay, or model judgments
   without a complete independently established inventory of postal content.
   Some repairs omitted content, retained faulty input, or introduced defects.

The central lesson is **not** that this dataset is unverifiable. It is that
exact execution of a proposed edit is different from a correct interpretation
of the entire party block. A reviewer saying `accept`, a valid schema, a hash,
and a label appearing somewhere in a document do not close that gap.

No universal automatic semantic-correctness guarantee has been demonstrated.
An exact checker can enforce an approved ownership/content specification; it
cannot prove the specification's meaning merely by checking its hashes. This
distinction is a release requirement, not a disclaimer to append after a run.

This pass produced documentation, a read-only corpus census, and isolated
falsification experiments. It did **not** repair the corpus, launch inference,
modify training, or ship a new production acceptance implementation. Results
and the failed stronger hypothesis are recorded in section 10.

## 2. Artifact map and preservation boundaries

Paths below are repository-relative. `G` denotes
`artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground015_cargo_text`.
`A` denotes `G/address-full-migration`.

| Artifact | Role and limitation |
| --- | --- |
| `artifacts/kie-training/datasets/mpci-bl-real1057-synthetic29910-recovered-mpci-aligned-v6-v2-ground015-working` | Older working data; not the final address repair. |
| Sibling with `_pre_address_archived` suffix | Preserved pre-address working copy. Do not overwrite. |
| `G/composed-candidate-v37-rehearsal` | Cargo-corrected candidate feeding address work; accepted train, validation, and independent cargo holds. |
| `A/repair-waves/work.sqlite`, `records` | Exact pre-address repair inputs, ownership/cards, proposals, reviews, requests, and billing evidence. Open read-only for audits. |
| `artifacts/kie-training/datasets/mpci-bl-ground015-addressline-v2` | Earlier applied derivative, superseded by V3; not fully approved. |
| `artifacts/kie-training/datasets/mpci-bl-ground015-addressline-v3` | Latest published address derivative; contains accepted and held partitions, **not** a whole-corpus sign-off. |
| `A/completion` and `completion_*` database tables | Later staged proposals and reviews. These do not constitute a published V4. |
| `G/party-source-role-probe` | Earlier pilots, role proposals, saved-render investigations, costs, and isolated outputs. |

Do not confuse a staged candidate, a mechanically valid proposal, a model
acceptance, and a published dataset. Do not infer success from filenames.
Source PDFs, historical synthesis artifacts, archives, and training history
remain immutable. Real OCR is immutable for this address pass. V37 already
contains two separately documented PDF-backed cargo/OCR derivatives; address
work must neither claim those as its edits nor undo them.

The address copy uses `6.1.0-address-line-repair`. Production schema/prompt and
training configuration migration are not established merely by that derivative
existing. The upstream synthesis catalog has not been comprehensively
recertified by these historical-dataset repairs.

## 3. Agreed target and input policy

### 3.1 Target

- `addressLine` is the complete printed postal address for one party occurrence,
  including buildings, units, plots, streets/sites, districts, administrative
  regions, city, country, and postcode **where actually printed**.
- Separate city/country fields remain. Their overlap with `addressLine` is
  intentional. No locality inference from a route, company incorporation
  statement, other party, or external lookup is permitted without the task's
  explicit evidence policy.
- Preserve printed wording, spelling, meaningful numbers, punctuation, and
  logical order. Physical line wrapping is not a semantic boundary. Fold line
  breaks to spaces; commas may separate genuinely distinct postal components,
  not split a wrapped street or building name indiscriminately.
- Exclude party names, headings, phones, fax, email, tax/customs/registration
  identifiers, business-purpose text, and nonpostal continuation/flavor markers.
- Explicit postal-caption values belong to the address; the caption itself
  normally does not. `P.O. Box 123` is a postal qualifier/value, not a tax ID.
- Select one complete occurrence, following explicit continuations. Do not
  concatenate two copies of the same address merely because both print.
- A source may print no street, postcode, country, or address at all. Completeness
  means completeness of available postal evidence, not filling every component.
- A `sameAs` relation must remain consistent with its referenced party; it is
  not permission to invent a separately printed address.

Example, nonpostal material excluded but all address content retained:

```text
10TH OF RAMADAN CITY - ZONE A6
BLOCK 16 18 20 INDUSTRIAL AREA B4
TAX ID :433392169
EGYPT
TEL: +201116600088
```

```json
{"addressLine":"10TH OF RAMADAN CITY - ZONE A6, BLOCK 16 18 20 INDUSTRIAL AREA B4, EGYPT"}
```

### 3.2 Input

Synthetic OCR can be corrected only within proven address-owned surfaces.
Preserve names, contacts, references, cargo, layout boundaries, and legitimate
repetitions. Real/validation OCR receives no address-pass rewrite.

Remove accidental duplicate postal components within one synthetic occurrence,
not all repeated words. Distinguish:

- `DAMIETTA ROAD` plus the city `DAMIETTA`: potentially legitimate.
- `NIGER STATE, NIGERIA`: distinct region/country, not duplicate spelling.
- A full consignee block repeated under notify or on another page: legitimate.
- `... EGYPT` followed by an accidental second country tail: a repair candidate.
- `TEN RAMADAN CITY` versus `10TH OF RAMADAN CITY`: an equivalence decision needs
  explicit evidence; approximate string similarity is not deletion authority.
- `PORT SAID FREE ZONE, PORT SAID`: named zone plus city, not automatically duplicate.

Synthetic geographic coherence must not contradict sampled/printed city and
country. A synthetic street need not be a real deliverable address. Do not
spend money proving real postcode geography or silently normalize a real source
to an external database. Previously approved removal of placeholder postcodes
must not be undone by an old generation artifact.

### 3.3 Repair precedence

Current approved corrections take precedence over original generation strings.
Use immutable baselines plus exact repair history, not “latest file wins.”
Restore saved facts only after checking party identity, source/layout version,
current edits, and all occurrences. Correct synthetic address facts can be
reused; a malformed bundle may be regenerated narrowly. Neither path may reset
other dataset repairs or silently restore source values.

## 4. Failure taxonomy, by layer

| ID | Failure mechanism | Manifestation / evidence | Required treatment |
| --- | --- | --- | --- |
| PA-01 | Label projection strips city/country | `12 EL MAHATTA STREET, DAMIETTA 34516` can lose the middle city while retaining the postcode. `training/address_projection.py` explicitly strips components and postal suffix/prefix combinations. | Derive full `addressLine` from final owned text; retire this projection for the new target. |
| PA-02 | Locality aliases defeat literal stripping/deduplication | `TEN RAMADAN CITY` inside a generated address and separately printed `10TH OF RAMADAN CITY`. | Resolve component identity, not global fuzzy word deletion. |
| PA-03 | Address and locality independently rendered | Full address already contains city/country; separate slots append them again. | One coherent postal bundle and explicit occurrences/dependencies. |
| PA-04 | Proportional/segmented insertion ignores meaning | Address words interleave with fixed city text or enter a source floor/unit/region slot. | Preserve semantic slot constraints; free wrapping cannot cross a typed boundary. |
| PA-05 | Fixed/source-only residue | Old district/country survives a route change, or generated street follows a fixed P.O.-box caption. | Inventory static gaps, auxiliary bindings, and caption meaning; update only with an approved contract. |
| PA-06 | Unowned or mixed spans | Company/name continuation, VAT, `File No.`, customs references, or phone suffix enters label; model drops real `Investors` from a building name. | Full postal/nonpostal partition, with exact boundary review where ambiguous. |
| PA-07 | Missing continuations | Postcode or street suffix occurs after contacts, after a blank line, or on another page (`FW>`, `+++`, `*`). | Full-document occurrence/continuation inventory, not fixed-size crops alone. |
| PA-08 | Repeated-block corruption | Whole later party blocks removed as duplicates; only first copy fixed; punctuation-different repeats missed. | Inventory each occurrence and preserve repeated documents/parties. |
| PA-09 | Missing-address coverage | Iterators select only existing `address` / `addressLine` fields. An omitted target is outside their universe. | Independently enumerate parties and explicit address-presence/absence dispositions. |
| PA-10 | Stale saved generation restores removed content | V3 `syn_tpl_be12d799032fa96a4d65ba84dda72a20b9fb71e0` restores `PIN-00000`; earlier input had already removed it. | Versioned lineage precedence; do not repair this solely with another special postcode regex. |
| PA-11 | Exact deletion damages separator | Removing locality text also consumes whitespace/comma and joins address to adjacent contact/reference. | Protected boundary segments and exact final reconstruction. |
| PA-12 | Shared-party conflicting edits | Consignee/notify aliases propose overlapping different replacements. | One joint edit plan; reject conflicts, never last-writer-wins. |
| PA-13 | Numeric retention too broad or too weak | Real postcode dropped; tax/phone number falsely treated as postal; set membership misses repeated-number multiplicity. | Typed ownership of numbers, occurrence/order checks rather than document-wide number sets. |
| PA-14 | Critic false positives / false negatives | Critic demands deliverable postcode; accepts good label while missing bad raw tail. | Exact question/evidence scope; semantic verdict is not an unconditional certificate. |
| PA-15 | Coordinate/provenance reconstruction defects | Old campaigns lack sidecars; diff alignment picks one of several identical occurrences. | Reconstruct exact render and edit lineage; ambiguous matching remains unresolved. |

### 4.1 Concrete inspected examples

**Correct bounded example:** V3 `train.jsonl` line 1044,
`syn_tpl_3261de425eb9b7b5c5016b85af20feea707c540b`. Shipper prints `UNIT 804,
8/F, HARBOUR VIEW / PLAZA, 22 HENNESSY ROAD, WAN CHAI, / HONG KONG`.
Its `addressLine` retains every postal word and number and excludes the company.
Consignee retains building 18, office 6, Port Said Free Zone and Port Said.
This direct inspection establishes these addresses, not all cargo or other fields.

**Confirmed stale restoration:** V3 `train.jsonl` line 25799, ID in PA-10,
source `doc_ed88ec5d74e496098a1b860b07a351be5c6f588a32e20c39df9bac0875e40945`.
Pre-address text has `UNIT 6, 18 KWAI CHEONG ROAD, / NEW TERRITORIES, /
HONG KONG, Z.T.E. 083`; saved address still contains `PIN-00000`.
Restoring that saved string revived a previously removed placeholder. `Z.T.E.
083` and the neighboring LUT reference also illustrate why preserving all
numbers without ownership is not a sufficient rule.

**Reported continuation repairs:** follow-up inspection recovered `1091 GM`,
`62815`, `11765`, and `FW>` continuation `9 DUKELSKA STREET, BENESOV CZECHIA`.
Receipts live in `A/followup-validation/inspected-corrections-v2.json`.

**Reported residual duplicate input:** completion review records examples such
as `SANTA FE, ARGENTINA, SANTA, SANTA FE / FE, ARGENTINA, ARGENTINA` and a
complete address followed by `P.O. BOX: <city>, <country>` without a box number.
These require occurrence-level review; the entire model-flagged census must
not be counted as established defects.

**Directly reconfirmed duplicate input:** V3 `train.jsonl` line 1057,
`syn_tpl_a1fa58135a0962bb4b6b8b389226362b017aace6`, delivery agent:

```text
47 AL-NASR AVENUE, HELIOPOLIS, CAIRO, A.R. EGYPT, A.R. - CAIRO
Egypt, A.R. EGYPT Tel.+20 2 2418 6307 Fax.+20 2 2418 6307
```

The label is already the reasonable first complete postal sequence,
`47 AL-NASR AVENUE, HELIOPOLIS, CAIRO, A.R. EGYPT`. Nevertheless the input
contains a repeated/malformed city/country tail. Label presence cannot validate
this input. The prospective local correction removes that tail, retaining the
entire Tel/Fax suffix byte-for-byte. It was tested in memory, not applied.

**A second directly reconfirmed duplicate:** V3 `train.jsonl` line 1090,
`syn_tpl_006c55242b3fdf9a1640d0e9b366f3641552b510`, consignee:

```text
NILE HARVEST PROVISIONERS/ALEXANDRIA EGYPT 17 EL NASR AVENUE, ALEXANDRIA, EGYPT .TEL:00201276148395
```

Company ownership ends before the slash. The postal locality/country occurs
both immediately after it and at the end of the street address. The existing
label correctly contains `17 EL NASR AVENUE, ALEXANDRIA, EGYPT`, while this
input still needs the bounded duplicate correction. Its current OCR hash is
identical to the pre-address baseline: this particular input defect survived,
rather than being introduced by, the address pass.

Exact selected spans, source IDs, current text/target hashes, labels, proposed
local edits, and before/after checker outcomes for these two records and PA-10
are in the [actual-corpus probe receipt](../artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground015_cargo_text/address-full-migration/checker-probes-20261001/existing-corpus-results.json).
These are three established **local** defects, not a claim about every other
field of the three records or a population prevalence sample.

### 4.2 Source/template/generation connections

The compiled byte template proves insertion coordinates, not that a value has
the right meaning at those coordinates. A `value_kind=address` binding can be
incomplete or semantically mixed; country bindings can refer to nonpostal
contexts. Existing `PartyAddressRoleCertificate` checks hashes, binding paths,
slots and selected fixed frames. Its `_address_bindings` includes only
`group_kind == party` and `value_kind == address`; coverage of this set is not
coverage of every postal byte, carrier, or source-only occurrence.

`complete_pipeline.py` and `descendant.py` historically call the training
address projection after rendering. A renderer can accurately express a full
scenario while that projection produces a narrower target. Conversely, a full
saved scenario can be internally plausible but incorrectly split into the OCR.
Correctness must be assessed at both levels.

## 5. Scope: distinguish measurements from prevalence claims

### 5.1 Corpus denominators

The current repair inventory has **31,066 records**: 29,910 synthetic and 1,156
real (including 100 validation). The synthetic campaigns are old/recovered
10,000, Egypt-heavy 9,980, and diversified 9,930. Do not substitute the earlier
“real1057” filename count for a live inventory count.

There are 1,306 synthetic source families; 651 overlap real-source identities.
The union is 1,811 source identities. Different revisions or structural variants
still require separate exact mappings. The 100 largest synthetic families cover
9,326 descendants; the 300 largest cover 15,506.

The pre-address inventory contains 129,938 party entries, of which 94,722 have
`address`; 6,273 entries use `sameAs`. There are 3,078 locality-bearing party
entries without `address` (historically 1,510 carriers and 1,568 other roles).
They include legitimate incorporation/locality-only entries, so **3,078 is not
a missing-label defect count**. Missing-label completeness still requires review.

### 5.2 Historical input exposure

- Saved synthesis targets contain 91,273 full party addresses across 29,910
  synthetic documents. This differs from the current 94,722 address-bearing
  targets because cohorts/real records/later target changes differ.
- A historical role sweep grouped rows into 21,058 wrong-role locality-screen
  hits, 3,700 source-preserved/cleared, 1,273 ambiguous, and 2,950 reconstruction
  holds. These sum to **28,981 accepted synthetic rows of that V37 snapshot**,
  not 29,910. The report's categories are screening/interpretation results,
  not a fresh adjudicated duplicate-error prevalence.
- The component-removal probe scanned 25,281 documents and 78,083 affected slots;
  it called 55,873 slot removals mechanical, flagged 10,130 nested-locality cases,
  11,007 unsafe-punctuation outcomes, and 1,073 uncertain edges. These are slot
  counts, with different scope from the record partition above. “Mechanical”
  is not permission to delete or proof a full document is repaired.
- Earlier exact replay reconstructed the original rendering of 19,910/19,910
  newer-campaign documents. It did not certify semantic ownership. Thousands
  had subsequent OCR edits needing current-coordinate reconciliation.

The new read-only census reproduced the actual historical projection against
cached saved-address/party pairs, using the current projection function with an
empty country-alias registry. Results are therefore a conservative count of
reproducible transformations, not a reconstruction of every historical alias:

| Exact reproduced transformation | Party values | Distinct synthetic documents |
| --- | ---: | ---: |
| At least one locality removal | 68,876 | 25,007 |
| City removal, including postal-city combinations | 61,998 | 23,916 |
| Country removal, including country-postal combinations | 53,521 | 20,687 |

City and country document sets overlap; do not add them. The first row is
83.61% of the 29,910 synthetic-document inventory. Among 91,017 available
saved/current address pairs, 21,657 were unchanged and 69,360 differed;
68,876 of those differences were exactly reproduced. The other 484 differences
were **not explained by this exact-replay test**, not automatically classified
as incorrect. Real records were not included in this projection calculation.

Examples from the replay:

| Saved generation address | Pre-address target | Removed meaning |
| --- | --- | --- |
| `Building 27, Nile Corniche, Giza` | `Building 27, Nile Corniche` | City. |
| `Via delle Robinie 18, 22044 Inverigo (CO), Italy` | `Via delle Robinie 18, 22044` | City/qualifier and country, preserving postcode. |
| `18 Al Nasr Avenue, Heliopolis, Cairo 11341, Egypt` | `18 Al Nasr Avenue, Heliopolis, 11341` | City before postcode and country. |

This is strong evidence of how widespread the **old labeling policy** was.
It does not establish that every saved string was printed completely, or that
25,007 inputs are malformed. The old projection was intentional under its old
contract; it is incompatible with the new full-address contract. Counting all
such changes as historical renderer defects would misdiagnose the problem.

For duplicate city/country in old raw text, the broad role screen above is an
exposure inventory, not a reliable prevalence estimator. There is no adjudicated
probability sample or complete independent ownership inventory from which to
honestly supply a precise duplicate-error percentage. That specific estimate
remains unavailable; do not replace it with the 21,058 screen-hit count.

### 5.3 Latest V3 partition and residual scope

| V3 status / check | Count | Interpretation |
| --- | ---: | --- |
| Address-accepted partition | 28,362 | Historical acceptance, now known insufficient as whole-party certification. |
| Address-held originals | 2,704 | Preserved, not declared irreparable. |
| AddressLine values in accepted partition | 85,449 | Field count, not documents. |
| Accepted train / validation / independently cargo-held | 27,452 / 95 / 815 | Training must not silently shrink validation to 95. |
| Repetition-screen candidate documents | 6,329 | Broad screen, includes legitimate repeats and nonpostal contexts. |
| Deduplicated review questions | 6,288 | Questions, not records. |
| Documents reopened by census | 2,629 | Includes uncertain/false findings; not 2,629 proven defects. |
| New mechanical flags | 51 | Includes false positives and genuine regressions. |
| Union of census and mechanical flags | 2,676 | Four overlap; 9.44% of historical accepted partition requires renewed attention. |
| Accepted with neither flag | 25,686 | Not an independently certified clean subset. |

Thus 5,380 records are held or newly flagged under those inventories, and 25,686
are unflagged. These are **workload bounds/categories, not an error-rate estimate**.
Additional missing-label coverage gaps are not fully represented in those counts.
This pass directly reconfirmed three residual local defects: two duplicate
postal inputs and the PA-10 restoration regression, with exact-row evidence.
This is a confirmed minimum, not an estimate that only three errors exist.
We must not report a “90.56% clean” rate from this table.

There are **946 independent cargo/fixed-context holds** across the corpus.
Address recovery cannot clear them. The V3 split has 815 address-fixed cargo
holds and 131 joint address/cargo holds.

## 6. What was tried, what it established, where it failed

| Stage | Recorded result | Limitation / lesson |
| --- | --- | --- |
| Initial 100-document address regeneration, 36 families | Mechanical replay 100/100; final model review 95 accept / 5 review; source-only/locality issues remained. | Two calls per document plus retries; no general source-coverage proof. |
| Compact deterministic vs GLM cleaning, 100 documents / 253 cards | Deterministic screen flagged 24; model screen flagged 49, including lost numbers/localities. | String rewrite and duplicate detection conflated distinct semantic decisions. Flags are not all established defects. |
| Slot/context enrichment | First100: 238 successful cards, 3 failed batches, 43 flags. | More context did not create a complete ownership contract; cost increased. |
| Stronger locality-retention instructions | 253/253 outputs but 67 flags; transfer50 98 outputs with 22 flags. | Successful response is not successful repair; over/under-retention remains. |
| 150-document combined pilot | 142 formatting paths; eight regenerated documents / 23 parties; 348 labels. | Useful components, not whole-corpus transfer proof. |
| Saved-scenario 300-document pilot | 669 addresses, 36 families; 15 targeted rewrites and seven touchups; 669/669 critic accept. | Narrow family coverage, two campaigns, party count per document lower than whole corpus. |
| Unseen-family transfer40 | Source model screen cleared 28/40; manual review disproved some downstream accepts; stricter run accepted 13/32 provisionally. | Whole-run rewriting deleted legitimate repeated blocks/continuations; source model claimed completeness incorrectly. |
| Real-label pilot30 | 80 accepted party proposals; at least 13 screening flags. Five-document retry still included a name/VAT and lost a building word. | A real-source extraction path needs full boundaries; synthetic success does not transfer automatically. |
| Full V2 | 26,823 address-accepted; 4,243 held; 80,345 labels. | Per-party review and exact edits missed later whole-document defects. |
| V3 follow-up | Recovered 2,685 holds, reopened 1,146 old accepts; 28,362 accepted, 2,704 held. | Improved corpus, not complete semantic sign-off; same model family used for multiple evidence views. |
| Completion attempt | Staged proposals/reviews plus full repetition census; no new V4 release. | Broad repeated review spent money while the acceptance boundary remained incomplete. |

### 6.1 Specific validation failures

`address_wave_validate.missing_printed_fragment` searches ordered label tokens
across the entire OCR. It does not require a contiguous owned postal occurrence
or inclusion of every postal component. Controlled probes demonstrate acceptance
of missing-building, wrong-party, and residual-duplicate-input examples by this
function. This statement concerns the function, not a full release replay.

`preflight.parties` only yields existing `address` entries; `address_blind_audit.party_map`
only yields existing `addressLine` entries. A blind reviewer receiving this owner
list is not a blind discovery audit of missing parties/addresses.

Some historical reviews also accepted source-preserved repetitions as faithful
rendering. That is not the same acceptance policy as the current requirement to
clean accidental within-address duplicates in synthetic text. Legitimate page
repeats remain valid. Both the historical policy boundary and genuine missed
defects must be accounted for rather than treating every later flag as a new
failure of an unchanged specification.

Exact edit receipts prove that specified edits occurred. They do not prove the
specified edits should have occurred. Whole-document word/number occurrence
counts can be satisfied by another party or legitimate repeated block.

Older synthesis strings are not authoritative after repairs. The placeholder
regex in `address_wave_restore` does not match `PIN-00000`; merely adding that
spelling would still leave the underlying precedence failure.

An ordered diff equal-block alignment provides one plausible coordinate mapping,
not necessarily a unique ownership mapping in repeated text. No semantic
certificate follows merely because one exact substring match exists.

### 6.2 Costs: historical estimates versus settled evidence

- Initial 100-document two-pass route: approximately $0.100–$0.150 per 100 at
  then-listed provider rates; naïve full-synthetic projection $30–$45. Full
  iterative pilot recorded approximately $0.364–$0.547. These were token-rate
  estimates, not account invoices, and excluded some source certification.
- Saved-scenario pilot300: successful-route estimate $0.04248 total. Multiplying
  by documents understated party work; per-party dual-critic projection was
  $5.80 before real records/transfer exceptions. Dropping its zero-incremental-
  finding label critic projected $2.42, **not** a validated full-corpus cost.
- V2 report: $7.7766 accounted, comprising settled and token-estimated charges.
- V3 follow-up: approximately $7.78 follow-up, approximately $15.55 cumulative
  under that report's accounting. Do not add cumulative totals together.
- After completion workers drained: **$17.6842764546** unique provider-reported
  charges in the current work database: $14.009653967 earlier waves and
  $3.6746224876 completion. There are 425 unpriced response receipts. This is
  a **lower bound**, not a fully reconciled all-history invoice; earlier pilot
  ledgers and interrupted/unpriced requests may be outside it.
- The ledger records 24,966 batch rows, 26,403 provider requests in usage fields,
  149,025,645 input and 24,390,592 output tokens. Usage counts are not unique
  billing identities. Repeated interpretation/review, long context, retries,
  and expanded semantic/geographic scope explain the escalation.

No historical estimate authorizes fresh spending. The working objective remains
approximately $5 or less for an efficient complete repair, with correctness
primary. A replacement cost projection must include source interpretation,
real/validation records, failed calls, all reviews, and exceptions—not only the
successful formatting call. This documentation/probe pass uses no paid inference.

## 7. Acceptance design to test, not assume

Use a complete, reviewed map of each source/layout variant: party owners,
postal occurrences, continuations/repeats, nonpostal segments and typed fields.
Reuse it across descendants only with exact current lineage/alignment checks.
No new mandatory per-building ontology and no separate call per component.

For correct current text, change labels only. For recoverable synthetic text,
re-render the address from reconciled facts. For genuinely damaged bundles,
regenerate the address only. All three routes share the same final acceptance
checks and preserve unrelated content. Real OCR remains unchanged.

The proposed checker must require:

1. A complete expected owner/occurrence inventory established independently of
   proposed labels; explicit absent-address and unresolved dispositions.
2. Exact current preimage and map version; no stale coordinates or scenario.
3. Every required postal segment included in the chosen occurrence, in order.
4. Every excluded segment explicitly nonpostal, formatting, legitimate repeat,
   or an approved duplicate with retained evidence.
5. Label equality to that expected content under a narrow formatting policy.
6. Final text reconstruction for every occurrence, including unlabelled tails.
7. Unchanged protected text and unrelated labels; real OCR immutability.
8. No outstanding conflicting evidence. Hashes bind decisions; they do not
   establish their semantic truth.

For synthetic regeneration, intended components must be known before rendering.
Do not let the same faulty selector define both the output and its expected
answer. For existing data, a proposed map needs semantic adjudication where
ownership is ambiguous. Different model calls are not infallible independent
truth sources. A bad but internally consistent map is a deliberate falsification
case below, not something to hide from the reported acceptance rate.

## 8. Experiment protocol and continuation rules

Experiments remain under `A/checker-probes-20261001`, separate from production
and dataset outputs. They must have no provider credentials/network use and no
database writes. Retain reproducible scripts, input hashes, exact expected
outcomes, results, runtime and memory. Documentation is in `docs/` for visibility.

Probe both positive and negative cases: omitted components/whole owners, wrong
party, duplicates outside the label, stale restorations, separator damage,
page continuations, legitimate nested localities, repeated blocks, nonpostal
contamination, same-as sharing, Unicode, formatting, and competing coordinates.
Include actual corpus rows and clearly distinguished constructed counterexamples.

Deliberately attack the specification too: remove a component from both the map
and label, or misclassify a postal segment as protected text. If this passes,
the exact checker is conditional on the map and cannot certify arbitrary OCR.
That boundary must remain visible in every subsequent report.

Do not claim that finite tests prove every unseen template. Mechanical invariants
must hold for every admitted record; unsupported or semantically unresolved maps
must not enter the accepted set. Family count alone is not coverage of structural
variants. A random audit can estimate risk, not guarantee each uninspected row.

To prevent another repair loop:

- Freeze the input and expected policy before bulk editing; version both.
- Maintain one accepted/defective/unresolved inventory per exact row hash.
- Turn each known false acceptance into a negative test before scaling.
- Test the checker before evaluating a repair method's pass rate.
- Reuse paid evidence only for identical or proven-equivalent questions.
- Correct underlying dataflow/ownership failures, not regex symptoms alone.
- Never close address, cargo, schema, and future-synthesis readiness with one
  undifferentiated `passed` flag.
- Publish exact accepted IDs and residual decisions; do not equate silence,
  a transport failure, or a model majority with acceptance.

## 9. Evidence index

- [Initial expensive pilot](../artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground015_cargo_text/party-source-role-probe/pilot100-address-repair-v2/REPORT.md)
- [Saved-scenario feasibility](../artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground015_cargo_text/party-source-role-probe/ADDRESS-REPAIR-FEASIBILITY-20260929.md)
- [Pilot300 results and limited cost projection](../artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground015_cargo_text/party-source-role-probe/ADDRESS-REPAIR-PILOT300-RESULTS-20260929.md)
- [Unseen-family transfer failures](../artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground015_cargo_text/address-full-migration/UNSEEN_FAMILY_TRANSFER_AUDIT_20260930.md)
- [Real-path and transfer audit](../artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground015_cargo_text/address-full-migration/STATUS_AND_TRANSFER_AUDIT_20260930.md)
- [V2 application report](../artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground015_cargo_text/address-full-migration/EXECUTION_REPORT.md)
- [V3 follow-up report](../artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground015_cargo_text/address-full-migration/followup-validation/REPORT.md)
- [Acceptance correction and final recorded spend](../artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground015_cargo_text/address-full-migration/completion/STATUS-AND-ACCEPTANCE-CORRECTION.md)
- [Upstream template/compiler/generation requirements](../artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground015_cargo_text/ADDRESS_TEMPLATE_AND_SYNTHESIS_REPAIR_REQUIREMENTS.md)
- [Old address projection](../src/document_ocr/training/address_projection.py)
- [Current role-certificate implementation](../src/document_ocr/synthesis/template_compiler/party_address_roles.py)

## 10. Probe results

All experiments ran locally with **zero provider calls and $0 inference cost**.
No dataset file was edited. Final receipts are linked in section 11.

### 10.1 Existing weak check: falsified as a sufficient acceptance criterion

The probe extracted the existing `missing_printed_fragment` function through
Python AST, avoiding repair-module initialization and database side effects.
It accepted all three constructed bad cases:

1. OCR includes `UNIT 804` but the label omits it.
2. Label is correct but input has an accidental extra `HONG KONG` tail.
3. Shipper's requested value is found only under the consignee.

It also accepted all three actual residual issues in section 4.1, including
the stale placeholder. These six results concern this support function, not a
replay of every historical release guard. They prove it cannot be used as the
sole criterion for complete party correctness.

### 10.2 Conditional exact checker: supported for its bounded contract

The experimental checker consumes an externally pinned interpretation map and
an immutable pre-edit document. The map partitions the complete input into
postal and protected segments, records owners/occurrences/expected final postal
text, and specifies which occurrence supplies each label. It checks full input
reconstruction, exact label projection, protected target hashes, structural
coverage, and real-input immutability. Offsets are Python Unicode character
offsets, not byte offsets. Label formatting in this experiment folds whitespace
only; a general comma-insertion policy was not implemented.

| Experiment | Result | What it supports |
| --- | ---: | --- |
| Actual V3 positive fixtures | 3 records, 7 selected addresses pass | The individually inspected postal occurrences work under exact mappings. Not whole-corpus certification. |
| Seeded constructed positive fixtures | 1,000/1,000 pass | Tested Unicode, contacts between address pieces, continuations, repeated blocks, same-as references, city words in roads, and distinct region/country names. |
| Deliberately corrupted candidates under fixed maps | 11,033/11,033 rejected | Eleven mutation families: lost text/owners, wrong party, nonpostal inclusion, duplicate text, stale postcode, separator damage, stale preimage, unrelated target edits, and unauthorized map changes. |
| Structurally malformed approved maps / real-input rewrite | 7/7 rejected | Gaps, unaccounted tails, unresolved role, duplicate/conflicting owner dispositions, protected rewrite, and real OCR mutation. |
| Three actual residual defects | 3/3 rejected | With reviewed expectations, bad input is caught even when the current label appears in it. |
| Narrow local corrections to those three, held in memory | 3/3 accepted | Only the selected corrections match those reviewed maps; protected content is retained. No changes applied. |

The seeded fixtures are **not 1,000 real templates**. The 11,033 negatives are
not independently discovered dataset errors. Much of the mechanical protection
comes from comparing with a fixed expected reconstruction; these tests cannot
establish that the expectations are semantically complete.

Runtime was 4.41 seconds for the main checker experiment, peak RSS 111.3 MiB;
3.83 seconds / 110.6 MiB for the actual residual probe; and 36.82 seconds /
111.3 MiB for the read-only full-corpus scope/ledger scan. File I/O is included;
these are local observations, not production throughput promises.

### 10.3 Stronger claim: deliberately falsified

Two **structurally complete but semantically wrong approved maps** passed:

```text
Shipper
ACME
UNIT 804
22 HENNESSY ROAD
HONG KONG
```

If the approved map marks `UNIT 804` as protected/nonpostal and defines the
address as only `22 HENNESSY ROAD HONG KONG`, exact reconstruction and label
equality both pass, despite the missing unit.

Likewise a map can mark the second `HONG KONG` in an accidental country tail
as protected text. It then accepts a clean-looking label and the still-bad
input. These are not failures to add another numeric or duplicate regex. They
show that **coverage by segments is not coverage by correct semantic roles**.

Thus the experiment does not support using this prototype to certify arbitrary
unseen templates. The correct conclusion is conditional mechanical assurance,
with semantic completeness still needing independent establishment. A model
review flag or map SHA alone does not supply that missing evidence.

A further controlled rebase example had two identical `HONG KONG` occurrences
after editing. Exact text search alone admitted two positions. An ordered diff
can choose one alignment, but equality by itself cannot identify the intended
party. Stable occurrence IDs and edit provenance are required.

### 10.4 What to keep, reject, and test next

**Keep:** exact preimages, protected-region preservation, a single joint edit
plan, explicit occurrence ownership, narrow label formatting, and a deterministic
checker against independently established expected content. Saved synthesis is
valuable provenance and often avoids regeneration.

**Reject as sufficient:** whole-document token presence, duplicate-word regexes,
all-numbers-retained checks, “model accepted,” saved-string equality, full byte
replay, and a total partition whose ignored regions have not been reviewed.
Use them as evidence or screens, never as complete semantic acceptance.

**Next bounded investigation, before another bulk repair:**

1. Use the existing 1,306 synthetic-family inventory and lineage receipts to
   group exact structural variants. Keep old-campaign reconstruction, directly
   replayable newer campaigns, real immutable OCR, and already-mutated V3 input
   separate. Do not pay again to interpret identical evidence.
2. Establish postal/nonpostal ownership for whole party blocks and continuations,
   independently of their current address labels. Review exclusions as carefully
   as selected spans. Reconcile template bindings, saved rendered values, and
   current OCR; any disagreement receives a specific unresolved disposition.
3. Challenge those mappings with the omitted-unit and protected-duplicate attacks,
   absent-label parties, static tails, interleaved identifiers, repeats, and
   known prior failures. Test different descendants and truly unseen families;
   do not merely replay the 36 pilot families or 1,000 constructed fixtures.
4. Select repair at the lowest valid level: label-only for correct input; a
   bounded postal-span replacement for malformed input with sound saved facts;
   one coherent address regeneration when those facts are themselves unusable.
   One call may generate the full address; no separate call per component.
5. Validate against the independently reviewed current expected input/labels.
   Preserve semantic decisions and mechanical results as different fields in
   the evidence record. Any unexplained source-only address text prevents a
   whole-party `accepted` disposition.
6. Reconcile the complete denominator before publication: every party, absent
   target, continuation, repeated occurrence, held record and validation record.
   Retain a versioned defect/unresolved list rather than relaunching broad
   critics after every partial repair. New evidence invalidates affected hashes,
   not silently all paid work.

The missing proof obligation is the map/adjudication layer, **not** another
formatting prompt. A useful next pilot must measure that layer's false accepts
and false holds directly. This audit does not promise that semantic judgments
over arbitrary documents can be made infallible. It gives a precise boundary
for rejecting unsupported `pass` claims and identifies what must be checked
before such a claim is trusted for a particular record.

## 11. Reproduction and handoff

- [Probe directory / commands / limits](../artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground015_cargo_text/address-full-migration/checker-probes-20261001/README.md)
- [Final full-corpus census, ledger and file hashes](../artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground015_cargo_text/address-full-migration/checker-probes-20261001/scope-results-final.json)
- [Checker results, fixed-map mutations and failed semantic attacks](../artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground015_cargo_text/address-full-migration/checker-probes-20261001/checker-results-final.json)
- [Per-mutation receipt](../artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground015_cargo_text/address-full-migration/checker-probes-20261001/checker-results-final.mutations.jsonl)
- [Three real-corpus residual issues and in-memory correction probes](../artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground015_cargo_text/address-full-migration/checker-probes-20261001/existing-corpus-results.json)

Use `.venv/bin/python` (3.12), not this host's Python 3.10, for the census's
`hashlib.file_digest`. An initial host-Python attempt failed before writing its
receipt; it did not alter any dataset. Earlier exploratory result files are
retained; use `*-final.json` for conclusions. The final census recognizes both
`city_postal` and `postal_city` rule names (similarly for country); initial
subtype counts omitted suffix-named rules. The aggregate 68,876 / 25,007 count
was unaffected. This audit-script correction is not a dataset repair.

Production source, datasets, templates, generation configurations, and training
remain unchanged by this pass. Pre-existing staged repository changes were not
modified. The remaining GROUND-015 cargo work is outside this address-checker
experiment and must not be marked complete by it.

Handoff validation: all four V3 dataset partitions re-hashed identically to
the census pins; all 17 local Markdown evidence links resolved; all four probe
Python files compiled; Ruff check and format check passed. The final mutation
receipt contains exactly 11,033 rows. A production test suite was not run,
because no production code changed and these are explicitly isolated probes.

## 12. Second experiment: whole-party coverage and saved-address reuse

This section records the **next experiment round**, not a new dataset release.
All experiments were offline, performed directly without subagents, and cost
**$0 in provider charges**. Production source, dataset, templates and active
training settings were not edited. In-memory edits below are experimental
receipts only. No row was promoted to training-ready by these tests.

Evidence lives beside the first probes, under
[`boundary-probes-20261001`](../artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground015_cargo_text/address-full-migration/boundary-probes-20261001/README.md).

### 12.1 Questions and experimental separation

The first probe established that an exact checker can enforce an interpretation
but cannot make a wrong interpretation true. This round tested the interpretation
boundary more directly:

1. Can one source-level description of a **whole party block**, including static
   tails and interleaved identifiers, transfer to fresh descendants?
2. Does that catch address content missed by checking only the old address slot?
3. Can saved synthesis values eliminate regeneration for a concrete malformed
   input class without restoring old defects?
4. Which incorrect candidates still pass, and what additional evidence is actually
   needed to reject them?

Five Egypt-campaign families were selected by deterministic hash outside both
the earlier 36-family pilot and the subsequent 40-family source-review set.
This is family-diverse diagnostic sampling, **not a random sample of the complete
dataset**. Their first source/exemplar texts were inspected to define 20 explicit
party-block boundary specifications. These are reviewed experimental metadata,
not an automatic party-discovery algorithm or new production special cases.

Initially three descendants per family were tested: 15 documents / 60 addresses.
These became development examples after their failures were inspected. The
revised extraction/screening code was then run on all 46 available descendants:
168 current address occurrences, plus the same 168 pre-address-repair values.
The additional **31 documents / 108 addresses** were fresh descendants when the
revised code was frozen. They are *same-family* transfer tests, not an additional
unseen-family test. Final manual adjudication occurred after the frozen screen;
it must not be folded back into an inflated automatic success rate.

| Family prefix | Documents | Address occurrences | Important structure |
| --- | ---: | ---: | --- |
| `9123af7c` | 7 | 21 | Full generated address plus separately rendered city/country; all remain in the old held partition |
| `096bccea` | 20 | 60 | Wrapped company name and inline address; ACID; repeated page headings; carrier registration/contact text |
| `428aa480` | 6 | 30 | Address unit after country; phone/passport; page-two delivery agent; locality words in building names |
| `7937adb6` | 7 | 21 | Multiline industrial addresses, footnote markers, postcode before city, forwarding-agent address |
| `22fa3887` | 6 | 36 | Country/code after tax ID, unheaded carrier address, address continuation after contact lines, second notify party |

The selected block list covers every nonempty *labelled* address in these 46
records. That cross-check is useful but cannot prove absence of an entirely
unlabelled postal block elsewhere in a document. Full-document discovery remains
a separate obligation.

### 12.2 What the transfer experiment actually found

The first screen produced 29 matches, 21 reviews and 10 boundary holds across
60 current addresses. The boundary holds exposed a bad experimental assumption:
an end heading may repeat across pages. The revised metadata pins the source's
expected heading count and uses the next appropriate heading after the reviewed
start. This removed those holds on the measured descendants, but is not proof
that arbitrary future page rearrangements are safe.

The revised frozen screen produced:

| Evaluation set | Address occurrences | Candidate matches | Review | Boundary holds |
| --- | ---: | ---: | ---: | ---: |
| Development | 60 | 39 | 21 | 0 |
| Fresh descendants | 108 | 90 | 18 | 0 |
| Total | 168 | 129 | 39 | 0 |

**`candidate_matches` is a word-retention screen, not an acceptance verdict.**
It compares the current label with all block content remaining after explicit
name, contact, identifier and reviewed-static exclusions. Unknown remaining
words are not automatically known to be postal. This deliberate distinction
matters: manual review of the actual blocks found four further ownership
questions among the 90 fresh matches.

The final address-block dispositions are:

| Disposition | Occurrences | Meaning |
| --- | ---: | --- |
| No address-content defect identified in selected-block review | 125 | Includes 86 fresh occurrences; scoped manual evidence, not full-party/dataset certification |
| Already-held, unmigrated address | 27 | 21 in the first family plus six in one other document; not newly corrupted accepted records |
| Company-name/locality ownership unresolved | 6 | Two caught by the screen; four more found manually; three documents with repeated consignee/notify blocks |
| Source auxiliary country/code role unresolved | 10 | Five accepted documents, shipper and consignee each; not ten proven bad labels |

The 125 do not certify their separate city/country/name targets, geographic
deliverability, every postal occurrence in the document, or unrelated cargo.
Receipts pin the specific block and label hashes and retain this scope explicitly.
This is a useful bounded reviewed set, not the requested eventual full-corpus
quality sign-off.

### 12.3 Concrete discoveries and why the repair level differs

**A. Real duplicate input plus stripped label: saved data can help directly.**

In `syn_tpl_0d957e8b07906a2cee98e614b48816ede60fc94e`, the forwarding party has:

```text
SILVERFEN CARGO MANAGEMENT SHANGHAI CO., LTD.
SHANGHAI 72 Yuyuan Road, Shanghai, China
```

The saved synthesis address is `72 Yuyuan Road, Shanghai, China`. The current
held label is `72 Yuyuan Road, China`. Here the saved full address survives as
an exact substring of the current input; the preceding standalone `SHANGHAI`
is separately introduced, redundant locality text. The experimentally repaired
block is:

```text
SILVERFEN CARGO MANAGEMENT SHANGHAI CO., LTD.
72 Yuyuan Road, Shanghai, China
```

The proposed label is exactly that retained postal line, with whitespace folded.
The company-name occurrence of `SHANGHAI` stays untouched. No text is generated,
no saved address is blindly reinserted, and no unit/number/locality is invented.
All seven descendants in this particular family support the same operation.

The same family's other parties need a different operation: the shipper has
`SHAOXING` in both the generated address and a separately rendered locality,
and `CHINA ,CHINA`; the consignee has `6TH OF OCTOBER CITY, GIZA, EGYPT` followed
by `6TH OF OCTOBER GIZA, EGYPT`. The latter is an alias/formatting question,
not authority for deleting every repeated city word. These 14 addresses were
**not** silently passed by the narrow seven-address experiment.

**B. Saved names can themselves cross the postal boundary.**

`syn_tpl_3dacd698e58084dfada5d42897ab3efd4b8bf4be` prints:

```text
ORIENT GATE FOOD TRADING
AND SUPPLIES -
CAIRO - VILLA 7 - AL NASR ROAD -
HELIOPOLIS 11341 - EGYPT
```

Its saved and current name include `AND SUPPLIES - CAIRO`. Excluding the name
therefore excludes the only printed `CAIRO` from the address candidate. Agreement
with the saved target does **not** independently settle the name/address boundary.
The revised screen exposes the locality that only occurs in excluded content.

Manual inspection found the related case in two more documents:
`syn_tpl_ae155332dbab32ff832cf43f798ce7fac4d9c8c4` and
`syn_tpl_d750afef1d96bf389cadfd92325bde2e5f222cbd`. They also have a name ending
in `- CAIRO`, but a later `CAIRO`/`NEW CAIRO` makes the simple absence check pass.
Both consignee and notify occurrences require adjudication. Do not implement
the converse heuristic that every city in a company name belongs to its address;
many names legitimately contain a city. The repair level is **source/name and
postal ownership**, potentially followed by label changes, not indiscriminate
city deduplication. None of the six is declared a proven error without that decision.

**C. Country after a tax identifier is not automatically an extra postal country.**

The fifth family's original source already contains:

```text
... LUDHIANA(PUNJAB) INDIA
GSTIN: ...
India(IN)
```

and similarly `Tax ID: ...` followed by `Egypt(EG)`. Five accepted descendants
retain this shape. A whole-block extractor notices that their address labels
do not include the trailing country/code; that does not establish an omission.
It may be auxiliary jurisdiction/form metadata rather than a second postal line.
The right next action is one source-level ownership decision for these two
positions, using the source layout and binding evidence, then a reused decision
across descendants. Do not append or delete these strings based solely on their
being country names. Their printed presence in the original also disproves the
claim that every such repetition was introduced by synthesis.

**D. Repeated locality words and punctuation require different checks.**

`SUITE 804, CAIRO TRADE CENTER, ..., CAIRO, EGYPT` and `FLOOR 3, CAIRO MERIDIAN
HOUSE, ..., CAIRO, EGYPT` are not duplicate-city defects. Both were false-positive
duplicate screens during development. Component role matters more than word count.

Strict text comparison also found seven differences hidden by token comparison:
five `U.A.E.` labels correctly retain the final period that the **experimental
formatter** stripped, while two labels retain a leading `-` separator. The latter
two are within the name-ownership cases above, not an additional two documents.
This is evidence against using broad punctuation stripping as the final label
builder. It is **not** evidence that the five correct abbreviation labels need
repair. The exact retained spans should determine punctuation; only agreed
whitespace/line-break presentation may be normalized.

### 12.4 Falsification results: what a match still fails to establish

Fifteen basic contract tests behaved as expected: omissions of a unit, country,
whole address or post-contact postcode were rejected; numeric changes, a wrong
party's address and contact contamination were rejected; valid interleaved
contact/postal text and legitimate locality-containing buildings were retained.

However, all five deliberately stronger semantic attacks could still match the
word-retention screen:

| Attack | Why it escapes | Required independent evidence/check |
| --- | --- | --- |
| `ACME LTD UNIT 804` treated as the company name | Incorrect name boundary hides a real unit from the candidate | Review the excluded name span, not only the address or saved name equality |
| `CREDIT TERM THIRTY DAYS` included in both input and label | Unknown text survives the exclusion list and is mistaken for postal | Every retained region needs an address-role decision; unknown is review |
| `TEN RAMADAN CITY` and `TENTH OF RAMADAN CITY` both retained | Literal matching misses semantic duplicates | Explicit duplicate-occurrence decision; preserve legitimate named sites |
| Previously removed `PIN-00000` restored to both input and label | Input/label agreement cannot establish repair-history correctness | Actual per-party approved edit history and current preimage pin |
| Input `UNIT 8/04`, label `UNIT 8-04` | Tokenization discards meaningful punctuation | Exact selected-surface comparison, not word/token identity |

An entirely unenumerated second party outside the selected block was also
invisible to that block's checker. This is a sixth, separate **scope** failure,
not one of the five semantic attacks. The experiment also demonstrated that an
explicit prior deletion receipt blocks the known placeholder restoration; that
does not mean every historic receipt has now been integrated or audited.

These failures are preserved as negative research results. The screen is not
being promoted with exceptions patched in until the examples happen to pass.
Its useful role is cheap triage and content accounting, beneath an independently
reviewed interpretation and exact edit verifier.

### 12.5 Positive repair proof: deletion-only reuse, with its limits

The seven forwarding-address proposals in section 12.3A were executed in copied
Python records only. For each one the receipt records the exact preimage and
postimage hashes, removed character range, before/after block and address target.
Validation required:

- The complete retained address is an exact current-text substring matching the
  saved address under case/physical-whitespace presentation, not fuzzy content.
- The only extra region is the reviewed standalone city prefix, and that city
  occurs as a complete component in the retained address.
- No extra unit, district, postcode, contact or unclassified text is discarded.
- All numeric tokens in the block remain exactly unchanged.
- Restoring only the permitted address-label change and text deletion reconstructs
  the original whole record exactly; no other text or target changes are allowed.

Eight targeted tests behaved correctly: the valid duplicate-prefix case was
eligible; added unit/district, trailing postcode, stale saved street, contact
requiring another role contract, repeated saved address and a different locality
prefix all prevented an edit. This demonstrates a concrete zero-call repair
path with preservation proofs. It is **not** a general postal semantic oracle;
it still relies on the reviewed name boundary and duplicate interpretation.

The initial implementation attempt refused all proposals because its inherited
word-based name boundary omitted the final period in `LTD.`. Before any results
were written it was corrected to match the full printed name, including
punctuation. This reinforces the difference between a permissive screening
comparison and the exact spans used for edits. No dataset was modified by the
failed attempt or the corrected experiment.

### 12.6 Tool research and the process this supports

The [official libpostal documentation](https://github.com/openvenues/libpostal)
describes a statistical international address parser/normalizer, not a full
geocoder. It could provide another disagreement signal, but it does not establish
which shipping-form text belongs to a party, whether a city is part of a company
or building name, or whether a prior repair must take precedence. I did not
install it or claim a measured benefit on this corpus.

[PydanticAI output validators](https://pydantic.dev/docs/ai/core-concepts/output/)
can validate structured answers with context and request retries. Their context
is not automatically part of the model prompt. The relevant use here is a small
role/duplicate decision over supplied text spans, validated by host code—not
another unconstrained address rewrite. Structured validation does not supply
semantic truth; that is an inference from the experiments and the tool's scope.
No provider model was called in this round.

Recommended process, now supported by these probes:

1. **Separate discovery, interpretation, editing and verification.** Locate whole
   party blocks, repeats and continuations from source/synthesis evidence, not
   just current address labels. Treat source maps as explicit data, not hidden
   per-template regex branches. Whole-document coverage needs a separate check.
2. **Reuse decisions where the evidence is genuinely identical.** Review stable
   source captions, nonpostal fields and fixed fragments once. Distinguish them
   from variable free-text values whose generated content can cross roles.
   The latter must not inherit semantic approval solely from an unchanged slot ID.
3. **For sound input, only repair the label.** Derive it from selected current
   postal spans in their printed order. Retain numbers and punctuation. Company
   names, contacts and tax IDs have explicit exclusion reasons.
4. **For proven input duplication, delete the exact duplicate occurrence.** The
   seven-case experiment demonstrates this without copying old OCR or generating
   anything. A city substring inside a building or company name is not enough.
5. **For a genuinely unusable generated address, replace only its reviewed postal
   region.** Regenerate one coherent address only when exact reuse cannot preserve
   the needed information. Apply prior repair precedence; never blindly restore
   an earlier saved target or source fragment. This branch was not newly tested.
6. **Use semantic review only for the unanswered role decisions.** Provide the
   full party block, relevant source position, saved data and previous edits;
   request selected/excluded spans and duplicate pairs. No invented address or
   guessed geographic correction. An unclassified region prevents acceptance.
7. **Verify independently against the approved interpretation.** Exact final
   text/label agreement, every expected party/occurrence accounted for, no unrelated
   edits, no numeric/punctuation loss, and prior repairs preserved. Challenge
   both the interpretation and the writer; testing only the writer is insufficient.

For the measured ambiguous cases, the next high-value work is concentrated:
one family's name/locality boundary and one family's two post-tax country/code
positions. Resolving those source roles could settle multiple descendants at
once. The seven confirmed deletion-only candidates do not need another LLM call.
This is a reason to target repair at the source-role level when shared and at the
exact postal-span level when local—not to re-review all 30k full documents with
a broad critic.

### 12.7 Timing, cost, validation and launch boundary

The revised 46-document / 336-baseline-and-current-occurrence run took **9.08 s**,
peak process RSS **113.8 MiB**, including scans to locate selected records. The
evidence/deletion-replay pass took **0.033 s**, peak RSS **108.2 MiB**. These are
offline computation timings, **not end-to-end review throughput**; manual
source/semantic review time is excluded. No production hot path changed, so
these are not a production performance improvement benchmark.

No paid calls were made, and there is no new defensible full-corpus dollar
projection from five Egypt-campaign families. The cost-saving evidence is more
specific: unchanged facts and reviewed duplicate spans can be processed locally;
shared role questions can be reviewed once; only unresolved semantic questions
need paid assistance. A broader mixed-cohort measurement is still required to
price the remaining work honestly.

This round does not cover new real/validation families or the two other synthetic
campaigns. For real data, OCR remains immutable; only source-supported label
repairs are permitted. Existing repaired data is not replaced by any experiment.
The remaining cargo work and future synthesis-template corrections are not
declared complete by this address investigation.

Handoff checks for this round: all four dataset partitions match their previous
SHA-256 pins; repeated boundary and edit runs reproduced the per-case receipts
exactly; all four experimental Python files compiled and passed Ruff check and
format check; all 18 local report links resolved. The repeated boundary run took
9.75 s and 110.1 MiB peak RSS. `boundary-probes-20261001/verification.json` records
these assertions and script hashes. No production test suite was run because
no production implementation was changed.

## 13. Mixed-cohort 150-document follow-up — localization, metadata and exact edits

### 13.1 Questions, scope and the decision this test supports

The follow-up asks whether every sample needs repair, whether duplicates can be
localized to repeatable source slots, and whether postal text after contacts or
flavor text is handled as well as postal text after tax IDs. It authorizes a
150-document experiment, not publication of a new bulk-repaired dataset.

**Conclusion: check every document, but do not rewrite every document.** The old
city/country-stripping policy was reproduced in 25,007 documents (68,876 party
addresses; section 6). That is historical policy exposure, not the number of
current bad documents. The present working copy has already migrated many of
those labels. Neither its prior accepted status nor an absence of current flags
establishes semantic correctness. Duplicate inputs also cross campaigns and
families; the five previously probed families are not a complete affected set.

The correct unit of repair remains a reviewed party occurrence: label-only when
the input is sound, an exact duplicate-span edit when the input is demonstrably
redundant, and review when the company/postal/auxiliary boundary is unresolved.
Repeated blocks belonging to the same party must be checked together, while
identical consignee and notify-party blocks remain separate owners.

Experiment directory:
[`pilot150-probes-20261001`](../artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground015_cargo_text/address-full-migration/pilot150-probes-20261001).
The canonical results for this round are `screen-v3/`, `final-evidence/` and
`verification-final.json`; earlier numbered outputs are development receipts.

### 13.2 Selection and coverage

The frozen selection contains **150 documents from 17 source families**:

| Cohort | Documents |
| --- | ---: |
| Egypt-heavy synthetic | 60 |
| Diversified synthetic | 30 |
| Recovered/third synthetic campaign (`old` in the investigation database) | 50 |
| Real, including three validation documents | 10 |

It includes 46 documents from the previous regression probe and **104 additional
documents**. The latter include transfer across campaigns in familiar families,
six previously untested synthetic families, and six additional real documents.
The six synthetic families and six additional real records were selected with a
fixed SHA-256 ordering and campaign/split stratification. Source-block boundaries
were then inspected. This is a deliberately difficult diagnostic sample, not a
population-random accuracy estimate or a completely blind benchmark.

The current partitions represented are 92 training, 38 address-fixed/cargo-held,
17 held-original and three validation records. The experiment covers **545
party-owner scopes**. All existing labelled address owners have a mapped block;
there are zero unmapped labelled owners. This check alone does not prove that an
unlabelled party elsewhere in an arbitrary document has been discovered.

Six malformed shipper blocks stop before partitioning. The remaining scopes
contain 562 successfully partitioned physical occurrences, including repeats
and an address continuation. Counts of physical blocks, owner scopes and
documents must not be interchanged.

### 13.3 Where duplicates occur, and what the slot evidence establishes

The test confirms several recurring shapes:

| Shape | Concrete current example | Interpretation/action |
| --- | --- | --- |
| Standalone prefix plus complete generated address | `SHANGHAI 72 Yuyuan Road, Shanghai, China` | Delete only the reviewed prefix; retained complete address matches saved synthesis evidence. |
| Country alias retained inside an address plus full country | `AE - JAFZA SOUTH, LOGISTICS PARK 4, UNIT 216, UNITED ARAB EMIRATES` | The reviewed `AE - ` is redundant; preserve every site/unit component. |
| City across an address-line boundary | `73 ZEELAAN KOKSIJDE` / `KOKSIJDE 8670 BELGIUM` | Collapse the reviewed repeated locality, preserving postcode 8670. |
| Full address followed by separate city/country slots | `... MWENE-DITU CONGO, THE DEMOCRATIC REPUBLIC OF THE 2337,` / `MWENE-DITU, CONGO, THE DEMOCRATIC REPUBLIC OF THE` | Remove the redundant tail, not the postcode or internal punctuation of the retained country name. |
| Same city/country surface in adjacent standalone slots | `HONG KONG` / `HONG KONG` | One retained mention supports the unchanged city and country targets. |
| A legitimate place name inside a site or company | `CAIRO TRADE CENTER ... CAIRO, EGYPT`; `SUZHOU INDUSTRIAL PARK ... SUZHOU, CHINA` | Preserve it. Repeated words do not prove duplicate information. |

Three current catalog templates were inspected, with their hashes and complete
relevant bindings saved in `final-evidence/template-slot-evidence.jsonl`:

- **Arkas `1249175a`:** shipper country is `slot_0003`; the address occupies
  `slot_0004`–`slot_0005`, whose source text includes `AE - AFZ BUILDING C1,`.
  Thus a country-code surface is already embedded in an address binding while
  the country also has a separate binding. This explains why field-name-only
  checks do not isolate every country occurrence.
- **Danzas `b8982e77`:** shipper address occupies `slot_0006` and `slot_0008`, with
  city between them at `slot_0007` and country at `slot_0009`. Consignee and notify
  similarly have address plus separate locality slots. This makes the overlap
  localizable, but a full generated address inside one slot must not be assumed
  to contain only the original slot's street-level information.
- **`22fa3887`:** shipper country has three occurrences—`INDIA`, `India`, `IN`—at
  `slot_0005`, `slot_0007`, `slot_0008`. Consignee has the corresponding
  `EGYPT`/`Egypt`/`EG` occurrences. The delivery agent's city and country are
  `slot_0030` and `slot_0031`, **after** its contact fields. These are concrete
  source-position relationships, not merely a search for repeated words.

These are current catalog observations, not a claim that the latest catalog is
the byte-identical historical compiler state for every generated document.
The seven prefix repairs additionally reuse exact saved-generation evidence.

Across the selected 150 documents, the literal duplicate screen flags 379 party
owners in 135 baseline documents, versus 118 owners in 54 current documents.
The current flags span ten source families. **These are review-signal counts,
not confirmed defect counts:** named buildings and administrative hierarchies
produce false positives; aliases can produce false negatives. Three reviewed
Arkas aliases were missed by the literal screen despite input/label agreement.
This is why no duplicate-word count is permitted to authorize deletion.

### 13.4 Postal text after metadata: contacts, flavor, codes and continuations

**Yes, the requirement extends beyond tax IDs.** There is no rule that an address
ends at the first telephone, email, tax ID, flavor phrase, or even page boundary.
The relevant question is the role of each span within the complete party block
and its linked continuation—not its position relative to the first caption.

Examples checked:

1. The real `22fa3887` delivery-agent block prints office/site information,
   `1234`, a telephone and email, then `New Cairo` and `Egypt`. All postal
   components must survive; neither the contacts nor `1234` justify discarding
   subsequent address text. Six current selected documents still have locality
   text following contacts; nineteen baseline documents have that arrangement.
2. Real `e0d915cb` prints `AS AGENTS ONLY E&OE`, a contact person and a job title
   near the agent address. The source-reviewed flavor/title spans are excluded,
   while the villa, street and district remain. Company-name exclusions are
   separately accounted for, not conflated with address words.
3. Validation document `e42f749e` ends its main shipper address with `#`; a later
   page prints `# INDIA PHONE: 91-22-44770000`. The current address label omits
   `INDIA`. The reviewed continuation contributes **INDIA only**, leaving the
   phone in the input and outside `addressLine`. The real OCR is unchanged.
4. Thirty-eight selected current owner scopes in nineteen documents have
   country/code text after an identifier (`India(IN)`, `Egypt(EG)` and generated
   variants in one family). Whether that trailing surface is postal or an
   auxiliary jurisdiction statement still requires a source-role decision.
   Being after a tax ID neither includes nor excludes it automatically.

The experiment's `after_reviewed_nonpostal` signal also counts carrier legal
captions and company-registration lines, not just informal “flavor.” Its 50
owners/49 documents therefore must not be reported as 49 address defects.

The generic screen was extended to recognize labelled customs references and
exact international phone values appearing inline. This removed 40 screening
mismatches without editing data. It does not mean that 40 labels were repaired.
The screen still cannot replace a reviewed semantic partition.

### 13.5 What was actually repaired and verified in the experiment

**24 party repairs across 23 documents** were executed in isolated research
copies, with explicit edit receipts:

| Repair | Party scopes | Input changes |
| --- | ---: | --- |
| Previously reviewed Shanghai prefix regression | 7 | Exact duplicate deletion |
| Newly reviewed alias, locality-tail, adjacent-city and same-surface duplicates | 8 | Exact localized edit |
| Unchanged carrier address recovered into held legacy address labels | 8 | None |
| Real validation address continuation | 1 | None |

The eight new duplicate decisions are in `reviewed-duplicate-decisions.json`.
They are **review data for these frozen examples**, not hard-coded production
rules for particular countries or document IDs. The common edit executor uses
exact preimages and offsets. Reusing a decision on another sample would require
that sample's own matching evidence and role checks.

Every repair was checked for:

- Exact pre-edit record hash and matching edited text.
- Preservation of all printed numeric tokens, including postcodes, units,
  telephone numbers and identifiers.
- Exact expected full address after the authorized changes.
- No unrelated text or target-field changes, including unchanged city/country
  targets and contacts.
- Real-source OCR immutability.
- Composition of multiple approved party changes in the same document without
  one overwriting another; 23 composed research copies were replayed.

Restoring only the authorized address fields and text changes reconstructs each
original record exactly. The outputs retain `wholeDocumentAccepted: false`:
repairing one address does not resolve another party or a cargo hold elsewhere
in that document. No training dataset, source template, production flow or
training configuration was modified during this test.

### 13.6 Falsification results and explicit limits

There were **120/120 rejected edit-verifier attacks**: adding label content,
omitting the last address component, changing unrelated text even with its hash
updated, changing another party field, and using a stale preimage. This proves
the tested exact-delta checks, conditional on the reviewed edit interpretation.
It is not a statistical estimate of semantic error rate.

Twenty-eight metadata-position fixtures (seven metadata types at four positions)
retained the complete address as expected. A twenty-ninth positive check retained
a city-containing building while flagging, rather than deleting, its repeated
word. A thirtieth fixture deliberately demonstrates that unknown business text
survives the permissive screen. It is a documented limitation, not a successful
semantic extraction.

Three explicit semantic counterexamples are preserved:

| Constructed counterexample | Observed weakness of the screen | Required evidence |
| --- | --- | --- |
| `CREDIT TERM THIRTY DAYS` appended to a postal block | Unknown business text survives as candidate postal text | An exhaustive reviewed role partition, not a blacklist of familiar captions |
| `PHONE: +20212345678 11835 CAIRO EGYPT`, with 11835 designated postal | A broad phone regex consumes the postcode along with the number | Exact contact-span boundary from source/synthesis evidence or explicit review |
| Incorrect existing name `NORTHSTAR LTD UNIT 8/04` | Trusting that name boundary removes the real unit | Review exclusion boundaries; existing name/label agreement is not independent evidence |

The second is a **constructed boundary stress test**, not a claim of an observed
additional postcode-loss population in the dataset. All three are prohibited
from becoming authorized repairs by the experiment's reviewed-plan requirement.
The experiment has not implemented an automatic semantic solver for them.

Manual inspection also reconfirmed unresolved name/locality ownership in the
`096bccea` family, including `EL - OBOUR` in the real record; six new-family
shipper blocks repeat the company name inside a street fragment; and several
otherwise matching blocks contain redundant `,,` separators. These observations
are reasons not to convert `candidate_match` into `accepted`.

After joining continuations and accounting for repeated blocks, the final owner
screen is **349 candidate matches, 190 review candidates and six boundary/content
holds**. These sum to 545 owner scopes. They are **not** 349 certified addresses
and 196 proven defects. Fifty-two owner scopes flag locality words inside the
party name; other flags concern named sites, and many are correct. Conversely, exact surface agreement
(392 scopes) can include duplicated data on both sides. This experiment is not
a 150/150 successful end-to-end automatic repair claim.

### 13.7 Practical path supported by this round

1. Run cheap coverage/disagreement checks over **all** current records. Do not
   regenerate all parties, nor limit discovery to previously flagged families.
2. Group work by whole-block and slot arrangement: address with separate locality
   slots; alias-bearing address slots; contacts interleaved with postal text;
   repeats/continuations; uncertain company/postal boundaries.
3. Review shared source roles once, and validate every descendant's actual content
   against that contract. Variable content that crosses a role boundary returns
   to review; the same slot ID is not semantic approval.
4. Reuse saved synthesis text only where it matches the current retained postal
   content and respects prior edits. Never restore an old target blindly.
5. Apply label-only repairs or exact duplicate-span deletions with the demonstrated
   verifier. Keep all other text, labels and previous repairs unchanged.
6. Use a narrow semantic decision for unresolved roles or genuinely damaged input.
   It must establish the spans, not freely rewrite an entire party and then
   validate itself by agreement with its own output.

The test demonstrates useful new zero-call repairs and broader localization; it
also rejects a blanket regex/word-match approval strategy. Full-corpus publication
is not authorized or certified by this result. The remaining work is explicit
role adjudication and descendant coverage, not making the exact editor more
elaborate or repeatedly generating addresses that are already present.

### 13.8 Cost, performance and reproducibility

- **Provider calls: zero. Additional API cost: $0.** There is no new full-corpus
  semantic-review cost estimate from this offline, purposively selected test.
- Loaded-data screening: 0.188 seconds, 109.2 MiB peak process RSS. The final
  evidence/edit/attack pass: 0.098 seconds, 106.3 MiB peak RSS. These exclude
  selection, manual review, report writing and full-file hash verification.
- The screen's initial measured pass was 0.182 seconds/110.6 MiB; after the
  metadata extensions it was 0.188 seconds/109.2 MiB. This tiny workload does
  not support a meaningful throughput improvement/regression claim.
- Both runs replay identically, apart from timing/RSS, through `verify_probe.py`.
  All four current dataset partitions retain their prior SHA-256 hashes.
- Five experimental Python files compile and pass Ruff check/format check.
  No production code was changed, so no production performance claim or new
  production-unit-test sign-off is made.

## 14. Controlled dataset execution after the 150-document probe

### 14.1 Scope, authority and preserved state

The user approved proceeding with the controlled process: preserve the corpus,
apply reviewed corrections, screen the complete corpus, group and adjudicate
shared roles, and validate each edit without converting weak screening matches
into acceptance. The execution directory is
`artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground015_cargo_text/address-full-migration/execution-20261001`.

The original V3 corpus is unchanged. An additional verified copy is at
`artifacts/kie-training/datasets/mpci-bl-ground015-addressline-v3_pre_boundary_repair_archived`.
All original files, not only the four JSONL partitions, were hashed and checked
after copying. Its archive notice explicitly warns about unresolved defects.

The updated working dataset is
`artifacts/kie-training/datasets/mpci-bl-ground015-addressline-v4-boundary-repair-working`.
All 31,066 IDs remain: 29,910 synthetic and 1,156 real, with 100 original validation
IDs across the historical partitions. No training configuration was repointed.
No source OCR, template catalog or production code was changed in this execution.
The existing staged production/test changes belong to preceding work and were
not altered here.

### 14.2 Repairs actually applied

| Intervention | Party scopes | Documents in that intervention | What changed |
| --- | ---: | ---: | --- |
| Previously reviewed pilot and residual corrections | 27 | 26 | Apply the 24 pilot decisions and three earlier confirmed residuals to real files |
| Synthetic shipper name intrusion, family `e46b2c49` | 99 | 99 | Remove two extra organization names from the postal span; resolve repeated address/locality/country suffixes with actual and saved evidence |
| Recipient and notify duplicates in the same family | 32 | 16 | Remove the extra city/country pair after the complete address; convert old stripped labels to full addresses |
| Fixed carrier registered offices | 568 | 568 | Restore 472 entirely missing address labels and complete 96 partial labels; text remains unchanged |
| Standalone duplicate country suffixes, family `9805d7fe` | 41 | 41 | Remove only redundant `IND`, `USA`, etc. or a reviewed country-name fragment after an already complete postal address |
| Held Shaoxing/6th-of-October locality bundles, family `9123af7c` | 14 | 7 | Preserve street, district, province and postcode while removing the extra printed locality bundle and completing the labels |
| One reviewed notify duplicate | 1 | 1 | Remove the second postal `CAIRO`, keeping `NASR CITY`, postcode, name, person and contacts |

Interventions overlap at document level. The final unique totals are:

- **735 documents / 782 party scopes / 57 source families.**
- **158 synthetic inputs edited.** All 1,156 real OCR inputs remain byte-identical.
- **664 address labels changed** in 617 documents. The remaining repaired scopes
  already had the desired label but needed their printed input corrected.
- 18 real records received label-only changes, including four in the current
  validation JSONL. No validation membership changed.
- Changed-document cohorts: Egypt 214; diversified 269; recovered/old 234; real 18.
- Changed historical partitions: train 508; validation 4; cargo-held 14;
  address-held 209. **These changes do not promote held records.**

The actual saved receipts are `repair-receipts-initial.jsonl`,
`repair-receipts-reviewed-wave.jsonl` and
`repair-receipts-locality-tail-wave.jsonl` in the working dataset. Each includes
the pre-edit and post-edit record hashes, exact text spans and replacements,
expected address label, owner, and reviewed evidence. Intermediate preimages
are retained in the execution's staging directories. A changed address does
not confer whole-document or whole-party approval.

### 14.3 Why this was more efficient than rewriting every address

The fixed-office review covered **1,279 carrier address scopes**, represented by
11 actual printed surfaces. The office sits between an explicit company heading
and a website or shipper heading. Repeated page occurrences were inspected
against the reviewed surfaces. Of these, 711 were already complete and 568
needed labels repaired. Original spelling, punctuation and postcode prefixes
were preserved; no geocoding or country inference was performed.

This also found a blind spot that address-only checks miss: **472 records had no
address label at all despite the carrier address being explicitly printed**.
Screening only existing address values cannot detect this omission. Future
coverage needs an inventory of printed roles, not only the keys already present
in the target.

For `e46b2c49`, all 100 current family records were inspected, not a sample
extrapolated to the rest. The 99 synthetic shipper blocks repeated their generated
company name twice inside the street address. Saved generation values corroborated
the intended single address, but were not blindly restored:

- Seven Egypt-cohort targets omitted the separately printed `US`; that printed
  country remains in the repaired full address.
- Two Egypt inputs put `SC` before `COLUMBIA`; the current printed order remains.
- One saved Hong Kong address itself repeated Hong Kong. The reviewed repair
  retains one printed occurrence rather than copying the saved defect.
- Five inputs repeated the entire address, including its street/postcode numbers.
  These have explicit numeric-deduplication receipts; one complete copy remains.
- Real source `doc_e46b2c4947579f84b6fddb2779f7ac4f445c6957a12865fa365e3a1216d74f07`
  contains two organizations and NVOCC text. Its unresolved postal ownership is
  retained for adjudication; it is not rewritten to resemble a synthetic party.

### 14.4 Concrete before/after behavior

One synthetic input contained:

```text
KANSAI FLOW CONTROL KK
29 SENRI KANSAI FLOW CONTROL KK
KANSAI FLOW CONTROL KK
CENTRAL AVENUE, TOYONAKA,
TOYONAKA, JAPAN JP
GOVERNMENT REFERENCE NUMBER: US-EIN:184129362
```

The repaired postal content is `29 SENRI CENTRAL AVENUE, TOYONAKA, JAPAN`;
the company remains once at the top, and the government reference and contacts
remain unchanged. The old label `29 Senri Central Avenue` becomes:

```json
"addressLine": "29 SENRI CENTRAL AVENUE, TOYONAKA, JAPAN"
```

Another input had `PLOT 8 INDUSTRIAL ESTATE ROAD, KARO, INDIA , IND` while its
address label already stopped at `INDIA`. Only the redundant printed ` , IND`
was removed. A street containing a city name, such as
`12 REWA INDUSTRIAL LINK ROAD, REWA, INDIA`, was kept intact.

The carrier example is a label-only repair: a printed
`12-14, chemin Rieu, 1208 GENEVA, Switzerland` now becomes the complete
`addressLine`, instead of `12-14, chemin Rieu, 1208` or no address at all.

### 14.5 Complete-corpus discovery and what its counts mean

Both before and after repairs, the local census visited all **31,066 records**
and all **129,938 party objects**, including objects without an address. Metadata
joins were exact by record ID, not inferred from layout similarity. Current
provenance spans 1,811 source IDs. The 17 reviewed source maps were replayed on
all 376 matching records, not just the 150 pilot records.

The final raw screen still flags 10,589 documents for one or more review signals.
This number is **not a defect count, rejection count, or new hold count**.
Examples include a legitimate city name in both a street and its locality,
optional locality-only targets, and address components separated by metadata.
No screen-negative record was newly certified by its absence of flags.

| Final screen signal | Documents | Source families | Interpretation |
| --- | ---: | ---: | --- |
| Existing old `address` fields | 2,681 | 611 | Migration/ownership work remains; historical hold partition still contains 2,704 records |
| Address not contiguous after format normalization | 3,133 | 655 | Could be a genuine omission/mismatch, or legitimate intervening metadata/continuation |
| Repeated city word in label | 4,114 | 948 | Includes valid street, district and company/locality contexts; never authorize global word deletion |
| Repeated country word in label | 335 | 232 | Requires positional postal/auxiliary ownership review |
| Locality with no address label | 2,732 | 145 | May be legitimate locality-only data; check printed role coverage before adding a label |
| Zero-postcode label candidate | 7 | 7 | Requires preimage/role review, not automatic numeric deletion |
| Repeated comma separators | 139 | 84 | Presentation review; not equivalent to semantic corruption |

These sets overlap. Per-record owners and reasons are in
`final-census/corpus-inventory.jsonl`; per-source counts are in
`final-census/family-groups.json`. The narrower mapped-source screen has 107
review scopes in 48 documents; its five name-boundary holds were independently
adjudicated as screening false positives below. An exact candidate match still
does not exclude unmodeled nonpostal text.

### 14.6 False positives and boundaries still needing adjudication

All five residual `name_missing_or_repeated` holds were investigated against
their complete current blocks. The permissive word matcher found a second
company-name occurrence *inside its email domain*. For example,
`RIO FRONTERA MEATS` also matched `RIOFRONTERAMEATS` in
`ORDERS@RIOFRONTERAMEATS.COM`. There was only one printed organization header.

`name-email-adjudications.json` pins the five current records, actual name hits
and independently delimited email spans. This is an explicit false-positive
decision, not a blanket bypass of name errors. Genuine repeated names outside
contact spans still need inspection. The historical screen is preserved rather
than silently rewriting its evidence or treating its name matcher as a perfect
semantic parser.

Other remaining role decisions are substantive: organization/locality overlap
in the `096bccea` family; country lines after tax/contact text in `22fa3887`;
alias/locality tails such as `HYD-BAD-34`, where the numeric fragment must not
be thrown away as a duplicate; and the real two-organization NVOCC source.
Existing labels, slot IDs, saved targets and whole-document presence are not
independent authority for these boundaries. They require source-role decisions
before applying the same exact editor. No part of this execution declares those
records irreparable.

### 14.7 Validation and performance

`final-verification.json` independently checks the actual serialized dataset:

- Every record replays exactly through its ordered repair receipts; all
  untouched records remain byte-identical.
- All original and archived files retain their pinned hashes.
- IDs, row counts and partition membership are unchanged.
- All 1,156 real OCR inputs remain identical, including validation inputs.
- All non-address target values are unchanged: names, city/country fields,
  contacts, identifiers, cargo, relationships and every other label.
- All 782 repaired addresses pass the existing strict `AddressText` validator.
- All 31,066 targets pass the strict V6 model after an **in-memory compatibility
  projection used only for validation** (`addressLine` renamed to `address`,
  schema version pinned to V6). This does not certify that mixed held records
  already form a finalized native address-line training schema.
- All 99 repaired shipper postal blocks equal their labels between the reviewed
  name and government-reference boundaries; all 1,279 fixed offices retain
  exact scoped input support.
- Six functional transaction tests exercise stale hashes, wrong IDs/preimages,
  overlapping edits, duplicate owners, numeric loss, real-OCR mutation, invalid
  address strings and unrelated output changes. These are behavioral tests,
  not configuration-value assertions.

The original 150-record probe is separately replayed against unchanged V3 into
`prior-pilot-replay-verification.json`; its 120 edit-verifier attacks and known
semantic counterexamples remain part of the evidence, not a universal guarantee.

Measured local timings: initial archive/apply 18.53 seconds; corpus screens
21.22 seconds before and 20.21 seconds at the end; main 699-scope application
plus serialized replay 22.89 seconds; final 56-scope application 21.26 seconds;
independent final all-record verification 26.42 seconds. Peak measured process
RSS remained approximately 109–117 MiB. These numbers exclude human review and
documentation; no production performance claim is made.

**New provider calls: 0. New API cost: $0.** No new full-corpus semantic-review
cost projection is inferred from this deliberately evidence-rich repair wave.

### 14.8 Upstream and publication status

The findings are appended to
`repairs/ground015_cargo_text/ADDRESS_TEMPLATE_AND_SYNTHESIS_REPAIR_REQUIREMENTS.md`.
In the current catalog, `e46b2c49` auxiliary slots `slot_0010` and `slot_0011`
represent NVOCC relationship/continuation text but carry organization-valued
auxiliary generation. The generic organization path can reuse the target party
name; it cannot represent that relationship by copying the name into both
positions. A typed relationship/continuation contract and real-source ownership
adjudication are needed before that template is reused.

This execution completed a verified, applied repair wave and full-corpus
discovery. It **did not complete the entire 31k semantic repair**, promote the
template catalog, finish independent cargo holds, or prepare a training release.
No old acceptance flag has been silently carried forward as a new sign-off.
The next work is grouped role adjudication and retained-content repair, using
the recorded current pins and receipts—not another blind regeneration pass.

## 15. Repair versus rebuild: finite workload and an actionable classifier

### 15.1 What this investigation did

The user's question is whether address work can be bounded and automated, or
whether rebuilding synthetic addresses would be faster than repeated repairs.
This pass is diagnosis and planning, not another dataset rewrite. It re-hashed
all four working partitions, joined the complete current record inventory to
saved generation cases, and revalidated all 1,185 existing source-role proposal
pins and their exact address-binding/slot coverage against the catalog files.
No paid request, synthesis, production edit or dataset change occurred.

The reproducible audit is `execution-20261001/strategy_scope.py`; its isolated
outputs are `execution-20261001/strategy-scope/summary.json` and
`source-work-manifest.jsonl`. The manifest contains exact source IDs and every
current real/synthetic descendant ID, not a sample extrapolation. Source-role
proposals are explicitly **not semantic approvals** or whole-party maps.

The successful census took 23.95 seconds and 111.8 MiB peak RSS. Ruff check and
format checks passed. The six existing edit/replay behavioral tests also passed;
they prove transaction properties, not correctness of proposed source roles.
An initial scope assertion correctly prevented output when the generation index
included 20 records outside the current dataset. The corrected inventory keeps
these 20 IDs explicit and excludes them from current repair counts; it does not
restore them to the dataset. All current synthetic IDs still require exact
campaign/source joins and an existing saved target.

### 15.2 Exact current work queues, without overlapping screen counts

| Existing evidence | Synthetic source families | Current synthetic documents | Meaning |
| --- | ---: | ---: | --- |
| Address-slot proposals contain postal roles, with no `contact_or_name` or `unresolved` role | 1,015 | 23,413 | First batch for source-boundary completion and descendant checking, not presumed clean |
| Proposals include mixed/nonpostal or unresolved roles | 170 | 4,149 | Explicit source-role adjudication needed before an edit is authorized |
| No recorded role proposal | 121 | 2,348 | Prepare source evidence/ownership mapping; lack of a proposal is not a defect |
| Total synthetic | 1,306 | 29,910 | Every current synthetic record has its saved generation target |

All 1,185 proposals matched their recorded source/template hashes and exact
address-binding groups/slots. This prevents reuse of stale evidence; it does
not prove that an old model's semantic classification is correct or that static
postal text outside address bindings was inventoried.

There are separately 1,156 immutable real records. Across real and synthetic
records there are 1,811 unique source IDs: 1,306 have synthetic descendants and
505 are real-only. The real sources shared with synthetic families can reuse
source interpretation, but their labels must be checked against actual real
OCR and their inputs cannot be rebuilt.

The largest 50 synthetic source families account for 6,197 documents; the largest
100 account for 9,326. This supports source-level batching. It does not mean that
one reviewed descendant certifies all variable generated addresses in a family.

### 15.3 The filter must return an action, not an unqualified score

Recommended per-address dispositions, rolled up to document level:

1. **Keep unchanged:** independently established postal ownership and complete
   printed coverage; the current address label equals the selected postal text
   under the fixed formatting policy; all repetitions and auxiliary spans have
   resolved roles. Missing labels, including absent-party/address cases, are
   explicitly accounted for. Passing a word-presence screen is insufficient.
2. **Deterministic repair:** those same boundaries are established and a unique
   edit is known, such as completing a label or deleting a proven redundant
   locality occurrence. Every removed span has a reason and a retained equivalent
   where it is a duplicate. No unrelated text or target changes are permitted.
3. **Rebuild one synthetic address:** replacement boundaries are established but
   the current postal contents are fragmented or contaminated enough that
   reconstructing one coherent address is simpler. Reuse sound saved content;
   generate new content only when needed. Construct the label from the final
   rendered postal content. Do not restore prior removed placeholders or clobber
   intervening identifiers, contacts, fixed context, or separate parties.
4. **Source/ownership decision:** boundaries, owner, continuation, or meaning of
   a repeated span is genuinely unresolved. This is a specific grouped decision,
   not a request to keep rerunning a formatter or generating more candidates.

A source plan specifies party headings, postal and nonpostal regions, separate
locality occurrences, fixed postal text, repeats and continuations. Plans are
data executed by the common editor, not a growing set of source-specific Python
repair branches. Existing proposals bootstrap the plan; they are not the plan's
independent approval. Dynamic generated text still needs checking for intruding
names, localities or auxiliary material that invalidates a source-only assumption.

PydanticAI/GLM can prepare compact span/role proposals from complete source-party
context, and resolve narrowly described disagreements in batches. Output should
be offsets/piece IDs and roles, with exact quote checks, not unconstrained rewritten
addresses. A separate source-coverage review must inspect both included and
excluded text; two calls agreeing on an incomplete crop do not establish coverage.
Semantic disagreement or a counterexample goes to a concrete decision rather
than becoming another broadly phrased repair request.

An address-ready **document** requires every party address/occurrence to be
accounted for. An individual repaired address does not promote other parties.
The scope remains extraction quality: no deliverability certification, postcode
geocoding, or new requirements on the administrative granularity of an address.
Real OCR remains immutable. Numeric and nonpostal preservation remains mandatory.

### 15.4 Why unconditional regeneration is not a demonstrated shortcut

The hard shared prerequisite is knowing what text may be replaced. Fresh words
do not resolve a mixed company/address boundary or an address continuation after
contact information. They also do not fix a renderer that inserts a complete
address and then inserts city/country again.

The existing historical whole-postal-run reconstruction test is relevant:
`synthetic-postal-run-transfer-transfer40-safe-v4.summary.json` processed 13 of
32 families and held 19. The holds included ten families with separate postal
runs, seven without cleared postal boundaries, one coordinate-rebase failure,
and one changed party identity. This was reconstruction from saved data, not a
new paid-generation benchmark or a current quality approval. It demonstrates why
whole-address replacement still needs the same ownership evidence.

Consequently the preferred path is **shared source discovery, then choose repair
or address-only rebuild per party using the same verifier**. Blanket regeneration
would add new text to validate and risk undoing earlier repairs. Address-only
rebuilding should be preferred locally when it demonstrably simplifies a malformed
synthetic party; it should not be prohibited merely because repair was attempted
first. There is no measured evidence here that regenerating everything is 10x
faster, nor a new all-in repair/rebuild cost projection.

### 15.5 Finite execution and stopping rules

The next execution should consume the complete manifest, not hand-select another
50- or 100-record repair cohort. First use existing reviews/proposals to assemble
and verify shared source plans; batch the missing and ambiguous source decisions.
Then classify **every descendant** into one of the four actions above. Before
bulk mutation, report counts of unchanged-ready, exact-repair, address-rebuild and
specific source decisions, with their IDs and the evidence needed for each.

Keep source-decision work separate from execution retries. Once the source plan
is resolved, execute it across all compatible descendants. If a local address is
not repairable by that plan, choose a bounded address-only rebuild where its
boundaries are known. If the boundary is unknown, more regeneration cannot fix
that prerequisite. Report the exact evidence/decision needed rather than cycling.

Validation is fixed before execution: complete address/party coverage, exact
retained postal content, explained duplicate removal, protected content/number
preservation, immutable real OCR, and exact text/label replay. Challenge source
interpretation with missing-city/unit, wrong-owner, auxiliary-text contamination,
alias repetition, continuation and restored-placeholder counterexamples. A blind
quality sample is additional evidence, not a substitute for full-record checks.

Retain current hash-bound decisions. A later change invalidates only the affected
evidence/party contracts, rather than reopening the entire dataset. A discovered
defect class requires a regression check and one sweep over its complete affected
set before that class is closed. Neither a family majority nor absence of a screen
flag creates automatic approval.

This defines a finite, auditable repair workflow. It is not a claim that an
arbitrary OCR semantic decision can be mechanically guaranteed correct. The
current implementation has the evidence inventory and exact editor; the complete
four-action classifier is the remaining integration deliverable, not something
this diagnostic pass silently declares finished.

## 16. Correction: routing categories do not supply a semantic classifier

The user correctly challenged section 15: listing keep/repair/rebuild/review does
not explain how to establish the reference that makes those decisions reliable.
The historical model flow already returns exact line/substrings, duplicate pairs,
roles/explanations and explicit review outcomes (`address_wave_model.py`). Merely
recommending PydanticAI span selection and a second reviewer is not a new remedy.

Read-only code inspection and a fresh isolated repro confirmed these particular
limitations (no model calls or dataset changes):

- `address_wave.context` locates context from existing name/address and saved
  address matches. Completeness is not established independently of those values.
- `address_wave_model.verify` protects numbers found in the old address. A printed
  postcode absent from the old address does not become a required number through
  this rule. A fixture with old `10 MAIN ROAD` and printed postcode `11765` showed
  that this guard protects `10`, not `11765`.
- `address_wave_validate.missing_printed_fragment` returned no missing fragment
  for three constructed examples: omitted `UNIT 804`, a consignee address used
  as a shipper label, and a correct label beside a duplicate raw locality tail.
  These reproduce defects in that support check, not a new end-to-end replay of
  every historical reviewer and release gate.
- `address_blind_audit.party_map` enumerates only parties already carrying
  `addressLine`. It cannot itself discover a wholly missing address annotation.

Thus stronger prompts are not enough: selected snippets, exact replay and a
critic can agree while sharing an incomplete reference. A complete *source-party*
interpretation is still a semantic task; hashes and model agreement do not prove
it. The 23,413-document postal-proposal queue must not be promoted to a reliable
automatic keep/repair count on the strength of its role names.

### 16.1 Regeneration comparison must be fair

The 13-of-32 historical test exercised saved-address reconstruction against the
then-existing template/ownership machinery. It was not a comparison with a newly
corrected template/synthesis implementation. It therefore cannot establish that
**fix templates first, then regenerate** would fail or be slower. That alternative
is valid and was inadequately represented by the earlier response.

The shared prerequisite remains mapping postal versus protected source content,
including repeats and continuations. Once that is solved, construction-based
synthetic replacement can avoid an additional semantic repair decision for each
malformed descendant. The exact cost/speed advantage has not been benchmarked.

### 16.2 Revised decision recommendation

Do not launch another bulk retrospective clean/not-clean classifier based on
these proposals. Prefer a **source-first address construction comparison**, with
current preservation/repair available where the evidence is already sufficient:

1. Define each source party's complete postal occurrences and protected regions,
   using full source context rather than old address labels as the discovery
   inventory. Reuse existing evidence and propose missing spans with a model;
   independently resolve uncertain ownership. This common step cannot be
   eliminated by either repair or regeneration.
2. For synthetic parties, use one authoritative postal bundle and one explicit
   rendering plan for that bundle. City/country occurrences refer to those same
   facts; the renderer must not append separate localities to an independently
   generated full address. Repeated whole party blocks remain separate legitimate
   occurrences. Interleaved names, IDs and contacts stay protected.
3. Produce `addressLine` from the known final postal rendering, not by asking
   another model to extract from its own edited text or stripping old components.
   Retain printed order/punctuation under the agreed whitespace/line-join policy.
4. For already reviewed existing addresses, keep or perform exact repairs. For
   remaining malformed synthetic groups, compare reusing sound saved content
   with generating a new coherent postal bundle through the corrected plan.
   Replacing postal content is an explicit address-rebuild operation, not a
   licence to undo earlier unrelated repairs or regenerate full documents.
5. Real OCR is immutable and still needs source-grounded extraction/annotation;
   construction is not a substitute for reviewing its actual printed information.

The measurable comparison must include all protected-field and full-occurrence
checks, the adversarial cases above, costs including source review and failed
calls, and counts of unresolved source decisions. A successful toy formatter or
another critic acceptance percentage is insufficient. The source-plan coverage
must be measured across the complete fixed inventory before bulk replacement.

This is a recommendation for the next bounded implementation/measurement step,
not a claim that corrected-template reconstruction has already been implemented,
benchmarked, or proven corpus-wide. No paid run or dataset write was authorized
or performed by this diagnostic follow-up.
