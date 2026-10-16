# Country presentations: end-to-end probe and implementation scope

Date: 2026-10-16. **Probe only. No production code, active policy, template/source
authority, dataset, training configuration or training process was changed.**

## Outcome

The proposed separation works in the contained test:

1. Keep the sampled country identity fixed.
2. Choose an eligible presentation for each owned occurrence.
3. Preserve source framing, captions and repeated postal occurrences.
4. Keep the address target faithful to the newly printed address, while all
   country targets use one pinned preferred English name.
5. Verify exact edits, unrelated values and coordinate preservation independently.

**180 counterfactual documents passed**, derived from **60 current synthetic
documents across 30 templates**, covering **80 country identities**. This is
strong evidence for the mechanism and its integration boundaries—not a claim
that production integration or the full dataset migration is already complete.

Evidence:

- [Probe configuration](analysis/country-endtoend-probe-20261016/config.json).
- [All results, source IDs, native proofs and protected-file hashes](analysis/country-endtoend-probe-20261016/results.json).
- [Independent saved-output replay](analysis/country-endtoend-probe-20261016/validation.json).
- [Original synthetic inputs and all 180 positioned variants, with target differences](analysis/country-endtoend-probe-20261016/INSPECTION.md).
- [All-413-template inventory and adjudications](analysis/country-endtoend-probe-20261016/template-review.md).
- [Occurrence-specific ambiguous-country adjudications](analysis/country-endtoend-probe-20261016/adjudication-country-holds.md).
- [Current and older-UAE synthetic migration scope](analysis/country-endtoend-probe-20261016/synthetic-scope.json).

This extends the [previous rendering probe](kie-country-rendering-probe-2026-10-16.md)
and [country-policy follow-up](kie-country-policy-followup-2026-10-16.md).

## 1. Policy validated by this probe

| Field | Proposed contract |
| --- | --- |
| Party `addressLine` | Preserve the owned printed address and punctuation, with the agreed uppercase/line-break normalization. An alias-rendering change updates the same country span in this label. |
| Every `.country`, including parties, route, issue and freight/payment locations | Normalize an explicitly supported, unambiguous identity to the pinned preferred English name. No independent synonym choice by the LLM. |
| Locality `.name` | Preserve the printed locality spelling; do not translate `EL ISKANDARIYA` to `ALEXANDRIA`. |
| Unsupported/ambiguous country | Explicit review hold; no city, telephone, company-name or unrelated-role inference. Do not add absent optional country fields. |

Examples:

```text
Printed address: GIZA, EGITTO
addressLine:     GIZA, EGITTO
country:         EGYPT

Printed issue place: MOMBASA,KE
name:                MOMBASA
country:             KENYA

Printed locality: EL ISKANDARIYA
name:             EL ISKANDARIYA
country:          absent unless supported for that role
```

The [249-name proposal](analysis/country-policy-followup-20261016/preferred-names-proposal.json)
uses forward-order names such as `DEMOCRATIC REPUBLIC OF CONGO`, `IRAN`,
`UNITED STATES` and `UNITED KINGDOM`. This is a proposed contract change for
party countries, whose current description instead copies printed wording.
It is not the separate, later ISO2-only-target training experiment.

The earlier 48 party printed-wording findings should therefore **not** first
be repaired to codes under the old policy and then expanded again. Under this
proposal their country-name targets can be correct; own-role identity and the
printed address still require validation.

## 2. What was exercised

For each of two existing descendants per template, the probe produced three
country-only profiles:

- Weighted eligible-family selection.
- Code exposure where the source frame permits it.
- Localized-name/international-initialism exposure where eligible.

The three profiles are **stress tests**, not proposed production frequencies.
They use existing sampled geography and linguistic values; no new goods,
parties, routes or quantities were generated. All 30 source blueprints were
compiled through the real ownership/auxiliary paths, and their native no-op
render reproduced source text exactly. Current span mapping used pinned native
render receipts, not a document-wide country-string replacement.

| Presentation family | Owned decisions across all profiles |
| --- | ---: |
| English country name | 282 |
| ISO2 | 330 |
| ISO3 | 170 |
| Dotted ISO code | 119 |
| Context-eligible localized name | 149 |
| Approved international initialism | 30 |
| Total | 1,080 |

Of these, **863 actual spans changed**. The probe verified **918 country target
facts**, **5,988 unchanged non-country/non-address values**, and **26,559 physical
lines**. There were 546 postal-target projections, of which 438 changed text;
the remainder were explicit no-ops. All figures count the three counterfactual
profiles separately, not independent source documents.

Additional checks included:

- Twelve owned `NAME(CODE)` frames and six explicit code-caption occurrences.
- Separate ISO2/ISO3 width, dotted typography, compact/spaced/dash/newline frames.
- Repeated postal owners, multiple pages, and different styles in different
  fields of the same document.
- Country-before-postcode and exact road-code negatives (`BR-101`/`BR-116`).
- Existing title/uppercase name rendering, uppercase normalized labels, and
  opaque uppercase ISO/initialism forms.
- A postal label bound to an owned country occurrence even when that party has
  no separate `.country` target. Its address still changes; no new country key
  is invented.
- Contradictory name/code pairs, unavailable presentation families, invalid
  weights and unknown identities produce visible errors.

All 180 outputs passed frozen direct-output JSON Schema, bound task
canonicalization and target-casing checks. A second pass reopened the original
rows and saved variants, replayed exact edits, rebuilt the expected target and
positioned text independently, and verified every output hash.

Automated checks cover every variant. Manual readback covered the country and
address edits for one weighted descendant from each of the 30 templates, plus
the separately documented ambiguity/mapping controls. This is not represented
as a fresh human line-by-line review of all unrelated text in all 180 outputs;
that text is protected by exact edit replay.

## 3. Alias registry and language eligibility

Reuse the pinned, corpus-independent
[ISO/CLDR registry](../artifacts/registries/compiled/country-presentations-iso-cldr-48.2.3-v1/registry.json):
249 identities and 3,401 presentations across seven supported languages.
Every presentation was rechecked against the extended structural resolver.

Sample a **family first**, then a presentation. Registry row counts must not
become implicit family weights. Eligibility is constrained by identity,
language and source frame; unavailable families are explicitly reported and
weights are renormalized only under that configured policy. An impossible
request is an error, not a guessed alias.

Two-/three-letter aliases remain ISO codes or reviewed international initialisms
(`UK`, `UAE`, `PRC`, `DRC`, including dotted forms). Local shorthand such as
Spanish `RU` for UK remains excluded. `A.R.E` is not a globally sampled alias;
undotted ISO `ARE` still denotes AE.

The probe uses an explicit `en/it/fr/es` allowlist. English is the base; national
official-language metadata makes another language eligible, rather than proving
the actual document language. This is a configurable synthesis plausibility
rule. CLDR distinguishes national `official`/`de_facto_official` from regional
and minority statuses; the probe does not treat them as equivalent.
[CLDR territory-language specification](https://unicode.org/reports/tr35/tr35-info.html#Territory_Language_Information).

One reviewed source additionally carries literal `CMA CGM ITALY` issuer context.
That evidence must remain present in both source and current render before
enabling its Italian profile. The addressed party itself need not be Italian.
Regression probes explicitly cover `EGITTO` for EG and `EMIRATI ARABI UNITI`
for AE. Existing foreign names remain recognizable for extraction even when
their language is ineligible for new synthetic sampling.

Normal country names containing punctuation must remain atomic vocabulary
entries: `ROYAUME-UNI`, `PAYS-BAS`, `CONGO (DRC)` and `MYANMAR (BURMA)` are not
arbitrary name/code frames. An early structural probe caught that distinction;
the final resolver checks registered names before structural composition.

## 4. Exact dataset migration scope

The following counts are **representation changes under the proposed name
table**, not automatically bad labels under the old contract. These come from
the exhaustive target inventory, not extrapolation from the 180 variants.

| Cohort | Country facts | Directly resolvable candidate changes | Documents with such changes |
| --- | ---: | ---: | ---: |
| Current real training600 | 2,563 | 182 | 146 |
| Validation60 | 256 | 12 | 9 |
| Current synthetic7,000 | 29,042 | 1,081 | 562 |
| Source authorities413 | 1,713 | 108 | 87 |

Real changes comprise 179 party and three location names; validation changes
are party names. Synthetic changes comprise 669 party and 412 location names.
The real location renames are `CZECH REPUBLIC`→`CZECHIA`, not identity corrections.

The older-UAE sidecar contains 2,500 rows, of which 1,500 already appear in the
current 7,000. The 1,500-row addition sidecar adds no further unique rows.
All overlapping full records agree exactly. Together these files contain
**8,000 unique synthetic records /33,104 country facts**. The proposed preferred
names change **1,215 country facts across633 unique records**. All identities
resolve; there are **zero synthetic target-identity holds**.

This does not mean all 8,000 current raw texts have already passed country-span
migration. Target normalization, source ownership and current-input span mapping
are separate checks. Full rendering migration must account for the old-UAE
sidecar, deduplicate by document ID and propagate one identical repaired record
to every retained copy.

### Contextual country adjudication

The previous exhaustive scan held 46 real/validation/source country facts.
Occurrence-specific inspection supports:

- Nine real and two validation `KOREA` facts through explicit same-role country
  declarations, matching exporter identifiers or address continuations.
- Five source-authority facts through role-linked evidence.
- Two additional source `EGIPT` facts as separately reviewed local typo
  adjudications, not a global typo alias.

The remaining **28 hold rows represent17 distinct fields across13 documents**
after source/real overlap is deduplicated. They need a recorded review decision
before an all-or-nothing migration; this probe did not silently equate every
`KOREA` with South Korea or dotted `A.R.E` with either Egypt or UAE.

The **nine older postal-versus-registration holds are resolved under the new
policy**: their own addresses print the matching ISO2. This supersedes the
earlier statement that registration-country fallback was needed. No such
fallback is necessary or proposed.

See the [complete per-occurrence adjudication inventory](analysis/country-endtoend-probe-20261016/adjudication-country-holds.md)
and [independently replayed120 evidence spans](analysis/country-endtoend-probe-20261016/adjudication-validation.json).

## 5. Templates, owners and remaining mapping work

All **413 templates** were inventoried: **1,713 source country facts** and
**2,057 physical country-linked owners**. Country facts have matched owned tokens
in399 templates. The remaining14 sources trigger an identity, typography or
source-evidence review—not14 proven ownership rebases.

| Physical owner type | Count |
| --- | ---: |
| Whole postal region with matched country token | 1,177 |
| Precise country slot | 768 |
| Linked component without a matched country token | 106 |
| Whole location with matched country token | 5 |
| Other country-bearing composite | 1 |

The 106 unmatched components often have their country in a separate exact
owner; they are not106 missing countries. The audit retains2,563 owner→fact
edges, including19 existing reviewed co-bindings omitted by the older
contract-only count. Inventory accounting was corrected; no ownership changed.

**No blanket template rebase is justified.** Precise slots are reusable, and a
whole-postal owner needs a verified internal country span—not replacement of
its entire address by an alias. Source `NAME(CODE)` instances already have
usable name/code anchors; their surrounding frame must be retained.

The representative36-case current-input mapping screen holds seven physical
owners across six sources. Each was opened and has visibly printed country
evidence; generated wrapping or a `TEL` suffix breaks the old whole-line mapper.
The primary30-template selection separately encounters three explicit
line-frame holds. Readback of both descendants for each held source confirms
all eight held country occurrences are printed in their own postal blocks;
the country line shares a contact or commercial-register caption. See the
[six-document readback](analysis/country-endtoend-probe-20261016/template-primary-holds.md).
None was admitted by loosening a guard. Full migration needs
an independently proven current-owner projection or reviewed occurrence spans
for these cases; a global OCR country search is not acceptable.

Source authority/evidence controls remain separate from alias rendering:

- `01d86535`: `HONGKONG` postal text and separately declared `CHINA(CN)` are not
  interchangeable aliases.
- `fb241e5d`: source location targets `SI`/`TR` need name normalization; current
  descendants already have full-name country targets.
- `22fa3887`: loading source owner prints only `MUNDRA`; `India` occurs under
  issue place. Descendants explicitly render loading country, so they are not
  automatically defective. Source evidence needs its own decision.
- `c7060e22`: notify `ARE` versus source target EGYPT is a contextual identity
  issue, not permission to override undotted ISO3 ARE globally.

The [template inventory](analysis/country-endtoend-probe-20261016/template-review.md)
lists all18 conservative source fact-screen candidates, exact source byte
spans, current mapping holds and negative controls.

## 6. Required production implementation, in order

The probe establishes the following concrete work; none is marked implemented
merely because the shadow counterpart passed.

| Layer | Required action | Publication guard |
| --- | --- | --- |
| Policy/schema descriptions | Adopt the same pinned preferred-English country rule for parties and locations; retain printed address/locality policy. Update extraction/review/synthesis descriptions and frozen contracts together. | No mixed old/new country contracts; update schema hashes and task constraints. |
| Shared country registry | Promote the verified vocabulary, preferred names and provenance into a production consumer. Keep observed input-only spellings separate from sampling aliases. | Collision tests, atomic-name-before-frame parsing, pinned sources. |
| Sampling | Keep ISO identity fixed; configure family weights, language allowlist and reviewed issuer/field profiles. | Explicit eligibility record, deterministic seed, visible impossible-request errors. |
| Template/source presentation | Preserve exact code captions and `NAME(CODE)`/multiline framing; use existing exact owners wherever sound. Adjudicate only the documented exceptions. | Exact owner/provenance coverage; no blanket rebase or cross-role borrowing. |
| Postal/lexical generation | Generate and validate canonical geography first, then apply an explicit presentation plan to verified owned country spans. | Address text and every linked address target use the same selected presentation. |
| Validators | Teach validators the planned country identity/presentation and owned postal subspan; compare identity rather than raw name equality. | Retain locality, country-occurrence-count and contradictory-identity checks. |
| Final assembly | Normalize all country targets; project printed address targets; apply existing target casing. | Other values and optional-field presence unchanged. |
| Existing dataset migration | Cover current7,000 plus retained sidecar records, real600, validation60 and source authorities. Resolve/report exceptional fields before publication. | Exact inverse replay; deduplicated copies agree; manifests/schema/task pins regenerated atomically. |
| Positions | For these country-only, same-line edits retain every coordinate suffix and blank ` ||`. Ensure future native render receipts include the presentation operation. | No dropped text, line-count change or coordinate-coverage change. |
| Final validation | Run native fresh synthesis after integration as well as replay on existing records. | Text/target/ownership/frame/geometry checks on the actual production path. |

### Why current validators must change

Actual production controls show:

- `GIZA, EGYPT`: both current postal-owner and geography validators accept it.
- `GIZA, EGITTO` or `GIZA, EGY`: current validators reject valid identity-preserving
  presentations.
- `GIZA, EGYPT(EG)`: the owner check accepts it, but the geography validator
  counts it incorrectly and rejects it.
- `DUBAI, UAE`: current checks reject it against canonical UAE country text.

Therefore alias rendering cannot be bolted onto the sampler alone. The clean
boundary is **canonical shipment facts → canonical postal generation/validation
→ owned presentation plan → labels/rendering → plan-aware independent checks**.
Do not merely weaken existing validators to accept arbitrary text.

The counterfactual phase also found two probe-specific guard mistakes: requiring
unchanged normalized country targets to remain literal in aliased input, and
interpreting registered hyphenated/disambiguated country names as code frames.
Both were corrected and regression-tested before final certification. The
initial rejected output is retained separately; it is not a dataset defect count.

## 7. Validation, measurements and limits

- **138 combined tests passed in11.59seconds.** Ruff passes.
- All249 identities and all3,401 presentations preserve semantic identity.
- All180 frozen outputs pass schema/canonicalization/casing checks and independent
  edit/target/geometry replay.
- **1,663 protected dataset/template/source files** were rehashed unchanged,
  including both older-UAE sidecars.
- Complete30-template/180-variant probe: **16.91seconds**, peak **310.6MiB RSS**.
- Independent saved-output validation: **6.50seconds**. Target normalization
  median **0.117ms**, p95 **0.164ms** per variant; exact input/coordinate replay
  median **0.166ms**, p95 **0.405ms**. These measure the stated audit components,
  not total synthesis throughput or a production before/after benchmark.
- No paid calls, GPU work, model loading, production or active dataset changes.

Coordinates are retained as the existing coarse line-position signal; this is
not a claim that a longer country name has the exact centroid it would acquire
after physically re-rendering a PDF. No physical lines were added or removed,
and coordinate coverage is identical for every pair.

The probe validates deterministic presentation/label consistency. It does not
prove model accuracy for unseen aliases or settle final training exposure
weights. Those require a later paired training/evaluation. The production
integration must still close the explicit source/mapping holds and pass native
fresh-synthesis tests before certifying a complete migrated dataset.

## Reproduction

Production follow-through is now complete in the
[country-policy implementation and R27 migration report](kie-country-policy-production-repair-2026-10-16.md).
The results above remain the historical isolated probe; they are not the
production migration's sample counts or final certification.

```bash
uv run python docs/analysis/country-endtoend-probe-20261016/probe.py
uv run python docs/analysis/country-endtoend-probe-20261016/probe.py --verify
uv run python docs/analysis/country-endtoend-probe-20261016/probe.py --scope
uv run pytest --import-mode=importlib \
  docs/analysis/country-rendering-probe-20261016/test_registry.py \
  docs/analysis/country-rendering-probe-20261016/test_probe.py \
  docs/analysis/country-policy-followup-20261016/test_probe.py \
  docs/analysis/country-endtoend-probe-20261016/test_endtoend.py -q
```
