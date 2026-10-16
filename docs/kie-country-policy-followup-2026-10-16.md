# Country policy follow-up: language context, source frames and preferred names

Date: 2026-10-16. **Investigation/proposal/probes only. No active policy,
dataset, template authority, production code or training configuration changed.**

This builds on the [country rendering probe](kie-country-rendering-probe-2026-10-16.md)
and the [country/base-model audit](kie-country-presentations-audit-2026-10-15.md).
It tests the user's proposed distinction: preserve address/locality text while
normalizing every country field to one stable, readable English name.

Later evidence: [end-to-end180-variant probe and complete implementation inventory](kie-country-endtoend-probe-2026-10-16.md).
That follow-up resolves the nine former postal/registration holds from each
party's own printed ISO2, provides occurrence-specific ambiguity adjudications,
and includes the older-UAE sidecar in the proposed migration scope. Earlier
hold counts below remain the initial scan, not the final adjudication outcome.

## 1. EGITTO: which parties are Italian?

There are **three EGITTO country targets across two current real/validation
documents**. The addressed parties are Egyptian, not Italian.

| Document | EGITTO belongs to | Italian context |
|---|---|---|
| Real row257 / admitted source `89227fc3` | Consignee and notify: EGYPTIAN BRITISH CO. (EGYBRIT), Giza, Egypt | Shipper SASOL ITALY SPA, Milan, ITALY; Italian issuing carrier agent; Genoa loading/issue context |
| Validation row24 / `217e911a`, not an admitted synthesis template | Forwarding agent UNIFREIGHT GLOBAL LOGISTICS, Alexandria, Egypt | Shipper DRAGO SPA, Lessona, Italy; document also refers to Italian law/Milan |

Evidence:

- [Training source, shipper and the two EGITTO occurrences](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_89227fc3b249d56ae4d3bed3dcd8e4f67f463f1438b7fe3825b3e4236618217b/source.txt#L3).
  The country occurrences are lines23 and31; the Italian shipper address is lines3–5.
- [Source country targets](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_89227fc3b249d56ae4d3bed3dcd8e4f67f463f1438b7fe3825b3e4236618217b/target.json#L54).
- [Validation OCR, forwarding party at lines24–27](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_217e911aa93695904825147cdeef12acaedc9284bcbba37883e0407ea2dff428/ocr.txt#L24).

The validation document mixes presentations: its other Egyptian parties print
`EGYPT`, `(EG)` and `EGYPT - (EG)`. Therefore language need not be uniform even
within one document. Neither the addressed party's nationality nor the country
being rendered should be used as the sole language gate.

The previous English-profile restriction concerned **new synthetic alias
selection**. It was not a rule to discard an existing printed foreign country
name during extraction. An existing `EGITTO` can be recognized as Egypt regardless
of whether the synthesis profile would choose to introduce it.

## 2. A bounded, registry-backed language approach

Use a configurable language allowlist, then make a language **eligible** when
the sampled shipment has explicit relevant context: e.g. Italian shipper/origin
or a reviewed Italian issuer/template-field profile. Select a profile once for
the intended region(s); do not independently randomize the language of every
country token.

This means an IT→EG shipment can render the Egyptian consignee's country as
`EGITTO`. It does not require the consignee to be Italian. Conversely, CN→EG
without other Italian context does not justify introducing `EGITTO` merely
because Italian names are present somewhere in the registry.

The contained probe adds pinned CLDR territory-language metadata, rather than
hardcoding a country→language list. It uses national `official` or
`de_facto_official` statuses; regional/minority-language entries do not alone
enable a language for the whole country. English remains the explicit
international base profile. This is a **synthetic plausibility/eligibility rule**,
not a claim that the national language proves an actual document's language.

Illustrative allowlist for this probe: `en/it/fr/es`. The vocabulary still holds
German, Portuguese and Turkish names; they remain independently configurable.
This avoids automatically activating every language or using brittle
string-similarity rules to decide whether a translated name is "too different."

| Sampled context | Eligible probe profiles | Verified example |
|---|---|---|
| IT + EG | English, Italian | EG → `EGITTO` |
| IT + AE | English, Italian | AE → `EMIRATI ARABI UNITI` |
| CN + EG | English | Requesting Italian explicitly raises a visible error |
| FR + EG | English, French | French is eligible |
| US + EG | English | Spanish regional status does not automatically enable Spanish |

Source-style metadata can prefer an Italian presentation in the existing
EGITTO-owning field when its sampled context remains appropriate. An Italian
origin alone does not force every field to Italian. Repeated copies of one
owned field should use the same choice; separate fields can preserve their
different source styles.

Source: [CLDR territory-language documentation](https://unicode.org/reports/tr35/tr35-info.html#Territory_Language_Information)
and [pinned territoryInfo.json](https://raw.githubusercontent.com/unicode-org/cldr-json/91c267402229a59e3ef2774544f001bf959e8809/cldr-json/cldr-core/supplemental/territoryInfo.json).
Local snapshot:
`artifacts/registries/sources/countries/cldr-json-48.2.3-supplemental/territoryInfo.json`.
SHA256: `b50369a0e1f5434c942dfe45064b39f488604ca0cedac5117e993248c19f2c0a`.
No production consumer was switched to it.

## 3. NAME(CODE) and initialism punctuation

Yes: country alias choice and source typography are separate. Atomic country
names/codes belong in the vocabulary; whitespace, parentheses, line breaks and
separators belong in the source presentation frame.

Existing source evidence includes:

- [01d86535, line12](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_01d86535b4e4e74dfe15a62cfb50d6f1c4389dd741cb37ac8ee9c8064b705228/source.txt#L12): `China(CN)`; line18 `Egypt(EG)`.
- [22fa3887, line11](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_22fa38878d2a936f40e5736b562835b080f7aa1097450cf52bf398ef29b4c524/source.txt#L11): `India(IN)`; line28 `Egypt(EG)`.
- Validation217e911a: country and `(EG)` on separate lines, and `EGYPT - (EG)`.

The first probe vocabulary enumerated spaced `NAME (CODE)` composites. It did
not enumerate every compact/multiline spelling. This follow-up tests structural
parsing/rendering instead of multiplying registry entries for punctuation.
**30 frame probes** cover three country names, five observed presentation
families and both ISO2/ISO3 slots:

```text
EGITTO(EG)
EGITTO (EG)
EGITTO - (EG)
EGITTO
(EG)
EGITTO-EG
```

All resolve to EG and project to `EGYPT`. Equivalent Kenya and DRC controls
pass. Conflicting pairs such as `CHINA(US)` or `EGITTO - (US)` are rejected.
Code slots use the selected identity's actual ISO2/ISO3, not a random short
alias. Tests also preserve each original line's coordinate suffix when the
name changes without changing the frame's physical line count.

The existing 12 international-initialism presentations already include:

| Identity | Available initialism presentations |
|---|---|
| GB | `UK`, `U.K`, `U.K.` |
| AE | `UAE`, `U.A.E`, `U.A.E.` |
| CN | `PRC`, `P.R.C`, `P.R.C.` |
| CD | `DRC`, `D.R.C`, `D.R.C.` |

Dotted ISO forms such as `U.S.` and `U.S.A.` are separate from this initialism
family. Ambiguous `A.R.E` remains excluded from context-free sampling; undotted
ISO3 `ARE` remains UAE. No blind punctuation removal was introduced.

## 4. Proposed common country-target policy

Recommended wording, **not yet an active field-description change**:

> Country explicitly supported by this role's OCR country wording, code or
> recognizable unambiguous alias. Normalize to the pinned preferred English
> country name in uppercase. Preserve the printed country presentation in its
> owned address text. Do not infer a missing country from a locality, telephone,
> company name or another party. Flag unresolved identity or ownership for
> review rather than inventing a country.

Apply this to parties, route locations, place of issue and freight payment
place. Locality names remain source-faithful. For example:

| Printed input | Address/locality target | Country target |
|---|---|---|
| `GIZA, EGITTO` in a party address | Address retains `GIZA, EGITTO` | `EGYPT` |
| `GIZA, EGYPT(EG)` in a party address | Address retains its owned printed wording/frame | `EGYPT` |
| `Place Issued: MOMBASA,KE` | `MOMBASA` | `KENYA` |
| `EL ISKANDARIYA (ALEXANDRIA)` with no owned country | Retain that locality wording | Absent |

The fixed preferred-name table is the normalization authority, not an LLM's
choice among plausible synonyms. The
[249-name proposal and provenance](analysis/country-policy-followup-20261016/preferred-names-proposal.json)
uses customary CLDR English names with explicit, reviewed editorial choices:

- `DEMOCRATIC REPUBLIC OF CONGO`, not `CONGO, THE DEMOCRATIC REPUBLIC OF THE`.
- `REPUBLIC OF CONGO`, distinguishable from the DRC.
- `UNITED STATES VIRGIN ISLANDS`, not an inverted name.
- `IRAN`, `RUSSIA`, `SYRIA`, `UNITED STATES`, `UNITED KINGDOM`.
- Preserve established complete forward-order names such as `BOSNIA AND
  HERZEGOVINA`, `SAINT VINCENT AND THE GRENADINES`, `TRINIDAD AND TOBAGO` and
  `TURKEY` instead of introducing UI abbreviations or needless spelling churn.

Do not blindly use every CLDR short form: e.g. its short BA name is `Bosnia`,
and several display names use `St.` or `&`. This proposal deliberately keeps
complete existing forward-order names for those cases. Commas are not banned
as characters: `SAINT HELENA, ASCENSION AND TRISTAN DA CUNHA` contains a genuine
enumeration, not an inverted name requiring repair.

CLDR's own guidelines favor customary language-appropriate names rather than
inverted/formal database names. [Naming guidance](https://cldr.unicode.org/translation/displaynames/countryregion-territory-names).
The proposed table is not yet installed as a sampling or extraction authority.
Its editorial choices should be frozen before any full migration.

### Measured scope under that proposed table

These are **candidate representation changes**, not counts of bad existing
labels. Most old party targets correctly follow the old printed-wording policy.

| Cohort | Existing country facts | Candidate changed facts | Documents with candidate changes | Unresolved facts |
|---|---:|---:|---:|---:|
| Real training600 | 2,563 | 182: 179 party +3 location | 146 | 24 |
| Validation60 | 256 | 12, all party | 9 | 3 |
| Synthetic7,000 | 29,042 | 1,081: 669 party +412 location | 562 | 0 |
| Source/template authorities413 | 1,713 | 108 | 87 | 19 |

The three real location renames are `CZECH REPUBLIC`→`CZECHIA` under this
preferred-name proposal, not three wrong geographic identities. Existing
validation location countries already equal the proposed preferred names.
Synthetic renames principally replace formal/inverted registry spellings;
all 29,042 synthetic country identities resolve without a hold.

Exact candidate changes and holds are in the
[normalization inventory](analysis/country-policy-followup-20261016/normalization-inventory.json).
The 49 accepted earlier rendering variants were also copied into the isolated
probe and projected to this normalized-country policy. Their OCR, coordinates,
addresses, locality names, other targets and field presence remain unchanged.

### Remaining contextual adjudications

- Real: 24 bare `KOREA` facts across18 documents.
- Validation: two `KOREA` facts and one `A.R.E` fact across3 documents.
- Source authorities: 16 `KOREA`, two `EGIPT`, one `THE NETHERLANDSS` facts,
  across13 sources. These overlap real-source families; they are not another
  19 independent document defects.

The resolver intentionally does not globally equate `KOREA` with South Korea,
or silently autocorrect source misspellings. Inspection shows several Korea
documents print explicit same-shipment `KR`/`KOREA, REPUBLIC OF` declarations;
these can support a reviewed identity decision when role ownership is clear.
Other documents contain only bare KOREA plus cities/companies. A global alias
or city-based guess is not warranted by the probe.

The prior 48 party-wording mismatches would cease to be wording violations
under the newly proposed normalized-country policy, provided geographic
identity/role support is correct. They must not be "repaired" toward codes
first and then expanded back again. The nine related-declaration holds still
need ownership review; normalization alone does not resolve a postal-versus-
registration-country distinction.

## 5. Non-party localities: current clarity, consistency and coverage

The existing field description is already conceptually aligned:

- Locality `.name`: retain source spelling/distinguishing words and remove
  separately owned country/generic facility captions; do not translate or
  geocode the locality.
- `.country`: normalize explicit role-owned country wording/code/adjective to
  the conventional English country name. Omit when not established for that
  location; do not borrow a party's country.

Actual labels exercise locality preservation: **26 real location-name facts**
retain ISKANDARIYA wording (21 `EL ISKANDARIYA (ALEXANDRIA)`, four bare
`EL ISKANDARIYA`, one reversed parenthetical form); validation retains one
parenthetical example. In `2d77f899`, delivery is `EL ISKANDARIYA (ALEXANDRIA)`
while discharge is independently `ALEXANDRIA`. Both country fields are absent
even though a party address elsewhere prints EGYPT.

[Kenya source](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_08ec8a5ff38973d6b448d6210beae09dfa6daa1df935781d0bcdb31b567b958b/source.txt#L92)
explicitly prints `Place Issued: MOMBASA,KE`; current issue target is
`{"name":"MOMBASA","country":"KENYA"}`. That is already code normalization,
not inference from the city name.

| Cohort | Present location objects | Country present | Country absent |
|---|---:|---:|---:|
| Real600 | 2,406 | 898 | 1,508 |
| Validation60 | 228 | 105 | 123 |
| Synthetic7,000 | 28,393 | 10,110 | 18,283 |

These presence counts do not establish that every omission is correct.
They show why normalization must preserve missing-field semantics. All current
transshipment country fields are absent:11 real and161 synthetic location
objects. Do not fill them from city knowledge during this change.

Current real/validation location countries have **zero code-valued targets**.
Seven real country targets remain the unresolved broad spelling KOREA; three
others use the perfectly intelligible English alternative CZECH REPUBLIC.
Two admitted source-authority targets still use `SI`/`TR`; their current
descendants already print and target full country names.

### Concrete lost synthetic exposure

The fresh native-owner replay covers **all16 admitted source templates with
standalone location-country ISO codes**,34 physical source owners. Across their
242 current descendants, it replays **459 country occurrences /324 distinct
target facts**. Every occurrence prints a full name matching its target; none
retains source-code presentation. There are zero replay mapping errors.

Thus this cohort is text/target-consistent, but no longer supplies source-like
code→name training examples. Alias rendering with unchanged normalized country
targets directly addresses that coverage gap. This is measured missing
presentation exposure, not proof of how much model performance it caused.

Full [location audit and all459 receipts](analysis/country-policy-followup-20261016/location-audit.json)
and [reproduction](analysis/country-policy-followup-20261016/location-audit.py).
The audit additionally identifies a source boundary hold: `22fa3887` has a
loading-country target whose owned source locality span is only `MUNDRA`, while
a separate issue phrase prints `MUNDRA,India`. This needs ownership adjudication
before claiming that another location's country supports loading. Its current
synthetic expression explicitly prints the sampled country, so this is not
proof of a bad current synthetic country target.

## 6. Validation and limits

- **105 combined tests passed** in10.86seconds: existing registry/rendering
  regressions plus32 follow-up tests. Ruff passes.
- All249 preferred names are unique, idempotent and valid `CountryText` values.
- Every3,401 registry presentation retains its expected identity under the
  proposed input resolver. Reviewed observed spellings are **input-only**
  normalization evidence, never an extra sampling pool.
- 30 frame probes pass identity agreement; five frame tests preserve physical
  lines and their original position suffixes. Contradictory names/codes fail.
- All49 full-document country-target counterfactuals pass frozen direct-output
  schema, bound task canonicalization and casing checks. Other text, addresses,
  locality names, fields and coordinates are identical to their inputs.
- **1,660 protected file hashes** match before/after the full probe.
- Full normalization/probe pass:13.95seconds,811.0MiB process peak RSS, including
  existing production-helper imports. Separate location-owner replay:8.84seconds,
 811.0MiB peak RSS. These are audit runtimes, not production throughput claims.
- No paid calls, model/GPU use, or active dataset/source/pipeline edits.

[Probe results](analysis/country-policy-followup-20261016/results.json),
[read-only inventory/probe](analysis/country-policy-followup-20261016/probe.py),
[follow-up tests](analysis/country-policy-followup-20261016/test_probe.py).

This validates coherent deterministic labels/rendering and an integration
boundary. It does **not** demonstrate that the trained extraction model already
normalizes every country alias reliably. The next paired training/evaluation
must measure that, separately from any later ISO2-only-target experiment.

## 7. Recommended next implementation

1. Freeze one preferred English-name table and adopt the same country-field
   policy across parties and locations; retain printed address/locality text.
2. Resolve the narrow contextual/ownership inventory, without a global KOREA
   or malformed-country fallback.
3. Keep sampled ISO identity separate from language eligibility, alias family
   and source frame. Add the presentation plan at the owned-region renderer
   boundary; keep explicit COUNTRY CODE captions code-valued.
4. Update labeling descriptions, synthesis postal/target validation and final
   programmatic country normalization together. Source bindings need precise
   country/frame ownership, not wholesale template rewriting.
5. Migrate country targets once, preserving addresses/localities/other targets;
   then diversify synthetic country presentations, updating owned address
   targets only where the rendered address actually changes.
6. Validate fresh synthesis and full-set preservation before training. Preserve
   current country absences unless a separate ownership audit establishes a
   missing source-supported country.
