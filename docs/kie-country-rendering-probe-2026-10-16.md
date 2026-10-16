# Country-presentation registry and contained rendering probe

Date: 2026-10-16. **Probe and preparation only. No production synthesis,
template/source authority, dataset, training configuration or country-target
policy was changed. No model was loaded and no paid calls were made.**

Follow-up: [language context, source frames and the proposed shared normalized-country policy](kie-country-policy-followup-2026-10-16.md).
That later probe keeps printed addresses/localities while testing normalized
country targets for both parties and locations. This report's original target
projections remain evidence of the prior/current policy, not a silent migration.

## 1. Conclusion

Country-presentation variability is feasible without resampling geography or
changing localities. A pinned ISO/CLDR vocabulary supplies all 249 ISO country
and territory identities independently of the document corpus. The scoped
rendering probe passes **49 variants from ten existing synthetic documents**,
covering 20 identities. Two deliberately requested full-name/composite variants
are correctly held because their template explicitly prints `COUNTRY CODE`.

This is **not entirely a rendering-only repair** under the current contract:
location countries normalize to existing English targets, but party countries
copy printed wording. Party-country and affected postal-address targets must
therefore follow the selected presentation. Geographic identity remains fixed.

The future direct-ISO2-target experiment is now documented in
[proposed improvements](mpci-bl-proposed-improvements-2026-10-06.md), with links
to the [existing base-model and country audit](kie-country-presentations-audit-2026-10-15.md).
That experiment is not implemented or launched here.

Evidence and review material:

- [Final results, owner proofs, preservation pins and measurements](analysis/country-rendering-probe-20261016/results.json).
- [All 49 original-current-input versus variant pairs, including positions and exact country/address target changes](analysis/country-rendering-probe-20261016/INSPECTION.md).
- [Pinned compiled vocabulary](../artifacts/registries/compiled/country-presentations-iso-cldr-48.2.3-v1/registry.json)
  and [manifest](../artifacts/registries/compiled/country-presentations-iso-cldr-48.2.3-v1/manifest.json).
- Reproduction: [registry preparation](analysis/country-rendering-probe-20261016/registry.py),
  [document probe](analysis/country-rendering-probe-20261016/probe.py),
  [registry tests](analysis/country-rendering-probe-20261016/test_registry.py),
  [probe tests](analysis/country-rendering-probe-20261016/test_probe.py).

## 2. Target-policy check: what can remain unchanged?

The current source class and frozen R26 schema agree on these distinct roles:

| Field | Current contract | Effect of changing only the printed country presentation |
|---|---|---|
| `parties.*.country` | Copy the party's supported printed country wording/code; uppercase targets. | Project the selected printed presentation into the target. |
| `parties.*.addressLine` | Source-supported address, including its country. | Change the same owned country span in the address target; preserve the remaining address. |
| `route.*.country`, `placeOfIssue.country`, `freight.paymentPlace.country` | Normalize to the established English country target. | Keep the target unchanged if geographic identity is unchanged. |
| Location `.name`, party names, city, postcode, contacts, goods and all other targets | Existing contracts unchanged. | No change. |

For example, when `UNITED ARAB EMIRATES` becomes `UAE`, a party target becomes
`UAE`; a discharge-country target remains `UNITED ARAB EMIRATES`. Neither change
alters the destination or learns a city-to-country inference.

The fresh R26 recheck confirms all **57** previously inventoried country/address
facts remain unchanged after the 1,000-description refresh:

- **48 confirmed party-wording violations**, across 36 documents/nine sources.
  The input prints a code while the target expands it, despite the copy-printed
  party policy.
- **Nine related exporter/registration-country holds** remain separate. They
  are not automatically nine wrong geographic identities or bad labels.
- Source `fb241e5d` still has two code-valued location targets, `SI` and `TR`,
  against the location-normalization policy. Its current descendants already
  have full-name targets; this is a source-authority issue, not two bad current
  synthetic targets.

The exact evidence remains in the [earlier owner adjudications](analysis/country-presentations-audit-20261015/owned-country-checks.json).
No repairs were applied during this probe. The field descriptions are at
[party country](../src/document_ocr/label_schemas/bill_of_lading_v7.py#L596)
and [location country](../src/document_ocr/label_schemas/bill_of_lading_v7.py#L68).
Existing normalized targets sometimes use formal ISO registry names; this pass
does not silently replace those with CLDR's preferred customary names.

## 3. Vocabulary: contents, sources and eligibility

The shared, experiment-independent asset location is:

```text
artifacts/registries/sources/countries/cldr-json-48.2.3/
artifacts/registries/compiled/country-presentations-iso-cldr-48.2.3-v1/
```

Identity comes from the existing pinned `iso-codes-4.9.0-1` snapshot.
Presentations combine ISO names and codes with Unicode CLDR **48.2.3**, commit
`91c267402229a59e3ef2774544f001bf959e8809`. CLDR is a language-tagged display-name
authority, not a promise that every variant is appropriate in every shipping
document. Its own guidance distinguishes customary names, short forms and
context-dependent variants. [CLDR guidance](https://cldr.unicode.org/translation/displaynames/countryregion-territory-names),
[pinned release](https://github.com/unicode-org/cldr-json/releases/tag/48.2.3).

| Presentation family | Distinct records, all eligible languages | English/international records |
|---|---:|---:|
| Country names, including supported alternate/formal/common names | 1,427 | 470 |
| ISO alpha-2 | 249 | 249 |
| ISO alpha-3 | 249 | 249 |
| Dotted ISO presentations, optional terminal dot | 994 | 994 |
| Documented international initialisms, including their dotted presentations | 12 | 12 |
| English `NAME (ISO2)` presentations | 470 | 470 |
| **Total** | **3,401** | **2,444** |

Counts are deduplicated by country identity plus presentation, not by how many
languages or providers repeat the same spelling. Languages are
`en/fr/de/es/it/pt/tr`. This first registry intentionally uses Latin-script
names compatible with the current extraction text contract; it is not a claim
of exhaustive worldwide language or historical/OCR-alias coverage.

Each record contains `code`, `surface`, `family`, `languages`, and `provenance`.
Names are uppercase for target projection; existing document-casing rendering
should still operate on the chosen surface during production integration.
Source files, Unicode license, official-initialism evidence and compiled data
are hashed. The loader verifies the registry, ISO snapshot and every recorded
source hash. Recompilation is byte-identical.

Final registry SHA256:

```text
6d6a9a473199aed6c63940b22e28050b697eadc90810e46595cdb6148d55639d
```

### Short aliases: the user's restriction

Two-/three-letter abbreviations are eligible only as ISO codes or explicitly
documented international initialisms. Approved additions are UK, UAE, PRC and
DRC, with separately documented dotted presentation derivations.

- PRC identifies CN, supported by the [Chinese Embassy's formal-name statement](https://ng.china-embassy.gov.cn/eng/jrzg/ff/).
- DRC identifies CD, supported by [GOV.UK's DRC wording](https://www.gov.uk/foreign-travel-advice/democratic-republic-of-the-congo/safety-and-security).
- UAE is supported by [UAE MOFA](https://www.mofa.gov.ae/en/THE-UAE).
- UK is a pinned English CLDR short form; its ISO identity remains GB.

Excluded local-language shorthand: Spanish `RU` for the UK, Turkish `BK` and
`ABD`, Portuguese `EUA`, and French `R.-U.` and `É.-U.`. ISO `RU` still correctly
identifies Russia. This rule filters abbreviations, not genuine short names:
Turkish `Çad`, the country name for Chad, remains a language-eligible name.

`A.R.E` and `A.R.E.` are excluded from context-free sampling because Egypt's
official initialism collides with a dotted presentation of UAE's ISO3 `ARE`.
Undotted ISO3 `ARE` remains AE. No punctuation stripping or inferred alias
equivalence is used to resolve this collision.

### Language policy

English names and international codes/initialisms are eligible by default.
Other names require an explicit language profile. `EGITTO` is unavailable in
English mode and resolves to EG in Italian mode; `ÉGYPTE` resolves in French
mode. A party's nationality or a company's name does not establish document
language. None of the 413 admitted templates currently has structured language
metadata, so arbitrary localized names should remain disabled until the
relevant template/region profile has been reviewed.

The registry is decoupled from corpus frequencies. A separately reviewed
supplement could later cover a real OCR spelling without making it a globally
sampled canonical alias. This pass does not silently invent accent-stripped
or malformed-name variants.

## 4. Sampling probe and distributions

The [probe configuration](analysis/country-rendering-probe-20261016/probe-config.json)
separates **country identity** from **presentation family**. It does not change
the route, locality or origin/destination distributions. A family is selected
first, then an eligible alias within it. Drawing uniformly from all registry
rows would overrepresent dotted forms simply because each code has several
dot styles: the earlier flat-row diagnostic produced 41.66% dotted forms.

Illustrative weights, not proposed production defaults or fitted real rates:

| Family | Configured weight | Observed in 10,000 diagnostic draws |
|---|---:|---:|
| Name | 70% | 72.56% |
| ISO2 | 12% | 12.60% |
| ISO3 | 8% | 8.28% |
| Approved initialism | 4% | 0.11% |
| Dotted ISO code | 3% | 3.45% |
| Composite | 3% | 3.00% |

Country identities are uniform **only in this registry stress diagnostic**.
Most countries have no approved international initialism. That unavailable
family is explicitly recorded and its configured weight redistributed over
eligible families; there is no invented initialism or silent fallback.
Requesting only unavailable families raises an error. Consequently a 4% family
weight does not mean 4% global initialism exposure across arbitrary countries.
Production frequencies should account for the countries actually sampled.

## 5. Actual-document probes and owner handling

The production compiler rebuilt ten existing source blueprints. All ten native
no-op renders exactly reproduced source text with zero edits. The probe then
used native edit receipts and the established current-owner mapper to reach
the corresponding text in current R26 descendants. It did not use global
country-string replacement.

| Source prefix | Country-rendering stress covered | Accepted variants |
|---|---|---:|
| `b54eac08` | Code-only party with explicit `COUNTRY CODE` caption | 3; two non-code profiles held |
| `dd8cea1c` | Country in postal owner plus separate repeated country | 5 |
| `92da45e4` | Generated postal codes and shared/repeated consignee/notify owners | 5 |
| `08ec8a5f` | Parties and route/issue countries; UAE and DRC initialisms | 6 |
| `66dc4377` | Many repeated country occurrences; protect Brazilian road `BR-101` | 5 |
| `7c103968` | Parties, discharge, delivery and freight-payment countries | 5 |
| `e9120c73` | Repeated party and route-country owners | 5 |
| `2d4248d3` | Postal owners and source country-before-postcode control | 5 |
| `23a49094` | Repeated addresses; French spelling available only through explicit language control | 5 |
| `22fa3887` | Separate name/code bindings inside one `NAME(CODE)` phrase | 5 |
| **Total** | **51 requested profiles, 49 accepted, two appropriate holds** | **49** |

The 49 accepted cases contain 412 owned country-span edit proposals, 312
country-target projections and 169 address-target projections. Proposals can
be no-ops; 286 scalar target values actually change across these counterfactuals.
The root audit inspected every final country's edit/projection and every
postal-target projection. Full paired documents remain available in the
inspection Markdown. This is a focused country-change audit, not a new semantic
certification of every unrelated field in those historical samples.

The handling is deliberately constrained:

1. Start with the known sampled identity and a certified country/postal owner.
2. Match exact eligible presentations inside that owner; longest phrases take
   priority and punctuation is preserved.
3. Inside postal owners, require country-component evidence. A road token
   `BR-101` is not a country; `HK` before numeric postcode `999077` is.
4. Treat separately owned `NAME(CODE)` fragments as one country presentation
   when their shared target-path evidence proves the pairing.
5. Reuse the selected presentation across repeated occurrences in this
   controlled probe. A production plan must retain per-occurrence style
   constraints, including explicit code captions, while sharing identity.
6. Derive the role-specific country target and update only the country's
   portion of an affected postal target. Do not change locality names.
7. Hold ambiguous inner spans, unsupported language/family choices and code
   caption conflicts explicitly.

### Probe defects found and rejected before acceptance

The first lexical-only candidate incorrectly replaced `BR` in `BR-101`, and
independent name/code replacement created nested/repeated composites. That
candidate and its original results are retained under
`rejected-lexical-only-v1-*`. Neither reached training data. General owner-scoped
component checks and combined phrase handling now cover both failures, with
negative tests.

A later inspection identified full country names under an explicit code
caption. Those unconstrained sample artifacts are retained in
`unconstrained-caption-v2-samples`; only paths referenced by final `results.json`
are accepted probe outputs. The final runner visibly holds both incompatible
profiles rather than altering the caption or silently changing the request.

### Separate authority/ownership controls

- `01d86535`: postal text identifies Hong Kong, while a separate country
  declaration identifies China. HK and CN remain distinct identities, not
  interchangeable aliases. This needs ownership/policy adjudication before any
  repair and is not a positive alias-rendering example.
- `fb241e5d`: the two source location targets remain a documented authority
  hold. Their current descendants are not relabeled by this probe.

## 6. Validation, performance and preservation

- **73 tests passed in 9.35 seconds**; Ruff checks and formatting pass.
- Every eligible registry presentation round-trips to its own identity and
  exact text span for all seven configured language profiles.
- All accepted probe targets pass the frozen direct-output JSON Schema,
  bound task canonicalization and target-casing checks. No `documentPatch`
  envelope or schema metadata was added to the stored targets.
- Exact inverse edit replay preserves all text outside proven country spans.
  Target shape and every unrelated scalar target are preserved.
- Every physical line and original position suffix is preserved. This is
  deliberately **not** a new centroid computation or geometry-reflow claim.
  Longer country words at the same anchors remain synthetic presentations,
  not asserted pixel-perfect PDF renderings.
- **1,660 protected file hashes** match before/after: R26 real/synthetic/
  validation/schema/constraints, active manifest/ownership/auxiliary and all
  413 source/target/template/contract authority bundles.
- Full probe: **13.81 seconds**, **811.0 MiB peak RSS**. The larger memory
  footprint includes importing the existing production compilation/mapping
  dependencies; the registry-only process uses **28.9 MiB**.
- Registry benchmark: **498,000 resolve/span operations in 0.468 seconds**,
  versus 0.484 seconds before the short-alias refinement; load 0.295 versus
  0.299 seconds. Memory differs by only 44 KiB.
- Isolated owner guard: 30,000 calls per approach, lexical-only 0.0450 seconds,
  scoped postal matching 0.1348 seconds, **+2.99 microseconds per owner**.
  Lexical-only is the deliberately insufficient baseline. No production
  training or synthesis throughput claim is inferred from this CPU probe.
- **Zero paid calls, zero GPU use, zero dataset or production-code changes.**

Reproduce locally:

```bash
.venv/bin/python docs/analysis/country-rendering-probe-20261016/registry.py prepare
TMPDIR=/tmp .venv/bin/python -m pytest -q docs/analysis/country-rendering-probe-20261016/test_registry.py docs/analysis/country-rendering-probe-20261016/test_probe.py
.venv/bin/python docs/analysis/country-rendering-probe-20261016/probe.py
```

## 7. Recommended integration boundary and remaining work

Use one identity/presentation registry for parties and locations, with a small
role-specific **target projection**. Keep canonical ISO identity in sampling;
choose an eligible presentation independently; bind it to the owned occurrence;
then derive the existing contract's target. Addresses need a verified inner
country slot or explicit assembly using that selected presentation.

This does not require rebasing every template. Existing exact standalone
country owners can be reused. Whole-address owners need the scoped inner-span
proof, and split name/code phrases need their common ownership represented.
Explicit code captions constrain allowed families. Language metadata should
be an optional reviewed template/region profile, defaulting to English plus
international forms; do not infer language from country or company name.

The current production path cannot be fixed by merely calling `random.choice`:
curated postal owners are rendered before ordinary `surface_values` overrides,
and direct country lexical values currently undergo target-expression agreement
checks. The presentation choice, postal wording validation, role-specific target
projection and final validator must share the same explicit plan.

Before dataset migration/production admission:

1. Apply the current printed-party policy coherently to the 48 confirmed
   mismatches; adjudicate the nine related-declaration holds and two source
   location-authority labels.
2. Integrate presentation planning and projection into the normal sampler,
   wording request, renderer and validation boundaries, preserving geographic
   identity and all other field sampling. Do not infer country from a city.
3. Run normal fresh synthesis with mixed per-occurrence styles and language
   controls, then audit the broader 413-template owner inventory. This probe
   does not certify all postal layouts or alphanumeric postcode formats.
4. Re-run the dataset with exact preservation checks and coordinate handling
   appropriate to that migration. Keep the future ISO2-target experiment
   separate so rendering diversity is not confounded with a target-policy
   experiment.

No unclear ownership, malformed spelling or unsupported language is silently
repaired. The present deliverable establishes the vocabulary, tested boundary
and concrete integration work, not a claimed completed production migration.
