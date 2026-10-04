# Direct extraction: field clarification and same-panel rerun

Completed 2026-10-02. This is the requested second **first-extraction** experiment,
not a bulk relabel, training run, or reviewer/corrector campaign. All original OCR,
comparison labels, first-pilot artifacts and datasets remain unchanged.

## Result

| Measurement | First pilot | Clarified pilot |
|---|---:|---:|
| Same frozen documents | 20 | 20 |
| Model / reasoning | gpt-6-luna / high | gpt-6-luna / high |
| Requests, including rejected answers | 20 | 20 |
| Completed JSON responses | 20 | 20 |
| Responses passing application validation | 17 | 18 |
| Estimated total API cost | $0.05926528 | $0.059874375 |
| Mean cost per attempted document | $0.0029633 | $0.0029937 |
| Median request latency | 48.71 s | 43.36 s |
| Run execution time | 315.59 s across two batches | 135.45 s, one batch |
| Maximum simultaneous documents | 5 | 8 |

All **20 new raw responses validate against the actual transmitted JSON Schema**.
The two application rejections concern nonconforming container identifiers, not
malformed JSON. All 20 wire requests explicitly used `json_schema`, `strict: true`
and reasoning `high`. No SDK retry, validation retry, extra reviewer call or PDF
request ran. Peak observed document concurrency was eight.

Field clarification helped concrete mistakes, but this was **not a uniformly
better extraction**: eight of 13 targeted old-error checks improved, five remained,
and four additional paired regressions were confirmed. These diagnostic checks are
neither a field-F1 estimate nor a count of completely correct documents. Both
batches are preserved; the new answers have not replaced the old ones as gold.
One stochastic run per prompt cannot establish that wording changes caused every
individual difference.

## What changed and why

### Field semantics, not a larger system rulebook

In [the V7 models](../src/document_ocr/label_schemas/bill_of_lading_v7.py):

- Address: retain postcode values but exclude `POSTAL CODE`, `ZIP`, and `ADD`
  captions; include owned continuations across contact/customs text and page
  headers. Competing addresses require a clear primary address or null for review,
  never concatenation.
- Name: retain legal-name suffixes, but exclude paid-in capital and registration
  boilerplate.
- Phone: include uncaptioned party-owned continuation numbers; never supply a
  missing prefix; fax-only values remain excluded.
- Voyage: exclude a separately headed leg/direction token.
- Marks: exclude table captions such as `TOTAL`; product codes need marks context.
- References: explicitly include party tax/import/export references within party
  blocks, not only references in the cargo section.
- Placement: an explicit shared-goods/container-list declaration supports each
  membership even if some package allocations are unknown.

The extractor prompt only adds the general absence convention: optional objects
with no facts are null. It grew from 117 to 126 whitespace-delimited words.
The unbound Pydantic schema grew from 31,147 to 31,709 serialized characters
(about 1.8%); no fields, categories or target shapes were added or removed.

### What constrained output does—and did already

[The flow](../src/document_ocr/labeling_agents/direct.py) already used PydanticAI
`NativeOutput(output_model, strict=True)` through OpenAI Responses. It was not
using ordinary JSON mode or merely asking the model to obey a schema in prose.
The pinned package registry is supplied in that native schema.

Two first-pilot failures were an integration mismatch:

```json
"freight": {"paymentArrangement": null, "paymentPlace": null}
```

and a `contactDetails` object whose four fields were all null. Both are permitted
by the transmitted schema. Python's after-validators separately demanded a
populated fact inside those objects. The third failure was a copied source ID
`INLLU4102226`, rejected by the ISO validation implemented in Python.

**Native decoding enforces the supported JSON Schema, not arbitrary Python
validators or source truth.** The current identifier's ISO shape/check-digit
checks live in an `AfterValidator`; they are not expressed by its string schema.
Cross-reference checks and source ownership likewise still need application or
semantic validation. See [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
and [PydanticAI native output](https://pydantic.dev/docs/ai/core-concepts/output/).

### Narrow integration fix

The existing PydanticAI capability now normalizes a nullable, wholly fact-free
object to null **before JSON validation**, based on the actual Pydantic field
definitions. The same path covers extraction and typed correction responses.

- Only nullable model objects with nullable child fields qualify.
- Unknown keys, required facts, invalid identifiers, empty strings/arrays and
  empty list members are not erased or accepted.
- Zero-valued measurements remain populated facts.
- Required section/target roots cannot disappear.
- Distinct correction response types are traversed only when their keys identify
  a unique model; ambiguous unions remain for normal validation.
- Raw provider text is retained. Any object-to-null normalization gets an explicit
  `nullObjectNormalizations` receipt with path and before/after values.
- No populated scalar may be changed by this operation.

The normalization takes place on JSON text through a PydanticAI output-validation
hook, preserving strict JSON-mode handling of tuples and dates. An initial local
before-validator experiment failed that requirement; tests caught it before any
paid rerun, and it was replaced rather than loosening strict validation.

Offline replay of the **old** 20 responses now passes 19, versus 17 without
normalization, with all non-null scalar paths/values unchanged. The invalid ID
still fails. The new pilot needed **zero** null-object normalizations: it emitted
null directly in those two formerly problematic places.

Failures now retain the underlying exception chain, not only the uninformative
`Exceeded maximum output retries (0)` outer exception. Concurrency is configurable
up to 16, defaults to eight here, and the one-time pilot runner enforces the same
limit globally across documents.

## Concrete improvements

Links below contain complete OCR and both raw answers, not just selected snippets.

| Document | Before | After |
|---|---|---|
| [01, 0600699f](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20-r2/comparisons/01-0600699f.md) | Address included `POSTAL CODE: 62815`; shipper tax reference missing | Caption removed, `62815` retained; `TAX ID: 91320594728739895L` recovered |
| [08, 096bccea](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20-r2/comparisons/08-096bccea.md) and [19, 6fe428c6](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20-r2/comparisons/19-6fe428c6.md) | Carrier name included `au Capital de 234 988 330 Euros` | `CMA CGM Société Anonyme`; capital wording excluded |
| [09, c9c96562](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20-r2/comparisons/09-c9c96562.md) | `TOTAL` was a shipping mark | No such mark |
| [10, b25a3616](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20-r2/comparisons/10-b25a3616.md) | No shared-cargo memberships | All five printed containers linked, allocation quantities left null |
| [15, a202c173](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20-r2/comparisons/15-a202c173.md) | Tax ID appeared only inside the longer ACID, not as its own reference | Explicit `TAX ID: 100483143` retained |
| [17, e47eaf30](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20-r2/comparisons/17-e47eaf30.md) | Voyage `02509/S` included separate leg | Voyage `02509` |

Additional observed improvements outside those 13 narrowly scored checks:
document 02 recovered B/L `MLTRLS2418021`; document 04 recovered the owned carrier
website; document 07 changed receipt place from `DELHI` to `DADRI, UP` and put
`DELHI` in freight payment place. Source layout review remains appropriate for
flattened route tables; these are not grounds for declaring every route field clean.

Several useful cargo behaviors also remained intact: paper reels stay one shared
goods entry across six containers; the five independently counted frozen-meat
cuts remain five goods; the long lubricant list remains 16 independently quantified
items; document 20 retains three goods with the 46/30/20 package totals and their
supported container portions. The DG case retains UN 1993, class 3, PG III and
50°C flash point, with 28 drums rather than seven outer pallets.

## Remaining errors and regressions

### Five persistent targeted observations

1. **01:** `(86-512)62589948` is still omitted from shipper phones. The OCR has no
   fax caption for that line. This is a completeness failure despite the new
   continuation instruction.
2. **10:** `SOKHNA EGYPTIAN SEAPORT` still becomes port name `SOKHNA` plus country
   `EGYPTIAN`, losing proper-name wording under the current location contract.
3. **13:** the delivery-agent postal continuation `EGYPT` after the next page
   header remains absent from the address and country.
4. **14:** the consignee address still concatenates `Port Tawfik ... Suez Egypt`
   and `568 El Horreya Avenue ... Alexandria Egypt`.
5. **20:** the same competing-address concatenation persists for consignee and
   notify party. The new instruction was not sufficient to make it abstain.

### Four confirmed paired regressions

- **04, yarn:** the OCR prints two container rows of 680 cartons, followed by two
  yarn specifications in a shared description. Previously the model emitted one
  shared goods with 1,360 cartons and two 680-carton placements. It now assigns
  each specification to a different container with 680 cartons. The source text
  does not establish those product-to-container associations. Equal row counts
  and order are not evidence of that mapping.
- **10, incomplete numeric total:** only four `20 BOXES / 20960 KGS / 40.9400 CBM`
  portions appear for five containers / 100 boxes. The new answer emits 83,840 kg
  and 163.76 m³ as whole-goods facts. These are sums of the four visible portions,
  not established complete totals. The former null mass/volume was safer. The
  simultaneous improvement in memberships does not validate these totals.
- **11, seal completeness:** source row prints `MRKU6255050 ML-IN0800111 ...`,
  followed by `Customs Seal: BOLT50714395`. Previously both seals were retained;
  the new answer keeps only the explicitly captioned customs seal.
- **12, postal boundary:** `NASR CITY F.Z. BL. J` moved from the address into the
  company name. The new address is only `11816 CAIRO EG`. The prior split was
  better under the postal-zone/address rule.

These four checks were added **after** inspecting the paired differences and
source text, and are labeled post-run regression checks in the machine report.
They are not a preregistered aggregate quality benchmark. Other unsettled areas
include categorical equipment aliases, cargo marks/product-code ownership, and
flattened route/freight tables. A scalar diff alone is not an error: for example,
omitting an unsupported inferred port country can improve a target.

### The two application rejections

- **06, 99adb051:** the joined source string is
  `TEMU95234567CM08875356/20GP/...`. The new answer uses container
  `TEMU95234567` and seal `CM08875356`. The 12-character container fails ISO
  validation. Earlier human adjudication identified the extra `7` as a mistyped
  separator, yielding container `TEMU9523456`; this pilot did not inject the
  previous correction as source evidence. The previous raw answer had the valid
  container but seal `7CM08875356`, also conflicting with that adjudication.
- **16, 298806fe:** the answer again faithfully copies printed `INLLU4102226`.
  That is not a valid ISO-shaped identifier. We did not silently correct, truncate,
  omit or replace it to make the answer pass.

Both full raw answers and detailed validation errors are retained, but no valid
`target.json` was exported for them. Adding a shape regex alone would not resolve
the second source: forcing a compliant string could replace faithful extraction
with an invented repair. These need explicit source-conflict handling/adjudication,
not a claim that native JSON decoding was disabled.

## Cost, validation and reproducibility

Usage across **all 20 calls**, including both rejected answers:

- Input: 232,134 tokens, including 187,625 cache-read and 44,449 cache-write tokens.
- Output: 104,872 tokens, including 78,482 reasoning tokens.
- Estimate: **$0.059874375**, or **$0.002993719/document**.
- Both 20-document experiments together: **$0.119139655** estimated.

Pricing is the official standard short-context [GPT-6 Luna pricing](https://developers.openai.com/api/docs/models/gpt-6-luna):
$0.10/M input, $0.01/M cache read, $0.125/M cache write and $0.50/M output.
Cache subsets are subtracted from total input, and reasoning is already included
in output; neither is counted twice. These are usage-derived estimates, not an
account invoice. Cost changed by about 1%, not an order of magnitude.

Validation performed:

- **516 tests passed in 14.64 s**, covering direct PydanticAI flow, existing
  labeling agents, V7 training integration and historical label schema suites.
- Actual SDK serialization tests verify literal OCR text, strict native schema,
  complete required-key declarations, descriptions, reasoning and receipt usage.
- All 20 live serialized request bodies independently inspected; all 20 raw
  responses checked by JSON Schema Draft 2020-12 against their actual wire schema.
- Old raw outputs replayed with the normalization: 17→19 application passes,
  **zero populated scalar changes**. Invalid IDs remain rejected.
- Tests reject unknown null keys, missing required mass facts, empty strings and
  lists, empty list members, bad IDs and targets without facts. A zero-temperature
  value survives; normalization is checked for corrections as well as extraction.
- Old pilot artifacts and source datasets match their pre-run SHA-256 snapshots.
- The 291 persisted historical V7 targets still validate and serialize unchanged;
  this is a compatibility check, not semantic reapproval of those old targets.
- Ruff passes on changed production modules and tests.
- Same-process traced microbenchmark over 2,000 validations: validation alone
  0.555 s; normalization plus validation 2.141 s (about 0.79 ms extra/document).
  Peak traced allocation: 63,493 versus 59,733 bytes. The extra local work is
  negligible relative to the 43.36-second median model request. This is not a
  claim of faster local validation; end-to-end batch throughput improved through
  concurrency, with provider latency variability also contributing.

Artifacts: [manifest](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20-r2/selection.json),
[analysis](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20-r2/analysis.json),
[paired OCR/answer comparisons](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20-r2/comparisons/),
and [one-time runner](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20-r2/rerun.py).
Each `runs/<document>/wire-request.json` is the actual API JSON body; authentication
headers were not captured. The offline analyzer uses a temporary `jsonschema`
installation outside the project environment; no project dependency was added.

## Interpretation / next decision

The requested implementation and rerun are complete. Concise field definitions
helped basic boundaries, and the null-object integration mismatch is repaired.
Native output is now verified at the real wire and returned-schema levels, not
inferred from a configuration name.

The evidence does **not** support treating one-pass extraction as gold. The next
useful experiment is the already planned section review/correction stage on these
same preserved drafts: measure whether it detects the documented omissions,
competing addresses, incomplete totals and unsupported allocations, while retaining
correct fields. That is a falsifiable review test, not another blind full relabel.
The two rejected drafts require an explicit source-conflict entry path or
adjudicated correction before the typed refinement command can accept them. No
additional paid stage has been started in this pass.
