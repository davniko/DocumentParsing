# Country presentations: R26 dataset and active template audit

Date: 2026-10-15. **Audit only: no country-policy, production-code, source/template, target or input changes.** Sections1–9 describe the CPU inventory/ownership audit. Section10 records the separate, explicitly requested offline GPU base-model experiment.

Snapshot note: the parallel goods-wording task subsequently refreshed1,000 R26
inputs/descriptions. Every country target and owned postal/location byte remains
unchanged. This audit's original file hashes resolve to the exact
[pre-refresh R26 backup](analysis/goods-description-refresh1000-20261015/backup-r26-before-refresh/manifest.json);
the country findings and base-query set are not a silent post-refresh rescore.

Follow-up preparation on2026-10-16: [country-presentation registry and contained rendering probe](kie-country-rendering-probe-2026-10-16.md).
It adds no production or dataset edits. Its vocabulary follows the user-approved
restriction to ISO short codes and documented international initialisms, with
localized names behind explicit language eligibility.

## 1. Findings

1. The current extraction contract intentionally treats parties and locations differently. Party countries retain the printed wording/code; location countries normalize to conventional English short names. The final MPCI form expects ISO alpha-2 codes. These are three distinct representations, not interchangeable targets.
2. All **18 previously reported standalone-code party-country violations remain**. The broader postal check finds 57 representation differences across 41 documents/12 sources. Final context adjudication confirms **48 party-wording violations across 36 documents/nine sources** (30 beyond the original 18); **nine additional facts have explicit related exporter/registration-country declarations outside the postal block and are reserved for policy adjudication**, not silently counted as bad labels. Geographic identity remains coherent.
3. Synthetic targets remain exclusively full registry names: **18,932 party-country facts and 10,110 location-country facts**, covering 181 country/territory identities. Real/validation expose codes, abbreviations, combined names/codes and language variants that synthetic supervision does not reproduce.
4. The ISO-only registry contains 249 identities and 920 lookup aliases, but many actually printed country variants do not resolve. **`A.R.E` is a concrete collision:** the current punctuation-normalizing ISO lookup returns UAE (`AE`), whereas its validation occurrence is the abbreviation for Arab Republic of Egypt. It must not be handled by blindly stripping punctuation and using the resulting ISO code.
5. One admitted source (`fb241e5d`) still has two code-valued location targets, `SI` and `TR`, contrary to the location full-name rule. Its current synthetic descendants have full-name targets. This is a source/template authority consistency issue, not two bad R26 synthetic labels.

Artifacts:

- [Inventory and source-country owners](analysis/country-presentations-audit-20261015/inventory.json).
- [Observed mapping-test inventory](analysis/country-presentations-audit-20261015/observed-aliases.json): **245 unambiguous observed surfaces**, two context-dependent surfaces and two unresolved source spellings.
- [Exact owned-region checks and final contextual adjudication of all 57 synthetic representation differences](analysis/country-presentations-audit-20261015/owned-country-checks.json).
- Reproducible read-only [inventory derivation](analysis/country-presentations-audit-20261015/audit.py) and [mapped-owner probe](analysis/country-presentations-audit-20261015/owned_probe.py). They print derived JSON; they do not edit the dataset or registry.

## 2. Contract and actual execution

| Layer / field | Current expectation |
|---|---|
| `parties.*.country` | Recognizable country wording/code printed for that party, preserving source wording in uppercase. No country inference from city, telephone or company name. |
| `route.*.country`, `placeOfIssue.country`, `freight.paymentPlace.country` | Country supported by that location's own wording/code/unambiguous adjective; conventional English short country name in uppercase. |
| MPCI `Party.country` / `Location.locationIdentifierCountry` | ISO alpha-2 code, supplied through an explicitly configured downstream resolver. |

The first two descriptions are present in both current source classes and the saved R26 `schema.json`. The reduced task's `CountryText` is free text up to 128 characters, not an ISO enum. The input-only training prompt is not secretly giving the model the full field descriptions: R26 `prompt-schema.json` has country string constraints but no country descriptions.

Authorities: [party field](../src/document_ocr/label_schemas/bill_of_lading_v7.py#L596), [location field](../src/document_ocr/label_schemas/bill_of_lading_v7.py#L68), [saved R26 schema](../data/curated/mpci-bl-real600-synthetic7000-dg-llm-wording-r26-positions-v1/schema.json), [MPCI form snapshot](../artifacts/mpci-ai-schema/mpci-cuscar-canonical.schema.json#L119). The [projection boundary](../src/document_ocr/label_schemas/mpci_projection.py#L154) requires an explicit country resolver and raises on unresolved countries; it does not infer a code itself.

## 3. Target distributions

Scope: latest R26 `real-train.jsonl` (600), `validation.jsonl` (60), `synthetic-train.jsonl` (7,000). Data/source hashes are recorded in the inventory. Sidecars are not added to these counts.

| Cohort | Party-country facts | ISO2 | ISO3 | Familiar undotted abbreviations | Punctuated abbreviations | Other printed/full-name targets | Location-country facts |
|---|---:|---:|---:|---:|---:|---:|---:|
| Real 600 | 1,665 | 28 | 17 | 25 | 8 | 1,587 | 898 |
| Validation 60 | 151 | 4 | 1 | 1 | 2 | 143 | 105 |
| Synthetic 7,000 | 18,932 | 0 | 0 | 0 | 0 | 18,932 | 10,110 |

The validation punctuated count includes `A.R.E`; its country identity is deliberately separated as context-dependent in the mapping-test artifact. The familiar abbreviations are chiefly `UAE` and `UK`: neither is the corresponding ISO alpha-3 code (`ARE`, `GBR`).

Real party-country facts have 110 distinct printed values; validation has 30. Synthetic party and location targets each have 181 distinct full-name values. Real and validation location targets have 41 and 18 distinct values respectively, **with no code-valued location targets**.

The real count was historically 1,666. It is now 1,665 after the reviewed stale-notify removal; this is not an unexplained country loss introduced by R26. R26's DG wording migration did not change the country contract or these synthetic totals.

Representative target forms:

- Codes: `EG`, `US`, `IT`, `CN`, `HK`, `TR`, `DE`, `ET`, `ES`, `USA`.
- Familiar/punctuated forms: `UAE`, `UK`, `U.A.E.`, `U.K.`, `U.S.A`, `A.R.E`.
- Language/legacy forms: `EGITTO`, `EGIPTO`, `EGYPTE`, `ÄGYPTEN`, `BRASIL`, `HELLAS`, `ESPAÑA`, `TÜRKİYE`, `TURKIYE`, `CZECH REPUBLIC`.
- Combined wording: `CHINA(CN)`, `EGYPT(EG)`, `INDIA(IN)`, `USA-US`, several `TAIWAN ... R.O.C ... TW` spellings.

These are explicit country fields or role-owned country/postal occurrences. They are **not** counts of every `IT`, `IN`, `US`, etc. substring found somewhere in a document.

## 4. Synthetic representation differences and contextual adjudication

### Method

1. Enumerate every synthetic party-country target and compare it with that party's own labeled postal address, with word boundaries and case-insensitive comparison.
2. For the 57 targets absent from their own address, load the exact archived native candidate from the R25 provenance index and verify its candidate SHA and inverse/forward source proof.
3. Compile the current source blueprint and map **all relevant own-country and own-postal regions** to current R26 text through the existing scoped owner mapping. This incorporates R24 phone/R25 identifier repairs rather than rerendering an old candidate over the current dataset.
4. Check the full-name target inside those owned occurrences. A full name elsewhere in a route or another party does not establish this party's printed target.

All 57 cases mapped successfully; **zero owner mapping failures and zero full-name matches in the mapped own-country/postal occurrences**. Twenty-six full-name targets are absent even from the whole document. The other 31 required contextual inspection: 22 print the name only in unrelated routes, other parties or company names, while nine explicitly print an exporter/registration-country declaration related to the shipper. Those nine are a separate policy decision, not automatic repairs. This is why neither own-address absence alone nor a whole-document substring match is sufficient adjudication.

| Source family | Own-postal representation differences |
|---|---:|
| `b54eac08` | 18 |
| `92da45e4` | 13 |
| `ce1f474c` | 6 |
| `b2b9200d` | 4 |
| `0bdef3ca` | 4 |
| `1b796092` | 4 |
| `04f208b1` | 2 |
| `5d7bd82a` | 2 |
| `67b1e5a1` | 1 |
| `6a62368c` | 1 |
| `e28ff6f7` | 1 |
| `f36409a0` | 1 |
| **Total** | **57** |

Final classification: 48 confirmed party-wording violations; nine related-declaration holds. The nine holds are five `92da45e4` shipper facts, plus one shipper fact each in `ce1f474c`, `6a62368c`, `e28ff6f7` and `f36409a0`. Their exact declarations and line contexts are retained in the owner-check artifact. Explicitly declared registration/export country is not automatically the same concept as a postal country, but it is also not an invented geographic inference; the policy should decide whether that same-party declaration may supply the target wording.

Examples from current R26 inputs:

| Descendant / role | Printed inside owned region | Target |
|---|---|---|
| `syn_full_v7_136aa583001515fd5daa7e7e`, shipper | `GRANVAGEN 14 / SOEDERTAELJE / SE` | `SWEDEN` |
| Same document, consignee and notify | `... PORT OF SPAIN / TT` | `TRINIDAD AND TOBAGO` |
| `syn_full_v7_3eb5dad3fd508514fdd8998e`, consignee and notify | `14 MARKET STREET, KINGSTON JM.` | `JAMAICA` |
| `syn_full_v7_b8f14bcea1e4667f0373212c`, shipper | `... QIBAO, SHANGHAI CN` | `CHINA` |

The original 18 are a genuine confirmed subset: `b54eac08` prints a standalone country code, has no postal address, and targets the expanded registry name. The other 39 are generated postal text using a code while the scenario target remains the full name: 30 are confirmed mismatches and nine have the separate declaration caveat above.

These are **57 proven postal/target representation differences, with 48 confirmed current-policy violations**, not a claim that every possible synonym, truncation or country ambiguity across natural language has been adjudicated. It is also not evidence that the geographic identity is wrong or that this alone explains model performance changes.

## 5. Template coverage and source presentation

All **413 admitted templates** compiled for this audit. The scan includes explicit country owners and whole postal owners connected to the same party's country field. It finds 2,544 country-to-owner edges, representing 2,057 physical owned regions; shared/repeated regions are not counted as separate physical occurrences just because they support several targets.

The source targets include:

- **45 undotted code/abbreviation-valued party fields across 37 templates**, underpinning **705 current descendants and 852 corresponding country facts**. These source/exposure counts reproduce the earlier audit. Many descendants now legitimately print full names, so this is lost code exposure, not 852 bad targets.
- Nine additional punctuated abbreviation source-party fields. Combined short forms span 46 templates.
- **Six standalone party-country code occurrences across four templates**, consistent with the previous standalone-owner inventory.
- **34 physical standalone location-country code occurrences across 16 templates** (36 country-fact edges). There are 26 ISO2 edges and 10 ISO3 edges; two physical occurrences each support both loading and issue country. These use the current compiled ownership and explicitly separate physical occurrences from target-fact edges; they are not interchangeable with older counts using different owner/type classes.

Two source location labels are inconsistent with the stated location rule: [fb241e5d target](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_fb241e5dc87e28957c2ca8b3e72703dadcec45a9f7ebef6e605cb1023bc1ab3a/target.json) uses `SI` and `TR` instead of `SLOVENIA` and `TURKEY`. The synthetic sampler expands these; current synthetic location targets are full names. No source fix was applied during this audit.

The source registry also contains printed/target spellings `EGIPT` (two party fields in `884975ff`) and `THE NETHERLANDSS` (one field in `c2180108`). They are retained in the unresolved mapping inventory rather than silently corrected. Bare `KOREA` appears in ten source party fields and six source location fields; real/validation also contain bare `KOREA`. This needs a separately agreed context/alias policy if converted to ISO, not a guess based on a city.

## 6. Why synthetic country presentations are narrow

The production flow is explicit:

1. [ScenarioCatalog.load](../src/document_ocr/synthesis/curated_scenarios.py#L1086) loads the pinned **ISO-only** country registry. The separate reviewed-alias loader exists, but the sampler does not use it and does not sample country aliases.
2. Port and locality records receive `(entry.common_name or entry.name).upper()` as their printable country. Country sampling chooses identities/localities, **not printed presentation families**. [Registry locality construction](../src/document_ocr/synthesis/curated_scenarios.py#L1121).
3. [Party scenario assignment](../src/document_ocr/synthesis/curated_scenarios.py#L1378) writes this full registry country string into every emitted party-country target.
4. The LLM is given both name and ISO2 context. [Postal requirements](../src/document_ocr/synthesis/curated_campaign.py#L600) request a sampled country/locality; some generated postal values choose the code. Most choose the full name. Source standalone alpha-2 party owners explicitly preserve alpha-2 via [ownership_surfaces](../src/document_ocr/synthesis/curated_ownership.py#L681).
5. [validate_postal_country_owner](../src/document_ocr/synthesis/curated_templates.py#L59) accepts either full name **or the certified code** inside the owned postal value. That is a geographic-identity check. It does not project the code presentation back into the party's target.
6. [validate_sample](../src/document_ocr/synthesis/curated_campaign.py#L1263) requires a party target to equal the sampled full registry name. Together, these checks can accept `CN` in the owned input and require `CHINA` in the label despite the extraction schema's source-wording rule.
7. Casing can vary document-level uppercase/title, while targets are uppercased. [Casing policy](../src/document_ocr/synthesis/curated_casing.py#L25) changes letter case only; it does not generate ISO2/ISO3/alias variation or repair country representation.

Thus this is not simply a missing random-choice call. Identity authority, presentation generation and target projection currently disagree for the copy-printed party role. Adding aliases only in text would increase that inconsistency unless the target rule is handled too.

## 7. Mapping probe inventory and resolver limitations

The [observed-alias inventory](analysis/country-presentations-audit-20261015/observed-aliases.json) provides each unambiguous observed surface with expected ISO2, ISO3, ISO registry name, registry common name, evidence kind, cohort/family frequency and examples. **119** unambiguous cases have real/validation or standalone-source evidence; the remainder also include full-name synthetic/source values. Full-name no-op cases must be scored separately from true alias/code conversion.

The inventory distinguishes `currentSamplerCountryTarget` from the policy wording “conventional English short name.” The ISO snapshot sometimes has formal registry names, such as `IRAN, ISLAMIC REPUBLIC OF`, `SYRIAN ARAB REPUBLIC`, and `RUSSIAN FEDERATION`. Do not assume these automatically establish one approved extraction spelling.

| Input | Current ISO-only lookup | Audit handling |
|---|---|---|
| `US`, `USA`, `U.S.A` | `US` | Unambiguous |
| `UK`, `U.K.` | unresolved | Reviewed alias: `GB` / `GBR` |
| `UAE`, `U.A.E.` | unresolved | Reviewed alias: `AE` / `ARE` |
| `TÜRKİYE` | unresolved | Reviewed alias: `TR` / `TUR` |
| `REPUBLIC OF KOREA` | unresolved | Reviewed alias: `KR` / `KOR` |
| `BRASIL`, `EGITTO` | unresolved | Reviewed aliases: `BR` / `BRA`, `EG` / `EGY` |
| `A.R.E` | **`AE`** | Context-dependent; validation occurrence is Egypt abbreviation. Punctuation-insensitive collision with `ARE`. |
| `KOREA` | unresolved | Context-dependent: unqualified surface does not uniquely identify `KR` versus `KP`. |
| `EGIPT`, `THE NETHERLANDSS` | unresolved | Separate malformed/source-spelling controls; no automatic target repair. |

Registry authority: [normalization](../src/document_ocr/synthesis/country_registry.py#L29), [ISO-only loader](../src/document_ocr/synthesis/country_registry.py#L221), [reviewed alias loader](../src/document_ocr/synthesis/country_registry.py#L234). The resolver rejects conflicting normalized aliases when constructing a registry; adding an Egypt `A.R.E` alias would collide with the existing UAE `ARE` key. A punctuation-sensitive/context-aware decision is needed for that case, not a broader silent fallback.

Revision note for the base-model experiment: the earliest observed inventory had 242 unambiguous cases. The final 245 adds `REPUBLIC OF KOREA` and `TAIWAN, CHINA` as reviewed mappings plus the newly observed `U.S.A.` source spelling. Existing 242 ISO2/ISO3 expectations are unchanged. Two malformed source spellings were also explicitly inventoried as unresolved, not added to graded unambiguous cases.

## 8. Recommendations and validation gates

1. **Settle the target contract before changing country sampling.** Keeping the current policy means party targets must come from an explicitly owned printed occurrence, while location targets normalize. Alternatively, choosing one canonical ISO representation for all country targets is a separate policy/schema/dataset migration requiring explicit approval. Do not silently mix these approaches.
2. Under the current contract, fix the 48 confirmed party mismatches and the sampler/renderer/validator split together; first adjudicate the nine explicit related-country declarations. Keep ISO identity in the scenario, but choose and prove a printed country presentation independently, then project the correct role-specific target. Preserve legitimate full-name/code alternatives printed for the same party; do not take an unrelated country elsewhere in the document.
3. Add configurable full-name/ISO2/ISO3/reviewed-alias presentation sampling only after targets follow that presentation contract. The existing country identity registry already supplies alpha2/alpha3/name; a small reviewed alias vocabulary can extend presentations without donor dependence. Ambiguous aliases need explicit handling rather than substring/geographic inference.
4. Review the two `fb241e5d` source location targets and bare `KOREA`/malformed spellings separately. Do not expand these into blanket changes based on place names or phone codes.
5. For the base-model mapping experiment, score direction and input family separately: ISO2→name, ISO3→name, name→ISO2, name→ISO3, and genuinely alternative alias→name. Exclude context-dependent cases from one-answer exact-match denominators. Separate whole-generation format compliance from first-answer semantic correctness and from harmless official/common-name alternatives.
6. Any implementation should be tested on code-only parties, full-name parties, name+code parties, repeated occurrences with different printed styles, location code→name normalization, title casing, postal continuations after contacts and the negative `A.R.E`/`ARE` collision. Required invariants: no party-role reassignment, no locality geocoding, no unrelated target/text changes, and explicit source/current owner proofs.

## 9. Audit limitations

All target distributions and all 413 template compilations are exhaustive for this frozen scope. The 57 synthetic mismatch checks use exact native mapped owners and cover all synthetic party targets absent from their own labeled address. They do not certify every possible real-country annotation boundary or every alias in arbitrary non-extraction boilerplate. Real cases whose target country is absent from `addressLine` were inspected as a separate warning class: an explicit same-party continuation can supply a full name, so absence from address alone is not sufficient to declare a bad label.

No country repairs were applied. The inventory/ownership audit loaded no model and launched no inference. Its observations are distinct from the subsequent experiment below.

## 10. Base-model mapping experiment — separate from extraction evaluation

The actual pinned **google/t5gemma-2-270m-270m**, revision
`7c38f16641f455ef0685b18431faf1b17722d5a1`, was loaded directly with
`AutoModelForSeq2SeqLM` in the existing trainer image. **No adapter, EVA,
fine-tuning, answer grammar, country list, retrieval or lookup was supplied to
the model.** This is a pretrained encoder–decoder, not an instruction-tuned
assistant. Its authors explicitly release pretrained checkpoints, so poor
instruction following must not be equated with absence of geographic knowledge.
[Google model card](https://huggingface.co/google/t5gemma-2-270m-270m),
[original paper](https://arxiv.org/abs/2512.14856).

The local-evaluation skill guided backend selection, GPU preflight and the
small smoke test. The experiment uses native Transformers for this custom
seq2seq task rather than a generic causal-LM evaluation adapter. Cached weights
were mounted read-only, networking disabled, and only the probe output folder
was writable. Greedy decoding explicitly overrides the cached sampling defaults;
decoder BOS2, EOS1 and PAD0 match the actual installed model/tokenizer.
The first launch exposed a read-only/non-executable Triton cache mount, resolved
by configuring an executable temporary cache; no model or training setting changed.

### Design and controls

- All249 ISO identities, four mapping directions, two prompt styles:1,992 queries.
- All245 unambiguous observed surfaces, two styles:490 queries. The two
  ambiguous and two malformed/unresolved forms are excluded from single-answer
  correctness denominators, not assigned guessed gold.
- Twelve copy controls in both styles:24 queries. **Total:2,506**.
- Plain instruction and six-example prompts. Every few-shot query excludes its
  own country from the demonstrations, including alternate names/codes.
- Strict full-response correctness is separate from a consistently extracted
  first nonempty answer line. Canonical country-name answers are distinguished
  from weaker semantic identity matches: returning a code is not credited as
  returning the requested country name.
- A second, explicitly non-generative diagnostic scores four candidate answers
  by their conditional content-token likelihood. Distractors are seeded random
  countries; none of the candidates appear as options in the encoder prompt.
  This diagnostic tests accessible factual signal, not a production extraction
  or249-way lookup success rate.

Frozen design, every prompt and raw output, runtime, and the single reproducible
script are preserved in
[probe-design.json](analysis/country-presentations-audit-20261015/probe-design.json),
[base-probe-results.json](analysis/country-presentations-audit-20261015/base-probe-results.json),
[base_country_probe.py](analysis/country-presentations-audit-20261015/base_country_probe.py),
and [likelihood diagnostic](analysis/country-presentations-audit-20261015/base-likelihood-diagnostic.json).
The earlier242-alias pilot and its raw outputs are retained under `initial-*`.

### Results

|Direction|Instruction: first canonical answer|Held-out few-shot: first canonical answer|
|---|---:|---:|
|Country name → ISO2|0/249|104/249 (41.8%)|
|Country name → ISO3|1/249 (0.4%)|46/249 (18.5%)|
|ISO2 → country name|16/249 (6.4%)|69/249 (27.7%)|
|ISO3 → country name|12/249 (4.8%)|75/249 (30.1%)|
|Observed surface → country name|88/245 (35.9%)|190/245 (77.6%)|
|Copy control|2/12|12/12|

**Strict entire-response success is0 for every task/style.** The model continues
tables or demonstrations rather than stopping after the answer. This is why the
copy controls and separately reported first-answer rates matter: a headline
“zero country knowledge” would be demonstrably misleading.

The observed-surface aggregate is also not a pure alias-conversion score:
179/245 inputs are already ISO country names. In few-shot mode the genuinely
alternate groups score:ISO2 11/15; ISO3 1/1; common abbreviations2/2;
punctuated abbreviations6/6; language/legacy aliases18/23; combined names/codes9/9.
The small abbreviation denominators do not establish universal reliability.

Four-way likelihood results on71 seeded countries per direction are much
stronger than free generation: **name→ISO2 69/71 (97.2%)**, **ISO2→name59/71
(83.1%)**, versus25% random-choice chance. These scores establish useful learned
mapping signal, but random distractors make this easier than arbitrary
country-code generation or full249-way classification. They are not directly
comparable to extraction field F1.

A stronger follow-up scores **all249 country answers**, not three random
distractors, on those same71 questions per direction. With summed conditional
content-token likelihood, name→ISO2 is **40/71 (56.3%) top1**,59/71 top5;
ISO2→name is **31/71 (43.7%) top1**,45/71 top5. Mean-token scoring gives
26/71 top1 for ISO2→name; including the end token gives28/71. Thus the strong
four-way result substantially overstates unrestricted mapping reliability.
Only one designated registry name per country is scored; this is a candidate
ranking diagnostic, not unconstrained-generation or extraction accuracy.
Every candidate rank and scoring variant is preserved in
[all-country ranking](analysis/country-presentations-audit-20261015/base-all-country-ranking.json).
This additional test took45.35s and peaked at2,968MiB allocated GPU memory.

The main2,506-query run took **123.24s**, including loading, and peaked at
**1,598MiB** allocated GPU memory. The separate likelihood diagnostic took5.69s.
No model training was launched; the base process exited and released its GPU allocation.

### Interpretation and recommended next experiment

The base model contains useful country associations, particularly for common
observed aliases, but **not a proven reliable general-purpose country normalizer**.
The four-way result is evidence that a small, explicitly supervised normalization
experiment is worthwhile; it is not evidence that postprocessing can safely be
removed today. Test held-out countries/aliases and real role-owned snippets,
not just known names in synthetic full-name addresses. Keep raw/canonical output
and role ownership separate, and retain negative ambiguity tests such as
`A.R.E` versus `ARE`.

For presentation variability, the existing pinned ISO snapshot already supplies
names and alpha2/alpha3. Unicode CLDR supplies customary, short and variant
territory names, with explicit disambiguation guidance. A pinned, collision-audited
English subset plus reviewed corpus aliases is a suitable candidate registry;
neither all translations nor every historic spelling should be sampled blindly.
This is a recommendation, **not an implemented country-policy migration**.
[Unicode territory-name guidance](https://cldr.unicode.org/translation/displaynames/countryregion-territory-names),
[official CLDR English data](https://github.com/unicode-org/cldr-json/blob/main/cldr-json/cldr-localenames-full/main/en/territories.json).
