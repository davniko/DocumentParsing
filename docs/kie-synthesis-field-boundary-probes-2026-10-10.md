# Synthesis field-boundary diagnosis and repair probes — 2026-10-10

## Outcome and scope

The company/name/postal errors from the random audit have identifiable roots.
The principal address/name contaminations are present in the **saved LLM
responses**, before rendering. The renderer and label projector then faithfully
copy those incorrectly scoped values. These are not solved by changing comma
formatting, country sampling, or container arithmetic.

Other audited defects have different roots: wrong or missing source occurrence
ownership, incomplete old-to-current package-path rebasing, an over-wide approved
description block, inherited postal noise, and missing generation ownership for
composite party identities. An instruction cannot affect a name that the system
never requests the model to generate.

This pass investigated all **2,500 current synthetic records and 300 effective
templates**, then tested proposed changes on in-memory copies and in a small
GLM wording experiment. It did **not** change production code, templates,
dataset rows, training configuration, or the running training process.

The credit top-up unblocked the experiment. The completed pass now contains
**86 wording outputs across 14 shipment scenarios**, followed by **eight targeted
postal-only correction outputs**. Every output was inspected. Total actual API
cost is **$0.03739969**, including the earlier $0.00935564; the resumed work added
**$0.02804405**. No credit blocker remains.

Field descriptions alone were insufficient: the completed first candidates
exposed caption contamination, a personal-name shape change, and values shifted
between requested fields. One request also instructed the model to generate a
description that was absent from its output schema. A scope-aligned candidate
with semantic field keys produced **28/28 outputs without company/address,
tax/contact-caption or contact/product ownership leakage** in this experiment.
It still changed two personal names into businesses and one product passage
repeated accounting; these findings were not counted as clean overall outputs.
Explicit source-certified identity kinds resolved the personal-name test.
Existing downstream postal checks selected eight follow-ups: six passed the
bounded correction automatically and two were explicitly adjudicated. All eight
then passed the postal checks with unrelated generated values unchanged.

The deterministic template repairs below also passed their stated in-memory
probes. This is a completed diagnosis/testing pass and an actionable repair plan,
**not a claim that production templates or the dataset have already been edited**.

Starting point: [random100 source/descendant manual audit](kie-synthesis-random100-manual-audit-2026-10-10.md).
Current dataset: `data/curated/mpci-bl-real600-synthetic2500-v7-positions-v1`.
Registry: `artifacts/synthesis-templates/mpci-bl-v7-reviewed`.

## 1. Whole-current-population boundary inventory

The selectors are deliberately **review selectors**, not automatic deletion or
whole-document acceptance rules:

1. The complete normalized party name appears inside its own address.
2. Party names contain explicit tax/contact/manager captions or broader suspicious
   vocabulary, including captions without a colon.
3. Goods descriptions contain an explicit party-contact/address caption.
4. Postal values contain legal-entity or business vocabulary. This captures a
   different company name inside an address, which exact own-name matching misses.

All **89 candidate documents** were inspected. Their dispositions are:

| Disposition | Documents | Meaning |
| --- | ---: | --- |
| Confirmed field-boundary defect | **39** | Actual identity/tax/contact content in the wrong generated field |
| Named-site/care-of ownership review | **5** | A literal company/name match cannot authorize deletion |
| Cleared for the triggering predicate | **45** | Genuine site, street, county, or company-name vocabulary |

The 39 confirmed documents span **26 source families**:

| Confirmed issue | Documents | Families | Affected target paths |
| --- | ---: | ---: | ---: |
| Company identity in postal address | **24** | **16** | **37** |
| Tax/contact/manager caption in party name | **12** | **9** | **16** |
| Delivery-agent contact in goods description | **3** | **1** | **3** |

Repeated consignee/notify targets can share one generated value; affected paths
are therefore not the number of model calls or independent mistakes. These groups
use the assigned primary finding; a record can also contain a different boundary
error, as in the source46 shipper-name example below.

The broader selector mattered: besides the exact-name candidates, it identified
12 clear postal contaminations involving a different company, a shorter member
of a composite identity, or a legal/trading prefix. It also caught five explicit
manager/contact statements without a colon. Conversely, blindly deleting
`TRADING`, `CO.` or `ROAD` would damage valid addresses and names.

Examples cleared after inspection include `MILLBROOK TRADING PARK`, `CO.
KILDARE`, `CO. TIPPERARY`, `CANAL SHIPPING ST.`, and the genuine company name
`ATLAS METALLURGIQUE DU RIF S.A.R.L.`. The last example shows why the mere word
`RIF` is not a sufficient tax-identifier test.

The five review cases are two generated estate/processing-zone names also used
as postal site names, and three descendants whose source address already contains
`C/O BP EXPLORATION (DELTA) LIMITED`. They are not automatically counted as
new company-prefix errors. Full IDs, values, paths and dispositions are in
[adjudication.json](analysis/synthesis-field-boundary-probe-20261010/adjudication.json).
The raw selector inventory is in
[inventory.json](analysis/synthesis-field-boundary-probe-20261010/inventory.json).

These counts are a complete inventory **for the tested selectors plus individual
adjudication**, not a proof that every possible multilingual or uncaptained
semantic contamination has been enumerated. No-hit documents have not been
silently declared completely correct. The purpose is to bound concrete repairs,
not to substitute token matching for semantic certification.

## 2. Company identity inside an address: confirmed generation root

For the clean-source cases, the original postal spans and effective bindings do
not contain the newly generated company identity. The saved generation response
does. Rendering uses that response directly; `addressLine` is built from the same
postal value. Thus raw text and label can agree exactly while the field itself
has the wrong meaning.

Audit source46 is a particularly useful counterexample:

```text
Generated consignee name:
DELACOUR FRIGO SARL

Generated consignee postal value:
POINTE NORD SARL
RUE DES TROIS CEDRES
MONT-DORE
NEW CALEDONIA
```

Checking only whether **DELACOUR FRIGO SARL** repeats in its address misses the
different identity **POINTE NORD SARL**. The same saved response also generated a
shipper name containing a plot/road phrase, crossing ownership in the opposite
direction.

Expected behavior:

```text
Name: DELACOUR FRIGO SARL
Postal region: RUE DES TROIS CEDRES / MONT-DORE / NEW CALEDONIA
```

The exact field-description candidate tested was:

> Return postal address components only. The party identity generated in the name
> field is context, not part of this address. Include no company or person identity,
> tax/registration identifiers, contact details or field captions.

This does not forbid street/site names or prescribe punctuation. The supplied
locality/country and the existing postal-assembly constraints remain in effect.
No source-specific company, country or template exception is added to the prompt.

## 3. Tax/contact text in names: confirmed generation root

Audit source26's original identity is a company plus an **uncaptioned** personal
identity. The agreed labeling policy permits that shape; it does not permit the
generator to invent a tax or attention statement inside `name`.

Saved examples:

```text
POLIMEROS DEL ORINOCO C.A.
RIF J-40528716-3
```

```text
COPAN DENTAL EXPORTACIONES S.A. DE C.V.
ATN: LIC. MARLENY ORTEGA
```

Other current descendants contain `CONTACT`, `GERENTE GENERAL`, or `MANAGING
DIRECTOR`, including suffixes without a colon. Their source examples do not
contain corresponding captions. This is generation changing an identity example
into a captioned tax/contact/manager statement, not a template inserting those
words after a clean name.

Expected generated names contain only the new legal/trading identity or personal
party identity. Generate explicitly requested contact names in their own fields.
Do not create a new contact or tax target simply to preserve an accidental
generation addition. Uncaptioned proprietor text may remain uncaptioned as part
of the identity, consistently with the source policy.

The first tested name clarification said:

> Return the fictional party identity only, matching the example's company/personal-
> identity structure. Exclude postal components, tax/registration identifiers and
> explicitly captioned contact information.

It prevented the original RIF/ATN pattern in the completed repeats, but one output
still returned:

```text
POLIMEROS ORINOCO COMERCIAL, C.A.
AGENTE: ROSA J. MARQUEZ
```

The current value validator accepted that output. Therefore an instruction-only
fix using that first candidate is not ready to be declared a complete remedy.

The second identity-description candidate states:

> Return only the new organization's legal/trading name or the individual party's
> personal name. An uncaptioned proprietor in the example may remain part of that
> identity, also uncaptioned. Do not add a contact/manager/agent caption, postal
> address, communication channel or tax/registration data.

This candidate completed eight outputs after the credit top-up. It removed the
observed AGENTE issue in both repeats, but one different response shifted values
between roles; that response passed the existing value gate. This falsifies the
claim that refining the identity description alone is enough.

The scope-aligned, semantic-key candidate subsequently avoided those ownership
mixups in all 28 inspected outputs. However, both repeats of the personal-party
control still generated a business name for the example
`AHMED ALSAYED GABR ALNASHAR`. Generic “keep the example's structure” wording was
not enough. A targeted source-certified field description was then tested:

> Value kind: individual party. Generate a new person's full personal name, not an
> organization or trading-business name.

Both outputs were personal names: `MOSA KOLLIE TARR` and `TARBOKO KOLLIE MORLU`.
Two organization-plus-uncaptioned-proprietor controls retained that two-part
identity without inventing contact/manager captions. These are four completed
outputs, not an automatic classifier for unknown source identity kinds.
Ambiguous/composite source identities still need their meaning established once;
there is no proposed runtime source-ID exception or blanket ban on personal names.

## 4. Paid wording probes: completed results and limits

Implementation: existing **PydanticAI native strict output**, actual dynamic field
descriptions, plain-text wording context, GLM-5.3-Flash through the configured
Fireworks route. Provider reasoning setting was **low**. All 94 successful calls
used 171,125 input tokens and 26,411 output tokens, of which **296 were reported
reasoning tokens**. Reasoning was approximately 1.12% of output tokens, not the
large reasoning share observed in some earlier synthesis pilots.

The same current-registry source ownership and supplied facts are used for each
comparison. Historical receipts predate package-policy repairs, so the probe
uses current published cargo facts, retaining saved geography/commodity choices.
This is a wording-ownership test, not a resampling or a new physical shipment
publication. Inputs and provenance are saved with each response.

The original comparison completed all 42 planned outputs after credit was
restored. Eight sharper-description controls, 32 scope/key controls and four
identity-kind controls were then completed. These **86 outputs contain 628
generated fields**, all inspected. Conditions are deliberate tests, not repeated
attempts silently replacing an unfavorable outcome.

| Condition | Outputs | Postal components | Identity fields | Product fields | Contacts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing instructions | 14 | 46 | 39 | 14 | 3 |
| First boundary-description clarification | 28 | 92 | 78 | 28 | 6 |
| Refined identity description | 8 | 26 | 18 | 8 | 2 |
| Refined description + aligned scope, original keys | 4 | 14 | 12 | 2 | 2 |
| Refined description + aligned scope + semantic keys | 28 | 92 | 78 | 28 | 6 |
| Semantic candidate + certified identity kind | 4 | 16 | 14 | 2 | 2 |

Important observations:

- A fresh baseline reproduced the generated shipper company in its address.
- The first clarification still produced `AGENTE:`, `ATTN:` and a care-of manager
  statement in names; one output shifted product/name/postal values between keys.
- The refined-only candidate also had a shifted-fields response accepted by the
  existing validator. Native string schemas do not certify string meaning.
- The scope-aligned semantic candidate had no company/postal, tax/contact-caption,
  or contact/product leakage across its 92 postal, 78 identity, 28 product and six
  contact fields. Two identity-shape changes remained; the explicit-kind test
  resolved the tested individual/company-plus-proprietor controls.
- One semantic candidate's product text independently added `4 PALLETS CONTAINING
  146 BOXES IN TOTAL`; the existing accounting guard rejected it. This is a
  generator-scope finding, not a new defect published in the dataset.
- No fresh product response repeated the observed delivery-agent-contact caption.
- The completed source48 repeats omitted `ENVNT`. Its baseline's `ATELIER NDEYE`
  remains a named-site/identity review; cleaning the source example is still the
  root solution, not relying on the model to ignore arbitrary source noise.

### The scope contradiction and output-key experiment

`host_product_wording()` supplies exact regulated-chemical or whole-unit product
identity itself. `wording_request()` correctly omits that product from generated
fields, but still unconditionally includes “Generate all description regions” and
`DESCRIPTION ASSEMBLY ... {v006}` in the request context. In the excavator control,
the output schema contains names, postal regions and a personal contact—**no v006**.
The model is thus asked to generate a region the schema cannot return.

The contradiction is directly observable. It is not proven to be the sole cause
of every field mixup. Removing the irrelevant description-generation instructions
and clarifying field meaning eliminated the observed shifts in six subsequent
excavator controls. Both original-key scope controls and descriptive-key controls
worked for that case. Semantic keys make ownership easier to read, but this small
experiment does not isolate their necessity independently of the other changes.

Example native keys are `shipper_name_component_1` and
`shipper_postal_component_1`. The host maps them back to unchanged internal
template keys; sampled values, target paths and occurrence ownership do not
change. Goods/route context can remain compatibility context when no generated
goods field exists, without asking for an absent output.

### Full postal assemblies, not just accepted strings

All 86 outputs were also passed through the current `assemble_lexical_target()` /
`postal_wording_errors()` path. Five semantic outputs, one original-key scope
control and two identity-kind controls were flagged. These eight outputs were
given exactly **one postal-only correction request each**, covering 13 components.
No names, contacts, product descriptions or sampled geography were regenerated.

The flags include missing country, duplicated locality, and a country name plus
an extra ISO-code mention. The last is a strict current synthesis-contract/style
failure, not proof that the extraction label contains a wrong country.

Six corrected outputs passed automatically. Two failed safely:

1. One output returned literal `\\n` separators. The value gate rejected it.
   Manual adjudication converted exactly those separators to physical line breaks,
   preserving every address token.
2. One output retained `Fonuakau Wharf Road, Alofi North / ALOFI / NIUE`, still
   violating required-locality-once. Manual synthetic wording adjudication used
   `Fonuakau Wharf Road, North District / ALOFI / NIUE`, retaining the district
   level. This is not removal of a real source address's district.

All eight final adjudicated postal outputs passed the same production postal
checks and the value gate; all unrelated generated values remained identical.
The two adjudications are explicit receipts, **not silent runtime normalization
or evidence that all eight model proposals were immediately correct**.

Actual total provider charge is **$0.03739969** for 94 successful calls. The eight
postal-only calls cost **$0.00096275**. The 22 preserved historical credit-rejection
receipts cost $0; there were no resumed credit failures. The resumed calls used a
2,048-token output maximum. Accounting sums individual unique call receipts, not
overlapping runner ledgers.

Measured mean call latency: baseline 3.914 seconds; initial clarification 3.946;
semantic candidate 4.421; identity-kind controls 5.096; postal-only corrections
2.106. These small, nonrandomized measurements are probe timings, not a production
throughput benchmark. The semantic candidate's longer keys/descriptions add token
overhead; production should use succinct replacement descriptions rather than
stacking all experimental instructions. No production performance improvement or
parity is claimed, because nothing was installed.

These results establish a tested component recipe and known rejection paths,
not a statistically established future error rate or blanket shipment approval.

Evidence: [exact responses and inputs](analysis/synthesis-field-boundary-probe-20261010/responses/),
[results](analysis/synthesis-field-boundary-probe-20261010/results.json),
[inspection and gate reproduction](analysis/synthesis-field-boundary-probe-20261010/probe-evaluation.json),
[refined controls](analysis/synthesis-field-boundary-probe-20261010/refined-results.json),
[scope/key controls](analysis/synthesis-field-boundary-probe-20261010/scoped-results.json),
[identity-kind controls](analysis/synthesis-field-boundary-probe-20261010/identity-kind-results.json),
and [postal follow-up adjudications](analysis/synthesis-field-boundary-probe-20261010/postal-followup-adjudication.json).

## 5. Why existing checks did not catch the identity errors

Native structured output guarantees requested keys and string types. It does not
guarantee that a string contains only postal information or only a legal identity.

The global wording prompt already says to keep postal text separate from names,
contacts and tax IDs. The field-specific requirements emphasize geography and
example structure but do not consistently spell out the field ownership boundary.
The proposed clarification places that boundary directly in the native field
description and the corresponding plain-text instruction.

The current name-caption regex covers only a narrow ATTN/ATTENTION line pattern;
it misses ATN, RIF, inline/suffix captions, manager/agent captions and some spaces
after a colon. Postal checks validate geography and placeholders, not identity
exclusion. Product checks do not catch the explicit delivery-agent caption found
in these saved descriptions.

A local falsification test submitted **19 actual bad target fields**—16 name paths
and three product paths—to the existing isolated value gate. **All 19 passed**.
This tests that gate alone, not a replay of their whole historical semantic review.
It establishes a concrete missing guard rather than assuming schema validity
implies correct field meaning.

A candidate explicit-caption review selector detected **19/19** of those actual
bad fields, plus **4/4** freshly observed AGENTE/attention/manager caption failures,
and selected **0/59** individually cleared site/county/name-field controls.
Its role is to produce a review/hold, not delete words or certify all
unflagged fields. Company-in-postal ownership still needs a semantic decision;
legitimate named sites cannot be safely rejected by matching company vocabulary.

Recommended accompanying controls:

1. Explicit caption/identifier scope violations in a generated name/product field
   must raise or create a publication-holding finding, not be silently cleaned.
2. Cross-field company/address matches should select semantic review; a name match
   alone cannot distinguish a legitimate named estate from a duplicated identity.
3. The rendered reviewer must explicitly check each requested name/postal/contact/
   product region's ownership. Exact text/label equality is necessary but cannot
   clear a semantically wrong shared value.
4. Preserve the agreed uncaptioned-person rule and legitimate site names. Test
   negative controls before making a stricter gate a production requirement.

No unvalidated new gate or wording patch was installed in production this pass.

## 6. Other defects: template-root probes and scope

The systematic source sweep found **65 current packaging/description defects
across ten families**, plus a latent repeated-package-category defect in one
additional family. These are not all LLM wording failures.

| Root | Families | Confirmed current descendants | Exact repair |
| --- | ---: | ---: | --- |
| Package type bound to a heading instead of declarations | 2 | 8 | Rebind actual declared and written-out package nouns; freeze the heading |
| `TOTAL PACKAGE 1` treated as number of schema rows | 1 | 9 | Bind the printed total to declared package quantity, not entity cardinality |
| A standalone repeated source quantity has no owner | 3 | 21 | Add the exact source occurrence as a typed declared-quantity owner |
| Package allocation also owns a fixed legal-clause number | 1 | 6 | Correct both current and historical occurrence lists |
| Approved description includes a leading package declaration | 1 | 10 | Narrow the product block to its actual product wording; keep count in OCR |
| Repeated container nouns retain dead pre-rebase paths | 2 | 11 | Rebind source-proven same-level rows to the current declared category |

One other family's six current CARTON descendants are consistent, but a BAG
counterfactual renders BAGS/CARTON disagreement. That family needs the same
explicit rebase before a future draw; its six current records are not counted
as bad category examples.

**54 current-campaign candidate rerenders passed** using the proposed ownership
corrections in memory. Twenty-two older affected descendants were separately
inventoried but not rerendered from their older receipts in this test. The tested
54 include currently unaffected controls and the latent family's six records.
This distinction is preserved in the evidence.

Examples:

```text
Wrong: DESCRIPTION OF CARTONS AND GOODS / 941PACKAGES
Right: DESCRIPTION OF PACKAGES AND GOODS / 941CARTONS
```

```text
Wrong: TOTAL PACKAGE 1, beside an actual 17-crate shipment
Right: TOTAL PACKAGE 17
```

```text
Wrong: Carrier's Receipt (see clause 1 and 1039)
Right: Carrier's Receipt (see clause 1 and 14)
       Actual container quantity 1039 remains unchanged.
```

```text
Wrong description: 89 PALLET STAINLESS STEEL TUBE AND PIPE FITTINGS...
Right description: STAINLESS STEEL TUBE AND PIPE FITTINGS...
                   The 89-pallet loading declaration stays in OCR/packages.
```

Seven falsification controls show why fixing the source contract matters:
fixing only one of the two clause-number owners still produces the wrong clause;
the latent mixed-package render passes today's replay/coverage checks; choosing
the party-owned first `TIRES` instead of the product-owned second occurrence is
correctly rejected by the overlap guard.

These corrections use existing declarative ownership mechanisms, not runtime
source-ID branches, global substitutions, or expensive regeneration. Source38
requires a source description-label/block correction; other probed package
fixes keep all targets unchanged. Full before/after declarations, exact source
IDs, affected current IDs and tested edits are preserved in
[template probe README](analysis/synthesis-field-boundary-probe-20261010/template_probes/README.md),
[results](analysis/synthesis-field-boundary-probe-20261010/template_probes/probe-results.json),
and [repair proposals](analysis/synthesis-field-boundary-probe-20261010/template_probes/repair-proposals.json).

Reproduction took approximately **6.52 seconds**, with **983.36 MiB** process peak
RSS including imports/compilation; package rerenders themselves took approximately
0.30 seconds. API cost is **$0**. There is no production performance delta because
production was unchanged.

## 7. The duplicate country is an ownership defect, not generation duplication

Audit source14 `9b35d0b6` generates a complete postal value containing the new
country once. A separate historical country binding then prints the same country
again. The postal label is already deduplicated; the raw input is not.

All **six current descendants** are affected. The existing source-owned `delete`
mechanism removes only the active extra country occurrence. All six in-memory
probes passed: the generated postal country remains, every target stays unchanged,
and exact byte replay verifies nothing else is removed.

This cannot be remedied by asking the LLM not to repeat country: it already did
not repeat it inside the generated region.

## 8. Inherited postal noise and omitted name generation

### ENVNT: source/example contamination

Source48 `88deddad` prints the delivery-agent postal block with Alexandria twice
and a final `Envnt` line. Its source label and complete lexical example retain
`ENVNT`. **Two of its six current descendants** copy that noise into new-country
postal values; the other four do not.

Exact affected IDs:

- `syn_full_v7_3b68e516600b5897d89b19b1`
- `syn_full_v7_a1c149f30a9a73f1b87fec6c`

Root action is to certify the source postal components, exclude the noise from
the source label/example and rendering ownership, and remove it from the two
existing affected generated regions. Do not interpret `ENVNT` as a country or
apply a global arbitrary-token remover. The fresh postal-field prompt probes
omitted the noise, but that does not certify or replace the source cleanup.

### Missing name ownership: four families / 24 descendants

The audit initially exposed source42's unchanged `VARDHMAN TEXTILES LIMITED A/C
SONVIGO INTERNATIONAL LTD`. The complete **effective** registry scan found five
unrequested name targets across four families, with six current descendants
per family:

| Family | Missing generation target |
| --- | --- |
| `0bd3ce7b` | Consignee and independently printed notify company-plus-person identities |
| `48973902` | Composite shipper / on-behalf-of identity |
| `600a1e89` | Notify company plus uncaptioned personal identity |
| `f256749b` | Composite shipper / A/C identity |

These **24 records** can have input/label agreement while retaining source names
under newly sampled geography. Fix the source's exact identity-fragment ownership
and composed name expression, preserving intervening address/contact fields.
Then the ordinary name generator can see and replace those regions. Merely
strengthening an LLM field description does nothing when the field is absent.

This scan loads current ownership and auxiliary overlays. Looking only at the
historical contract overcounts supposedly unbound fields already repaired by
those overlays. The four families above are the effective runtime omissions.

## 9. Secondary identifier inconsistency is a separate cleanup

Audit source39 contains importer `VAT:643-668-997` in party blocks and `IMPORTER
VAT NO:643668977` in its customs declaration. The discrepancy already exists in
the real source. The auxiliary generator normalizes punctuation and derives a
new number from the original identity string: hyphenated643668997 shares an
identity with compact643668997, but not with the different digits643668977.
Consequently it produces two fictional values for a captioned same-type importer
identity. This is not evidence that all importer/exporter/registration numbers
should be collapsed to one ID.

The current recipe's `identifier` label is not the random identity seed; changing
that label alone will not resolve the mismatch. The root remedy is a reviewed
same-entity/same-identifier alias declaration and support for that explicit
canonical identity. Do not infer alias equivalence from similar digits.

The earlier random100 audit also observed mutable secondary identities or
references in **80 descendants from 40 sources**. That is the original audit's
observed scope, not a new census of all2,500. It is separate from the39 wrong
field-boundary documents and should not be presented as80 wrong extraction labels.
Fixed carriers, agents, legal clauses and independently owned third parties must
not be swept up in anonymization.

## 10. Practical repair and validation sequence

1. **Apply the proven declarative fixes** for ten current package/description
   families, the latent category family, and source14's duplicate country. No
   paid generation is needed. Rerender every affected old/new descendant from its
   own saved facts; do not redraw cargo/route values.
2. **Implement the tested generation-ownership changes**: generate only fields
   actually present in the native schema; use clear role/value-kind descriptions;
   keep name/postal/contact/product ownership distinct. The blocked experiment is
   now complete. A source-certified individual/organization/proprietor kind helps
   preserve identity structure. Test any succinct production version on the same
   saved controls before installing it; do not assume an abbreviated prompt is
   equivalent merely because this experimental version worked. Accompany it with
   explicit scope-violation holds and the existing bounded postal correction path.
3. **Correct the enumerated generated boundary values**, reusing sound address
   components and product text. Where text boundaries are clear, remove only the
   exact unauthorized prefix/caption; otherwise regenerate only that requested
   field and inspect it. No blanket address or full-shipment regeneration.
4. **Repair the four omitted identity owners and noise source**, with source-
   specific certified fragments rather than source-specific runtime logic. Keep
   the five genuine named-site/care-of cases visible until adjudicated.
5. **Handle secondary aliases/anonymization separately**, after extraction defects,
   in accordance with the user's priority. Do not reopen non-issues about HS/product
   exactness, cone fills or payload arithmetic that do not contradict targets.

Publication validation must run on the actual current dataset mirrors, not only
candidate receipts: exact same sampled facts, correct field meaning, repeated
count/unit consistency, descriptions retaining product specifications, identifiers
and labels in their proper regions, schema round-trip, casing, train/synthetic
mirror equality, hashes, and no validation leakage. Rebuild positioned inputs
through the existing lineage-aware position renderer when input regions change;
then verify raw-text recovery, line mapping and geometry. The in-memory raw-text
probes above are **not** a claim that new positioned repairs have been published.

Validation completed for this diagnosis/testing pass:

- **198 existing regression tests passed** in 9.05 seconds:
  `test_curated_campaign.py`, `test_curated_templates.py`, and
  `test_curated_source_preparation.py`.
- All analysis scripts pass Ruff and Python compilation checks.
- Offline inspection/check replay took approximately 3.67 seconds and 983.36 MiB
  peak RSS including imports; no API requests were needed to reproduce it.
- All raw responses, unfavorable outputs, credit errors, manual decisions and
  call costs remain preserved. Neither production dataset rows nor production
  templates/code were overwritten. The running training process was untouched.

The identified extraction/render classes currently cover **112 distinct synthetic
documents**:39 generated boundary cases +65 package/description cases +six country
duplicates +two noise cases, with no overlap between those sets. Adding the24
missing-name-cast records gives **134 distinct records**, because two overlap
the other findings. These are enumerated repair populations, not an estimated
global semantic failure rate. Secondary residue and unresolved site/care-of
interpretations are deliberately not folded into that total.

## Reproduction and preservation

Offline commands:

```bash
.venv/bin/python docs/analysis/synthesis-field-boundary-probe-20261010/probe.py scan
.venv/bin/python docs/analysis/synthesis-field-boundary-probe-20261010/adjudicate.py
.venv/bin/python docs/analysis/synthesis-field-boundary-probe-20261010/evaluate.py
.venv/bin/python docs/analysis/synthesis-field-boundary-probe-20261010/template_probes/run_template_probes.py
```

The comparison and follow-up inputs can also be replayed from the saved successful
response caches. These commands have a paid-call path if a required response is
missing; do not confuse them with the fully offline commands above:

```bash
.venv/bin/python docs/analysis/synthesis-field-boundary-probe-20261010/probe.py run --max-output-tokens 2048
.venv/bin/python docs/analysis/synthesis-field-boundary-probe-20261010/probe.py refine --max-output-tokens 2048
.venv/bin/python docs/analysis/synthesis-field-boundary-probe-20261010/scoped_probe.py
.venv/bin/python docs/analysis/synthesis-field-boundary-probe-20261010/identity_shape_probe.py
.venv/bin/python docs/analysis/synthesis-field-boundary-probe-20261010/postal_followup_probe.py
```

Paid probe commands are separate explicit actions; `scan` and all deterministic
checks make no API calls. Exact accepted responses, rejected-call receipts,
messages, output schemas and cost accounting remain under the neighboring
analysis directory. Nothing was substituted for an uncompleted request.
