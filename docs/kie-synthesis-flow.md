# Current V7 synthesis flow

Implementation and operating guide. Verified against code and the published
full-scenario pilot on **2026-10-07**. This describes the active implementation,
not the earlier 30k generator or the first, limited-variability pilot.

- [Investigation, repairs, results and costs](kie-synthesis-v7-pilot-2026-10-06.md)
- [Original OCR and registry-sampled rendered examples](../artifacts/kie-synthesis-production/curated-v7-registry-pilot72-v1/samples.md)
- [Position-enriched examples](../artifacts/kie-synthesis-production/curated-v7-registry-pilot72-v1/positions-v2/samples.md)
- [Active campaign configuration](../configs/synthesis/mpci_bl_curated_v7_registry_pilot72.yaml)
- [Registry integration and validation](kie-synthesis-registry-integration-2026-10-07.md)
- [Active CLI implementation](../src/document_ocr/synthesis/curated_campaign.py)

**Latest expanded catalog:** the reviewed 100-source / 200-sample campaign is
documented in [the expansion report](kie-synthesis-expansion100-2026-10-07.md).
Its reusable assets are in
[`artifacts/synthesis-templates/mpci-bl-v7-reviewed/`](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/README.md),
with [its own campaign config](../configs/synthesis/mpci_bl_curated_v7_expansion100.yaml).
The 24-source examples below remain historical worked examples. The current
500-sample release, after joint reflow, has 82.36% coordinate coverage overall
and 67.90% on goods-description lines. See the
[500-sample geometry audit](analysis/real600-synthetic500-results-20261008/POSITION_AND_NEGOTIABILITY_FOLLOWUP.md).

**Party policy update, 2026-10-08:** negotiability is always emitted and depends
only on the OCR consignee instruction: order wording is negotiable, a readable
named consignee without it is non-negotiable, and unavailable instruction is
explicit `null`. A notify-only occurrence cannot restore a missing consignee.
Each emitted notify party has an explicit
reference or `sameAs: null`. Source instructions are preserved through generation.
Optional exact document quotas can control negotiable and notify-reference
prevalence; see [configuration and semantics](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/README.md#party-instruction-policy-and-sampling-2026-10-08)
and the [migration/validation report](analysis/party-instruction-policy-20261008/REPORT.md).

## 1. Overall design

We start with a reviewed real document's **existing OCR and current V7 labels**.
We produce new OCR-like text and matching labels. We do not create a PDF, run
OCR again, or ask an LLM to re-extract labels from the text it just generated.

```text
Current training OCR + labels       Pinned registries + training observations
             |                                      |
Reviewed source contract + old exact-span hints      |
             |                                      |
Compile mutable-region ownership                    |
             +---------------+----------------------+
                             |
              Sample coherent shipment facts
     route, party localities, goods, packages, loads, equipment
                             |
             Generate host-owned IDs and dates
                             |
       LLM writes names, postal text and product wording
             (three independent variants per call)
                             |
       LLM writes emails/websites for the generated companies
                             |
           Optional bounded postal correction
                             |
        Finalize numeric/dependent facts and V7 labels
                             |
       Render owned regions into the source OCR layout
                             |
          Deterministic checks and complete replay
                             |
              LLM full-rendered document review
                             |
       Resolve findings; re-render/re-review changed samples
                             |
                Validate complete campaign
                             |
        dataset.jsonl + samples.md + manifest.json
                             |
        Optional source-edit-anchored positional input variant
```

The **template** supplies structure and locations of facts. The **sampler**
chooses new shipment facts. The **LLM** writes language expressing those facts.
It does not independently choose package totals, weights, equipment classes or
the goods-to-container graph.

### Target and rendered casing

The full campaign now accepts an independent presentation policy:

```yaml
casing:
  target: uppercase
  render_styles: [uppercase, title]
```

`target` is `uppercase` (the current dataset convention) or `preserve` (an
explicit opt-out, not a different guaranteed-normalized convention). It is
applied after lexical generation/target assembly. `render_styles` is a nonempty
list of distinct `preserve`, `uppercase`, or `title` styles, sampled equally and
reproducibly using the campaign seed and sample ID. One choice applies across
the document's owned party, locality and vessel text, including repeats.
Unowned source text and headings are not recased. Product wording, equipment
codes, package units, identifiers, emails and websites are not title-cased;
embedded alphanumeric postcode/plot tokens are also preserved. This is casing
augmentation, not normalization of spelling or punctuation.

The published pilot's config explicitly retains `render_styles: [preserve]`
so its existing artifacts remain replayable. For a **new** campaign, use the
mixed policy above with a new output directory. Changing render casing changes
candidate hashes, so old semantic review receipts cannot approve the new text.
Review instructions and their hashes also follow the selected target policy.
The 72 mixed-case diagnostic outputs are recorded in the
[casing/goods probe directory](analysis/synthesis-goods-casing-probe-20261007/).

## 2. Inputs, dependencies and selected scope

The active command is `python -m document_ocr.synthesis.curated_campaign` with
`configs/synthesis/mpci_bl_curated_v7_registry_pilot72.yaml`. The older `curated`
CLI/config targets the earlier pilot and is not interchangeable, although
`curated.py` supplies shared contracts, request execution and cost accounting.

| Input | Purpose |
|---|---|
| `data/curated/mpci-bl-real-v7-reviewed-r16-paddle-positions-660/train.jsonl` | 600 training records: selected layouts, donor cargo, vessel and tare support |
| Same directory's `validation.jsonl` | 60 held-out records: exclusion/preservation checks; not cargo donors |
| `joinedRawText` and `target` | Plain source OCR and current reduced V7 labels |
| `configs/training/production/contracts/mpci_bl_real660_reduced_v7_r16_positions/task-constraints.json` | Hash-pinned task/schema vocabulary |
| `artifacts/kie-synthesis-production/curated-v7-pilot24/sources/<id>/contract.json` | Current-source variable/target bindings and source/target hashes |
| Historical catalog `mpci-bl-production-template-catalog1212-v38b-docb7-frozen-provisional/cases/<id>/template.json` | Exact source-region hints, **not** old label authority |
| [Ownership declarations](../configs/synthesis/contracts/curated_v7_full_pilot24_ownership.yaml) | Reviewed binding corrections, repeated occurrences, complete regions and surface recipes |
| [Auxiliary declarations](../configs/synthesis/contracts/curated_v7_full_pilot24_auxiliary.yaml) | Customs, source-only contacts/identifiers, dates and other dependencies |
| Config's pinned registries | ISO countries, UN/LOCODE, World Port Index, GeoNames, UK Global Tariff HS, commercial phrases, HMT/ECICS DG |

Registry files are local hash-pinned snapshots, not live web queries during
generation. Startup checks the task constraint hash, registry content,
source-contract hashes, training membership and exact capability coverage.
Donor IDs cannot overlap validation IDs; selected source OCR is additionally
checked for exact validation OCR overlap. This is not a fuzzy shipment-level
near-duplicate detector.

All sampling registries now live under `artifacts/registries/`: original
authorities in `sources/`, normalized versioned snapshots in `compiled/`, and
commercial HS phrases in `phrases/`. `relocation-receipt.json` maps historical
paths to unchanged bytes. Historical experiment receipts are not rewritten.

The pilot explicitly selects **24 sources × three variants**: 15 ambient,
three chilled, one frozen, three vehicle/machinery, one DG chemical and one DG
vehicle. It does not randomly select arbitrary old catalog templates.

Live support inventory at documentation time:

| Support | Count |
|---|---:|
| Eligible one-goods/one-package training observations | 567 of 600 |
| Ambient / chemical DG / frozen / chilled / vehicle / DG vehicle observations | 518 / 13 / 14 / 10 / 11 / 1 |
| Maritime port entries / countries represented | 2,594 / 183 |
| Countries with both ports and localities, before contact exclusions | 182 |
| Localities admitted by registry preparation | 31,924 |
| Validated HS6 commercial phrases | 5,612 |
| Eligible ECICS-to-maritime-HMT chemical candidates | 386 |

Source and donor eligibility do not constitute fresh manual certification of
every one of the 600 real labels. The current source topology is **one goods
accounting group, one positive typed package row, and zero or more containers**.
Multiple HS identities can belong to that one goods group.

## 3. Prepare the template and compile ownership

Code: [`curated_ownership.py`](../src/document_ocr/synthesis/curated_ownership.py)
and [`curated_templates.py`](../src/document_ocr/synthesis/curated_templates.py).

### Representation

The original OCR remains the layout authority. A contract names exact quoted
substrings, their one-based occurrence numbers, semantic roles, and target
paths. These resolve into UTF-8 byte regions. This is not unrestricted global
search-and-replace or a free-form Jinja template.

Simplified illustrative contract fragment:

```yaml
variables:
  - key: consignee_postal
    kind: postal
    value: '12 HARBOUR ROAD ALEXANDRIA EGYPT'
    meaning: 'Complete consignee postal address'
    required_literals: []
    occurrences:
      - text: "12 HARBOUR ROAD\nALEXANDRIA EGYPT"
        occurrence: 1
        presentation: text
targets:
  - path: documentPatch.parties.consignee.addressLine
    expression: '{consignee_postal}'
```

One variable may own multiple printed occurrences and multiple label paths.
Repeated consignee/notify information can therefore share one generated value.
An address interrupted by contacts may use multiple postal regions, but those
are generated **jointly**, not in separate calls for every street/district.

Expressions interpolate `{key}`; numeric baseline expressions can use explicit
sums. No arbitrary Python expression execution is involved. Presentation modes
cover text, printed numbers and spelled-out integer counts.

### Compilation sequence

1. Check the saved source OCR and target hashes.
2. Reconstruct current baseline labels from the source contract.
3. Verify quoted text/occurrence positions against current OCR bytes.
4. Translate historical structural names to existing V7 paths, without
   resurrecting dropped fields or the old cargo graph.
5. Apply reviewed declarations: correct/add/retire owners, enlarge complete
   lexical regions, or specify source-dependent date/package rendering.
6. Merge nested historical spans as evidence of the complete owner, not as
   additional overlapping physical edits.
7. Reject unresolved partial overlaps, conflicting owners, unknown paths and
   omitted repeated occurrences.
8. Add the declared auxiliary dependencies and caption edits.

Unchanged fields may stay fixed. **Every changed target leaf must have a
rendered owner.** A successful compile establishes that contract; it is not
proof that an arbitrary unseen document's semantics were annotated correctly.

The original preparation helper,
[`rebase_curated_v7.py`](../scripts/synthesis/rebase_curated_v7.py), is a bounded
first-pilot tool with fixed input configuration. It drafts source contracts
from current labels, old spans and numeric hints. It is not a universal
template-certification command. Normal full-campaign generation consumes saved
contracts plus the new ownership/auxiliary declarations; it does not rerun
that helper or require rerunning the historical full-catalog numeric sidecar.

## 4. Sample geography and party context

Code: `ScenarioCatalog._geography` in
[`curated_scenarios.py`](../src/document_ocr/synthesis/curated_scenarios.py).

Every source/variant gets a deterministic random stream derived from seed,
source ID and variant. Named substreams separate geography, cargo and other
choices. A fixed seed reproduces host sampling under unchanged authorities;
fresh uncached LLM wording is not promised bit-for-bit reproducible.

1. Choose an origin country with eligible maritime ports and localities.
2. Choose a **different destination country** from that eligible domain.
3. Reject countries missing required phone-generation metadata, recording
   rejection reasons. South Georgia (`GS`) is currently excluded this way.
4. Choose a port for each endpoint.
5. Direct-route receipt/loading fields follow origin; discharge/delivery/final
   destination follow destination. Place of issue and goods origin follow origin.
   A source with transshipment instead declares its route-node ownership:
   receipt, loading, intermediate port, discharge, delivery and issue place can
   follow different nodes or deliberately share one node (see section 16).
6. Freight payment place follows its declared side, or prepaid → origin /
   collect → destination. The payment arrangement itself remains source-fixed.
7. Sample party localities in their assigned countries. Shipper defaults to
   origin; other roles default to destination. Current forwarding-agent
   declarations explicitly use origin.
8. Repeated source party identities share locality; `sameAs` references are
   preserved rather than independently sampled.

**No selected source pins origin or destination to Egypt.** Sampling is
country-first, then port/locality within that country, not weighted by real
trade volume. Party locality need not be near the selected port. This does not
validate live carrier schedules or country–commodity trade likelihoods.
Populated transshipment fields require an explicit `route_topology` capability.
The sampler now supports this contract, including loading at the hub and repeated
destination captions. A blank transshipment heading does not enable it.

## 5. Joint cargo, packaging, equipment and quantity sampling

### Compatible training donors

A donor supplies empirical package/equipment/load relationships, **not** the
new document's layout. Eligibility considers cargo family, required measures,
container presence, valid HS support, configured package/equipment domains,
and thermal/ventilation requirements. Container donors require a complete,
coherent size/type pair. Container-free templates use container-free support.

For ordinary profiles, the package category and mass/cube per package come
from that donor together. This is stronger than random independent numbers,
but does not prove actual material density or commercial packability for every
newly worded product.

`physical_profile: source_bundle` keeps the source's own observed package/load
basis while allowing new registry identities and scaled inner quantities. It is
useful for nested packaging whose outer level stays fixed in source-only text:
do not combine four fixed outer pallets with an unrelated full-container donor.
An ambient source without printed HS codes needs an explicit reviewed HS domain
for this profile; private sampling identities do not create absent public HS
fields. This restriction does not constrain ordinary donor-supported profiles.

### HS identities and meaning

Each source specifies how many identities it supports: `33546e11…` has two,
`01d86535…` has four; most have one. These remain one accounting group.

Ordinary goods are drawn from registry HS6 domains within compatible observed
headings, using the earlier pipeline's ambient chapter scope. Exact source or
donor HS codes are not the candidate list. This restores registry variability
without detaching package/equipment/load support. Frozen and chilled food
identities use separate registry-wide thermal domains with explicit profiles;
observed produce supplements those domains with its own exact settings.
Whole-unit sources retain explicit restricted HS6 domains because their printed
unit count, engine/body wording and unit mass need a compatible whole-vehicle
contract. Chemical DG uses independently registry-linked chemical identities.
The old `explore_within_heading` switch has been removed; passing it is an error.

A different source identity is preferred when available; narrow domains can
legitimately retain an identity. Wording now receives a compact commodity brief,
not the sibling/national tariff tree. Exact HS-to-product classification is not
the extraction-training objective: plausible assortments, brands, models and
technical qualifiers are allowed. Printed codes and labels must still agree,
and DG/thermal and shipment accounting facts remain constrained. HS codes are public
only where the source supplies a public HS field. Otherwise HS is private
generation context and is not injected into labels or text as a new code field.

### Counts and measures

Ordinary package counts follow:

```text
nominal count = donor package count
              × source container count / donor container count
              × load scale
```

Container-free profiles use one for those container-count factors. Current
scale is **0.60–1.20 in 0.01 increments**. Counts round half-up to a positive
integer/configured multiple, unless the source fixes its unit count.

Weights and volume use donor amount-per-package × new quantity, converted to
the source's existing units and precision. Net cannot exceed gross. Each load
must fit configured equipment payload/cube bounds. Invalid candidate bundles
are rejected and redrawn, up to **512 attempts**; exhaustion is an error.

Container count and placement structure remain source-defined. Existing
explicit per-container package counts are reapportioned to total exactly,
retaining positive rows. Membership-only placements remain membership-only.
Private equal shares may be used for physical planning where printed counts
are absent; those assumptions do not become public quantity labels.

### Special families

| Family | Additional coupling |
|---|---|
| Chilled/frozen | Registry goods and configured carrying profiles determine temperature/ventilation; food load donors supply package/equipment/measures, not product identity. Observed produce extensions retain their exact settings. |
| DG chemical | Eligible HMT/ECICS identity supplies coherent HS, proper shipping name, UN, class, packing group and supported subsidiary hazards |
| DG vehicle | Preserve the supported liquid-fuel family; no electric/hybrid wording paired with its liquid-fuel declaration |
| Whole vehicle/machinery | Source-whole-unit or explicitly reviewed donor profile, fixed count and restricted product identity |

For the four whole-unit profiles, code supplies the registry commercial phrase;
the LLM does not invent model/body/engine specifications around a fixed load.
Chemical DG likewise uses the exact sampled proper shipping name as product
wording. Registry sampling changes the chemical; language generation cannot
change its physical form, invent a mixture, or substitute a solution that
requires a different UN entry. Route/party/contact wording remains generated.
All four currently have one physical donor each. Thermal food domains are no
longer limited to observed donor products. Explicitly chilled non-live food is
chosen when a tariff entry also permits live/fresh forms. HS exclusions such as
"not frozen" do not authorize a frozen profile. Frozen and chilled generic food
profiles use closed fresh-air ventilation; commodity-specific produce keeps its
observed ventilation. These facts are synchronized into existing printed slots.

DG packages use allowed categories (normally drum/carton/box) with non-bulk
bounds of 400 kg and 0.45 m³ per package. This is not full legal DG packaging
certification or the older package/overpack generation algorithm. Capacity
checks are scalar bounds, not 3D stowage or actual CSC-plate certification.

## 6. Generate host-owned IDs, dates, vessels and contacts

Code: `public_identity_updates`, `host_variable_values`,
[`curated_identifiers.py`](../src/document_ocr/synthesis/curated_identifiers.py)
and auxiliary rendering.

- Containers retain the source owner/category prefix with new serial/check
  digits. The same new ID is propagated into all placement references.
- B/L numbers, voyages and seals follow source character patterns. Repeated
  occurrences share a generated identity.
- Declared VIN/chassis policies generate supported fictional identifiers;
  a valid pattern/checksum is not manufacturer issuance.
- Issue/on-board dates shift together by 30–759 days, preserving intervals.
  Source date typography is handled by verified date recipes, not guessed by
  the wording model.
- Vessel names are sampled from eligible current training names, excluding
  the source vessel; they are not freshly invented by the LLM.
- Phone strings follow sampled-country support. Contact-person names belong
  to lexical generation. Emails/websites are generated after the company name
  exists, as described below; the old hash-based placeholders are no longer used.
- Source-only customs/registration values use explicit auxiliary recipes;
  they are not automatically added as public fields.

## 7. Generate language in one constrained call per source batch

Code: [`curated_wording.py`](../src/document_ocr/synthesis/curated_wording.py)
and `Campaign.generate`.

Configured runtime: **GLM-5.3-Flash, OpenRouter/Fireworks, low reasoning,
native strict PydanticAI output, concurrency eight**. These are local config
values, not fresh remote availability/pricing claims. Provider fallback is
disabled; timeout is 300 seconds and output cap 16,000 tokens. Automatic agent
and transport retries are both zero in the current configuration.

A call normally covers three independently sampled variants of one source.
Partial resumption requests only missing/stale variants. The request contains:

- Readable route and party-locality context.
- Compact commodity phrases, all source description fragments, and their explicit
  target assembly. Continuation fragments are not separate HS identities.
- Host-rendered packaging, measures and equipment facts as compatibility context.
- Sampled cold-chain context, with empirical produce extensions identified separately.
- The requested region's role, concise requirements and original structure
  example (hierarchy/line span, not source facts to copy).
- Postal assembly expressions explaining generated components versus
  country/locality fragments supplied by code.

It does **not** provide the complete source OCR or PDF to this wording call.
The reviewer later receives complete rendered OCR. Source OCR is not wrapped
in a JSON object and accompanied by a request for full extraction.

Native schemas require exactly `s0`, `s1`, `s2` and the requested region keys.
Fields have descriptions; unknown fields are forbidden. The model returns
**strings**, not quantities, full labels, copied source hashes or per-scalar
evidence. For example, schematically:

```json
{
  "s0": {
    "shipper_name": "NEW FICTIONAL TRADING COMPANY",
    "shipper_postal": "PLOT 7, INDUSTRIAL LANE\nMALE MALDIVES",
    "goods_wording": "NEW COMMERCIAL PRODUCT WORDING AROUND THE COMMODITY BRIEF"
  },
  "s1": {"...": "the second sampled shipment"},
  "s2": {"...": "the third sampled shipment"}
}
```

Actual keys are source-contract keys, not necessarily these illustrative names.

Goods have no character or line quotas. Natural generated paragraph/list boundaries
are kept; a short source word does not imply a narrow rendering column. Multiple
owned fragments are generated together and assembled into one description target;
repeated occurrences reuse one generated value. Intervening customs, totals and
boilerplate remain outside those edits. Shipment packing, fill weights and transport
settings belong to host-controlled facts rather than invented product prose.
Validation rejects whole-description source copies, identical generated descriptions
across a batch, and explicit shipment-total/tariff captions inside generated products.
It also rejects literal escaped newlines and explicit unsampled per-package fill
claims. One bounded, recorded model correction can repair a rejected new output;
a second failure remains an error. Both calls are billed in the ledger.
These deterministic checks complement, not replace, final rendered semantic review.

### Addresses specifically

All postal components for a party are generated together with the requested
locality/country, roughly preserving the source hierarchy and line span.
Punctuation may change. Street deliverability is not required. Company names,
tax IDs, contacts and captions stay outside postal regions.

The complete address must contain the requested locality once. Country appears
once where that party has a country target; otherwise the source omission is
kept. A separately printed country supplied by code already counts. Code
assembles the **same generated components** into `addressLine`, normalizes
whitespace/newlines to spaces and applies current uppercase text policy. It
does not strip cities or insert commas merely because physical lines break.
Targets remain `addressLine` plus `country`, not a separate city target.

Local wording checks reject incomplete key coverage, empty strings, copied
source identities, detached line-leading punctuation, zero-placeholder
postcodes and specified contact captions inside name regions. Postal checks
normalize case/diacritics/punctuation before counting the requested locality
and country. These checks enforce the declared generation contract; they do
not establish every address component's semantics or postal deliverability.

## 8. Explicit correction, not an endless automatic retry loop

### Company contacts after wording

[`curated_contacts.py`](../src/document_ocr/synthesis/curated_contacts.py) sends
plain-text company names, countries and original email/website style examples
to a narrow native-output call. The response contains only requested contact
strings. Repeated roles for one company share values. Email and website domains
are independent: corporate and free-mail mailboxes are both allowed, even when
the source used one common domain. Website syntax/style remains source-conditioned.

`generate` includes this stage. `contacts` runs it separately when the rest of
the shipment is already generated. Receipts bind the company/context/schema;
`render` rejects missing, stale, malformed or incomplete contacts. Validation
checks syntax and ownership; it does not generate strings with regexes or
perform DNS/deliverability checks. A realistic invented domain can coincide
with an existing domain, so these values are not authorized contact endpoints.

The 2026-10-07 bounded repair reuses prior full-document approval only after
exact non-contact delta checks plus explicit manual contact review. Its review
receipts label this method; they do not claim a new full-document LLM review.
Newly generated shipments still use the normal full-render review stage.

`correct-postal` is an optional bounded stage. It assembles addresses, selects
the failed parties' mutable postal regions and asks for just those replacements.
Other wording and shipment facts stay unchanged. Before/after values and
post-check results are saved. A still-failing proposal remains inspectable in
the receipt and the stage reports failure; it is not automatically rolled back
or approved for publication.

Other errors are adjudicated at the appropriate authority: wording,
ownership/auxiliary contract, or scenario policy. Changed candidates must be
re-rendered and re-reviewed. There is no autonomous “rewrite everything until
the reviewer agrees” stage, nor a manual-adjudication CLI.

The completed pilot retained **13 manual wording corrections across 12
records**. Their reasons and before/after values are preserved. This is not a
claim of perfect unattended first-pass generation.

## 9. Final physical and auxiliary preparation

[`curated_physical.py`](../src/document_ocr/synthesis/curated_physical.py)
reconciles the sampled plan with what the source can actually print:

1. Select a common decimal precision representable in repeated totals/rows.
2. Allocate row weights/cube by exact arithmetic and largest remainder, so
   rounded rows still sum exactly.
3. Synchronize totals, number words and package nouns.
4. Render canonical/ISO equipment forms and aggregate `N × TYPE` receipts.
5. Preserve supported source tare for unchanged equipment; changed equipment
   needs matching train-observed tare or fails.
6. Align thermal prose, setpoints, ventilation and DG declarations.
7. Recheck capacity and net/gross after rounding.

The **final candidate target**, not the earlier scenario JSON, is authoritative:
printed precision may cause final numeric values to differ from the initial
plan. Physical preparation cannot silently add public fields.

A printed measure absent from labels may still need updating. Its private
support is recorded as a donor measure or a unique source-owned per-package
measure. Printed zero/unknown values remain unasserted; private support does
not authorize inventing public labels. VGM cases without an explicit contract
are rejected rather than inferred.

[`curated_auxiliary.py`](../src/document_ocr/synthesis/curated_auxiliary.py)
then handles declared customs captions, country/code repeats, references,
source-only contacts and dates. Egyptian ACID wording can become generic import
reference wording while linked values follow the new geography. These are
owned-span changes, not global replacement of every country name. Unrelated
carrier/legal/third-party text can remain fixed. This is fictional trade text,
not simulation of each country's national registration rules.

One documented source-evidenced recipe adds issue/on-board dates omitted from
that source's real labels into its **synthetic blueprint**. It is an explicit
exception to ordinary field-presence preservation; the real labels are untouched.

## 10. Render text and labels together

The renderer merges accepted wording, host values and physical/auxiliary
surfaces. Conflicting owners fail rather than following last-writer precedence.
Exact source regions are replaced; all other bytes remain unchanged. Explicit
generated newlines are respected; otherwise wrapping approximates source line
span, preserving tokens and avoiding detached punctuation. This is OCR-like
text layout, not authentic PDF coordinates.

Current labels are updated directly from the chosen facts and lexical
expressions, then casing-normalized and checked against the reduced task.
There is no second extraction call trying to rediscover the generated facts.

Checks cover changed-field rendering ownership, package nouns in their actual
package region, postal country support, current schema, sampled HS agreement,
container foreign keys, explicit allocation sums, unchanged quantity-field
presence, repeated party consistency and actual route variation.

Candidates retain source/variant IDs, scenario, text/target, render values,
physical/auxiliary/identifier receipts, wording hash, blueprint inventory and
edit proof. The proof records byte ranges and before/after text. Edit replay
must reproduce OCR exactly. A stronger authority replay reconstructs the
**complete candidate object** from current source/config/wording and requires
equality, including metadata.

This catches stale or altered outputs. It is not an independent semantic proof
that the source contract or accepted free wording was right.

## 11. Review the complete rendered shipment

`review` first replays the candidates and then sends one source's three
variants to a separate call. Inputs are complete final OCR as text, current
target as YAML, party localities, registry commodity scope and thermal context.

The reviewer checks product/HS meaning, DG/thermal compatibility, stale facts,
postal ownership/geography, repeated information, quantities, unsupported
labels and missing supported values. Instructions distinguish normalized
casing/aliases and private printed facts from actual target defects.

The schema requires one `s0/s1/s2` object each, containing a `findings` list.
Findings contain `field`, `problem`, short quoted `evidence` and `correction`.
Empty means no concrete defect was reported. Host code supplies real sample
IDs, rather than asking the model to copy long hashes.

The reviewer uses the same configured model family as wording; these are
separate calls, not an independent-model ensemble. Structured output guarantees
shape/coverage, not correct semantic judgment. The reviewer does not edit text.

### Resolution and approval

- Real defect: fix the correct authority, re-render and obtain a fresh review.
  An adjudication marked `fixed` cannot approve changed bytes under an old review.
- False positive: explicitly reject it with a reason and exact current OCR
  quotations, bound to the finding, review and candidate hashes.
- Unresolved/missing/stale reviews block publication.
- `review` can finish with exit code zero **and findings**: successful review
  execution is not approval.

This pilot also received independent full inspections in three
`audit/final-review-*.json` receipts. **Publication code does not consume those
extra inspection reports.** They are additional manual release evidence for
this pilot, not an automated prerequisite guaranteed for every future run.
Hashes of manual corrections likewise establish freshness, not correctness
of the human semantic decision.

## 12. Complete-scope publication

[`curated_publication.py`](../src/document_ocr/synthesis/curated_publication.py)
requires exactly the configured sources × variants: no missing/extra candidate
files, wrong IDs or duplicate generated OCR. It reruns full candidate authority
replay and checks current review coverage and prompt/schema hashes. Every
finding must be resolved under the rules above. Source train/validation files
are hashed before and after validation and must not change.

| Published artifact | Contents |
|---|---|
| `dataset.jsonl` | `documentId`, `sourceDocumentId`, `joinedRawText`, `joinedRawTextSha256`, `target` |
| `samples.md` | Original OCR followed by all published variants |
| `manifest.json` | Scope, schema/config/review hashes, source snapshots, per-sample receipts and file hashes |

Pending files are replaced with the **manifest committed last**. Existing
identical outputs are idempotent; differing publication files are refused and
preserved. This is not one atomic directory-wide transaction. The manifest is
the completion marker.

`validate` executes the same checks and constructs expected publication bytes
without writing publication files. Neither `validate` nor `publish` pays for
API calls. Neither merges the real dataset, makes train/validation splits,
adds positions, or starts training.

## 13. Stage commands and operational behavior

From the repository root, these are the individual stages. This is a runbook,
not authorization to start another campaign.

```bash
# This revision preserves the earlier published pilot.
SYNTHESIS_CONFIG=configs/synthesis/mpci_bl_curated_v7_registry_pilot72.yaml

# Paid only when compatible successful wording is not cached:
.venv/bin/python -m document_ocr.synthesis.curated_campaign generate \
  --config "$SYNTHESIS_CONFIG"

# Optional contact-only refresh; generate already includes this:
.venv/bin/python -m document_ocr.synthesis.curated_campaign contacts \
  --config "$SYNTHESIS_CONFIG"

# Optional paid correction of detected postal failures only:
.venv/bin/python -m document_ocr.synthesis.curated_campaign correct-postal \
  --config "$SYNTHESIS_CONFIG"

# Local render + deterministic checks:
.venv/bin/python -m document_ocr.synthesis.curated_campaign render \
  --config "$SYNTHESIS_CONFIG"

# Paid only when the exact successful review request is not cached:
.venv/bin/python -m document_ocr.synthesis.curated_campaign review \
  --config "$SYNTHESIS_CONFIG"

# Resolve reported errors/findings before local validation/publication:
.venv/bin/python -m document_ocr.synthesis.curated_campaign validate \
  --config "$SYNTHESIS_CONFIG"

.venv/bin/python -m document_ocr.synthesis.curated_campaign publish \
  --config "$SYNTHESIS_CONFIG"

# Local-only, separate positioned inputs after plain publication:
.venv/bin/python -m document_ocr.synthesis.curated_campaign positions \
  --config "$SYNTHESIS_CONFIG"
```

A new campaign needs a new output/configuration; do not change a published
run's scope and expect to overwrite its manifest. There is no single
auto-repair-until-done command. Stage summaries retain successful partial work
and per-source errors; processing errors give nonzero exit status. Publication
still requires the entire configured scope.

### Concurrency, cache and retries

The paid runtime uses PydanticAI `NativeOutput(..., strict=True)` and a shared
concurrency semaphore. A nonblocking `.campaign.lock` prevents simultaneous CLI
writers. `.env` supplies credentials, not prompt content.

There are two cache layers:

1. Per-sample wording binds its instruction/native prompt/schema hash; cached
   text is checked again before rendering. Compatible correction metadata stays.
2. Paid-call keys include stage, source identity, provider config, complete
   system/user prompts and output schema. Successful exact calls are reused.

Changing prompt/schema invalidates corresponding reuse. A provider-config
change affects paid-call cache keys; an already valid per-sample wording
receipt is not forced to regenerate solely to switch providers.

Cached failures remain explicit failures. `--retry-rate-limited` permits
retries of cached HTTP 429 rejections, preserving numbered attempt receipts.
`--retry-invalid-output` similarly permits one new attempt for a cached invalid
native response. Neither flag enables an unbounded retry loop; billed failures
remain in the ledger. Valid contact receipts can be reused without either flag.
There are no silent provider fallbacks or semantic retry loops.

### Budget accounting

The current configured cap is $5. Before a paid request, code reserves a
conservative bound using prompt/schema bytes and maximum output tokens; the
check includes simultaneous reservations and recorded spending. Returned
provider costs/usage replace estimates. Known rejections without usage cost
zero; uncertain timeout/disconnection billing keeps an upper-bound reservation.
This is not crash-proof external billing reconciliation after abrupt process
death or receipt-write failure.

Stage summaries and manifest cost include **all recorded campaign attempts**.
The retained-path cost is separately attributed in the cost audit. Additional
calls change the campaign ledger and expected manifest, so publication
idempotency assumes the ledger/config also remain unchanged.

### Artifact map

Under `artifacts/kie-synthesis-production/curated-v7-full-pilot24-contacts-v2/`:

```text
calls/                 Attempt inputs/outputs, messages, usage, errors and costs
wording/               Accepted per-variant text, scenario and correction records
contacts/              Company-conditioned email/website batch and sample receipts
candidates/            Complete rendered records and replay/physical proofs
reviews/               Hash-bound review results, with explicit manual-delta provenance where used
adjudications/         Created when explicit false-positive decisions are needed
audit/                 Extra pilot inspections and negative-control probes
*-summary.json         Generation/correction/render/review outcomes
dataset.jsonl          Published examples
samples.md             Original/rendered inspection gallery
manifest.json          Publication completion record
positions/             Preserved earlier anchor-only publication
positions-augmented-v1/ Configured augmented dataset, geometric receipts and gallery
```

Files such as `plan/`, product inventories and sampling probes are investigation
artifacts, not additional mandatory runtime stages.

## 14. Historical worked example and prior pilot result

The figures in this section describe the **previous contacts pilot**, not the
new registry-sampling run. Current results, correction history and costs are in
[the registry integration report](kie-synthesis-registry-integration-2026-10-07.md).

Source `33546e11…`, first variant, retains one goods group, two HS identities
and one-container placement. It samples Maldives → Belgium, compatible donor
HS853650/HS845090, and load scale 0.69. The final result contains:

- **1,173 cartons, 11,353.5 kg, 45.553 m³**.
- 40-foot high-cube general-purpose equipment.
- New switches/washing-machine-parts description and fictional party wording.
- Container **MSMU5623798**, seal **6464227**; the same ID is used by the
  container and goods placement, which assigns all 1,173 cartons to it.

Other variants use DR Congo → Italy with footwear/plastic parts, and Ecuador →
Marshall Islands with medical equipment, without changing the accounting shape.
See the [complete revised text pairs](../artifacts/kie-synthesis-production/curated-v7-full-pilot24-contacts-v2/samples.md).

| Pilot measure | Result |
|---|---:|
| Sources / accepted variants | 24 / 72 |
| Origin / destination countries | 60 / 59 |
| Country pairs / Egypt destinations | 71 / 1 |
| Sampled / publicly printed distinct HS6 | 55 / 32 |
| Package categories / equipment pairs | 14 / 4 |
| DG / thermal / ventilation documents | 6 / 12 / 8 |
| Descriptions differing from source | 68/72 |
| Manual wording corrections retained | 13 across 12 records |

Recorded pilot evidence: 98 targeted tests and lint passed; render/validation
took approximately 3 seconds for 72 after initialization, CLI peak memory about
433 MiB. Full paid review at concurrency eight took approximately 31 seconds
including initialization. These are pilot measurements, not new performance
benchmarks from this documentation pass.

Full-restart development API cost was **$0.24554292**. Retained generation,
postal-correction and final-review calls cost **$0.06081975** for 72, projecting
to **$8.45/10k** at observed billing or **$12.67/10k** at configured conservative
prices. Neither includes manual work, template engineering or future retry
overhead. [Exact cost receipt](../artifacts/kie-synthesis-production/curated-v7-full-pilot24/audit/final-cost-and-variability.json).

On 2026-10-07, this documentation pass reran **read-only validation**: 72
expected, 72 valid, 24 sources, no failures or unresolved findings. Dataset hash
`2574fe51caddc5289341b9c450a98beaeb421c742fee7c65b06ef6a1192ed34a`
and manifest hash
`28f64921d79051076060f3101e4227b385b20b9de2d25762ba06da66fbf6a7e2`
match that historical publication. The later contact/position extension is
documented in section 11 of the [pilot report](kie-synthesis-v7-pilot-2026-10-06.md);
the old pilot remains intact.

### Source-anchored positions

[`curated_positions.py`](../src/document_ocr/synthesis/curated_positions.py)
transfers normalized source coordinates through the renderer's exact UTF-8 byte
edits. Unchanged lines retain their source anchors. Changed line counts now use
verified measured geometry through `curated_position_regions.py`. Interpolation
is accepted only at plausible measured line spacing; local expansion can use its
share of the adjacent vertical gap. It cannot consume a neighboring region's space.
An expansion of one line no longer repeats one coordinate for every new line.
If the region cannot fit, its text is retained with explicitly unknown positions.
This local transfer is the baseline, not the final placement in campaigns that
enable joint reflow. Unknown source positions remain ` ||` rather than borrowing
another field's coordinates. Page markers and blank lines remain structural,
without suffixes.

The current 100-source campaign enables **joint elastic reflow** between transfer
and page augmentation, implemented in `curated_reflow.py`,
`curated_reflow_geometry.py` and `curated_reflow_policy.py`. It connects exact
edit/region owners, estimates replacement widths using measured source text and
font advances, samples coherent block spacing from pinned source priors, consumes
blank gaps and shifts dependent source blocks. Shared rows, columns, reading
order and clearance are checked, as are the final rounded line centres.

`positions.reflow` pins the calibration and font path/hash, plus the seed,
row-lock tolerance and column/bottom guards in source-glyph units. Calibration
lives in the reusable template catalog. The font cache is bounded. No LLM call,
text regeneration or target-driven coordinate inference occurs. Unknown ownership
and missing source anchors are recorded as held regions. Rejected page proposals
retain the whole baseline placement with explicit reasons; no partial reflow is
published for that page. Source text and labels are unchanged.

This is intentionally approximate layout conditioning, not reconstructed PDF
glyph geometry. Text-only re-matching against Paddle and absolute output line
index copying are not used. Source positioned-text hashes, edit replay, page
ownership and suffix-only preservation are checked before writing a separate
`positionedText` field. Plain text and targets remain intact. The historical
24-source contacts pilot, under its then-current placement policy, had
6,043 positioned lines out of 7,168 (84.31%) after correcting three auxiliary
clauses that contained escaped rather than physical newlines.

The production stage now applies the validated coherent page augmentation from
[`curated_layout.py`](../src/document_ocr/synthesis/curated_layout.py). The explicit
`positions` configuration sets output subdirectory, seed, minimum/maximum scale,
maximum translation and bounded proposal count. The current pilot uses scale
0.95–1.05, at most 20 units of translation per axis on the 0–1000 grid, and 32
scale proposals. All known lines on a page share one transform; none receives
independent jitter. The seed incorporates sample identity and page, not labels.

Source dataset and Paddle parquet hashes are verified. Geometry must agree
with the source alignment's measured regions and centroids. Bounds include
**all** recognized source regions, including those not matched to the GLM text.
Every accepted integer output preserves axis order, equal-axis alignment and
complete nearest-neighbour sets, including ties. If no scale proposal qualifies,
the page uses an explicitly recorded exact integer translation. A page with no
known coordinates remains unknown; an immovable page records zero changed points.
No coordinates are silently clamped or borrowed. Original text and labels are
unchanged, and rerunning a different policy cannot overwrite an existing result.

The historical contacts-pilot output is `positions-augmented-v1/`; its old
`positions/` remains intact for comparison. All 72 samples / 120 pages were published and independently
verified, with 118 scaled pages and two translation-only pages. See
[implementation and validation](kie-synthesis-v7-pilot-2026-10-06.md#13-production-positional-augmentation--2026-10-07).
The earlier [geometry experiment](kie-synthesis-v7-pilot-2026-10-06.md#12-coordinate-synthesis-experiments--2026-10-07)
remains historical evidence. The newer
[goods/layout pilot](kie-synthesis-goods-layout-pilot-2026-10-07.md) enables bounded
local reflow without character-width reconstruction. Its 24 full samples / 48 pages
are published separately under `curated-v7-goods-layout-pilot24-v1/positions-v2/`.
These are coarse source-layout anchors, not measured synthetic glyph boxes.

For the active 100-template / 200-sample campaign, the configured output is now
`curated-v7-expansion100-v1/positions-reflow-v1/`, preserving `positions-v2/`.
The whole-page transform bounds and validates the **reflowed** envelopes, not the
old source envelopes. Publication has per-sample receipts, a complete manifest,
positioned JSONL and a text gallery. The reusable
`scripts/synthesis/audit_curated_positions.py` audits the saved files and produces
source/previous/reflow/final diagrams. See
[integration results, commands and plots](kie-synthesis-position-reflow-investigation-2026-10-07.md#production-integration-and-200-sample-regeneration).

## 15. Boundaries to preserve when scaling

| Dimension | Current behavior |
|---|---|
| Countries/ports/localities | Sampled at both ends; no Egypt pin |
| Names/addresses | Generated within reviewed ownership and sampled geography |
| Cargo/package/equipment | Joint constrained donor-supported sampling, not arbitrary full-registry combinations |
| Goods/package row counts, container counts | Fixed per source |
| Placement topology and quantity presence | Fixed; identifiers/explicit quantities change coherently |
| DG/thermal presence | Source-dependent, not randomly switched on/off |
| Whole-unit wording | Host-controlled restricted product phrase |
| Negotiability, freight arrangement, many handling facts | Source-fixed unless explicitly owned/sampled |
| Carrier/legal/unrelated third-party text | May remain fixed |
| Customs | Generic fictional references, not national compliance simulation |
| Positions | Optional coherent page augmentation of source-layout anchors; 72 samples published and validated |
| PDF/images | Not generated or re-rendered |
| Catalog readiness | These 24 reviewed families, not the historical full catalog |
| Validation meaning | Exact declared invariants plus model/manual semantic inspection, not an infallible classifier |

Adding a source needs a current contract, reviewed mutable public/private
ownership, supported capability domains, baseline replay, representative
variants and semantic inspection. Adding variants to a prepared source does
not require manually editing each template again, but generated examples still
go through validation and review.

Hashes are reproducibility/freshness checks under trusted local authorities,
not cryptographic signatures proving semantic truth. Capacity checks do not
prove regulatory compliance, and address checks do not prove deliverability.
The pilot's extra manual inspection is not silently replaced by model agreement
when scaling. Distribution quotas, a broader audited layout library, broader
commodity/load support and a training ablation of the new positional inputs
remain separate work, not results implied merely by successful publication.

## 16. Transshipment integration and validation pilot — 2026-10-07

### Diagnosis and source scope

The historical renderer had transshipment bindings, but the current scenario
sampler explicitly rejected a populated `route.transshipmentPort`. Simply
removing that rejection would have silently assigned the intermediate port to
the destination. The required fix was explicit route ownership, not another
generated place name.

The current real dataset inventory contains **11 labeled training sources and
zero labeled validation sources** with this field. A broader textual screen
finds **57 documents**: those 11 plus 42 training and four validation documents
with transshipment-related wording but no such target. Those mentions largely
include blank captions, contact headings and legal conditions, so they are not
57 confirmed intermediate-port examples. The inventory is reproducible with
[`prepare_transshipment_pilot.py`](../scripts/synthesis/prepare_transshipment_pilot.py)
and recorded in [source-inventory.json](../artifacts/kie-synthesis-production/curated-v7-transshipment-pilot/source-inventory.json).

| Source prefix | Observed route/shape | Scope in this pass |
|---|---|---|
| `551657f3` | Manila receipt → Hong Kong loading/via → Port Said discharge/delivery; issue place Hong Kong | Complete new source contract; three full variants |
| `8579256f` | New York Elizabeth → Aliaga → Port Said West; destination also repeated in customs/party-adjacent text | Complete rebound source contract; three full variants |
| `978a3990` | Gothenburg → Hamburg → Alexandria, onward El Dekheila; repeated feeder vessel/voyage distinct from main vessel | Real-text feeder recipe probe; not full-document publication |
| `ce1f474c` | New York → Aliaga → Alexandria, route context spread across pages | Inventoried; source onboarding remains |
| `d684b168` | New York → Mersin, no discharge target | Inventoried; missing-discharge topology tested |
| `30ad41fb` | Chicago receipt, New York → Mersin → Port Said | Inventoried; inland receipt needs its own node |
| `6c14020d` | Chicago receipt, New York → Aliaga → Alexandria | Inventoried; inland receipt needs its own node |
| `a67afe13` | Rio → El Dekheila; transshipment caption also names El Dekheila | Inventoried; repeated-destination topology tested, not forced to a third distinct port |
| `5a4230e0` | Port Klang → Jebel Ali → Sokhna; partial container/quantity detail | Inventoried; cargo ownership must also be onboarded |
| `e44ccf71` | Boppard receipt, Antwerp → Ambarli → El Dekheila; barge pre-carriage | Inventoried; pre-carriage and inland ownership must be onboarded |
| `13280227` | Norfolk → Aliaga → Alexandria; whole-loader cargo | Inventoried; whole-unit cargo profile also needs admission |

Mention-only examples `6c5e700d` and `a5265d59` print a transshipment-related
caption with a place already labeled as discharge; they do not establish a new
intermediate node. `ffb526f7` prints `AS PER MTO`, not an extractable intermediate
port. No real labels were changed or inferred from these captions.

### Implemented end-to-end flow

1. **Compile exact source ownership.** The two source contracts pin the current
   OCR/target hashes and every edited span. `551657f3` had no historical compiled
   template, so its ownership YAML supplies the complete new binding map.
   Rebinding drafts cannot be used when required ownership is missing or the
   original target cannot be replayed.
2. **Sample endpoints as before.** Both origin and destination countries vary;
   neither pilot template pins Egypt. Existing direct-route random streams and
   serialized scenario receipts are unchanged.
3. **Sample the additional node.** `RouteTopology` maps every populated route
   field to `origin`, `destination`, or a named node. Extra ports come from the
   pinned UN/LOCODE + World Port Index domain; inland nodes use the locality
   registry. Country dependence is explicit. The pilot's independent hub must
   differ from both endpoint countries and is drawn from Hamburg, Hong Kong,
   Aliaga and Mersin. This domain is configurable. An unregistered port ID fails
   configuration validation rather than being accepted as arbitrary wording.
4. **Keep related facts coherent.** Loading can equal the intermediate port;
   receipt can remain the earlier origin. Issue place follows its declared
   node. Repeated `FROM`, `TO`, `VIA`, discharge clauses and country declarations
   are bound to those same draws. A repeated destination caption can share the
   destination node; it must not become a fabricated additional port.
5. **Generate the complete remaining shipment.** The existing registry cargo,
   physical quantities, equipment, parties, names, addresses and contact stages
   run normally. The lexical brief receives actual loading and intermediate
   ports. It does not independently choose geography. Generic customs captions
   and source-owned references vary with the sampled countries.
6. **Render all owned occurrences and public targets together.** New auxiliary
   placeholders expose intermediate port, country and code. Missing intermediate
   geography is an error. The optional `transport_leg` recipe independently
   samples a feeder vessel/voyage from training support; it preserves the
   source voyage's character shape and makes repeated mentions identical. It
   cannot overwrite the main vessel or invent a new target field.
7. **Validate and review before publication.** In addition to exact text-edit
   replay and normal schema/numeric/ownership checks, every route component
   must equal its sampled node. Missing fields, stale ports and invented country
   components fail. Native-structured model review sees the complete rendered
   sample and sampled facts. Final files and replay hashes are committed through
   the existing publication stage; coordinates are written separately.

Implementation: [`curated_routes.py`](../src/document_ocr/synthesis/curated_routes.py),
[`curated_scenarios.py`](../src/document_ocr/synthesis/curated_scenarios.py),
[`curated_campaign.py`](../src/document_ocr/synthesis/curated_campaign.py), and
[`curated_auxiliary.py`](../src/document_ocr/synthesis/curated_auxiliary.py).
The runnable [six-sample config](../configs/synthesis/mpci_bl_curated_v7_transshipment_pilot6.yaml)
contains both source capability declarations and links to their ownership files.

### Additional defects found during the full run

- **Nested package load:** the first source prints 146 boxes on four pallets,
  684.460 kg and 8.608 m³. Borrowing an unrelated large cargo load while retaining
  four outer pallets was unsuitable. Its explicit `source_bundle` profile now
  scales this observed inner-box/load relationship, while drawing electronic
  goods from registry headings 8534/8541/8542. Published variants have 156, 134
  and 91 boxes with corresponding weights/volumes; the four source-only outer
  pallets stay fixed. The source does not print an HS code, so these sampled
  identities remain private instead of creating unsupported HS labels.
- **Product model versus tariff code:** the lexical checker treated `MODEL
  HS-1881` as an HS declaration. It now distinguishes that product-model syntax
  while still rejecting unauthorized `HS 854231`, `HS:854231`, `HS-854231`, `HS CODE-854231`
  and `TARIFF CODE 854231` declarations. Regression tests cover both sides.
- **Country-less postal slots:** a generated address inserted `FALKLAND
  ISLANDS` when that slot should omit country. The registry spells the country
  `FALKLAND ISLANDS (MALVINAS)`, so the old exact check missed it. The omission
  guard now also recognizes the canonical name without its optional parenthetical
  qualifier. This only tightens forbidden-country detection; it does not weaken
  positive grounding. The existing bounded postal-correction stage corrected
  the affected wording, followed by rerender and rereview. Manual inspection
  found this even though the earlier model review reported no findings.
- **Receipt stability:** inactive topology fields are omitted from direct-route
  metadata. This keeps the existing 72-candidate authority replay identical,
  rather than forcing regeneration because new optional fields were added.

### Published examples

| Variant | Receipt/origin → intermediate port → discharge/destination |
|---|---|
| `551657f3 / 1` | Europoort, Netherlands → Aliaga → Georgetown, Guyana |
| `551657f3 / 2` | Kralendijk, Bonaire → Hamburg → Khanom, Thailand |
| `551657f3 / 3` | Koper, Slovenia → Hamburg → Basra, Iraq |
| `8579256f / 1` | Mongla, Bangladesh → Aliaga → Pago Pago, American Samoa |
| `8579256f / 2` | Port Stanley, Falkland Islands → Mersin → Funafuti, Tuvalu |
| `8579256f / 3` | Acajutla, El Salvador → Mersin → Moudi Terminal, Cameroon |

For the first family, loading and issue place follow the middle node, not the
receipt/origin. The first published example therefore renders `T/S CARGO FROM
EUROPOORT, NETHERLANDS / TO GEORGETOWN, GUYANA VIA ALIAGA` with matching labels.
Goods, parties, quantities, seals, identifiers and other sampled fields also
change; this is not a route-only string-replacement demonstration.

- [Original OCR and all six rendered samples](../artifacts/kie-synthesis-production/curated-v7-transshipment-pilot/samples.md)
- [Coordinate-enriched samples](../artifacts/kie-synthesis-production/curated-v7-transshipment-pilot/positions-v2/samples.md)
- [Plain dataset](../artifacts/kie-synthesis-production/curated-v7-transshipment-pilot/dataset.jsonl)
- [Publication manifest, costs and exact hashes](../artifacts/kie-synthesis-production/curated-v7-transshipment-pilot/manifest.json)
- [Offline stress, mutation, geometry and benchmark results](../artifacts/kie-synthesis-production/curated-v7-transshipment-pilot/offline-validation.json)

### Validation and measured impact

| Check | Result |
|---|---|
| Full generated records / source families | 6 / 2, all published and replay-valid |
| Final rendered model review | Zero unresolved findings; no adjudication overrides |
| Unit/regression tests | 180 passed (`test_curated*.py` plus `test_synthesis_curated.py`) |
| Real-source deterministic stress | 1,000 scenarios, each reproduced exactly |
| Deliberate route corruptions | 3,000/3,000 rejected: stale hub, missing field, invented country |
| Stress geography coverage | 181 distinct origin and 181 destination countries; four configured hubs |
| Goods diversity in the two 500-draw cohorts | 18 / 320 distinct HS6 identities; 61 / 288 package quantities |
| Existing direct-route regression | 100 complete serialized scenarios equal to pre-change baseline; all previous 72 published candidates replay and validate |
| Source-only feeder probe | Both original `BIANCA RAMBOW 941 S` occurrences become the same `NICOLA 393 D`, distinct from main vessel |
| Coordinates | 381/502 content lines positioned; 121 explicitly unknown; 12/12 intermediate-port mentions positioned |
| Geometry preservation | Nine coherently scaled/translated pages; text, targets and structural lines unchanged; all known coordinates within 0–1000 |

The coordinate stage also retains its existing measured-source-region bounds,
axis-order/alignment and nearest-neighbour/tie checks. Its 121 unknown positions
are preserved as ` ||`; text is not removed and another field's position is not
borrowed. These are inherited approximate layout anchors, not reconstructed
synthetic PDF glyph positions.

Interleaved warm direct-route sampling measured **0.1211 ms before / 0.1286 ms
after** per sample: +0.0074 ms. Incremental traced Python allocation peaks were
**41,900 / 42,180 bytes**, a 280-byte difference. Cold catalog loading was 4.87 s.
The new topology-enabled sources measured 0.631 / 0.652 ms per warm scenario;
that includes their additional node selection and source-specific cargo work.
The stress process peaked at 433.83 MiB RSS, including both old/new catalog model
copies; this is not an isolated per-worker memory estimate. Timings and memory
measurements are recorded, not inferred from test success.

The full new campaign ledger is **$0.00735508** (about 0.74 US cents), including
the recorded generation attempts, correction and review calls; approximately
$0.001226 per published sample. Six documents are too small for a dependable
large-campaign cost projection. No paid calls were used for stress tests or
replaying the previous pilot.

The real **600 training / 60 validation records remain unchanged**. Their SHA-256
values are respectively `ca15c382bd1a36e72db978a0acb34f9dec64e8ea6c7e00639e62bccc98058305`
and `b8c0d4bddd4b3a452f901f3fc5e08e54d580a82768eddd5df33c97ab2c85da5a`.
No training run or dataset merge was started.

### Readiness and limits

These **two transshipment families are ready for more variants through the
normal pipeline**, alongside the previously validated direct-route pilot.
The remaining nine labeled source families need the same explicit source
ownership/onboarding as any additional template; this pass does not certify
their unrelated cargo or party bindings. The feeder mechanism is integrated
and real-text tested, but its whole source is not one of the six publications.

The route contract models extraction roles and coherent synthetic geography,
not verified carrier services, realistic voyage duration or optimal trade
routes. Some uniformly sampled country combinations are circuitous. Likewise,
the current public schema has one intermediate-port field, not a complete
ordered list of arbitrary shipping legs. Neither claim is needed for this
pilot's label/layout-learning objective.
