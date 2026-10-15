# Goods-description prompt scope: manual comparison

Date: 2026-10-15. **Completed, probe-only.** No production prompt, schema,
template, training configuration or dataset was changed. Country diagnosis and
GPU/model experiments remain outside this pass.

**Follow-up completed:** qualifier-rich sources were tested next, including a
real-training occurrence of `99.7% AS PER ISRI` and an admitted `3930X50 KG`
packing expression. Explicit source-content profiles improved detail retention;
generic additional prose did not. See [the follow-up findings below](#follow-up-source-like-qualifiers-without-expanding-bare-names).

**Wider-source validation also completed:** 12 fresh templates and two repeated
long sources with different sampled goods; 80 proposals reviewed. Both requested
polishes ran on every case. Automatic profile inference was separately tested,
without manually supplied source profiles. See [the wider test](#wider-source-robustness-check).

**Latest: classifier-free retry completed.** All25 previous source/scenario pairs
were rerun with the simple count baseline and the general-detail variant plus a
single contextual-boundary clarification:100 proposals, all directly reviewed.
No added tariff caption was found; some catalogue/detail contraction persists.
No classifier or source profile was used. See
[the latest findings](#classifier-free-contextual-boundary-retry-2026-10-15).

## Explicit source counts improve description scope

The useful refinement is to state the **measured source word count in the
goods field description**, while keeping it approximate. Adding only a generic
word-count sentence to the system prompt did not reliably control expansion.
The explicit reference produced short identities for short sources and much
fuller wording for the longest source, without hard length constraints.

The additional format/content instruction improved some long-list presentation
and length, but was not uniformly better: two outputs inserted a host-owned HS
caption and were correctly rejected. The simpler word-count-anchored variant
is the better default candidate from this probe.

[All six sources, previous accepted wording and all 60 raw variations](analysis/goods-description-scope-probe-20261015/SOURCE_VARIATIONS.md)
are available for inspection. Failed proposals are preserved unedited.

## Current request and experimental controls

The production system prompt is `WORDING_PROMPT` in
`src/document_ocr/synthesis/curated_wording.py`. Per-goods requirements and
commodity/physical context are constructed by `wording_request` in
`src/document_ocr/synthesis/curated_campaign.py`. The native Pydantic schema
also carries each field's description.

The existing request already supplies:

- The actual source goods wording as a structure example, with its complete
  reviewed description ownership. All six selected examples cover their
  source description exactly; none is an accidental first-line-only excerpt.
- Sampled HS6 commodity references, classification context and required
  product facts. Exact tariff-to-product classification is not scored here.
- Sampled package and equipment categories, explicitly described as transport
  compatibility context. Therefore adding package categories was unnecessary;
  the second treatment clarifies how that existing information should be used.
- Instructions separating product wording from host-owned shipment totals,
  equipment settings, references and separately printed HS captions. Incidental
  per-unit capacity/fill remains allowed, as previously agreed.

Six ambient, LLM-worded sources were selected: two short, two medium and two
long. Each arm generated two variations per source in one native structured
request, using **GLM-5.3-Flash, low reasoning**, the unchanged provider/native
decoding settings and no stronger model. The request contains goods fields
only: these are language probes, not newly generated complete B/L samples.

For each source, all arms use the same archived sampled commodity scenario,
the same complete source example, physical context and output keys. The
historical accepted wording/current R25 description is included as an
additional comparison, not mislabeled as a fresh current-prompt control.
An actual fresh control uses the unchanged current production prompt.

There is no truncation, padding, character/line quota, hard word-count
validator, semantic repair or replacement of failed outputs. All output text
was opened and reviewed directly. Existing production wording checks were
also applied independently to each generated variation.

### A: approximate word count

The following text was inserted after the existing descriptive-depth paragraph,
with the rest of the system prompt, request and schema unchanged:

> Use approximately the same total word count as the supplied source description
> fragments. Contract or distill the supplied goods wording, or expand it as
> needed, to reach a comparable overall length while retaining the sampled
> commodity subjects and required physical form.

### B: A plus source-like information and presentation

B uses exactly A's system prompt, adding:

> Match the source's presentation and kinds of product information as well.
> Synthesize plausible, realistic product details analogous to those in the
> source, adapted to the sampled goods. The supplied package and equipment
> categories are compatibility context, not a request to add packing wording
> to the description.

### A+ and B+ use measured source word counts in field descriptions

The initial A/B results still expanded the two-word examples substantially.
A single follow-up tested whether an explicit source-length reference in the
field description helped. It did not change commodity facts or add restrictions:

> The supplied source description contains N words in total. Aim for
> approximately N words across the requested description fields, distilling
> or expanding the sampled goods wording as needed.

`N` is calculated from the supplied source description examples, not the
classification brief, full OCR, transport context or generated text. It is
guidance, not a required count. A+ uses A's system prompt; B+ uses B's.
Both expose this same reference through the native schema's field description
and the existing plain-text field requirements.

The request/schema equality checks verify that control/A/B differ only in
their system prompt, and A+/B+ differ only in their system prompt. A+/B+
intentionally have the additional field-description guidance compared with
their unanchored counterparts.

## Exact word-count results

Counts use whitespace splitting on complete generated description fragments.
Each slash-separated pair represents the two variations, not a range or mean.

| Source description → sampled goods | Source | Previous accepted | Fresh control | A | B | A+ | B+ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `OILFIELD EQUIPMENT` → polystyrene | 2 | 25 | 34 / 30 | 29 / 26 | 41 / 33 | **2 / 2** | **2 / 2** |
| `INSHELL WALNUTS` → electric storage radiators | 2 | 196 | 42 / 53 | 40 / 42 | 61 / 51 | **4 / 4** | **4 / 4** |
| Glucose-test models list → synthetic monofilament | 34 | 13 | 26 / 23 | 14 / 17 | 74 / 60 | 23 / 23 | 23 / 25 |
| Used/new equipment inventory → refrigerator-freezers | 34 | 36 | 46 / 41 | 66 / 65 | 59 / 59 | 20 / 24 | 27 / 26 |
| Telecom parts list → optical fibre cables | 58 | 57 | 42 / 40 | 41 / 43 | 72 / 59 | 39 / 38 | 47 / 50 |
| Compressor/serial inventory → propylene copolymer resins | 132 | 62 | 9 / 16 | 61 / 54 | 81 / 86 | **126 / 106** | **130 / 130 rejected** |

The previous polystyrene wording contains 25 whitespace-delimited words;
earlier narrative estimates are not reused as measurement inputs here.

| Arm | Mean absolute source-word-count difference | Existing wording checks |
| --- | ---: | ---: |
| Fresh control | 38.50 words | 12 / 12 pass |
| A | 34.17 words | 12 / 12 pass |
| B | 33.83 words | 12 / 12 pass |
| A+ | **10.08 words** | **12 / 12 pass** |
| B+ | 5.17 words | **10 / 12 pass** |

B+'s length statistic deliberately includes its two rejected raw outputs;
it must not be presented as an all-valid improvement. A+'s absolute difference
is 73.8% lower than the fresh control in this small diagnostic set. This is a
description-scope measure, not extraction F1 or a demonstrated training gain.
The median length ratio alone would be misleading: the control's median is
near one despite severe expansion of short sources and contraction of the
longest source.

## Direct review findings

### Short descriptions: explicit reference is the material difference

The generic A sentence left polystyrene at 26–29 words and radiators at
40–42. B expanded these further. A+ and B+ instead produced:

```text
POLYSTYRENE RESIN
POLYSTYRENE GRANULES
```

and:

```text
ELECTRIC STORAGE HEATING RADIATORS
```

Four words rather than two for the radiator identity is a sensible approximate
result: it retains the complete sampled commodity rather than discarding
meaning to satisfy an arbitrary exact count. Both radiator variations use the
same concise identity. This is a wording-diversity observation, not a label
error: independently sampled shipments may legitimately share a product name.

### Medium descriptions: still approximate, not a universal improvement

The medium monofilament source is a five-line models list. B generated three
detailed variants of 60–74 words; anchored variants stayed around 23–25 but
often used one specification passage instead of five short model entries.
Thus length guidance does not guarantee equivalent list granularity.

For household appliances, A overshot to 65–66 words, while A+ contracted to
20–24 and B+ to 26–27. Complete appliance identity, separate doors/drawers and
plausible technical attributes survived. A+ was not closer than the control
in every medium case. This is why the recommendation is not an exact-count
policy or a claim of uniform semantic improvement.

### Long descriptions: information depth can be restored without fixed lines

For the telecom source, B better retained a vertical catalogue/list character.
B+ yielded 47–50 words with a heading, cable variants, technical qualifiers
and a new batch identifier. A+ was shorter at 38–39 words. This is a real
benefit of presentation guidance, although not every medium example followed
the same pattern.

The longest source exposes the control's opposite problem: this fresh call
contracted 132 source words to 9–16. A+ supplied coherent grade/specification
passages of 106–126 words, rather than truncating or repeating text to hit a
quota. B+ created a longer seven-grade list but introduced the separate
failure below.

### Two rejected B+ proposals: HS context became product text

Both B+ resin outputs start:

```text
POLYPROPYLENE COPOLYMER RESINS IN PRIMARY FORM (PELLETIZED), HS 390230:
```

The host, not this wording field, owns printed HS codes and their formatting.
The existing unmodified validator rejected both with:

```text
v003: host-owned tariff caption inside generated description
```

No correction or deletion was applied to make this treatment look successful.
The two full outputs were also identical, and contain a cosmetic `BAG BAG`
repetition. These are further reasons not to select B+ just because its raw
word count is closest to the source. The rejection demonstrates that the
existing guard acts on this actual generation path.

### Compatibility context can still become unnecessary prose

One unanchored B radiator output adds:

```text
... EACH UNIT SUITABLE FOR CRATED PACKING AND 40FT HIGH CUBE GENERAL PURPOSE CONTAINER STOWAGE
```

This is transport-context spillover, not analogous information from its
two-word source. It passes the narrow deterministic wording checks; direct
review finds it outside the intended lexical scope. Some other outputs also
append generic pallet/container-transport phrases. These are scope/style
issues, not proven arithmetic contradictions or current-dataset label defects.
No full documents or labels were published from these probes.

Incidental bag fills, spool sizes and appliance capacities were not rejected
merely for being newly synthesized. They are permitted product qualifiers
under the agreed policy; no generated explicit shipment net-total omission
was found. Source serial/batch identities were not retained in the reviewed
new text. Exact HS/product-category purity was not used as a rejection reason.

Minor raw casing such as `ARMoured` and `POLYcore` remains visible in the
inspection file. Production's existing final target-casing normalization was
not changed or disabled; this report preserves raw wording for transparency.

## Recommendation

Use **A+ as the leading integration candidate**: the requested approximate
word-count guidance plus the measured source count in the actual field
description. Keep the count soft and joint over the complete description,
preserve every sampled commodity, and allow longer natural identities when
needed. Retain the existing source-depth/list instructions and product versus
host-fact boundaries. In a joint goods/party request, derive the count only from
that description group's owned product examples, never names or addresses.

Do not make B+'s extra invention/presentation instruction the universal default
from this evidence. It can improve long-list fidelity, but this probe found
HS leakage and did not establish consistent medium-list fidelity. Nor does
this small sample prove an improvement in extraction training; it demonstrates
a practical way to improve the generation-length distribution for testing.

Any later integration should verify the joint goods-plus-party request path
and complete rendered samples as well. This pass intentionally changes neither
production requests nor current synthetic wording.

## Accounting, validation and preservation

- **60 descriptions**, 30 successful native calls; all native outputs have
  the requested structure. Existing semantic wording checks pass 58 and reject
  the two HS-caption proposals. All 60 raw texts were directly reviewed.
- Three HTTP429 transport rejections were preserved and explicitly retried
  without changing model/provider/decoding settings. They cost zero. No output
  repair or quality-based resampling occurred.
- All-attempt provider-reported cost: **$0.00950190**. Successful calls used
  **42,588 input tokens**, **7,917 output tokens**, including **349 reasoning
  tokens** and **7,568 visible output tokens**; 7,040 input tokens were cache reads.
- Successful-call median latency by arm: control 6.18 s; A 8.38 s; B 9.75 s;
  A+ 1.85 s; B+ 1.92 s. These phases ran under different transient provider
  load/cache conditions, including rate limiting. They are observed timings,
  not proof that the anchored prompt intrinsically makes inference faster.
- **Six targeted probe-design tests pass**; Ruff passes. Checks establish
  isolated prompt differences, unchanged source examples, native field
  descriptions carrying the soft count, preservation of rejected text and
  absence of hard length rejection.
- Hash checks preserve all **357 production/dataset files**, the selected
  source-case assets and historical wording receipts. No GPU, model training,
  dataset migration, template edit or country-policy change occurred.

Evidence: [design and exact prompts](analysis/goods-description-scope-probe-20261015/design.json),
[request/schema artifacts](analysis/goods-description-scope-probe-20261015/requests),
[all results](analysis/goods-description-scope-probe-20261015/summary.json),
[metrics and all-attempt accounting](analysis/goods-description-scope-probe-20261015/metrics.json),
[provider call receipts](analysis/goods-description-scope-probe-20261015/calls).

The probe runner and tests remain in the analysis folder. Source IDs, sampled
HS6 subjects and historical receipt pins are recorded per case in `design.json`.

## Follow-up: source-like qualifiers without expanding bare names

Completed 2026-10-15. **Manual probes only; no production or dataset changes.**

[All seven sources and all 56 raw proposals](analysis/goods-description-details-probe-20261015/SOURCE_VARIATIONS.md)
show the actual wording, not automatically repaired examples.

### Selection and ownership

The motivating paprika `1680X25KG.` example is a validation document, so it
was not used as a synthesis source. Instead, an admitted lentils template
supplies the comparable `PACKED IN 3930X50 KG` expression. The copper
`99.7% AS PER ISRI` wording also occurs in **real training row 546**, source
`55d71731…`; that training occurrence was used in a manual wording request,
without admitting a new template. No validation source entered the experiment.

The seven cases are:

- The same two bare-name controls from the previous experiment, each two
  source words: `OILFIELD EQUIPMENT` and `INSHELL WALNUTS`.
- Garnet with mesh grade, bag capacity, conductivity and chloride limits.
- Cork variants with model codes, dimensions and construction/finish wording.
- Zinc with a numeric purity qualifier and material/packing form.
- Lentils with crop/origin and a compact multiplied packing expression.
- Copper scrap with numeric purity/grade and an industry reference.

Purity in the zinc example is in the approved goods block, not the nearby
Marks column; the source PDF ownership was checked. Source names/IDs,
source targets, complete actual product-field examples, archived scenarios
and receipt hashes are preserved in
[design.json](analysis/goods-description-details-probe-20261015/design.json).

The lentils template illustrates why full source descriptions and generated
regions are not interchangeable. Its LLM-owned examples contain **nine words**;
the full approved description has **15**. Origin, package noun, equipment
wording and fixed framing are assembled separately. In contrast, the
`3930X50 KG` qualifier is inside an LLM product region. This experiment does
not secretly reassign those owners or claim that generated fragments alone
are fully assembled descendants. The garnet `(6 CONT.)` source phrase is
also inside its existing product region, which matters for the failure below.

### Treatments

Each treatment generated two variations per case, using the same cheap
GLM-5.3-Flash/low-reasoning native PydanticAI output path. The two variations
share one call; they are not 14 independent calls per arm.

1. **Previous B+:** fresh control using the prior second treatment and explicit
   approximate source word count in field descriptions.
2. **Source-shaped system guidance:** replace B+'s generic presentation
   paragraph with instructions to mirror the source's amount and kinds of
   detail, while leaving bare identities concise.
3. **Field-anchored general guidance:** retain A+'s system prompt and put the
   source-information-pattern instruction directly in each goods field.
4. **Explicit source-content profile:** retain A+'s system prompt/count and
   add a short field-description summary of the source's information roles.
   For example, `commodity identity, numeric grade/composition qualifier,
   industry grade/standard reference`, rather than asking generally for
   "more realistic detail".

The fourth treatment's profiles were **manually identified for this probe**,
not inferred by an already-implemented automatic classifier. They contain
information roles, not copied purity, count or capacity values. Short controls
have `commodity identity only`; detailed profiles ask for corresponding
detail appropriate to the newly sampled goods. No character limits, fixed
line counts, hard word-count validator or post-generation text repair is used.

The initial pair differs only in its system paragraph; request text and
native output schemas are checked identical. Treatments three/four deliberately
change field descriptions and return to A+'s system prompt. They are **not**
a pure test of moving identical wording from system to field description.

### Results: detail retention, not only length

| Source pattern | Previous B+ examples | Explicit source-content profile examples |
| --- | --- | --- |
| `99.7% AS PER ISRI` | Clean aluminium offcuts; no industry reference in either variation | `ALLOY 6063 SERIES, PER ISRI TELIC` and `99.5% PURITY PER ISRI TALON` |
| `3930X50 KG` packing qualifier | `ON PALLETS, BULK` / `PALLETIZED, LOOSE` | `ROLLS OF 50 METRES` / `REELS OF 100 METRES` |
| Zinc purity/form | Coils with `0.8MM` / `1.2MM` thickness | `C 0.08% MAX` / `GRADE AISI 1008`, with material form |
| Cork models/dimensions | Both variations already preserve models/dimensions | Both retain models/dimensions and add construction details such as `PIGTAIL 2,5 SQMM` |
| Garnet technical/packing detail | Yarn denier/tenacity; generic pallet wording | Yarn denier/tenacity/elongation, `FIVE 2KG CONES` and other capacity wording; one proposal rejected for extra container count |

These are analogous commercial descriptions for different sampled goods,
not literal preservation of paprika/copper/lentil values. The aim remains
extraction training, not proving exact HS/product equivalence or metallurgical
certification of every newly written standard/grade reference.

For example, the copper training source:

```text
COPPER MILBERRY SCRAP GRADE 99.7% AS PER ISRI
```

produced for the manually supplied aluminium-scrap commodity brief:

```text
ALUMINIUM SCRAP CLEAN EXTRUSION SOLIDS 99.5% PURITY PER ISRI TALON
```

The admitted multiplied-packing example produced this coherent pair of
LLM-owned regions for tubing:

```text
COATED COTTON HOSEPIPING
INNER DIAMETER 25 MM, RUBBER-LINED

ROLLS OF 50 METRES
```

This restores a numeric per-unit detail rather than generic `PALLETIZED`
wording. It does **not** establish that the model can safely originate a new
shipment package count in a `count × fill` expression. Such a count must come
from the actual sampled shipment, with host-owned assembly where applicable.

### Short controls: concise, but not literally unaffected

All counts are whitespace-delimited words per complete set of requested
LLM-owned regions. Slashes separate the two variations.

| Source → sampled goods | Source words | Previous B+ | System refinement | General field refinement | Explicit content profile |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bare equipment → polystyrene | 2 | 2 / 3 | 7 / 4 | 2 / 5 | 5 / 5 |
| Bare walnuts → radiators | 2 | 4 / 4 | 4 / 4 | 4 / 4 | 4 / 4 |
| Garnet detail → yarn | 25 | 20 / 23 | 16 / 17 | 27 / 22 | 36 / 33 |
| Cork models → brushes | 23 | 24 / 23 | 24 / 22 | 16 / 15 rejected | 19 / 20 |
| Zinc purity → steel | 8 | 7 / 8 | 8 / 7 | 8 / 8 | 12 / 12 |
| Lentil product regions → tubing | 9 | 11 / 10 | 11 / 13 | 11 / 10 | 12 / 12 |
| Copper purity/reference → aluminium | 8 | 9 / 9 | 6 / 7 | 7 / 8 | 11 / 10 |

The explicit-profile short outputs are simply:

```text
POLYSTYRENE RESIN IN PRIMARY FORMS
OTHER POLYSTYRENE IN PRIMARY FORMS
ELECTRIC STORAGE HEATING RADIATORS
ELECTRIC STORAGE HEATING RADIATORS
```

There is no return to 30–60-word descriptions, packing paragraphs or invented
specification lists for bare-name examples. Nevertheless, polystyrene grows
from two/three words to five. **Zero short-output change was not demonstrated.**
For a production candidate, keep the simpler A+ path unchanged for genuinely
bare-name sources and selectively supply richer patterns only to sources that
contain corresponding detail. Do not apply an unconditional "add detail"
instruction everywhere. That selective production routing is a recommendation,
not an implementation claimed by this probe.

### Rejected proposals and validator limits

The system-only refinement did not consistently preserve purity/reference
information and expanded a short control. The generic field refinement was
also insufficient: its two brush descriptions appended standalone package
declarations (`2 CTNS OF 20 PCS`, `2 BOXES OF 10 PCS`), which existing guards
correctly rejected. One aluminium proposal also has malformed `BALESD` wording.

The explicit-profile yarn proposal contains:

```text
POLYPROPYLENE FILAMENT YARN, SINGLE, UNTWISTED, 900 DENIER (12 CONT.)
```

The archived sampled shipment has **six**, not twelve, containers. This is a
real contradiction, unlike harmless differences in incidental fill weights or
exact HS classification. Narrow wording checks pass it; direct comparison with
the sampled shipment rejects it. The source example itself contains `(6 CONT.)`
inside its current LLM region, while the model-facing compatibility context
provides equipment categories but not the actual container count. The probe
therefore exposes a product/transport boundary risk, not a current dataset edit
or evidence that full rendering is already safe for this proposal.

All 56 raw proposals were opened and reviewed. The
[manual-review inventory](analysis/goods-description-details-probe-20261015/manual-review.json)
records the two deterministic rejections, the one extra inventory contradiction
and the narrower wording/style concerns. Failed proposals were not rewritten
or quality-resampled. No sample from this probe was added to training.

### Conclusion and measured cost

The useful improvement is **explicit source information roles in the actual
field description**, alongside soft length guidance—not simply more global
instructions to invent details. It restored grade/reference and numeric
packing specifications in the intended direction. Bare-name examples remained
concise, but did not retain identical word counts, and the one transport-count
failure prevents calling this a universally safe drop-in replacement.

If integrated later, the small source-content profile belongs with reviewed
template description metadata. Keep host accounting separate and test complete
rendered descendants, including multiplied packing expressions, before any
bulk description reprocessing. No extra per-descendant review/generation call
is inherently required by the tested profile: it is input metadata supplied
in the existing wording call. Automatic profile derivation is not proven here.

- **56 descriptions**, 28 successful native calls, plus three preserved
  zero-cost HTTP429 rejections explicitly retried with unchanged provider/settings.
- **$0.00632289 total**, including all attempts. Input: **38,223 tokens**;
  output: **2,976**, comprising **232 reasoning** and **2,744 visible** tokens;
  cache reads: **7,488** tokens.
- Existing wording guards: previous B+ **14/14**, system refinement **14/14**,
  general field refinement **12/14**, explicit profiles **14/14**. Direct review
  additionally rejects the explicit-profile container-count contradiction.
- Median successful-call latency by arm: **2.71 / 1.99 / 4.24 / 5.65 seconds**.
  Different transient load/cache conditions make this an observed cost/latency
  record, not proof of intrinsic throughput improvement. No production hot path
  changed. Peak process RSS across invocations: approximately **193.3 MiB**.
- **12 targeted tests pass**, including the prior probe's six tests. Ruff and
  `git diff --check` pass. Tests cover arm isolation, field/schema guidance,
  actual source selection, exclusion of validation sources, preserved failures,
  no hard length limits and content profiles without source numeric values.
- Hash verification preserves all **357 production/dataset files**, selected
  template assets and historical receipts. No training, GPU/model loading,
  template migration or data repair occurred.

Reproducibility: [probe runner](analysis/goods-description-details-probe-20261015/probe.py),
[follow-up design](analysis/goods-description-details-probe-20261015/follow-up-design.json),
[profile design](analysis/goods-description-details-probe-20261015/profile-follow-up-design.json),
[metrics](analysis/goods-description-details-probe-20261015/metrics.json),
[all-attempt receipts](analysis/goods-description-details-probe-20261015/calls).

## Wider-source robustness check

Completed 2026-10-15. **Probe-only; production, templates and datasets unchanged.**

[All sources, automatically inferred profiles and raw variations](analysis/goods-description-robustness-probe-20261015/SOURCE_VARIATIONS.md)
and the [manual review inventory](analysis/goods-description-robustness-probe-20261015/manual-review.json)
contain the complete results, including rejections and holds.

### Do we already know which sources are detail-bearing?

We know the **actual source examples and their reviewed ownership**. The current
template/extraction structures do not contain a validated descriptive-depth or
bare/detail flag. `description_blocks` defines membership, `description_groups`
ties related generated regions together, and cargo family distinguishes ambient,
thermal, DG and vehicle handling. None is an information-density classifier.

The previous successful content profiles were manually supplied; that is not an
existing automatic production capability. Word count is automatically measurable,
but cannot replace this semantic distinction: the nine-word cotton source has
composition, yarn count, construction and end use, while the 24-word auto-parts
inventory is largely a list of bare component names. A qualitative description
can also contain substantial detail without any numeric specification.

An explicit binary flag is **not required to run either wording treatment**:
the model already sees the source. This round therefore tests both polishes
without hand-labeling a detail-bearing switch, and separately tests whether an
automatic source-content profile could replace the previous manual profiles.

### Scope and experimental controls

The selection contains **12 new admitted ambient sources**, with no overlap
with either prior probe's source IDs or with validation. It spans three bare
commodity names, compact technical descriptions, qualitative construction/use
wording, single-region lists, and lists distributed across several owned regions.

Two previously tested long sources are added as **explicit repeated-source
controls**, this time with different archived sampled goods: the compressor
inventory becomes bamboo seats, and the telecom inventory becomes a phosphonate
polymer. They are not misreported as fresh source templates. The selected actual
LLM-owned examples span **1–132 words**, and requests span **1–7 product regions**.
Fresh admitted ambient sources above 60 words were unavailable after the prior
source exclusions; the two repeated controls provide the long-description stress.

Every case uses the same archived sampled facts across treatments. No source
selection labels or manually written information profiles are sent to the models.
Only ambient LLM-owned goods wording is tested; this is not a DG/thermal-host
rendering test or a generation of complete new training documents.

The three arms are:

- **A — first polish:** the prior A+ approximate word-count prompt and measured
  source count in field descriptions.
- **B — second polish, general:** A plus the prior general field-level request
  to reproduce analogous source information/detail roles. It requires no
  classification stage or manually supplied profile.
- **B — automatic profile:** A plus a source-content profile inferred in a
  separate call from the actual source regions alone. The profiler sees neither
  the sampled replacement goods nor manual selection labels. Each profile records
  an identity/qualified/list/uncertain class, an information-pattern instruction
  and excluded non-product context. It is inferred once per source, not once per
  descendant. The experiment does not silently replace uncertain profiles with A.

All calls use the unchanged **GLM-5.3-Flash, low-reasoning, native PydanticAI**
provider settings. There are two variations per wording call. These share a call
and are not statistically independent model runs. There is no output truncation,
padding, length rejection, semantic correction or quality-based resampling.

### All requested first/second-polish cases completed

A and general B each produced **28 descriptions across all 14 cases**. Automatic
profiles were inferred for all 14 sources, covering 24 owned regions. Two profiles
declared uncertainty, so that arm produced **24 descriptions on 12 sources** and
explicitly held the other two. Thus 80 proposals were generated, not the 84 that
would have resulted without holds. These holds do not leave either requested
first/second-polish comparison incomplete.

| Actual source example | Source words | A: first polish | B: general second polish | B: automatic profile |
| --- | ---: | ---: | ---: | ---: |
| Bulgur name → plated steel | 3 | 3 / 3 | 4 / 5 | 3 / 3 |
| Compressors name → vehicle parts | 1 | 6 / 6 | 5 / 5 | 5 / 5 |
| Book name → mixed-fibre fabric | 1 | 4 / 9 | 9 / 11 | 3 / 4 |
| Cotton count/composition → resin products | 9 | 8 / 8 | 7 / 8 | 11 / 11 |
| Liner model/dimension list → looms | 17 | 15 / 14 | 13 / 20 | 21 / 17 |
| Tobacco grade/crop → ethylene polymers | 8 | 14 / 11 | 13 / 9 | Held |
| Four bodycare identities → four packaging subjects | 7 | 27 / 25 | 12 / 15 | 17 / 15 |
| Split fan-model list → analytical instruments | 18 | 25 / 26 | 24 / 31 | 27 / 28 |
| Seven packaging-product regions → pump subjects | 16 | 26 / 31 | 33 / 33 | 21 / 27 |
| Glucose-meter catalogue → fabric variants | 55 | 36 / 38 | 52 / 29 | 54 / 59 |
| Qualitative pastry-bag wording → coated textiles | 24 | 14 / 15 | 18 / 17 | Held |
| Component inventory → electric motors | 24 | 31 / 30 | 37 / 37 | 18 / 20 |
| Long compressor inventory → bamboo seats | 132 | 163 / 137 | 130 / 126 | 124 / 134 |
| Telecom inventory → phosphonate polymer | 58 | 40 / 41 | 63 / 63 | 43 / 51 |

Counts refer to requested LLM-owned examples, not independently rendered host
pieces or auxiliary identities. For example, the packaging-list target has
additional brand wording and the pastry-bag target has separate non-stackable
wording. Those are not secretly added to the LLM word-count reference.

Mean absolute source-word-count difference is **9.07 for A** and **7.04 for
general B**, on the same full 14-case set. Automatic-profile B has 5.04 on its
12 completed cases. On those same 12 cases, A is 9.42 and general B is 7.42.
The automatic-arm figure must not hide its two holds or profile errors. These
are scope measurements, not extraction F1 or demonstrated training gains.

### Short inputs remain concise, with sensible identity-length exceptions

All three bare-source controls remain commodity identities rather than long
specification/packing paragraphs. For the one-word `BOOK` source, the replacement
is an inherently multiword fabric subject. The automatic-profile output is:

```text
WOVEN ACRYLIC-WOOL FABRIC
WOVEN MODACRYLIC-MIXED WOOL FABRIC
```

General B instead spells out the full fibre identities in 9–11 words. That is
more verbose than A/automatic B, but it is not a new 40-word specification
passage or a proven label defect. For the one-word `COMPRESSORS` source, all
arms need 5–6 words to express the new vehicle-parts identity. Forcing one word
would discard the sampled subject; the word-count rule correctly remains soft.

### Source matching helps some lists, but is not a universal improvement

For the liner catalogue, automatic profiles produced a heading with distinct
new model/reference/dimension rows:

```text
SHUTTLE-TYPE WEAVING LOOM
TLM-820H (207311) REED WIDTH 190 CM
TLM-820H (207458) REED WIDTH 230 CM
TLM-820H (207502) REED WIDTH 230 CM
```

A more often produced a single model passage. For the long bamboo-seat stress,
all arms produced genuinely detailed descriptions; automatic B's 124/134-word
lists retained new model/serial/dimension structure particularly well. General
B also stayed close to source length, but its tariff leakage below invalidates
using word count alone as the success criterion.

The mixed-region tests reveal other, narrower style issues: A turned the final
product slot of one list into `PACKED IN CARTONS`; general B added more spare-part
and carton filler than the bare source identities request. Automatic B kept those
regions product-bearing and more concise. Conversely, the automatic arm did not
consistently improve the auto-parts inventory, and its meter-catalogue output
added container transport prose. The printed 20-foot GP category matches the
actual sampled equipment, so that is context/style spillover, **not a proven
inventory contradiction or OCR/label disagreement**.

### The automatic profile flag is not reliable enough to route unreviewed

Direct review found:

- The cotton profile correctly describes count/composition/process/use, but labels
  the region `identity_only`. Those two outputs contradict each other.
- The fan-model list is also labeled `identity_only`, despite its own profile
  explicitly describing model, part-number and serial-range entries.
- Tobacco grade/crop and qualitative pastry-bag construction are marked uncertain,
  although their reusable information patterns are apparent in the source. They
  are avoidable review holds for this task, not unknown shipment facts.
- The meter-catalogue profile returns the placeholder `description_pattern`,
  not an actual information-pattern instruction.
- The long compressor profile makes an overreaching claim that used-condition
  wording is not a product qualifier. That should not become a general rule for
  deleting condition words from reviewed descriptions.

All inferred profiles are preserved unedited; no manual profile was substituted
to improve the automatic arm. The experiment therefore **does not establish a
trustworthy automatic bare/detail switch**. It does establish that automated
profile proposals sometimes supply useful structure, and identifies specific
classification/metadata problems that manual profiles had hidden.

### Real scope violations versus validator false positives

General B inserted separate **tariff captions in four of 28 proposals**:

- Both bamboo-seat proposals: `HS 9401.52` or `HS REFERENCE 9401.52`.
- Both phosphonate proposals: `HS REFERENCE: 391120` or
  `HS HEADING REFERENCE: 391120`.

These are actual product-field scope errors under the existing description
policy, not complaints about exact HS/product matching. Existing wording guards
catch **one** and miss the other three caption variants. A and automatic B have
no confirmed tariff-caption leak in this set. That is a measured observation,
not a general guarantee of zero errors.

The same general failure already appeared in the preceding long-resin B+
proposals. Seeing it again with bamboo seats and a different polymer subject
means it is not confined to one sampled commodity or one particular source
product. It is a recurring boundary between supplied tariff context and the
generated product field, not evidence that exact tariff/product matching is
required for this training objective.

The existing guards also produce two different holds that must not be counted
as confirmed semantic failures:

- A's `UNIT WEIGHT 4,5 KG` is held as ambiguous numeric notation. It is a
  plausible decimal-comma per-unit mass, not evidence of an arithmetic or net-total
  mismatch. No text was rewritten to make the guard pass.
- Automatic B's `HS-2600 HPLC DETECTOR MODULE` is flagged as a tariff caption.
  In context it is a model name alongside serial identifiers, not a customs
  declaration. This is a review-confirmed false positive.

Mechanical wording passes are therefore **27/28 for A, 27/28 for general B,
23/24 for automatic B**, but those numbers are not semantic approval rates.
The root review distinguishes actual scope violations from conservative/incorrect
guard holds, length/style drift and cosmetic or technical-plausibility concerns.
No new explicit shipment-net-total omission or confirmed invented container count
was identified in these 80 proposals. Incidental capacity/fill and exact tariff
classification were not treated as failures merely because they were newly written.

### Decision supported by this round

The **first polish remains the simplest leading candidate**. Its length guidance
transfers to fresh source examples, including multiword identities and substantially
different sampled goods. It needs neither a hand-labeled detail flag nor an extra
profile call. This is wording evidence, not yet a complete-rendering/training test.

The second polish is valuable for some detailed/list layouts, but this broader
set does not support applying it indiscriminately. Explicit source-content
profiles can help, but the automatically generated category cannot currently be
trusted as an unreviewed routing decision. A source-level profile proposed during
template compilation and reviewed once is more defensible than pretending we
already have a reliable per-descendant detail classifier. Whether a simpler
profile representation can remove the observed conflicts is **not proven here**.

Before integrating richer wording broadly, the concrete issues are now bounded:
source-profile semantics/placeholder detection, product versus transport scope,
and distinguishing real tariff captions from legitimate model names. This pass
does not patch production validators or relabel current data under the guise of
a prompt experiment. Both requested broader tests are completed and audited.

### Accounting and preservation

- **80 raw proposals reviewed directly by the root**, plus all **24 inferred
  region profiles**. One read-only helper located source/receipt candidates;
  it did not generate, edit or adjudicate these proposals.
- **54 successful native calls**: 14 profile calls, 14 A calls, 14 general B
  calls and 12 automatic-B calls. Seven zero-cost upstream HTTP429 rejections
  were preserved and explicitly retried without changing provider/model settings.
- All-attempt cost: **$0.01486835**. Profile inference accounts for **$0.00128218**;
  A **$0.00429125**; general B **$0.00490101**; automatic B **$0.00439391**.
  Generic B generation cost is about 14.2% higher than A in this probe. Cache,
  input length and transient provider conditions matter; this is not a universal
  price/throughput forecast. A template profile could be amortized across descendants.
- Tokens: **74,531 input**, **10,219 output**, including **801 reasoning** and
  **9,418 visible** output tokens; **11,840** cache-read input tokens.
- Median successful-call latency: profile **3.95 s**, A **3.83 s**, general B
  **4.71 s**, automatic B **7.16 s**. Peak process RSS approximately **195.3 MiB**.
  No production hot-path performance changed or GPU workload was started.
- **19 offline tests pass**, including both preceding probe suites; Ruff and
  `git diff --check` pass. Checks cover input isolation, exact native region
  coverage, unchanged commodity/source facts, soft counts, uncertain-profile
  holds without fallback, fresh-source exclusion and explicit repeated controls.
- Hash verification preserves all **357 production/dataset files**, selected
  source assets and archived receipts. Existing phone/container worktree changes
  were preserved. No generated text was published into a dataset or template.

Evidence: [frozen design](analysis/goods-description-robustness-probe-20261015/design.json),
[source-only audit reference](analysis/goods-description-robustness-probe-20261015/source-review-reference.json),
[metrics](analysis/goods-description-robustness-probe-20261015/metrics.json),
[manual inventory](analysis/goods-description-robustness-probe-20261015/manual-review.json),
[all-attempt receipts](analysis/goods-description-robustness-probe-20261015/calls),
[runner and experimental controls](analysis/goods-description-robustness-probe-20261015/probe.py).

## Classifier-free contextual-boundary retry, 2026-10-15

This completes the requested retry without a classifier, source-specific profile,
extra model call or routing decision. The source/scenario union covers **25 cases
from23 sources**, preserving the exact sampled facts from the preceding scope,
detail and robustness tests. The two long sources appear with their two distinct
previous commodity draws; duplicated identical short controls are tested once.
Each case receives two raw proposals per arm: **100 proposals total**.

- **A:** unchanged first word-count polish, rerun as a contemporaneous control.
- **B:** the same base plus the previous general source-information/detail
  guidance and this one added field clarification:

> Use surrounding shipment facts to check compatibility, not to expand the
> description: do not add HS/tariff declarations, container/equipment or routing
> details, or shipment totals. Follow the source's product information instead,
> retaining analogous attached specifications and per-unit packing where shown.

The source regions remain plain text. Sampled goods, HS classification context,
host-owned shipment facts and output schema are unchanged. Approximate word count
is guidance, not a rejection threshold; no character or line quota was introduced.
No hand-written detail flag, inferred profile or classifier output is supplied.
This tests **new wording**, not a revised rule to delete approved source wording
from real extraction targets. In particular it does not reopen the policy on an
already printed specification or a genuinely embedded customs-code clause.

### Findings from direct review of all100 proposals

The contextual clarification addresses the concrete leakage motivating the retry:
**no newly added HS/tariff caption was found in either50-proposal arm**. On the
same14 broader robustness cases, the previous general-detail arm had4/28 such
leaks; the new B has0/28. These are fresh stochastic generations, not a proof of
zero future risk or an isolated causal estimate. No invented container inventory
count or explicit shipment-net-total omission was identified in this new set.
All100 pass the existing mechanical wording checks; direct review is still needed
because a wording-check pass does not measure source-like richness or scope fully.

The improvement does **not** require policing exact HS-to-product matching.
Incidental plausible product specifications, per-unit fill and capacity are
allowed, as agreed. In particular `25 KG BAGS`, product dimensions and operating
pressure are not treated as errors merely because the model writes them.

|Case|LLM-owned source words|Fresh A words|New B words|Direct observation|
|---|---:|---:|---:|---|
|Purity / ISRI|8|9 /8|9 /9|B produces an ISRI-qualified aluminium description in one proposal and99.7% grade in the other; it does not reliably combine both details|
|Compound packing / tubing|9|14 /16|16 /12|B adapts packing to25m rolls or coiled tubing, without copying the old shipment count|
|Model / dimensions / standard|23|16 /17|23 /23|B preserves richer model/dimension/packing structure|
|Long compressor list → resins|132|114 /117|141 /156|B writes grade/MI lists and per-bag packing, without the earlier tariff-caption leak|
|Same long source → bamboo seats|132|135 /132|131 /129|Both produce substantial new model/dimension catalogues; B has no added HS caption|
|Telecom list → phosphonate polymer|58|49 /48|42 /43|B retains technical grades/properties and synthetic batch wording, without the previous HS reference|
|Meter catalogue → fabric|55|26 /27|**9 /11**|B over-condenses to the commodity identity and loses the source-like catalogue/detail richness|
|Bare `BOOK` → blended fabric|1|9 /11|2 /2|B stays short, but drops the sampled fibre composition; A retains it|

The meter-list and bare-book results matter: the new clarification improves
context boundaries, **not every aspect of generation**. We cannot confidently
attribute their contraction to the added sentence alone; two joint-call proposals
per arm are not enough to disentangle stochastic variation. They are nevertheless
observed counterexamples to claiming that general B consistently preserves the
source's information pattern. Cotton composition/process and the liner-model list
also show that source-like detailed structure is not reliably reproduced in all
short/medium examples. Short polystyrene/radiator controls remain short, while
multi-region bodycare/packaging cases expand to name their multiple new subjects.
That necessary expansion is not automatically an error.

Fresh A also illustrates why the boundary sentence is useful independently of
the richer-detail treatment: its split packaging-list proposals add a20ft
container phrase, and its long resin proposals add generic container-stowage
wording. B avoids those additions. No new equipment count was invented in them;
the distinction is unnecessary contextual wording versus a shipment contradiction.

### Decision and experimental scope

**A remains the simpler baseline. B with the new boundary is a viable richer
candidate, not a universally better replacement.** The retry succeeds at removing
the previously observed tariff leaks without introducing a classifier. Before a
bulk description regeneration, the unresolved choice is how much source-like
detail/list structure to encourage without over-condensing some cases. The
proposals and their caveats are preserved for that discussion; no production
prompt, template, label, rendered input or coordinate was changed in this pass.

These calls generate only the native lexical regions. They do not constitute
new complete rendered documents or a geometry/training experiment. Existing
coordinate behavior therefore cannot be credited or blamed for this result.

### Measurements and preservation

|Measure|A|B|
|---|---:|---:|
|Raw proposals / mechanical passes|50 /50|50 /50|
|Mean absolute difference from source word count|7.72 words|7.30 words|
|Median successful-call time|8.57s|7.50s|
|Successful calls|25|25|
|All-attempt cost|$0.00722288|$0.00864650|

Mean word difference is a descriptive diagnostic, not an acceptance score.
In particular it must not conceal the55→9/11 catalogue contraction. Latencies
are observed provider times under concurrency2, not proof that B is faster.
B costs19.7% more in this bounded probe, principally because its request is longer.

Total cost is **$0.01586938** for50 successful GLM-5.3-Flash low-reasoning calls.
Six additional HTTP429 rejections cost zero and were preserved, then explicitly
retried with the same requests/settings. Successful-call token totals:
**84,180 input;10,079 output**, including **577 reasoning** and **9,502 visible**
tokens;14,976 cache-read input tokens. Peak process RSS was approximately195.2MiB.
No quality-based reroll or post-generation text repair was used.

Hash verification preserves357 production/dataset files plus the selected source
assets and original receipt pins. All100 proposals were directly read by the root;
the inventory records qualitative caveats rather than certifying them solely from
validator passes. The focused probe suites and training metric tests are recorded
with the final validation results below.

Evidence: [all sources, context and unedited paired proposals](analysis/goods-description-context-probe-20261015/SOURCE_VARIATIONS.md),
[frozen design and exact instructions](analysis/goods-description-context-probe-20261015/design.json),
[manual review inventory](analysis/goods-description-context-probe-20261015/manual-review.json),
[metrics](analysis/goods-description-context-probe-20261015/metrics.json),
[all-attempt receipts](analysis/goods-description-context-probe-20261015/calls),
[probe runner](analysis/goods-description-context-probe-20261015/probe.py).

Final offline validation: **65 tests passed in26.42s**, covering all four probe
suites and the production metric/relation/order tests. The separate training
audit replays every saved native field/category/relation metric without changing
historical gold. This is validation of the probe's controls and measurements,
not a claim that every generated proposal preserves every desirable detail.
