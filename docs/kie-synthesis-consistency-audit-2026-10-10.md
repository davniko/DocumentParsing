# Synthesis consistency and repair-durability audit — 2026-10-10

## Outcome

The approved targeted repairs are complete. **One synthetic document now labels
its explicitly printed 24,800 kg shipment net total**, in both mirrored dataset
files. The two future-synthesis gaps are fixed in reusable source contracts,
sampling and render preflight. **541 relevant tests pass**, including rejection
controls, and **1,200 fresh plans across all 300 sources pass**.

The original audit flagged four descriptions. The subsequent user policy
clarification distinguishes extraction correctness from physical plausibility:
the two 50 kg bag-fill examples and the yarn cone-fill example have no confirmed
input/target inconsistency, so **those three examples remain unchanged**. They
are not classified here as three additional dataset repairs still needed.

The repaired document's OCR, description and coordinates are unchanged. Exact
reversal of the one-field edit restores the original hashes of both complete
dataset files, proving no unrelated row edits. All **3,160 current targets** pass
the extraction schema. Original campaign candidates/reviews remain historical
evidence; the dataset records a separate target-only repair receipt.

The broader net-wording screen also identified three older real sources and six
descendants with NET/NW declarations whose units depend on context. The follow-up
in section 11 resolves all nine under the existing unit-ownership policy:
**keep net weight absent because no net-owned or shared-heading unit is
established**. They are not nine confirmed missing-label defects or nine
outstanding repairs.

**No training process was stopped, restarted or modified. No subagents, provider
calls, model loading or GPU work were used. API cost: $0.** The already-loaded
training snapshot is unaffected by the in-place file correction; subsequent
loads use the corrected label and updated config hash.

## 1. Exact population and evidence

| Population | Audited scope |
| --- | ---: |
| Current training | 3,100: 600 real + 2,500 synthetic |
| Current validation | 60 real |
| Shared admitted templates | 300 |
| Newest template cohort | 100 sources; 600 descendants in the latest campaign |
| Latest synthesis campaign | 1,000: original100 × 2 + second100 × 2 + newest100 × 6 |
| Inherited synthetic | 1,500: original100 contributes 800, second100 contributes 700 |
| Newest admission pilot | Separate 200 examples, not included as extra training rows |

Active dataset:
[real600-synthetic2500](../data/curated/mpci-bl-real600-synthetic2500-v7-positions-v1/README.md).
Campaign:
[coverage1000 config](../configs/synthesis/mpci_bl_curated_v7_coverage1000.yaml).
Registry:
[shared template registry](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/README.md).

Pre-repair audit snapshot SHA-256 (retained for exact reversal checks):
`66a3b60f5595fe2645f3555b907d7fe8504fe49fda7d6264a204df29a5796131`.
Synthetic:
`6c4941f24892d5e9d191c889e0a853e973069d82230423d6724ff2f133570643`.
Validation:
`9f3b4c36fe5d6310b3fb79a6d380a06d24d2fe8583acf80748d83861b2fe94da`.

Fresh receipts and executable probes are collected in
[the audit directory](analysis/synthesis-consistency-sweep-20261010/README.md).
The dataset manifest's **2,543 file/dependency hash pins** were also verified.

## 2. Checks that passed

### Latest 1,000: publication, rendering, reviews and ownership

The full production publication validator was rerun read-only, including:

- Replaying every current candidate from source, sampled scenario, physical
  facts, wording and contact receipts.
- Current schema and casing; exact cohort/sample identities and quotas.
- Independently reconstructing byte edits, verifying preimages and preserved
  unowned bytes, and projecting approved goods-description regions.
- Host physical row conservation and scoped current contact/equipment overlays.
- Binding every final review to current candidate hashes and current review
  instructions/schema. All 300 source review receipts remain current; the 134
  retained review findings have explicit hash-bound adjudications.
- Comparing published positioned/plain records with candidates and inherited
  records with their approved repair receipts.

Result: **1,000/1,000 mechanical publication checks passed**, zero replay
failures. This describes the pre-repair campaign snapshot. The deeper sweep
below found a missing net-weight target that this mechanical replay could not
detect; section 10 records its repair. The other three wording flags were
subsequently accepted under the extraction-training objective.

Receipt:
[current-publication-audit.json](analysis/synthesis-consistency-sweep-20261010/current-publication-audit.json).
Runtime: **47.917 s**, peak process RSS **979.988 MiB**.

### All current records and shared templates

- **3,160/3,160** targets satisfy the current schema/canonical casing contract.
- Raw and positioned input hashes agree with text; enrichment preserves every
  line's original OCR body, blank lines and page markers.
- All positional suffixes have the expected syntax and normalized range.
- Every record has one accounting goods item with a nonempty description.
- Allocation identifiers refer to retained containers.
- No short-phone fragments were found by the fewer-than-five-digits screen.
- A literal placeholder screen found no `TBD`, `INSERT HERE` or previous
  `company-<hex>.example` placeholder targets; 8,216 contact scalar values were
  included in the population scan. This is a formatting screen, not a claim
  that fictional domains or phone numbers identify real organizations.
- **1,800/1,800** case asset hashes and all four shared-file hashes agree with
  the registry; all **300/300** sources load through current production
  blueprint preparation.

The population contains 321,115 nonblank/non-page-marker lines; 270,633 have
coordinates (**84.28%**). Missing coordinates are explicit ` ||` suffixes.

### Inherited repairs were not lost

Comparing old generation candidates directly with current training labels
initially flags historic differences. Those must be resolved through the actual
repair chain, not treated as corruption or silently dismissed.

The independent reconciliation reconstructed **all 1,500 inherited targets and
inputs** from the saved description-policy stage, final source-description
decisions, package proposals and exact subsequent input-only edits. All passed.
It also independently re-extracted **all 660 real descriptions** from their
adjudicated OCR intervals; all match current labels.

Historical differences explained by this chain:

- 130 candidate/target discrepancies involve description and/or package policy
  repairs. They contain 102 description-field differences and 29 package-count,
  package-category and allocation-count corrections, with overlap.
- 24 duplicated candidate IDs occur in both a pilot and the actual production
  campaign. They are not duplicate training examples; production authority is
  recorded by campaign manifests.
- 20 saved-candidate/current-input differences reflect later description/input
  recovery or external-reference suffix repairs.
- Three old edits cannot be projected through today's stricter block endpoints
  directly; the later reviewed description receipts supply their explicit
  interval authority and match actual OCR and labels.
- The newest mixture additionally contains precisely the already-authorized
  **28 contact-occurrence + 3 equipment-surface input repairs**. Their labels and
  coordinates are unchanged. The other 1,469 inherited synthetic rows are
  byte-identical to the prior dataset.

Receipt:
[historical-reconciliation.json](analysis/synthesis-consistency-sweep-20261010/historical-reconciliation.json).

### Real source limitations are still explicit, not forced to balance

A blanket rule that every set of known container allocations sums to the
shipment total would wrongly flag seven real documents. Their source OCR retains
only part of the identifiable container rows. Examples:

- `bbf194b6`: total 45 packages, retained identified portions 14 + 6; another 25
  packages have no safely retained container ownership.
- `11a4fc7d`: total 70 pallets, one retained container and its 35-pallet portion.
- `ef438af5`: ten containers/1,000 packages declared, seven identifiable
  100-package rows retained.
- `05fd9e37`: 40 × 990 bags, but one OCR container ID is invalid; retain 39
  identifiers and their portions, not an invented corrected ID.

Two other real documents have known portions but no safe shipment total:
`6ce54cdf` has an incomplete octabin table; `67d49789` has 143 in product wording
versus a 144 declaration and 48 + 48 + 48. These are the **same previously
documented source limitations**, verified against unchanged OCR and unchanged
rows. They are not newly broken synthesis allocations.

Exact reasons, OCR and PDF links are retained in
[source-limitations.json](analysis/package-accounting-policy-audit-20261009/source-limitations.json)
and this audit's reconciliation receipt. The screen's nine initial flags were
explicitly reconciled, rather than using an unconditional fallback.

## 3. Initial wording flags and subsequent policy adjudication

The sweep searched all 2,500 synthetic descriptions for mass/fill language,
then inspected all ten flagged records against their package labels, physical
receipts, source product ownership and current instructions.

Six inherited examples have plausible packing details without a confirmed
arithmetic contradiction. For example, one declares 3,046 bags and net
76,150 kg, consistent with its printed 25 kg bags. A 155-carton description has
60 + 50 + 45 subcounts, still one accounting goods group. Those were not removed
merely because they contain packing or numbers; that would contradict our real
description policy. This check is not a new claim that every historical packing
detail was separately sampled by the host.

**Four latest-campaign examples crossed the original host/wording boundary.**
Their source product regions contain no such fill facts, their host physical
context supplies none, and their final reviews reported no corresponding flag.
The table preserves the original audit findings, not the final disposition of
all four examples.

| Sample | Rendered claim | Actual supplied facts | Finding |
| --- | --- | --- | --- |
| [a4ebe186](../artifacts/kie-synthesis-production/curated-v7-coverage1000-v1/candidates/syn_full_v7_a4ebe1862894e91eccb81d7b.json) | `IN 50 KILOGRAM PLASTIC BAGS` | 616 plastic bags; one 20-foot standard GP; no supplied mass/fill | Implied net 30,800 kg exceeds the pipeline's 28,300 kg payload limit by 2,500 kg, before packaging mass. |
| [fe433cf6](../artifacts/kie-synthesis-production/curated-v7-coverage1000-v1/candidates/syn_full_v7_fe433cf610ecab1993925d6c.json) | `IN 50 KILOGRAM PLASTIC BAGS` | 568 plastic bags; one 20-foot standard GP; no supplied mass/fill | Implied net 28,400 kg exceeds that limit by 100 kg, before packaging mass. |
| [6dfa8bf5](../artifacts/kie-synthesis-production/curated-v7-coverage1000-v1/candidates/syn_full_v7_6dfa8bf5eb0e442abfbb17c7.json) | `496 PP BAGS OF 50 KG NET EACH, TOTAL 24,800 KG NET, LOADED IN ONE 20FT ...` | 496 plastic bags; one 20-foot standard GP; no supplied mass/fill | Invented unit fill and shipment net total. OCR explicitly gives net 24,800 kg, but `netWeight` is absent from the label. |
| [35fb134f](../artifacts/kie-synthesis-production/curated-v7-coverage1000-v1/candidates/syn_full_v7_35fb134fab20f969b5070c67.json) | Three yarn variants `on 1.5 kg paper cones` | 538 cartons, gross 27,695 kg; no supplied cone count/fill | Unsupported inner-package fill, though no overload or public total mismatch is established. |

The first three descend from newest source
[`1b176673`](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_1b176673edbe31b74ba03995b85ee9be6d0d22218f2764df4648462f7cfd15fa/source.txt);
the fourth from second-cohort source
[`92bc5451`](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_92bc5451eb186398e16b01e3f2a183c2c67205e3fd3f87cec9e6ee70d3266330/source.txt).
They are synthetic JSONL rows **1949, 1950, 1953 and 1826**, respectively.

Scope: **4/1,000 latest samples (0.4%)**, **4/2,500 current synthetic (0.16%)**,
two families. Three are in the newest 600 descendants. These are not evidence
that the whole current dataset is malformed.

Final disposition under the user's clarified objective:

- `a4ebe186` and `fe433cf6`: leave unchanged. The calculations concern physical
  payload plausibility, not a mismatch between printed extraction values and
  their targets. No shipment net total is printed; do not manufacture a
  structured net total by multiplying a bag count by its capacity.
- `35fb134f`: leave unchanged. Cartons are not cones, so the printed 1.5 kg cone
  capacity establishes neither a shipment-total contradiction nor a missing
  shipment net-weight label.
- `6dfa8bf5`: add `netWeight = {unit: kilogram, value: 24800.0}`. Here the OCR
  explicitly states the shipment total, regardless of whether it originated in
  host sampling or lexical generation. Preserve the OCR and description.

Thus **one actual input/target omission was repaired**, not four descriptions
rewritten. This source template has no net-weight target or owner, and other
descendants do not consistently print a shipment net total. A one-off target
repair is appropriate; adding net weight to every descendant would invent
labels unsupported by their OCR.

### Why these passed

The generator's instruction and product field requirement already say that
packing, unit fill, totals and transport claims belong to supplied host facts.
For source `1b176673`, the actual request explicitly provides:

```yaml
goods:
- numberAndTypeOfPackages:
  - packageQuantity: 616
    typeCategory: PACKAGE_BAG_PLASTIC
equipment:
- sizeCategory: TWENTY_FOOT_STANDARD_HEIGHT
  typeCategory: GENERAL_PURPOSE
printed_measure_totals: {}
```

Therefore the missing 50 kg is not hidden in another supplied slot. This is not
an ambiguous instruction authorizing the model to choose package fill.

The deterministic guard in
[curated_wording.py](../src/document_ocr/synthesis/curated_wording.py#L539)
covers forms such as `25 KG PER BAG` and `BAGS 25 KG`, but not all modifiers and
word order: `50 KILOGRAM PLASTIC BAGS`, `1.5 KG PAPER CONES`, or `BAGS OF 50 KG
NET EACH`. The standalone accounting detector also misses this `TOTAL ... KG
NET` ordering. The physical validator sees only the sampled/private receipt;
it does not know the new mass claim introduced by lexical generation.

Under the original generation contract, the final pair review did not flag
these added claims. The actual
[`1b176673` review](../artifacts/kie-synthesis-production/curated-v7-coverage1000-v1/reviews/doc_1b176673edbe31b74ba03995b85ee9be6d0d22218f2764df4648462f7cfd15fa.json)
has `findings: []`, including all three affected current descendants. Two other
descendants of this same family had comparable added fill/accounting removed in
the previous repair. That correction was not generalized to all variants.

All description targets match their printed description regions. This is
sufficient for the three retained packing details under the clarified training
objective, but not for structured-field completeness: the explicit shipment net
total in `6dfa8bf5` also requires its separate net-weight target.

Receipt:
[packing-wording-adjudication.json](analysis/synthesis-consistency-sweep-20261010/packing-wording-adjudication.json).

## 4. Did recent fixes reach the reusable layers?

### Durable ownership/code fixes confirmed

| Repair class | Persistence checked | Earlier sample exposure |
| --- | --- | --- |
| Repeated allocation package nouns | Shared ownership binds each occurrence to the declared package category | Latest repaired family is newest; no inherited descendants |
| Stale CIF destination and exporter geography | Shared auxiliary recipes follow explicit sampled route/party ownership | These newest families have no old-1,500 descendants |
| Duplicate standalone locality/country | Exact redundant occurrence is owned/deleted; primary postal region remains | Newest family; no inherited descendants |
| Independently printed equipment receipt | Typed dependency owns the receipt; it no longer stays at the old size | Newest family plus separately repaired three inherited equipment rows |
| Spelled package total/repeated goods products | Every repeated noun/product occurrence is bound, not merely first-match replaced | Newest families; no inherited descendants |
| Contact-name repetition | Per-occurrence component recipes preserve combined target identity without printing it into every slot | All 28 affected inherited inputs in four families have exact applied overlays |
| Postal country completeness | Production correction and post-checks cover complete address and separately certified country owner | Code-level guard, not a one-off target edit |
| Measurement precision and proportional rows | Typed admission and common printed quantum/conservation logic | Covered by current replay, fresh plans and tests |

The seven late declarative template repairs cover 42 newest-cohort descendants;
all were replayed against registered assets. Their zero old-1,500 exposure was
verified by exact source ID, not inferred from sample names. The earlier six
newest-first50 ownership repairs likewise persist in shared registry files.
The 31 inherited input overlays remain present in the active mixture.

### Gap A: one private-measure requirement is campaign-only

**Historical finding, now fixed:** the registry and reusable expansion config
carry the requirement, and production source preparation rejects an omitted
requirement before sampling. The text and table below record the initial probe.

Source
[`5c2fdc31`](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_5c2fdc31e60e1b85600907b20b5303554de0aa6293feb162ff71c2c9f4eea90f/source.txt)
has newly certified positive private gross-weight and volume owners. The active
campaign requires complete donor support for both, but the registry's stored
`capability` does **not** carry `required_private_measures` for this source.

Comparing normalized capabilities across all 300 sources yields eight
differences. Seven are intentional campaign-specific bag/drum coverage
restrictions; this eighth is a structural requirement, not a distribution knob.

In-memory probes used the existing seed plus four fresh seeds, six variants
each, changing only this capability:

| Requirement source | Passed | Failed |
| --- | ---: | ---: |
| Current campaign capability | 30 | 0 |
| Stored registry capability alone | 13 | 17 |

The failure is explicit and occurs before generation:
`different-product donor lacks required private grossWeight; declare the source's
required_private_measures before sampling`.

Thus the current six examples are valid, and the runtime is fail-closed; but a
future config built from admitted registry capabilities can fail despite using
an admitted template. The durable fix belongs in the registry capability, plus
source-readiness checking that required positive private owners are covered by
the effective sampling contract. It should not be rediscovered per campaign.

### Gap B: mixed-equipment capacity uses donor bounds, not the printed mix

**Historical finding, now fixed:** a source-hash-bound equipment-mix contract is
in shared ownership and registry capability data. The sampler checks joint
feasibility and render preflight checks it again after rounding. Neither check
creates public size/type labels. The fixture below now fails as required.

Source
[`98f0e0e0`](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_98f0e0e0461007b238fa574b587e2dcc4614b2f72e3724b360bc3aa345049627/source.txt)
prints **1 × 20 DR + 3 × 40 HC**, but its four retained container identifiers
have no individually established size/type labels. The sampler applies the
first donor container's capacity across rows when those public categories are
missing; it does not constrain the fixed printed mixed multiset.

Independent feasibility enumeration checked all four possible assignments of
the 20-foot unit, jointly considering row weight and volume:

- **9/9 current descendants** are feasible: seven inherited plus two new.
- **100/100 fresh seeded plans** are feasible.
- An adversarial, in-memory physical fixture with **four equal 40 m³ loads**
  passes current production preflight, but cannot fit this source's printed mix:
  whichever row is the 20-foot unit exceeds its 33.2 m³ bound. Every row fits
  the donor's 76.4 m³ HC bound, and aggregate 160 m³ is below the mixed aggregate
  ceiling, so an aggregate-only check would also miss it.

The fixture is a falsification control, **not** a generated/training sample or
an observed failure among the 100 seeded draws.

This is a proven admission gap, not a claim that current nine samples are bad.
A private mixed-equipment certificate and row-to-multiset feasibility gate can
close it without inventing size/type labels or guessing an identifier's type.
Code path:
[curated_scenarios.py](../src/document_ocr/synthesis/curated_scenarios.py#L1390).

## 5. Falsification: manual wording repairs are not all durable guards

The audit replayed **34 exact before/after wording edits** from three recent
repair ledgers against current requests, wording validation and postal-owner
checks, without model calls.

| Replay | Rejected mechanically | Allowed mechanically |
| --- | ---: | ---: |
| Archived faulty before-text | 4 | 30 |
| Repaired after-text | 0 | 34 |

The four rejected before-values are two leaked requested field keys and two
`TBD` values. Many other previously fixed clauses remain mechanically allowed:
unsupported inner counts/fill, logistic totals, incompatible lexical packing,
copied neighboring commodity wording, foreign source-city residue, and product
coherence/typos.

This is a guard test, not a claim of 30 still-bad samples: those archived
values were repaired before publication. It establishes that a **manual
wording repair does not necessarily become a recurrence-preventing rule**.
These results describe guard coverage under the original host/wording contract,
not 30 current dataset errors. Following the policy clarification, unsupported
fill language alone is not sufficient to diagnose an extraction defect. The
explicit-but-unlabeled shipment net total is the actionable completeness error;
the three accepted examples do not justify a broader arithmetic-cleanup pass.

Receipt:
[recurrence-probes.json](analysis/synthesis-consistency-sweep-20261010/recurrence-probes.json).

Across all 300 sources, a separate sweep of four fresh seeds produced
**1,200/1,200 successful host-only plans and preflights**. This exercises
sampling/render preparation, not new LLM text. Its success therefore cannot
invalidate the lexical or alternate-configuration failures above.

## 6. Positional synthesis

Source audit: **300 sources, 29,227 content lines, 25,214 measured coordinates**.
It checks 1,303 repeated-text groups against measured region ownership; there
are **zero unproved reuses** of one measured region for different OCR occurrences.

Latest 1,000 coordinate coverage:

| Line class | Lines | Positioned | Coverage |
| --- | ---: | ---: | ---: |
| Goods description | 2,865 | 2,174 | 75.88% |
| Party address | 9,191 | 6,079 | 66.14% |
| Other | 87,214 | 76,128 | 87.29% |
| Overall | 99,270 | 84,381 | 85.00% |

Page receipts: 808 accepted reflows, 1,081 pages with no eligible reflow, and
three declined reflows. Declined pages preserve the proven baseline; they are
not partially transformed or silently assigned fabricated anchors.

All **1,500 inherited geometry receipts** were independently checked against
source boxes and actual current positioned rows. The 39 old receipt text hashes
affected by documented input-only changes were rebound only in memory, after
exact input-chain and unchanged-coordinate-suffix verification. No coordinates,
anchors, transforms or layout nodes were altered for validation.

Falsification rejects text mutation, off-page coordinates, duplicate emitted
lines and inconsistent anchors. No current coordinate defect was found.
Receipts:
[latest geometry/publication](analysis/synthesis-consistency-sweep-20261010/current-publication-audit.json),
[inherited geometry](analysis/synthesis-consistency-sweep-20261010/historical-geometry.json).
Inherited geometry runtime: **23.847 s**.

## 7. Target policy notes

The current schema/casing checks and description replay show that the earlier
main-product-block, Marks exclusion and declared-package-unit repairs are still
applied. They were not reverted when the newest cohort was joined.

Negotiability is present in every target. Current train counts: 181 negotiable,
2,868 non-negotiable, 51 null. Validation: seven, 51 and two respectively.
Across the complete population, the 53 nulls comprise ten real records and 43
synthetic descendants; all have no labeled consignee. That is consistent with
the agreed OCR-unavailable policy, not 53 missing required fields. This audit
does not infer negotiability from document titles or transfer a notify occurrence
into an unprinted consignee block.

The packing findings do **not** justify deleting embedded capacities from real
or synthetic goods-description labels. Preserve the agreed product passage.
Extraction labels follow the rendered OCR: an explicit shipment total needs its
structured target even if lexical generation introduced it. A per-package
capacity alone does not authorize calculating an unprinted shipment total.
Physical-capacity plausibility and extraction correctness are separate checks;
the accepted fill examples are not being rewritten solely for arithmetic or
container-payload realism.

## 8. Approved repair scope

No wholesale reannotation or full regeneration is indicated by this sweep.

1. Add the explicitly printed 24,800 kg net-weight target to the single affected
   sample, in place, with a recoverable before-row and exact-change receipt.
2. Leave the three accepted packing-description examples, their inputs and
   labels unchanged. Do not add inferred totals or a blanket fill-removal rule.
3. Persist `5c2fdc31`'s private gross-weight/volume requirements and enforce
   source readiness independently of campaign coverage overrides.
4. Certify `98f0e0e0`'s fixed printed equipment mix and reject infeasible row
   loads before wording generation and after physical rounding, without
   guessing public container categories.
5. Screen the full current train/validation population for other missing net
   totals. Discuss older context-dependent declarations separately rather than
   silently filling units or treating every NET token as a shipment weight.

These items were implemented/checked as recorded in section 10. The earlier
proposal to rewrite all four descriptions and add a broad lexical arithmetic
guard was superseded by the user's policy clarification and is not implemented.

## 9. Original read-only audit execution record

- Existing curated production and prior campaign-quality tests:
  **561 passed in 37.25 s**.
- New independent mixed-equipment checker tests:
  **7 passed in 10.20 s**; include exact-boundary acceptance, assignment
  permutations, no-fitting-row rejection, volume and payload failures and
  incomplete-layout rejection.
- Audit helper Ruff checks: passed.
- Population/fresh-seed sweep: **34.397 s**, peak RSS **1,017.785 MiB**.
- Targeted requirement/mixed-capacity/wording probes: **10.917 s**, peak RSS
  **820.742 MiB**.
- All protected dataset/registry files are unchanged. No throughput change to
  production is claimed because no production implementation changed.

The initial population receipt deliberately retains its flagged historical
differences. The reconciliation receipt supplies the exact explanation and
reconstruction, rather than overwriting that diagnostic history or labeling a
failed check successful without evidence. Read those historical receipts together
with section 3's final dispositions and section 10's current repair validation.

## 10. Completed targeted repairs and validation

### Dataset correction and preservation

Only `documentPatch.goodsItemDetails[0].netWeight` changed for
`syn_full_v7_6dfa8bf5eb0e442abfbb17c7`: `{unit: kilogram, value: 24800.0}`.
The change is mirrored at train row 2553 and synthetic row 1953. OCR, description,
coordinates, all other targets and validation are unchanged.

- Original full row:
  [net-weight-label-before.json](../data/curated/mpci-bl-real600-synthetic2500-v7-positions-v1/net-weight-label-before.json).
- Exact before/after authority:
  [net-weight-label-repair.json](../data/curated/mpci-bl-real600-synthetic2500-v7-positions-v1/net-weight-label-repair.json).
- Dataset manifest and next-launch config hashes are updated. Registry dependency
  changes are recorded separately as maintenance, not misrepresented as new
  semantic reviews of historical candidates.
- Reversing just the target edit recreates both original complete-file hashes.
  This verifies **one document, two mirrors, zero unrelated changes**, including
  preservation of the three examples accepted as non-issues.
- The running trainer's already-loaded snapshot is not changed. It will finish
  on the snapshot it loaded; later launches use the corrected dataset files.

### Private-measure prerequisite

`5c2fdc31` now has `required_private_measures: [grossWeight, volume]` in the
shared registry and reusable expansion config. Production source preparation
derives required positive private owners from the certified blueprint and
rejects a capability that does not cover them. Existing same-source
`source_bundle` / `source_whole_units` profiles retain their source's own numeric
bundle; they are tested explicitly and are not treated as different-product
donor profiles.

The original registry-only seeds now pass **30/30**, versus **13/30** before.
Removing the requirement is a negative control and is rejected at readiness,
before generation, with the missing dimensions named in the error.

### Mixed-equipment feasibility

`98f0e0e0`'s ownership contract records its exact source hash and unique printed
`1X20 DR / 3X40 HC` evidence, canonical equipment counts and no identifier
assignment. The registry carries this contract; production campaigns inherit
it from ownership and reject contradictory overrides. Counts must cover exactly
the retained identifiers. Public types remain absent.

The new `curated_equipment.py` matcher checks payload and volume jointly using
existing configured capacity limits and exact decimals. It finds a feasible
row-to-equipment matching, rather than using a donor's first equipment type or
an aggregate capacity bound. The witness is private feasibility metadata,
**not a classification of a container identifier**. Both sampling and actual
rounded render preflight enforce it.

- **9/9 existing descendants** remain feasible, agreeing with the independent
  original enumeration.
- **100/100 fresh plans** pass sampling and rounded preflight.
- Four 40 m³ loads are rejected; aggregate capacity alone would have allowed
  them. Tests also cover joint payload/volume constraints, boundaries,
  rounding-induced failure and valid reassignment after rounding.
- The matcher agrees with exhaustive enumeration on **500 random four-row
  controls**. Invalid source hashes/evidence/counts and invalid numeric inputs
  fail explicitly.

### Full checks, performance and cost

Receipt:
[repair-validation.json](analysis/synthesis-consistency-sweep-20261010/repair-validation.json).
Executable reproduction:
[repair_validation.py](analysis/synthesis-consistency-sweep-20261010/repair_validation.py).

| Check | Result |
| --- | --- |
| Strict extraction-schema validation | 3,160/3,160 current targets |
| Registry case assets and shared-file pins | 1,800 assets + all four shared files |
| Dataset file/dependency and training-config hashes | All verified |
| Fresh source plans/preflights | 1,200/1,200 across all 300 sources |
| Full rendered replay using saved wording/contact receipts | 4/4, two variants per affected source; identical OCR and public targets |
| Relevant curated + transport-capacity tests | 541 passed in 26.08 s |
| Ruff checks on changed code/tests/audit helper | Passed |
| Provider calls / API cost | 0 / $0 |

The offline production-path audit took **35.84 s**, peak RSS **983.36 MiB**,
including catalog/framework imports and the full population check. Its 1,200
plan/preflight loop took **3.83 s**. Median standalone matching overhead for
four containers was **6.76 microseconds**; no model call, new hot-path I/O or GPU
work is added. The pre-change narrow suite passed 112 tests in 10.27 s; the same
selection initially passed in 9.49 s after the production changes. Those test
timings are regression smoke measurements, not a claim of training speedup.

This uses the actual sampler, physical preflight and full render path with saved
model receipts. It is not represented as a fresh paid wording-generation run;
reusing wording isolates the two host-side changes without regenerating good
text or introducing unrelated lexical variation.

### Initial net-weight unit/context questions (resolved in section 11)

The broad population screen initially flagged **30 train + two validation
records** without a net-weight target. After the approved repair, **31 lexical
flags remain**. NET/NW words are not equivalent to missing shipment weights:
false positives include `.net` contacts, northwest addresses, model codes,
litre capacities, lengths and per-package fill weights.

No second generated, explicit shipment net total with its own mass unit was
identified among those flagged records. However, three older real sources and
six synthetic descendants print shipment NET/NW declarations without a unit
on that declaration. Their surrounding unit context differs:

| Source | Printed evidence | Scope / needed decision |
| --- | --- | --- |
| [`48325f63`](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_48325f63ba3ebeea284baa1b9c48eabee4df716417bb04dd2f7884c4fc88cf98/ocr.txt) | `NW: 17,600.000`; also `17.60 TONS` and a separate `19212,000 KGS` container gross row | One real record. Strong supporting mass context; review whether that context authorizes the NW unit rather than treating it as explicitly unitless everywhere. |
| [`3c2a9bfe`](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_3c2a9bfe15745f9dde9d84d60a99ce43e1f83b43de24f25129fb310594662412/ocr.txt) | Two rows with `G.W.27500.000 N.W.26500.000`; 208 and 174 slabs | One real record. No explicit mass unit found in OCR; a numeric magnitude alone cannot establish kilograms. |
| [`635b24d9`](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_635b24d9c8f04747ddd8f7ddea82423ce5a75ed8e714c6b35500b8d8a26d50f9/ocr.txt) | `NET WEIGHT:20553.30`; a separate container gross row has `20686.20 KGS` | One real + six synthetic records. Decide consistently when a separately printed gross-weight unit governs the shipment net field. |

These are not the repaired sample's newly generated net-total mechanism, and
their private numeric sampling does not itself authorize a public unit label.
The initial audit left them unchanged for unit/context review. Section 11
records the completed follow-up against the existing field description, not
a new unit-inference rule. Do not infer units solely from typical shipping
practice.

## 11. Net/NW follow-up: nine records resolved without label edits

The follow-up inspected the complete OCR for the three real sources, all six
current descendants, the active extraction schema and the admitted template's
numeric/unit owners. The governing rule was already explicit in
[`CargoMassV7.unit`](../src/document_ocr/label_schemas/bill_of_lading_v7.py#L246):

> A unit in a shared column heading can apply to its rows; a unit on a different
> measure does not transfer.

The same description says to preserve the printed magnitude/unit, never guess
an absent unit and not convert tonnes into kilograms. Both value and unit are
required inside a populated mass object; the enclosing `netWeight` is optional.
Therefore a numeric NET/NW declaration alone does not require a populated
net-weight target.

| Source | Final decision under existing policy |
| --- | --- |
| `48325f63` | Keep net absent. `NW: 17,600.000` has no unit; `KGS` qualifies the separately printed container gross mass. `17.60 TONS` is a separate product/loading mass without an explicit net designation. Linking that amount to NW through a presumed tonne-to-kilogram equivalence is not the current extraction rule. |
| `3c2a9bfe` | Keep both net and gross absent. The two GW/NW rows contain no printed mass unit anywhere in this OCR. Complete numeric rows do not authorize inventing kilograms before summing them. |
| `635b24d9` and all six current descendants | Keep net absent. `NET WEIGHT:...` is unitless; the only mass-unit occurrence belongs to a distinct container gross-weight row, not a shared mass heading. Retain the valid gross-weight targets. |

This closes **three sources / nine current records for this net-unit question**.
It is not whole-document recertification, and it does not reclassify the earlier
24,800 kg omission: that sample explicitly prints `KG NET`, so its added target
remains correct. The initial broad lexical flags were candidate questions, not
established label defects; the prior statement that all nine still needed a new
policy decision was unnecessarily tentative given the existing field rule.

### Template and pipeline disposition

The only admitted template of these three sources is `635b24d9`. Its net value
is correctly kept as a private rendered numeric owner, not exposed as a public
mass label. Both registry and campaign capabilities already require private
net-measure donor support. The unit on its separate gross-weight owner remains
independent. There is no justified schema, source, template or runtime code
change for these nine records under the current policy.

Sampling's internal physical units are not evidence that the extraction model
can read those units in OCR. If a future synthesis feature deliberately prints
a net unit, it must own/render that unit and populate the corresponding public
target together; adding only a label would be incorrect. Likewise, transferring
gross units into unitless net fields would be a policy change, not this repair.

### Targeted probes and preservation

Receipt:
[net-unit-policy-adjudication.json](analysis/synthesis-consistency-sweep-20261010/net-unit-policy-adjudication.json).

- Streamed all **3,160 current records**, locating exactly the three real
  sources and six descendants; schema-validated those nine and checked their
  complete OCR mass-unit occurrences against the reviewed contexts.
- Structural controls reject missing/unknown units and accept canonical
  kilograms/metric tonnes. Importantly, schema-valid kilograms alone are not
  proof of OCR ownership; the source-context adjudication supplies that decision.
- **12 fresh plans/preflights** for the admitted family retain its unitless net
  number privately without emitting an invented public net unit.
- **Two full render replays** using saved model receipts reproduce the existing
  OCR and targets exactly, with net weight still absent.
- Hashes of train, synthetic, validation and dataset manifest are unchanged.
  No changes to OCR, descriptions, coordinates, labels, templates or schema;
  no running-training intervention, provider calls or GPU work.
- Population scan: **0.773 s**. API cost: **$0**. No production performance
  change because no production behavior was altered.
