# Next 1,000 synthetic documents: coverage investigation and proposed recipe

**Completed 2026-10-10:** the approved 1,000 samples have been generated,
reviewed, corrected, positioned, independently audited and merged. The new
dataset has 3,100 training / 60 validation records. See the
[completed results and evidence](kie-synthesis-coverage1000-results-2026-10-10.md).
No training was launched. The remainder of this file preserves the original
investigation and proposal as historical context.

Original investigation date: 2026-10-09. At that stage there was **no paid
generation, training, dataset change or production-code change**. This plan retains the requested source
allocation and separates settings that are already supported from small pipeline
changes and genuinely absent source topologies.

The performance motivation is the [eight-run results ledger, section 13](mpci-training-results-and-next-experiments.md#13-corrected-description-and-declared-packaging-run-h).
The corrected-description/package run improves exact descriptions and allocation
restraint. Remaining content weaknesses include description omissions and
neighboring-field leakage, notify-address ownership, some route wording, volume
aggregation and incomplete container lists. This proposal does not change gold
annotations or claim a particular future training score.

## 1. Lock the source budget

| Source cohort | Templates | New descendants per template | New documents | Existing synthetic documents | Resulting synthetic documents |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original 100 | 100 | 2 | 200 | 800 | 1,000 |
| Second 100 | 100 | 2 | 200 | 700 | 900 |
| Newest 100 | 100 | 6 | 600 | 0 | 600 |
| Total | 300 | — | 1,000 | 1,500 | 2,500 |

The intended next training set is therefore **600 real + 2,500 synthetic =
3,100 training documents**, with the same 60 real validation documents. The
200 newest-template admission descendants are inspection evidence; they are not
silently counted as additional training records.

The three source selections are disjoint and exactly cover the admitted registry.
All 300 have current training labels, and none overlaps validation by ID or exact
OCR hash. All **1,800 registered case assets** were independently hash-verified
against the current manifest. This repeats the relevant checks rather than
assuming old config capabilities still match repaired registry contracts.

Current source authorities:

- [Original 100 selection](../configs/synthesis/mpci_bl_curated_v7_expansion100.yaml).
- [Second 100 selection](../configs/synthesis/mpci_bl_curated_v7_expansion200.yaml).
- [Newest 100 selection](../configs/synthesis/mpci_bl_curated_v7_template_expansion300.yaml).
- [Current shared registry](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/manifest.json).

The old `mpci_bl_curated_v7_expansion300.yaml` refers to a 300-document campaign,
not the newest 100-template cohort. The recipe must use current registry
capabilities, ownership, auxiliary declarations, cases and the corrected
description-block task constraints, not stale copies from older campaigns.

## 2. What the current configuration can actually control

The distinction is **source topology versus sampled shipment facts**.

| Layer | Existing control | What it does not do |
| --- | --- | --- |
| Template frequency | `template_sampling.samples` and exact `source_counts` | Does not create new layouts or change a source's instructions |
| Instructions | Source negotiability and notify `sameAs`; optional source-frequency margins | Does not rewrite a non-negotiable source into a negotiable one |
| Commodity identity | Pinned registries, cargo family, allowed HS headings/codes and ambient chapter eligibility | Cannot draw from an empty compatible identity pool |
| Package category | Per-source `allowed_package_categories`, eligible physical donors and DG packaging profile | Is not a global class-weight or per-variant quota setting |
| Equipment | Per-source `allowed_equipment_pairs`, eligible donors and capacity contracts | Cannot add absent size/type fields or bypass missing capacity support |
| Quantity/mass/volume | Joint donor bundle, quantity scale, source-owned physical rows, source units and precision | Does not change the container-list cardinality or invent unknown allocations |
| Routes and parties | Country/port/locality registries, source-owned route graph and role-locality assignments | Does not turn a direct-route source into a transshipment layout |
| Wording | Joint source-owned product/postal regions, examples, sampled facts | Currently has no configurable concise/specification/list distribution |
| Geometry | Source anchors, calibrated reflow and joint page transforms | Does not create a source product continuation on a new page |

Use an explicit 300-entry count map: two for every old source and six for every
new source. `variants_per_source: 2` then remains the **request batch limit**,
not the campaign source quota. The newest six descendants are processed in
manageable batches rather than one oversized model request.

`source_counts` cannot be combined with negotiability or `sameAs` marginal
fractions: those would allocate different source frequencies. Under the fixed
quota the instruction totals are exactly **48 negotiable**, **938
non-negotiable**, **14 OCR-unavailable**, and **216 notify-reference documents**.

There are **no current `package_weights`, equipment weights or global cargo-family
weights** in the strict configuration schema. Adding imaginary YAML keys is not
a solution: five falsification checks confirmed incorrect quotas, incomplete
source coverage, conflicting margins and unsupported weight controls are rejected.

## 3. Coverage inherited by the requested 6/2/2 mix

These are source-topology counts for the entire intended 1,000, not claims that
1,000 final rendered documents have already passed review. Overlapping rows must
not be added together.

| Coverage trait | Independent source families | New documents |
| --- | ---: | ---: |
| No containers | 5 | 22 |
| One container | 209 | 682 |
| Multiple containers | 86 | 296 |
| At least eight containers | 6 | 24 |
| Membership-only placements, no per-container quantity | 9 | 26 |
| Multiple approved description regions | 16 | 60 |
| Repeated product-region occurrences | 22 | 64 |
| Split postal regions | 33 | 130 |
| Name and address sharing a source OCR line | 26 | 72 |
| Volume: multiple container rows, no printed shipment total, gold equals row sum | 18 | 84 |
| Volume: container rows plus a separately printed shipment total | 18 | 48 |
| Temperature settings | 12 | 44 |
| Dangerous-goods family | 4 | 12 |
| Genuine cross-page product-description continuation | 0 | 0 |

The source families imply **914 ambient, 36 frozen, eight chilled, 30 ordinary
vehicle, ten chemical DG and two DG-vehicle** new examples. These preserve basic
coverage but do not amount to DG oversampling. Temperature settings remain tied
to compatible cargo/equipment. It is not valid to raise a thermal or DG weight
on an ambient template with no corresponding printed fields.

Volume coverage deserves the explicit distinction above. A repeated shipment
total is not a row-aggregation exercise. Four other multi-row sources have no
public volume target and contribute 12 descendants; they are not counted among
the 132 public-volume examples. The 84 genuine aggregation descendants add to
45 existing synthetic ones, giving **129**, while the 48 row-plus-total examples
add to 114 existing, giving **162**.

## 4. Description style: correct the distribution, not just length

The latest current-label measurements are:

| Cohort | Documents | Median description characters | 90th percentile | At least 200 characters |
| --- | ---: | ---: | ---: | ---: |
| Real training | 600 | 30 | 89 | 11 |
| Existing synthetic training | 1,500 | 477 | 745 | 1,412 |
| Newest-template admission pilot | 200 | 451 | 820 | 191 |

Long lexical descriptions are already abundant. The missing training signal is
more often **which region belongs to the product, where it ends, and how to
recover all product-owned wording**, not the ability to emit a long string.

The current request universally says “Generate a plausible commercial
assortment.” This is a concrete over-expansion pressure even when the source
is one short commodity identity. Other instructions correctly provide source
depth and joint description assembly, but the assortment request still pushes
toward lists and variants.

Recommended wording change, before generation:

> Match the source's presentation: a concise commodity identity, a specification
> passage, or a product list. Represent every sampled identity at that level of
> detail. Generate all product-owned regions jointly and preserve their complete
> meaning.

This is a general semantic instruction, **not a character count, line quota or
requirement to match exact HS-to-product classification**. Embedded capacities
and product specifications remain legitimate. Host-owned totals, declared
package facts, references and thermal/DG settings remain separately rendered.
Keep the already corrected description-block/Marks ownership policy unchanged.

Validate the wording change on concise, specification-rich and genuinely listed
sources before scaling: compare source/generated presentation, completeness,
neighbor contamination and geometry. Descriptive length is a diagnostic, not a
hard rejection threshold. No paid wording experiment was run in this investigation.

## 5. Reachable targeted package coverage

Current datasets still omit several rare canonical package classes from
synthetic labels, despite having valid vocabulary entries. Merely adding 1,000
ordinary draws leaves them absent.

| Canonical category | Current real training documents | Existing synthetic | Unchanged next-campaign reachability | Recommended action |
| --- | ---: | ---: | --- | --- |
| Plain paper bag | 0 | 0 | No eligible donor | Requires a legitimate physical/category profile, not a weight |
| Plastic bag | 1 | 0 | No eligible identity under current ambient scope | Add the tested grain eligibility and deliberately select compatible templates |
| Steel drum | 2 | 0 | Three compatible DG sources; default DG packaging overrides the subtype | Narrow those sources to their supported steel-drum subtype |
| Intermediate bulk container | 7 | 0 | No complete eligible profile | Recover certified private donor measures and establish the compatible commodity profile |
| Unpackaged single unit | 5 | 0 | No eligible admitted source/profile | Admit an appropriate whole-unit source/profile; do not relabel generic packages |
| Multi-wall paper bag | 1 | 0 | No eligible current donor/profile | Establish compatible identity support and source coverage separately |

### Two bounded configuration interventions already demonstrated

**Plastic bags:** the historical donor `88bab156` is HS `100710`, outside the
current ambient chapter list. Narrowly adding grain chapter 10 makes **125
templates** identity/physical-donor eligible while respecting their existing
capability constraints: 36 original, 51 second-cohort and 38 newest.

Four selected templates passed six full host preflights each:
`88bab156` (original, two actual quota slots), plus newest `147a2626`,
`157fadd2` and `1b176673` (six each). Their requested quota therefore gives
**20 deliberately plastic-bag examples**, without changing 6/2/2 source counts.
The native source test additionally pins its source bundle and HS heading 1007.
This is a genuine category/identity eligibility correction, not a guess that
more random draws will eventually produce plastic bags.

The final proposed recipe proved that the native source also works with only
the package-category restriction; extra source-bundle/heading constraints are
not needed. All seven targeted sources retain their other registered capability
fields unchanged.

**Steel drums:** compatible chemical-DG templates `21d3ced7`, `7d44dcc4` and
`fec3f655` passed **18/18** six-variant stress preflights with steel-only package
constraints. Their actual old/old/new quotas give **2 + 2 + 6 = 10** examples.
The underlying DG registry, package compatibility, capacities, UN/class/group
fields and source-owned rendering stay active. These are all ten chemical-DG
slots in the new campaign: existing synthetic DG examples retain the broader
generic package coverage. This within-DG concentration is deliberate and must
be checked during final rendered review.

The 20/10 targets come from the tested source domains and the user's integer
quotas. They are **bounded coverage interventions, not statistically proven
optimal class frequencies**. Existing general package coverage remains in the
1,500 retained synthetic documents and the other new source quotas.

### Full proposed-configuration replay

The [complete analysis-only recipe](analysis/synthesis-next1000-coverage-20261009/sampling/proposed-config.yaml)
was also tested across all 1,000 exact quota slots, not just the selected rare
sources. It adds chapter 10 and narrows the package category on the seven
demonstrated sources. **999 pass, with only the same precision-grid failure
described below; 24 plastic-bag and ten steel-drum plans pass.** Twenty plastic
slots are deliberately constrained; four more occur on other eligible templates.
Other commodity/donor choices can change under the expanded ambient domain.

All inherited instruction, container-count, omission, DG/thermal presence and
allocation-shape counters match the unchanged baseline. The configuration is
therefore targeted without altering those source topologies. It has a new,
uncreated output directory, not the previous pilot directory. Its inherited
**$2 budget belongs to the old 200-sample pilot and is explicitly not validated
for a 1,000-sample launch**. Provider prices, campaign budget and the two remaining
implementation items must be checked when preparing an actual production recipe.

### Why IBC is not another YAML-weight tweak

Source `37c96354` has approved private gross-weight and volume owners: printed
gross weight 21,120 kg and volume 28 m³. Its public target intentionally omits
gross weight. The donor fitter currently reads public goods values only, so the
required private gross-weight fact is not available to donor eligibility.

An isolated HS1518/chapter15 feasibility test still fails all six preflights;
blindly broadening food/chemical chapters or deleting the private-measure
requirement would not fix the data contract. The correct owning-layer extension
is to supply the fitter with **hash-pinned, approved private physical facts and
units**, preserve their private status, and validate their compatibility through
the existing physical renderer. That enhancement and a compatible commodity
profile are separate from this configuration-only investigation.

## 6. Equipment, list completion and absence examples

The ordinary full host plan increases untyped-equipment documents from 44 in
the existing synthetic set by 48, giving **92**. It adds 26 membership-only and
22 no-container documents, giving **77** and **38** respectively. These are
useful counterexamples to inventing equipment attributes or placement quantities.

Container-list cardinality is inherited from the source. The new mix adds
296 multi-container records, including 24 with at least eight containers; the
largest selected list is 13. It does not sample a longer list on a single-container
template. This is actual sequence/ownership coverage, not an equipment-class weight.

Preserving every old template keeps its existing capabilities in the data, but
does not keep all percentages unchanged. Explicit transshipment examples add
12 to the existing 45, giving 57/2,500 synthetic documents. Notify references add
216 to 403, giving 619/2,500. The source-budget choice slightly dilutes these
previously covered proportions; it is not evidence that either skill will regress.

Additional explicit equipment restrictions were host-tested: 40-foot high-cube
refrigerated, 40-foot standard open-top, 20-foot standard open-top and 20-foot
standard tank each passed six constrained variants. These are available options,
but there is no evidence to prioritize them over the demonstrated description,
package and volume errors in this next experiment.

A 40-foot standard refrigerated restriction failed six tests because its
source-supported **capacity contract** is absent. A nonempty donor list does not
prove renderability. Literal equipment aliases are source-style renderings,
not an independently weighted field; increasing category draws alone does not
guarantee a particular carrier-specific alias occurs.

No admitted sources currently cover fallback package text, missing package
quantities, multiple package rows or mixed known/unknown allocation quantities.
These require source/profile support; weights cannot synthesize absent fields
or safely invent their ownership. Preserve the real examples rather than forcing
them into supported canonical categories.

## 7. A precise pre-generation physical failure found and localized

The unchanged proposed seed `202610096` was run through the **actual host
scenario and production preflight** for all 1,000 quota slots. **999 passed;
one failed before any model request.**

Source `30116fcc`, variant 1, has six container-volume rows printed at integer
CBM precision. The selected donor has six pallets, 888 kg and 2.1 m³ total.
The source print lattice rounds that total to 2 m³. **Six positive integer-m³
rows cannot sum to 2 m³.** The renderer correctly rejects this impossible draw.

This is not a reason to change labels, delete the template, invent fractional
precision, set rows to zero, choose a lucky seed or weaken the guard. Package
quantity multiples 2/6/12 still produce the same failure class in two of 20
draws each; restricting to the source bundle fails all 20 because its compatible
identity pool is empty. Those are falsified workarounds.

The clean fix is **candidate-selection feasibility**: check the source's positive
row grids and printed measure precision inside the bounded host sampling loop,
reject the incompatible candidate with an explicit provenance reason, and
continue to an actually renderable donor/quantity bundle. If no candidate is
feasible, fail before paid generation. Re-run all exact planned slots after that
fix. This needs a small production-path change; it has not been implemented
or hidden by this planning pass.

Only diagnosed sample-incompatibility conditions should advance that bounded
candidate stream. Source-contract, ownership, schema and missing-unit errors
must remain hard failures. Regression tests must cover this exact six-row grid,
asymmetric row weights, multiple physical dimensions, no feasible candidate,
true source contradictions and deterministic replay of accepted sample IDs.

## 8. Source topology not provided by the 300-template catalog

All 300 approved description regions were checked, including internal page
markers. **None is a true product-description continuation across pages.**
Two regions on one page and a long generated product list are not substitutes.

Three current **real-training**, non-validation sources have already-audited
cross-page descriptions and are outside the admitted catalog:

- `3bc0bdd0`: two-page XLPE product sentence.
- `345a2a0b`: three-page laboratory-supplies list.
- `21883640`: two-page used-vehicle list.

Their exact identities, source hashes, spans and source links are retained in
the [topology inventory](analysis/synthesis-next1000-coverage-20261009/topology/inventory.json).
They are concrete candidates for a small later source-admission pass, not a
request to rebuild 300 templates. Admitting/reallocating new sources would alter
the current fixed cohort recipe and should be agreed separately. The next 1,000
can improve currently supported boundaries and list/quantity/category coverage,
but must not be advertised as directly covering this absent topology.

## 9. Recommended execution plan and validation gates

1. Preserve the exact **600/200/200 source allocation**, current registries,
   uppercase target normalization, variable rendered casing, input-only
   positional format, declared-package policy and current description ownership.
2. Implement the bounded physical-row feasibility fix and clarify source-faithful
   description presentation. Do not change optimizer/rank/labels in this coverage
   experiment.
3. Configure the demonstrated **20 plastic-bag and ten steel-drum slots**;
   retain general sampling for the remaining sources. Keep country sampling broad
   and party/route dependencies coherent, rather than pinning destinations to Egypt.
4. Run the full exact planned quota through host preflight before any paid call.
   Check source counts, category floors, unique identities, current asset hashes,
   validation exclusions and protected-dataset hashes. Zero late-preflight
   failures is the readiness requirement for the fixed campaign plan.
5. Test the wording refinement on source presentation strata. Review rendered
   OCR and complete targets together, with special attention to source-owned
   product boundaries, category specificity, quantity hierarchy, per-container
   lists, arithmetic, party boundaries and continuation completeness.
6. Generate and audit the full campaign using the existing complete-sample
   review/adjudication gate. Keep unresolved candidates out of publication until
   resolved; count them explicitly rather than replacing evidence with an older pass.
7. Validate actual generated positions after lexical reflow: page bounds,
   line uniqueness, reading order, collisions, column ownership and protected
   anchors; plot stratified complex and rare-category examples. Host planning
   alone is not a geometry audit of text that does not yet exist.
8. Merge only the final accepted 1,000 with the current 1,500 synthetic and
   600 real records. Confirm 3,100/60 train/validation, source attribution,
   explicit instruction fields, casing, schema/constraints, zero sample-ID
   collisions and unchanged validation. Record actual, not intended, distributions.

Measure the next model on all documents and matched parseable content, with
exact description/address correctness, contamination/omission subtypes,
package-category specificity, allocation FP/FN, exact container graphs and
volume-total versus row-selection errors. Keep the full-field metrics primary.
The unchanged evaluation set and the retained existing training records make
the next change interpretable; more data still does not guarantee a particular
metric improvement.

## 10. Evidence and measured diagnostic work

- [Host-plan and current-data distributions](analysis/synthesis-next1000-coverage-20261009/sampling/probe-results.json).
- [Targeted category/equipment tests and physical failure reproduction](analysis/synthesis-next1000-coverage-20261009/sampling/targeted-results.json).
- [Full configured coverage replay](analysis/synthesis-next1000-coverage-20261009/sampling/coverage-plan-results.json).
- [Topology, lexical-style and volume audit](analysis/synthesis-next1000-coverage-20261009/topology/REPORT.md).
- [Independent inventory and quota validation](analysis/synthesis-next1000-coverage-20261009/inventory-verification.json).

The ordinary 1,000-slot host probe took **26.22 seconds** including inventories
and targeted eligibility checks; the final configured replay took **26.59 seconds**.
Independent source/quota/hash verification took approximately **four seconds**,
with approximately **821 MiB peak RSS** (the existing
provider/training imports dominate the analysis process). The exact quota module's
tests pass **19/19 in 1.38 seconds**. Five deliberately invalid controls were
rejected. There were **zero paid requests and $0 API cost**. Production files,
registry assets and training/validation data remain unchanged by this investigation.

This is an actionable pre-generation plan, **not a launch-ready certification**:
the localized physical-selection fix and source-faithful wording refinement
still need implementation and their specified validation before generation.
