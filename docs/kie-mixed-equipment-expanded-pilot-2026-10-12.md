# Mixed-equipment synthesis: expanded 30-template pilot

## Outcome and scope

The contained experiment expanded from five to **30 source templates**, generating
**three primary descendants per source: two mixed and one uniform**. Another
**90 matched uniform controls** reuse the same accepted wording, parties, routes,
goods, identifiers, packages and allocation facts. This tests equipment variation
without attributing unrelated resampling differences to mixing.

All 90 primary and 90 control records passed final rendering, independent equipment
checks, exact byte replay, hash-bound final-text review/adjudication and pilot
publication. The publications are **inspection datasets only**. This pass does not
modify the current training dataset, activate production mixed sampling, change
the template registry or start training.

The same source supports mixed **and** uniform sampling. No template is marked
mixed-only, and no source label is relabelled as uniform to circumvent a guard.
The tested mixtures are compatible general-purpose equipment sizes/heights, not
arbitrary mixtures of GP, reefer, tank or dangerous-goods transport.

### Inspection entry points

- [Complete source inventory and template links](analysis/mixed-equipment-expanded-pilot-20261012/INVENTORY.md).
- [Original OCR followed by all 90 primary rendered descendants](analysis/mixed-equipment-expanded-pilot-20261012/campaign/samples.md).
- [Position-enriched primary descendants](analysis/mixed-equipment-expanded-pilot-20261012/campaign/positions-reflow-v1/samples.md).
- [Matched uniform-control gallery](analysis/mixed-equipment-expanded-pilot-20261012/control/samples.md).
- [Primary labels and raw text](analysis/mixed-equipment-expanded-pilot-20261012/campaign/dataset.jsonl).
- [Primary positioned inputs and labels](analysis/mixed-equipment-expanded-pilot-20261012/campaign/positions-reflow-v1/dataset.jsonl).
- [Frozen experiment configuration](analysis/mixed-equipment-expanded-pilot-20261012/config.json).
- [Final code, artifact and cost receipt](analysis/mixed-equipment-expanded-pilot-20261012/final-receipt.json).

## 1. Selection, coverage and exclusions

All **360 registered sources** were inventoried. This direct-owner ambient-GP
screen found **45 compatible sources**. Thirty were selected: the five previously
tested mixed sources plus 25 originally uniform sources. All are real-training
sources, and source identity/OCR-hash overlap with validation was checked.

| Scope | Count |
| --- | ---: |
| Selected source templates | 30 |
| Carrier/layout families | 9 |
| Source pages | 64 |
| Distinct source container rows | 93 |
| Primary descendants | 90: 60 mixed, 30 uniform |
| Matched controls | 90 uniform |
| Container-row instances | 279 per arm; 558 overall |
| Two/three/four/five/six-container sources | 13 / 9 / 2 / 4 / 2 |

Coverage includes per-ID dimensional tables, attachments, parallel OCR columns,
slash/reversed aliases, counted singleton rows, whole-shipment and subgroup
summaries, repeated copies and multi-page tables. The sample contains 19 Maersk
sources plus PIL, CMA CGM, Vasco, Turkon, ONE, MLH, Amass and Prism layouts.

Multiple containers alone are **not** sufficient admission evidence. Five sources
that appeared acceptable to the earlier aggregate checker were excluded because
they do not establish which individual ID owns which type. Other exclusions
include one-container sources, non-GP/thermal families, missing direct owners and
unsupported source syntax. These exclusions are recorded, not silently converted
into uniform output. The complete inventory gives each rejection reason.

The primary set exercises five GP categories:

| Canonical size/type | Container instances |
| --- | ---: |
| 20-foot standard GP | 57 |
| 20-foot high-cube GP | 35 |
| 40-foot standard GP | 60 |
| 40-foot high-cube GP | 64 |
| 45-foot high-cube GP | 63 |

There are ten sampled package categories: IBC, skid, carton, crate, paper bag,
coil, jumbo bag, wooden case, plastic drum and multi-wall paper bag. All goods
and packaging are drawn through the current registry-form/load-profile path,
not substituted observed cargo donors. Actual alias/count distributions are in
[matched-audit.json](analysis/mixed-equipment-expanded-pilot-20261012/matched-audit.json).

## 2. How mixed and uniform sampling work

1. Compile the source's exact per-ID equipment owners. A scalar shared across
   several rows imposes equality only on those rows; a counted summary is derived
   from its owned row set, not mistaken for a shared scalar.
2. Choose the draw's mode. This pilot explicitly requests `mixed`, `mixed`,
   `uniform`, with contrasts 20-standard/40-HC and 40-standard/45-HC. Its third
   descendant samples a uniform category. Mode belongs to the draw, not template.
3. For a mixed draw, choose a positive composition and assign complete owner
   groups deterministically from the seed. Both requested categories must occur.
   Assignment is linear-space, not exponential enumeration of all binary mixes.
4. Sample registry goods/load profiles and package counts using the capacity of
   **each assigned container**. Preserve ID-keyed allocations and source visibility.
   Matched controls additionally require those same loads to fit their uniform
   40-HC inventory; that extra constraint is specific to the experimental control.
5. Generate lexical text around the accepted shipment facts. Equipment, quantities,
   accounting weights and allocations remain host-controlled.
6. Render each equipment scalar for its own row, and each subgroup/full summary
   from its actual inventory. Select reviewed aliases independently by canonical
   pair; packages use their own declared-count context.
7. Replay actual bytes, review the complete rendered text and labels, adjudicate
   findings, publish inspection records, then synthesize/audit coordinates.

The source still determines **container cardinality and printed ownership shape**.
This pass changes compatible equipment assignments, not the number of containers
or arbitrary row/table topology.

Concrete example: the same five-container `b60ce9db` template produced
**2 × 20-standard + 3 × 40-HC**, then **1 × 40-standard + 4 × 45-HC**, and then
**5 × 20-HC**. Its IDs, allocations and all repeated summaries agree with each
draw. That is the requested mixed-or-uniform behavior, not a mixed-only template.

## 3. Integration problems diagnosed and tested

### 3.1 Registry-form sampling rejected mixed sources

The existing registry-form path assumes a homogeneous equipment observation.
Simply deleting that guard, or sampling uniform cargo then changing type labels,
would leave load feasibility and replacement state disconnected from the inventory.

The contained `InventoryCatalog` instead compiles ownership **before** sampling
loads and supplies the actual assigned inventory to bounds/allocation admission.
It preserves source labels and reuses current registry/profile/numeric compilers.
Mixed and uniform outputs were exercised for every selected source.

### 3.2 Package-alias projection read the wrong count owner

A historical package alias can carry merged shipment and per-container paths,
while the current curated target expression refers to one specific allocation.
Treating the merged historical paths as the count authority produced false holds
even when rendering and labels agreed.

The prototype derives a frozen count fact from the **current physical plan and
exact target expression**, binds it to the source occurrence and current target,
and validates the resulting alias against that count. Explicit source-owned
presentation recipes remain authoritative; optional alias projection cannot
overwrite them. Missing, contradictory, stale or forged context fails loudly.

Evidence: all 90 expanded plans passed; 393 canonical package-count contexts were
checked. The previous five-family probe also passes; 12 already passing outputs
remain unchanged, and three false holds are resolved. See
[PACKAGE-CONTEXT-AUDIT.md](analysis/mixed-equipment-expanded-pilot-20261012/PACKAGE-CONTEXT-AUDIT.md).

### 3.3 Rejected retries left stale cargo replacements

Independent tests found a genuine mutable-state defect: a rejected draw could
leave a row's replacement behind when the next accepted draw returned that field
to its source value. One three-container plan consequently labelled all rows as
45-HC instead of its accepted inventory of two 45-HC and one 40-standard.

The contained adapter rebuilds cargo replacement arrays from each draw before
validation/acceptance. A regression test deliberately rejects the first draw and
returns equipment **and quantity** to source values on the next. Source data is
not altered to hide the error. Production integration should likewise create
fresh replacement state per attempt rather than reuse conditional mutations.

### 3.4 A whole-shipment header was incorrectly owned by row zero

Source `11da8a8c` prints `02 CONTAINERS OF 20 DRY CONTAINING`. Its historical
`20 DRY` binding also occurs in row zero's type cell. Replacing both occurrences
as one scalar falsely claimed that both containers had the first row's type.

The prototype establishes a distinct **whole-inventory header owner**, requiring
an exact adjoining count owner, explicit whole-container dependency, agreement
with source cardinality and the original uniform type. Row-local occurrences
remain row-local and are reindexed exactly. The source itself is unchanged.

Correct mixed examples now read:

```text
1 x 40' HIGH CUBE GENERAL PURPOSE CONTAINER + 1 x 20DV CONTAINERS CONTAINING
1 x 45' HIGH CUBE GENERAL PURPOSE CONTAINER + 1 x 40GP CONTAINERS CONTAINING
```

The uniform descendant reads `2 x 40HQ CONTAINERS CONTAINING`. The plural noun
repetition in a long-form alias is cosmetic; inventory counts and labels agree.
Independent tests reject a forged all-first-type summary even when hashes and
edit replay are internally consistent. This is a semantic guard, not hash-only
approval. See [SAMPLER-AUDIT.md](analysis/mixed-equipment-expanded-pilot-20261012/SAMPLER-AUDIT.md).

### 3.5 Replay metadata used a tuple before serialization and a list afterward

Package count receipts initially returned tuple-valued `target_paths`. JSON
stores lists, so faithful saved receipts failed fresh equality/replay. No paid
review calls ran while this mismatch blocked replay. Receipt serialization now
emits lists explicitly; the replay check was retained, not weakened. Tests
compare fresh output, JSON roundtrip, stored candidates and independent replay.

## 4. Validation and matched-regression evidence

- **208 targeted tests passed** for inventory sampling, current package context,
  header rebasing, matched-treatment falsification and earlier equipment/proof validators.
- **352 existing production regression tests passed** for campaign, registry
  sampling, equipment aliases and package aliases, in 10.37 seconds.
- **18 geometry tests passed** in 8.12 seconds, including receipt-exact independent
  plot captions. The root's combined five-file experiment run also passed
  **189 tests** in 42.49 seconds before the final caption test was added; these
  overlapping runs are not added together as a unique-test total.
- Final 90 primary and 90 controls pass independent per-ID equipment decoding,
  row count/identity checks, scalar equality, subgroup/full summary checks,
  exact source-byte ownership and rendered-byte replay.
- The complete target is equal between matched arms **except size/type fields**.
  Parties, routes, goods, references, package types/counts, weights, container IDs,
  seals and allocation relations do not change.
- **7,698 owned rendered regions** were compared across the 90 pairs. Of these,
  7,387 are unchanged. The 311 differences are exactly 253 equipment scalars,
  39 equipment receipts, two full-inventory headers, 14 row tares and three
  aggregate tares. Each tare difference has an explicit owner and inventory-derived
  value; there is no blanket exemption for all cargo text.
- Full source OCR, descendant raw text and labels were read across all 30 sources
  and 90 primary examples in two manual semantic audit cohorts. Matched controls
  were checked for unchanged non-equipment content; changed cases were reread.
- A separate GLM native structured-output review covered **all 180 final texts and
  labels**. The one address cleanup received fresh reviews. All remaining findings
  were individually adjudicated against current candidate hashes and exact quotes.

Falsification includes aggregate-only false-admission sources, missing direct ID
owners, wrong count dependencies, overlapping header ownership, a forged summary,
stale retry replacements, forged package contexts and serialized metadata replay.
Passing here has explicit equipment/ownership/unchanged-content obligations; it
does not merely mean that a schema parsed.

Audit inventories:
[A](analysis/mixed-equipment-expanded-pilot-20261012/SEMANTIC-AUDIT-A.md),
[B](analysis/mixed-equipment-expanded-pilot-20261012/SEMANTIC-AUDIT-B.md),
[matched comparison](analysis/mixed-equipment-expanded-pilot-20261012/matched-audit.json),
[review decisions](analysis/mixed-equipment-expanded-pilot-20261012/review-adjudications.json).

## 5. Additional observations, resolved cases and honest boundaries

The bounded automated postal correction left six documents/eight regions for
explicit inspection touchups. Another reviewer identified an optional `AJMAN`
qualification in a `HALWAN` address; that optional region was removed without
changing the sampled locality/country or the separately printed route. Both arms
share the corrected wording and were rerendered/reviewed.

Three generated steel passages mixed a `NOT IN COILS` registry phrase with an
explicit sampled coiled physical form. A six-case prompt clarification probe
**did not fix this automatically**. The three actual passages were adjudicated
to `IN COILS`, preserving their original specifications; other probe proposals
were not applied. This is not an HS-classification repair requirement: exact
HS/product mapping is outside the agreed objective. The supported physical form
must nevertheless be internally coherent. The failed probe is retained in
[goods-form-probe.json](analysis/mixed-equipment-expanded-pilot-20261012/goods-form-probe.json).
Do not claim that this prompt alone is an upstream preventive fix.

Several automated review findings misinterpreted ISO aliases. **45G1 is 40-foot
HC**, while **L5G1/55G1 are 45-foot HC**; they are not literal dimensions. The
independent decoder passed the actual owned row/summary spans. Cross-checks:
[BIC size/type table](https://www.bic-code.org/size-type-code/),
[BIC table PDF](https://www.bic-code.org/wp-content/uploads/2018/01/SizeAndType-Table1-3.pdf),
[Hapag-Lloyd equipment codes](https://www.hapag-lloyd.com/content/dam/website/downloads/pdf/17038_Update_Container_Specification_engl_sRGB.pdf).
These incorrect findings were explicitly rejected, not used to corrupt aliases.

Some source-only Marks localities, contract/export references, registration
captions and Belgian VAT/trade-registration remnants persist outside the reduced
targets. They are unchanged across matched arms and already documented in the
semantic inventories. **This is not a complete auxiliary anonymization or
geographic-registration-authenticity certificate.** No inspection descendants
were silently inserted into training. The equipment experiment is complete;
broader auxiliary casting cleanup is a separate admission consideration.

## 6. Cost and measured runtime

All provider-reported costs, including validation retries, postal corrections,
the unsuccessful goods-form probe and repeated reviews, total **$0.12919498**:

| Paid stage | Cost USD |
| --- | ---: |
| Primary wording | 0.03134230 |
| Word-validation repairs | 0.00340098 |
| Business contacts | 0.00052005 |
| Postal corrections | 0.00271455 |
| Goods-form probe | 0.00105680 |
| Primary full-text reviews | 0.04529165 |
| Uniform-control full-text reviews | 0.04486865 |

119 requests accounted for 684,200 input tokens and 54,113 output tokens,
including 18,408 reasoning tokens. Provider: GLM-5.3-Flash, low reasoning,
PydanticAI native structured output. These are API costs, not total engineering
time or infrastructure cost. Control generation reuses accepted primary wording
with explicit lineage and revalidation; it is not billed as new generation.

Warm real `physical_render`, five interleaved rounds on 60 actual mixed plans
versus their 60 matched controls:

- Mixed: **1.071 ms/plan**; uniform: **1.049 ms/plan**.
- Difference: **0.0221 ms/plan**, approximately 2.10%; negligible relative to
  generation/review and not a claimed whole-campaign benchmark.
- Traced peak: **78,772 vs 78,737 bytes**. This is Python traced allocation,
  not process RSS or GPU memory.
- Retry-admission correctness adds approximately 48 microseconds/plan in its
  separate warmed microbenchmark; source planning and provider latency excluded.
- Final raw rendering: 4.28 seconds for 90 primary; 1.73 seconds for 90 controls.
- Position export: 5.83 seconds primary; 5.57 seconds controls.

## 7. Geometry audit

Both 90-record publications have position-enriched exports using the current
source-conditioned reflow/coherent-page transform. Source texts, labels and
source alignments are not modified to manufacture anchors. Rebased whole-inventory
header owners are included explicitly in equipment geometry coverage.
This validates informative, source-owned point-layout geometry, not pixel-exact
PDF typesetting or new word-level boxes for every expanded equipment alias.

The independent audit validates **192 pages per arm / 384 overall**, exact
source/edit ownership, normalized bounds, generated centroids, page transforms,
source-inherited versus newly introduced overlaps and text/label preservation.

| Position coverage, per arm | Known / total | Coverage |
| --- | ---: | ---: |
| All content lines | 7,392 / 8,668 | 85.28% |
| Equipment-owned lines | 333 / 363 | 91.74% |
| Container-ID lines | 309 / 339 | 91.15% |

Both arms have identical coverage and **zero known-to-unknown regressions**.
The 30 unknown equipment/ID lines per arm inherit incomplete source anchors;
they are not losses introduced by mixing. All 25 newly selected uniform sources
have complete relevant source anchors. Missing coordinates remain explicit
` ||`, never invented from another container's row.

All **8,668 ordinal line coordinates and source origins agree** between arms.
The stable-edit-owner matcher separately matches 8,662 lines exactly; its six
remaining matches differ only because a dependent tare is an identity edit in
one arm and a changed edit in the other. They were inspected, not ignored as
unexplained geometry differences.

Each arm has 250 independently verified generated centroids across 95 reflowed
blocks. The overlap checker permits only overlaps already present between
untouched source rectangles; it does not declare the source layout overlap-free.
All 90 direct-container and five generic foreign-owner corruption probes are
rejected. There are no unmatched non-equipment lines in the cross-arm comparison.
For the 25 new sources specifically, equipment coverage is **300/300** and
container-ID coverage **288/288**; the remaining relevant gaps are concentrated
in the previously tested mixed-source alignments, not the new templates.

Eleven final readable figures have exact per-arm captions checked against their
actual receipts and hash-bound in the audit. Parallel-column, whole-inventory
header and six-container representatives were visually inspected. No additional
generated overlap/collision was introduced. The apparent tiny-caption arm-text
reuse concern was falsified; larger captions improve inspectability rather than
repairing a data defect.

Evidence: [GEOMETRY-AUDIT.md](analysis/mixed-equipment-expanded-pilot-20261012/GEOMETRY-AUDIT.md)
and [geometry-audit.json](analysis/mixed-equipment-expanded-pilot-20261012/geometry-audit.json).
Example figures:
[parallel columns](analysis/mixed-equipment-expanded-pilot-20261012/geometry-plots/syn_full_v7_c84cfc7ed60d499d0766adf8.png),
[inventory header](analysis/mixed-equipment-expanded-pilot-20261012/geometry-plots/syn_full_v7_96717f19a0cc3464349a49b0.png).

## 8. Production integration recommended from this experiment

Promote the tested components, rather than replacing the production pipeline
with a second standalone implementation:

1. Compile a per-source equipment-inventory contract from current owners, including
   separately scoped summaries/header occurrences. Reject unresolved per-ID scope.
2. Add a per-draw uniform/mixed inventory choice before registry load/quantity
   bounds. Preserve uniform sampling for every compatible template. Use requested
   distributions in the synthesis run, not a permanent mixed flag on templates.
3. Admit loads against each row's actual equipment and preserve ID-keyed
   allocations. Keep thermal/DG eligibility separate; this pilot does not certify
   arbitrary equipment-family mixing.
4. Rebuild replacement state on every sampling attempt; port the deliberate
   rejected-draw/return-to-source regression test.
5. Integrate current target/count-expression context into package alias projection,
   retaining explicit source presentation owners and JSON-compatible receipts.
6. Render scalar cells, subset receipts and full-inventory summaries from their
   own scopes. Carry those owners into the position audit.
7. Port independent actual-text inventory decoding, whole-target matched controls,
   and geometry tests as regression gates. Keep final semantic reviews and
   explicit adjudication for genuinely ambiguous cases.

This task delivers the expanded contained experiment and proven integration
building blocks. Production activation and resampling the next UAE dataset are
not performed by this experiment.
