# Goods/package compatibility and auxiliary cargo-text correction

Date: 2026-09-02

## Diagnosis

The previous semantic plan sampled task-facing package categories from a package-role marginal and
sampled HS goods identities later. Those decisions had no common conditioning key. Consequently,
valid values could be combined into invalid pairs such as frozen cod with `PACKAGE_ROLL` or a
liquid dangerous good with `PACKAGE_BAG`.

This was a construction defect, not a cargo-language prompting defect. A text model could describe
the supplied pair faithfully but could not make the upstream pair coherent without changing the
authoritative structured target.

The auxiliary-text contract had a separate defect: one non-empty output string was required for
every source `additionalInformation` slot, even when the synthetic target contained no semantic
fact capable of replacing the source formulation, grade, or measurement. This forced filler rather
than allowing a source-private slot to be removed.

## Evidence and method selection

The pinned training partition contains 1,057 documents. Its package/goods support has:

- 1,241 cargo groups;
- 1,105 groups with a complete task-facing package signature;
- 722 typed groups with an HS heading;
- 37 typed groups linked to a temperature setpoint within the configured frozen/chilled ranges;
- 33 typed dangerous-goods groups;
- 517 distinct HS-heading/cardinality/signature support rows;
- 8 thermal-profile/cardinality/signature support rows; and
- 23 DG-hazard/cardinality/signature support rows.

Sixty-four fit groups contain at least one deliberately unresolved package category and are
excluded from compatibility fitting rather than silently coerced.

An isolated text classifier was rejected as the primary selector. On 681 singleton typed rows with
template-grouped folds, TF-IDF plus balanced logistic regression reached 0.361 top-1, 0.627 top-3,
0.772 top-5, and 0.934 top-10 accuracy. That is useful evidence that broad semantic retrieval is
possible, but not strong enough to own a training label.

SDV neural conditional sampling was also rejected for this decision. The field is a sparse,
high-cardinality semantic compatibility relationship rather than a dense tabular marginal, and
SDV documents that CTGAN/TVAE conditional sampling relies on rejection sampling. The implemented
method instead uses exact train-only relationships and the HS hierarchy. The first four HS digits
are a stable heading-level semantic key; national digits beyond the six-digit subheading do not
affect compatibility.

References:

- WCO Harmonized System FAQ: <https://www.wcoomd.org/en/faq/harmonized_system_faq.aspx>
- SDV conditional sampling: <https://docs.sdv.dev/sdv/single-table-data/sampling/conditional-sampling>
- PydanticAI native structured output: <https://pydantic.dev/docs/ai/core-concepts/output/#native-output>

## Implemented contract

### Ambient cargo

The generator now co-samples a whole package signature and one or more distinct HS identities.
It first draws a fit-observed `(HS heading, package cardinality, ordered package signature)` row,
weighted by fit-group occurrences, then uniformly samples registry identities within that heading.
Every generated pair therefore has direct fit-only heading-level evidence.

### Thermal cargo

For every structurally eligible cargo group, the generator now computes the profiles that are
actually realizable at that group's package and HS cardinalities. A packaged profile is eligible
only when the fit partition contains a complete package signature at the exact cardinality and the
registry contains enough distinct thermal goods identities. The configured frozen/chilled weights
are renormalized over that proven subset before selection. Goods, package signature, reefer
equipment, and temperature therefore share one dependency chain; a random seed cannot select an
impossible profile/package combination.

### Dangerous goods

When every hazard in a cargo group has one common fit-observed package signature, that empirical
intersection is sampled. When no common signature exists, the pipeline uses a reusable catalog
entry generated once per unique semantic context. The PydanticAI call uses provider-native strict
JSON Schema with:

- an enum containing only the 47 pinned task package categories;
- exact package-signature cardinality;
- one request and zero repair retries; and
- stored request, response transcript, usage, rationale, and immutable hashes.

The 50-plan catalog required one call for zinc nitrate and cost $0.00073205. The 100-plan audit
catalog required one call for trinitroanisole and cost $0.00107850. All supported contexts avoided
an API call.

### Auxiliary text

`additionalInformation` now returns one position per source slot, but each position is either a
grounded string or `null`. `null` is an explicit final-patching instruction to remove the source
slot. It is not a missing value or fallback. The prompt also requires fragmentary HS leaves such as
`Other` or `Of polyurethanes` to be interpreted with their supplied heading and chapter.

## Integrated probes

### Fifty semantic plans

The final rebuilt Compose image completed all 50 relation-v5 targets with strict schema validation
and relational inverse validation in 3.654774 seconds with 138.867 MiB peak RSS. Ninety-three
package-bearing groups resolved as:

- 89 fit HS-heading joints;
- 2 fit thermal-profile joints;
- 1 fit DG-hazard joint; and
- 1 constrained catalog decision.

### One hundred semantic plans

All 100 relation-v5 targets passed strict schema validation and relational inverse validation in
6.791739 seconds with 139.531 MiB peak RSS. The 142 package-bearing groups resolved as:

- 129 fit HS-heading joints;
- 10 fit thermal-profile joints;
- 2 fit DG-hazard joints; and
- 1 constrained catalog decision.

The exact 100-document run namespace that previously failed after selecting unsupported frozen
cargo with three package rows now also completes 100/100 in 6.653264 seconds. This is the direct
regression probe for seed-dependent profile selection.

Profiling also found and removed an accidental quadratic scan in ambient compatibility sampling.
The pre-optimization 50-document host run took 5.668333 seconds; after building one heading index
per cargo group, the equivalent optimized host run took 3.332395 seconds, a 41.2% reduction. The
older pre-feature semantic stage took 3.14--3.22 seconds, so the complete package/goods expansion
is within approximately 3--6% of its prior runtime instead of regressing materially.

Observed corrected examples in the final probes include:

- flammable liquid hexamethyleneimine: `PACKAGE_DRUM`, not `PACKAGE_BAG`;
- frozen poultry, frozen fish, and frozen vegetables: fit-linked carton signatures;
- chilled lamb and mushrooms: fit-linked carton and box signatures; and
- plastic film, electrical conductors, steel tubes, and machinery: package signatures drawn from
  their respective fit-observed HS headings rather than a role marginal.

Every realization records its basis, support key, fit occurrence count, and fit document count.
Of the 129 ambient package-bearing groups in the 100-document audit, 58 use a relationship observed
in one fit document. Those rows are not guessed or silently backed off, but the provenance makes
this sparse-support cohort directly filterable or reviewable in a future large-generation policy.

### Five constrained cargo-language calls

All five final calls were schema-valid and passed every deterministic validator with no repair
request. Total estimated cost was $0.00158048. The calls used 7,909 input tokens and 1,183 output
tokens, including 759 reasoning tokens. Mean latency was 4.445 seconds and concurrent wall time was
6.949 seconds.

The important qualitative results were:

- `Of polyamides` became `POLYAMIDE PLASTIC FILM, NON-CELLULAR` using its HS heading;
- the fragment `Machinery` was expanded to the complete heading-level goods identity;
- liquid hexamethyleneimine retained its coherent drum category;
- the source-only `211615MTS` auxiliary slot returned `null`, not filler; and
- every output preserved the requested field topology and categorical-surface exclusion.

The deterministic semantic validator now evaluates an HS leaf together with its heading, matching
the generation contract. This corrected a false rejection of `polyamides` versus `polyamide
plastic film` without relaxing dangerous-goods identity validation.

## Failure behavior and scope

There is no heuristic category fallback. Missing fit support and a missing constrained catalog entry
are errors. Fit/evaluation overlap, unknown registry categories, catalog vocabulary drift, reused or
missing catalog contexts, altered relation topology, and target/provenance signature differences all
fail before publication.

The later raw-text patcher must interpret auxiliary `null` as deletion of the corresponding source
slot. Printed package/container/HS surfaces remain intentionally deferred to that final linguistic
patching stage; this change owns semantic target compatibility only.
