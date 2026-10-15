# Egypt synthesis and UAE special cargo refresh — complete

**Completed and certified:** 900 new samples have been approved, and the new
[dataset](../data/curated/mpci-bl-real600-synthetic5500-egypt300-special-refresh-r21-positions-v1/README.md)
is published with 5,500 synthetic documents, 600 real training documents and
60 unchanged validation documents. There are no unresolved findings in the
new cohorts. The 400 retired samples are recoverably archived. Training was
not launched.

The [final certificate](analysis/synthesis-refresh900-20261014/final-audit.json),
[assembly publication receipt](analysis/synthesis-refresh900-20261014/assembly-publication.json)
and [real-versus-synthetic coverage comparison](analysis/synthesis-refresh900-20261014/coverage/final-comparison.md)
are the authoritative completion records. The
[granular coverage interpretation](analysis/synthesis-refresh900-20261014/coverage/final-coverage-interpretation.md)
details the remaining real-only supports and intentional distribution changes.
Total paid API cost was **$0.89186533**.

This campaign adds 300 Egyptian-import and 200 UAE-import rare-packaging
examples, and replaces 200 dangerous-goods and 200 temperature-controlled
examples in the latest UAE cohort. The resulting synthetic pool contains
5,500 documents. The original pool, the 400 retired examples, the 600 real
training documents and the 60 validation documents remain recoverable.

## Scope and starting inventory

The starting dataset is
`data/curated/mpci-bl-real600-synthetic5000-uae-coverage-r20-party-masterbill-positions-v1`.
Its latest UAE cohort contains 300 dangerous-goods examples and 300 thermal
examples. The full 5,000 synthetic pool contains 335 DG and 399 thermal
examples. The older UAE sidecar is not silently added to training.

The registry contains 413 admitted templates: 27 DG and 38 temperature-bearing
templates, alongside ambient, vehicle and explicitly non-operating reefer
sources. The expanded special-cargo repertoire supports source-owned optional
DG fields and ventilation without requiring them in every descendant.

The [frozen baseline](analysis/synthesis-refresh900-20261014/baseline-inventory.json)
records the dataset, shared registry files and six assets for every template.
The [coverage inventory](analysis/synthesis-refresh900-20261014/coverage/inventory.json)
compares real labels with the current and older synthetic pools.

## Retired examples

The retirement selection keeps 100 old examples from each special-cargo group.
It preserves all 13 thermal and four DG source layouts represented in the
latest UAE set, plus existing printed equipment aliases, temperature/unit
presentations, HS identifier widths, package categories and DG classes.

The retained thermal split is 67 frozen and 33 chilled; the retained DG split
is 83 chemical and 17 vehicle. The exact selections and original-row hashes
are recorded in the [thermal inventory](analysis/synthesis-refresh900-20261014/coverage/thermal-retention-100-archive-200.json)
and [DG inventory](analysis/synthesis-refresh900-20261014/coverage/dg-retention-100-archive-200.json).
Retirement means exclusion from the refreshed pool, not deletion of evidence.

## New examples

| Cohort | Documents | Destination | Purpose |
|---|---:|---|---|
| Egyptian import additions | 300 | Egypt | Broader layouts and retained extraction coverage |
| Rare packaging additions | 200 | UAE | Fill named package-category gaps |
| DG replacements | 200 | UAE | Expanded source-owned DG presentations |
| Thermal replacements | 200 | UAE | Expanded frozen, chilled and ventilation layouts |
| Total generated | 900 | | 500 net additions and 400 replacements |

The DG plan has 167 chemical and 33 vehicle descendants across all 27 DG
templates. Its printed fields include 123 packing-group, 48 flash-point and
20 subsidiary-hazard examples. UN-only and UN/class-only templates remain
partial presentations; hidden sampled properties do not become labels.

The thermal plan has 133 frozen and 67 chilled descendants across all 38
temperature-bearing templates. Each DG and thermal source receives at least
three descendants. Every planned UAE special-cargo descendant has passed the
ordinary production physical-render and wording-request preflight.

The 200 rare-package additions cover nine previously absent synthetic categories:
20 each of blocks, fibreboard boxes, fibre drums, octabins, flexibags, flexitanks,
granular bulk loads and powder bulk loads, plus 40 unpacked single units. These
are sampled through a separate, reviewed load-profile catalog; the original
eleven-profile catalog remains unchanged. [The package design and evidence](analysis/synthesis-refresh900-20261014/packages/README.md)
records the units, forms, limits and production-path probes.

Count-less tanker bulk liquid remains deferred. The real source prints 350 MT
without a package count or containers, whereas the current counted physical
sampler requires a package count. We did not invent a count of one to populate
that field. This is one remaining real-only category, not a failure of the
requested 200-example package addition.

Before any paid rare-package request, manual source inspection replaced one
host with fixed, unowned "5 PALLET" framing. Otherwise a flexitank descendant
could have inherited "1 FLEXITANK IN 5 PALLETS." The replacement preserves the
quota, seed, output identity and 180 unaffected plans. [Nine explicit unstarted-plan revisions](analysis/synthesis-refresh900-20261014/packages/revision1/plan-revisions.json)
pin both old and new configurations; the runner rejects revisions of any
already started cohort.

## Policy and validation

The [label policy](kie-real-baseline-label-policy-2026-10-05.md) governs description
boundaries, declared package units, supported allocations, OCR grounding,
party ownership, dates, casing, complete printed HS identifiers and private
identifiers. Incidental packing arithmetic and exact HS-to-product taxonomy
are not new extraction requirements. Explicit totals, incompatible carrying
conditions and missing or incorrectly owned labels are genuine defects.

Certification requires exact expected IDs, source/target edit replay,
complete rendered-text and label model review, adjudication of every finding,
private-value checks and position reconstruction. Unknown positions remain
explicit; no coordinate is invented to meet a percentage. Validation sources
and PDF hashes are excluded from synthesis. Publication follows validation;
assembling the refreshed pool does not launch training.

The provider is GLM-5.3-Flash with low reasoning, native constrained output,
eight concurrent calls and no provider fallback. Historical campaign costs
suggest approximately $0.90–$1.00 for 900 ordinary examples before extra repairs.
The aggregate paid-call ceiling is $3, including retries and reviews. Actual
all-attempt usage and cost will be recorded in the completion inventory. The
[official endpoint metadata](https://openrouter.ai/api/v1/models/z-ai/glm-5.3-flash/endpoints)
is checked against the pinned provider configuration before launch.

## Completion inventory

Generation, findings, repairs, coordinate audits and final realized coverage
are recorded under `docs/analysis/synthesis-refresh900-20261014/`. Publication
commits the dataset manifest only after verifying every output file's hash.

| Export | Records | Purpose |
|---|---:|---|
| `synthetic.jsonl` | 5,500 | Synthetic-only training pool |
| `real-train.jsonl` | 600 | Real training examples, separately retained |
| `train.jsonl` | 6,100 | Combined real and synthetic pool |
| `validation.jsonl` | 60 | Byte-identical holdout |
| `replacement-synthetic.jsonl` | 2,500 | Latest UAE pool after the 400 replacements |
| `egypt-addition-synthetic.jsonl` | 300 | New Egypt-import cohort |
| `package-addition-synthetic.jsonl` | 200 | New UAE package-coverage cohort |
| `archive/retired-uae-dg200.jsonl` | 200 | Exact retired DG rows |
| `archive/retired-uae-thermal200.jsonl` | 200 | Exact retired thermal rows |
| `previous-uae-synthetic.jsonl` | 2,500 | Historical sidecar, excluded from training |

The latest UAE pool still contains exactly 300 DG and 300 thermal documents:
100 retained plus 200 replacements in each family. Every original parent file
remains unchanged. Of the 4,600 retained synthetic rows, 4,587 are byte-identical
and thirteen have only the approved seal correction; 599 real rows are
byte-identical and one has only the printed seal-label additions. The excluded
older sidecar has seventeen matching seal corrections, with its other 2,483
rows unchanged. The 400 archive rows are exact originals, not corrected copies.

Independent assembly validation passed in 164.71 seconds at 1,656.45 MiB peak
RSS. The explicit publication pass passed in 167.66 seconds at 1,651.26 MiB;
both replay all 900 ordinary publications, validate direct targets, verify
position/raw preservation and enforce exact partitions, leakage exclusions and
approved retained-row deltas. The final read-only coverage check passed in
13.89 seconds at 800.41 MiB, verifying 1,018 pins without API calls or training.

### Realized coverage, including remaining gaps

Counts below are documents unless a row explicitly says otherwise. Coverage is
not a model-performance forecast.

| Trait | Before: 5,000 synthetic | After: 5,500 synthetic | Real training: 600 |
|---|---:|---:|---:|
| Represented source templates | 360 | 413 | — |
| Thermal goods | 399 | 439 | 24 |
| Dangerous goods | 335 | 355 | 17 |
| Printed DG packing groups | 0 | 138 | 9 |
| Printed DG flash points | 0 | 54 | 2 |
| Printed subsidiary hazards | 0 | 22 | 1 |
| Free-text package fallback | 0 | 25 | 10 |
| Mixed assigned container types | 577 | 617 | 9 |
| Multiple containers | 1,608 | 1,706 | 193 |
| At least five containers | 520 | 581 | 61 |
| At least eight containers | 228 | 244 | 26 |
| Transshipments | 107 | 113 | 11 |
| Negotiable consignee wording | 333 | 407 | 48 |
| Notify same-as references | 1,294 | 1,404 | 136 |
| Explicit net-weight targets | 876 | 1,090 | 135 |
| Master-B/L references | 34 | 36 | 3 |
| Ventilation targets | 67 | 63 | 5 |

Multi-container proportions are 31.02% synthetic versus 32.17% real;
transshipments 2.05% versus 1.83%; negotiable documents 7.40% versus 8.00%.
Thermal and DG are deliberately overrepresented for supervision. Mixed
equipment remains deliberately above the real proportion, consistent with
the earlier request for at least 500 synthetic mixed-container documents.

Nine previously absent named package categories now have supervision: blocks,
fibreboard boxes, fibre drums, octabins, flexibags, flexitanks, granular bulk,
powder bulk and unpacked single units. The dedicated cohort supplies twenty
of each of the first eight and forty unpacked single units. Other new cohorts
bring the final fibre-drum count to twenty-six. No previously supported package
category disappears. Replacement does reduce wooden ordinary boxes 50→16,
steel drums 60→35, cartons 1,178→1,165 and unpacked goods 75→59; their support
remains and the exact counts are visible rather than claimed unchanged.

Ventilation decreases by four documents: thirty-two retired and twenty-eight
fresh positives. All six ventilation-capable layouts remain represented, with
63 positives overall; this is a disclosed sampling trade-off, not removed or
incorrect ventilation labels. Every previously measured owned equipment alias
remains represented and `40RE` is added. Counts of individual aliases can
decrease after replacement. The alias inventory is a provenance-backed lower
bound, not an indiscriminate regex count of public tariff mentions.

Four real-only support gaps remain for a later augmentation pass:

- An explicit original-B/L **reference identifier**: one real example. This
  is not an ORIGINAL stamp or a count of original copies.
- `freight.paymentArrangement = payable_elsewhere`: two real examples.
- DG `hazardCategory = GASES`: one real example.
- Count-less `PACKAGE_BULK_LIQUID`: one real example, deliberately deferred.

The old UAE sidecar supplies none of these four missing supports. Adding that
sidecar alone will therefore not close them. Unseen literal names, addresses,
phone numbers, dates and continuous weights are not classified as missing
categories: those are expected to vary.

### Final coordinate certification

The [independent audit](analysis/synthesis-refresh900-20261014/geometry/final900/REPORT.md)
passed for all 900 new samples, 194 source templates and 1,731 pages. Exact
transfer, joint reflow and augmentation replay preserved every raw input and
direct target, with zero lost known coordinates or introduced unexplained
collisions. All 2,782 falsification controls were rejected, including wrong
duplicate-occurrence and cross-page ownership, off-page values and stale
in-bounds coordinates with recomputed hashes.

New-cohort coverage is 74,951/89,850 lines, **83.42%**, versus 81.82% from
anchor-only transfer: 1,435 additional positioned lines. Goods lines are
81.01% positioned, addresses 63.22%, and other content 85.58%. Most unknowns
come from incomplete or absent measured source anchors; they retain explicit
` ||` rather than fabricated coordinates. Twelve lines lack safe local space.
Two rejected reflow proposals retain independently verified source transfers;
their proposed geometry was not applied.

Across the entire final 5,500 synthetic pool, coverage is **83.95%**, versus
84.07% before, a 0.119-percentage-point composition difference. This is not a
loss of known anchors in the new samples. The
[24-document/48-page gallery](analysis/synthesis-refresh900-20261014/geometry/final900/GALLERY.md)
and montage were inspected, with six detailed page plots reviewed. Root also
opened the montage and three detailed panels. An apparent goods-column shift
was traced to the actual original right-half Paddle regions, confirming the
correct owner rather than treating the plot as proof of a defect.

The final combined code/audit suite passed **1,985 tests in 211.84 seconds**;
Ruff and the targeted diff whitespace check passed. Earlier rejected and
failed attempts remain preserved as historical evidence, not current holds.

### Audit corrections and their ownership

The independent review distinguishes extraction defects from incidental
commercial arithmetic. Product capacities, historical freezing/storage
temperatures and neighbouring HS wording are not automatically relabelled as
shipment totals, current container setpoints or classification mistakes.
The explicit carrying-temperature declaration determines the transport target.

Confirmed wording repairs are bounded to their owned fields. They remove an
incompatible bulk assertion from drummed goods, incorrect pallet declarations,
a contradictory current chilled-chain phrase for frozen cargo, and misplaced
identity text. Individual/proprietor names retain the source identity structure.
Corrected values are replayed through the ordinary renderer and receive a fresh
complete-text review; an approval against an earlier candidate hash is unusable.

The postal correction stage had a real upstream defect: its generic rewrite
request could replace an already-correct country-only region with a complete
address. It now retains the original field requirements and fixes exact,
already-correct country-only owners while jointly rewriting the faulty address
components. Wrong countries remain correctable. The actual affected source was
replayed with both positive and negative controls; the initial generation
prompt/schema hashes are unchanged. See the
[postal probe](analysis/synthesis-refresh900-20261014/manual/ambient-review/postal-country-probe.json).

The finite privacy inventory adds explicit owners for private signatories,
secondary organizations, commercial references and source-only dates. Public
carriers stay unchanged. Only the new private language fields are generated;
accepted shipment facts and previously corrected wording are frozen. The old
receipt and its actual generation context are retained rather than pretending
that the original model call already requested the added fields.

That scoped language phase completed successfully: 53 samples, 80 new fields,
53 metered calls, $0.00313498 incremental cost. The cumulative campaign ledger
was $0.82898013 at this boundary; final rerender/review costs are accounted
separately below. The root also inspected all 20 generated VIA-company/postal
pairs. Each replaces the original company and street, while retaining its
explicit Siauliai/Lithuania geography; no copy of the private original street
survives in those accepted values. This field inspection does not substitute
for the later complete rendered-document review.

One source (`f256749b`) exposes a separate target omission, not a cosmetic
privacy issue. Each of its two container rows prints a carrier seal followed by
`C/S` and a distinct `BOLT…` seal. Both additional seals are present in OCR and
in the PDF's container/seal column. The source's existing historical bindings
already identify their respective containers, but the current target omitted
them. The repair must reactivate those bindings and include all printed seals;
it must not merely anonymize the two values while leaving incomplete labels.
The original source OCR and measured positions remain immutable. Any historical
label repair or synthetic identifier replacement is separately pinned in the
new dataset revision, never silently applied to the parent dataset.

The seal correction is now applied upstream. Twenty ordinary fresh sampling
probes regenerated all 40 additional seal occurrences, without changing routes,
goods, quantities or wording requests. A closure sweep checked 709 historical
seal owners across all 413 sources: the only other two absent owners are literal
`NOSEAL` declarations, correctly unlabelled. The new dataset carries exactly
one real-label, thirteen retained synthetic-row and seventeen excluded-sidecar
repairs; original rows remain preserved. See the
[source repair](analysis/synthesis-refresh900-20261014/f256-source-rebase/README.md)
and [bounded seal-owner sweep](analysis/synthesis-refresh900-20261014/coverage/seal-owner-absence-sweep.md).

A cache-bookkeeping defect was also corrected in the production generator.
Reusing accepted wording must retain its original receipt byte-for-byte, not
overwrite its generation context with a later source-authority revision. That
overwrite invalidated the explicit scoped-private lineage checks; the checks
blocked replay instead of silently approving it. All 53 affected receipts were
recovered from pinned before-images and actual native responses. Ordinary and
scoped cache-reuse regression tests and the subsequent complete replay passed.
No accepted lexical values were lost or regenerated during that recovery.

The final package review caught one internally conflicting product passage:
the same oil was called both refined and crude. Removing only the second
`CRUDE` qualifier changed the description and its rendered occurrence, with
every other shipment fact unchanged. The fresh complete review has no findings;
a 900-description scan with this case as its required positive control found
no other such pair. This is a genuine wording contradiction, unlike the
explicitly waived product-code taxonomy or per-unit fill arithmetic.

### Completed text and target validation

All 900 candidates passed ordinary production publication, with zero unresolved
findings. The final 328 source/cohort review receipts cover every complete
rendered text and target; they are checked against the actual stored native
requests and responses, not inferred from an agent status. All 192 current
findings have specific, hash-bound dispositions and exact OCR evidence.

The manual source-to-descendant inspection covers 100 selected complete pairs:
40 DG, 40 thermal and 20 ambient/package pairs. Changed selected content was
fully reread; unchanged content has exact text/target proofs carrying forward
the original full reads. Every additional finding and confirmed repair was
reviewed separately. This does not claim that a human manually opened all 900
documents. Detailed component inventories are linked here:

- [DG final evidence](analysis/synthesis-refresh900-20261014/manual/dg-final-review.md).
- [Thermal final evidence](analysis/synthesis-refresh900-20261014/manual/thermal-review.md).
- [Package final evidence](analysis/synthesis-refresh900-20261014/manual/ambient-review/final/README.md).
- [Egypt final closure](analysis/synthesis-refresh900-20261014/design/egypt-review-adjudication.md).
- [Actual native-call and all-attempt ledger audit](analysis/synthesis-refresh900-20261014/native-review-ledger-certified900.json).

The final paid ledger is **$0.89186533**, including all generation, repairs,
reviews and retries: 1,333 requests, 4,665,599 input tokens (including 525,696
cache-read tokens), and 510,218 output tokens, of which 197,707 are reasoning
tokens. The final ledger audit made no API requests itself and passed in
145.70 seconds at 800.41 MiB peak RSS. It proves actual call coverage and
accounting; semantic approval is supplied by the separate dispositions and
ordinary publication gate.

The integrated curated-synthesis and campaign-audit suite passed **1,959 tests**
in 211.32 seconds. Four additional repair-context tests confirm that the manual
field executor uses the guard-validated effective source authority and rejects
missing, duplicated or invalid contexts without falling back to raw YAML.
Six manual-adjudication falsification tests pass with a hermetic fixture; the
old fixture incorrectly coupled an earlier approval to mutable live artifacts.
Production validators were not weakened to accommodate that test failure.

Measured supporting changes include 7.59× faster baseline hash-I/O validation,
3.09× faster scoped-private preflight, and ordinary receipt validation at
119.42 versus 120.27 ms in the paired control benchmark. Explicit source-revision
validation adds 1.863 seconds once per 900-sample preparation; it does not add
per-request model calls or hot-path generation work.

### Source geometry bundle completion

The first coordinate-publication attempt failed loudly with missing alignment
files. The target-only `f256` source-authority promotion had copied its six core
files but omitted the alignment directory, even though its unchanged manifest
already pinned every alignment. This was an incomplete artifact bundle, not a
coordinate-matching or text/label defect. No coordinate or label was invented
to bypass it.

All **704** original alignment files—644 training and 60 validation, 50,213,482
bytes—are now restored byte-for-byte under the revised authority. Its six core
files, active registry/index, original datasets, and all 4,301 existing files
across the 16 campaign outputs remained unchanged. The authority publisher now
validates the complete core/geometry inventory before writing and commits the
manifest last. Twenty-four focused tests reject missing, extra, corrupted,
swapped or symlinked assets and plain/positioned-input hash drift. No production
coordinate algorithm or fallback behavior was changed.

An independent full-bundle probe verified 72,442 source lines and 61,425
positioned lines and successfully loaded measured geometry from both original
and revised authorities. Only the intentional two-seal target revision differs;
all OCR, measured positions and alignment bytes remain identical. The after
probe passed in 37.92 seconds at 800.41 MiB peak RSS. Paired streaming inventory
validation medians were 2.778 seconds for the original bundle and 2.550 seconds
for the revised bundle; this is timing variance, not a claimed algorithmic
speedup. See the [restoration receipt](analysis/synthesis-refresh900-20261014/f256-source-rebase/alignment-restoration-applied.json)
and [independent after proof](analysis/synthesis-refresh900-20261014/geometry/source-bundle/after.json).

### Certification boundary

The requested additions/replacements are audited as 900 new samples. Existing
retained rows are protected against unrelated edits; separately approved,
source-proven corrections are recorded explicitly. The parent dataset and old
UAE sidecar remain available as original evidence.

This audit also documents two known historical, non-target private echoes:
`JIA WEIXIANG` and `CABLEX (EGYPT)` in older descendants. Their future template
bindings are corrected, but preservation of the old rows does not constitute
certification that the complete historical pool is fully anonymized. These
known flavour-text residues must not be conflated with missing or inconsistent
extraction labels, nor concealed under a blanket privacy certification.
