# Next 60 synthesis templates: three 20-source admissions

Date: 2026-10-10. Status: **complete — 60 sources / 120 reviewed pilot descendants admitted; registry 360**.

## Scope and experiment boundary

This work first audits the latest completed training experiment and then expands
the reviewed template registry from 300 to 360 sources. Preparation, generation,
semantic review and admission operate in three groups of 20. Every accepted
source must produce two independently generated descendants with reviewed labels,
complete rendered OCR and validated positional inputs.

The completed training experiment actually used **2,500 synthetic plus 600 real
training records**, not 3,000 synthetic. Its original run snapshot and validation
targets must remain distinct from the subsequently repaired current dataset.
The detailed results belong in
[the training comparison ledger](mpci-training-results-and-next-experiments.md).

The user's next-experiment outline is recorded, not launched by this work:

1. Expand toward 5,000 synthetic examples, targeting coverage gaps rather than
   equal per-source quotas. New synthesis must use UAE end/import localities.
2. Compare synthetic-only training with the current real/synthetic mixture.
3. Compare rank 64 with rank 48.

The stated next 2,000 examples would take the verified present 2,500 to 4,500;
reaching 5,000 instead requires 2,500 additional examples. The 120 admission-test
descendants in this task are evidence, not an automatic training-set extension.
Campaign size and experiment sequencing will therefore need an explicit count
reconciliation when that larger run is prepared. No training is launched here.

## Admission contract

- No existing registry source or current validation document, OCR/PDF alias or
  normalized B/L duplicate can be selected as a new source.
- Current R16 source labels, or an explicitly recorded corrected source authority,
  are the authority. Historical bindings are reusable
  only after exact source/target rebasing and current ownership validation.
- Description main-block boundaries, declared package accounting units,
  supported allocations, uppercase targets, consignee-based negotiability,
  notify `sameAs`, primary-party ownership and the DD-MM ambiguous-date policy
  follow [the current policy](kie-real-baseline-label-policy-2026-10-05.md).
- Compilation/replay, host scenario stress, complete rendered-text/label review,
  finding adjudication and independent geometry checks are distinct gates.
  Passing exact-match checks alone is not semantic certification.
- Repeated source lines cannot borrow the coordinates of a different occurrence.
  Explicit unknown coordinates remain explicit when no measured anchor exists.
- Admission preserves all previously registered assets and source declarations,
  verifies the installed assets and commits the manifest last. Every batch has
  its own inventory and review receipts.

## Progress

All three groups completed source preparation, actual generation, complete-text/
target inspection, independent semantic review and final geometry checks. All
three serial admissions succeeded, with a separate protected-registry check
before each transaction. Final registry receipts are listed below. There are
**zero unresolved new-template or pilot-descendant holds**.

## Completed training audit

The results ledger now includes run I and a reproducible complete-target audit.
The native best checkpoint is **970 / epoch 10**, with field F1 **0.8967**,
field accuracy **0.8562**, relation F1 **0.9348**, category F1 **0.8939**,
**59/60** valid JSON outputs and **52/60** schema-valid outputs. All four
evaluation checkpoints remain resumable; the exported adapter's hash matches
checkpoint 970, not the earlier checkpoints.

The matched comparison uses identical complete frozen reference labels, including
addresses and goods descriptions, rather than only unchanged annotation fields:

| Comparison scope | Previous H F1 / accuracy | Latest I F1 / accuracy |
| --- | ---: | ---: |
| All 60 validation documents | 0.8690 / 0.8059 | 0.8961 / 0.8555 |
| Same 54 documents parseable in both runs | 0.9036 / 0.8675 | 0.9102 / 0.8778 |

Of the 145 additional correct scalar values, 125 come from recovered document
parsing and 20 from improved content on the matched parseable subset. Container
identifiers, seals, party names, routes and volumes improve. Exact goods
descriptions on that matched subset decrease from 38 to 34: four improve, eight
regress, thirty remain exact and twelve remain incorrect. Coverage expansion
therefore needs to protect embedded product specifications, whole description
blocks and genuine continuations, not merely increase sample count.

Remaining quantity distractors include a **3,700 kg tare**, repeated **22GO
equipment codes**, container length and vehicle identifiers. Seven 45HC sizes
are predicted as 40HC in a ten-container document even though all ten identifiers
are recovered. These are useful coverage targets, not reasons to change correct
package/allocation labels. Native checkpoint progression, complete field/group
counts, source-checked description errors, punctuation sensitivity, confidence
intervals and training/runtime evidence are in
[section 14 of the results ledger](mpci-training-results-and-next-experiments.md#14-expanded-coverage-run-i-2500-synthetic--600-real).

The independent CPU audit reproduced both final native metric sets and passed
ten falsification controls. It took **5.46 seconds**, with **137.18 MiB** peak RSS
and **$0** API cost. The subsequently authorized reference repairs are rescored
separately rather than silently substituted into the historical run snapshot.

## Source preparation and exclusions

Preparation is additive: protected real/synthetic datasets and the existing 300
catalog entries are not modified to force a new source through admission.
Historical template syntax is reusable, but ownership, units, private copies,
description blocks and current target paths are checked anew against the exact
current source. Sources with genuinely unresolved identity/accounting boundaries
are replaced before paid generation.

Two initially selected sources print explicitly unit-bearing gross weights
absent from their current targets: `6c14020d` (19,394 kg) and `20cf0a0c`
(7,140 kg). They were initially held, then recovered through the isolated source
authority described below, with the exact printed gross values added. A third
source, `a3e3c1e3`, had consignee contacts copied from the goods continuation;
those unsupported party contacts were removed in that same authority. The
protected current real/mixed training sets were not edited. The frozen authority
retains all 600/60 source records; 2,642 other files are hash-identical to the
original, and all OCR/position inputs remain unchanged. A fourth source whose
issue date is visible only in PDF was excluded rather than assigned an OCR-
unsupported date. Additional exclusions/replacements are in the batch
inventories. This finite admission does not certify every unused real source.

## Confirmed production changes during preparation

### Complete metadata for newly introduced source owners

The compiler's explicit lexical-override path omitted `value_kind` and canonical
occurrence metadata. A location/owned-text/identifier owner could reproduce its
source while later physical rendering failed with missing metadata. The fix
records the already-measured UTF-8 byte spans and their exact source text, plus
the explicit override kind. It does not infer boundaries or relax ownership.
Three parameterized real-path regression tests exercise those owner kinds,
including non-ASCII byte offsets and location surface construction.

The isolated before/after benchmark uses 1,000 compiles per repetition, five
repetitions: no-override median 383.094→383.259 ms (**+0.043%, negligible**);
location-override 396.502→390.995 ms (**−1.389%**). Measured single-call peak
allocation changes 20,510→20,463 bytes. The previously failing downstream path
passes. Evidence: `analysis/synthesis-expansion360-20261010/benchmark_override_metadata.json`.

### Source-certified shared postal localities

Different company identities or optional address suffixes may legitimately share
one postal component. Independent locality sampling for those roles conflicted
with their source-certified shared lexical owner. The new `party_locality_groups`
capability names exact, independently printed postal occurrences and pins the
source hash. It verifies each occurrence against the party's address target and
compiled postal owner, rejects country/route conflicts, and reuses one existing
deterministic locality draw for the certified group only. Unrelated party draws
and configurations without a group retain their former behavior.

This is not grouping by similar company names, deleting duplicate addresses, or
allowing arbitrary parties to inherit geography. Missing/stale source evidence,
wrong owner, missing role, `sameAs` misuse and conflicting countries are tested
as failures. Actual source-specific render probes and benchmark receipts are in
`analysis/synthesis-expansion360-20261010/party-locality-probe/`.

### Corrected template-source authority

Narrow, proven source-label errors can be corrected for rebasing without changing
the protected current real/synthetic training set. The isolated authority retains
the full train/validation inventory, OCR, source positions and Paddle provenance.
Every changed target path has exact source evidence and before/after receipt;
each admitted template records its `sourceDataset`. Replay checks use that
recorded authority, never an old-label fallback. An authority must be frozen
before its campaign because the sampler fits the complete source inventory.

The final combined regression suite passes **781 tests in 22.47 s**. This includes
the production curated-synthesis modules, raw-text pipeline, batch admission,
host-render fixtures and repeated-fact regression tests. Batch-specific full-text
inspections and geometry checks are additional completed gates, not inferred
from this test count. Four initially failing older test fixtures were corrected
to supply already-required preparation fields and recognize the accepted `RE`
equipment alias; the underlying production functions were unchanged. No
production fallback was added to make those fixtures pass.

### Compact source-certified packing counts

The contained-count renderer required whitespace between a certified numeric
owner and its printed unit. Real source expressions such as `540CTNS/15PLTS`
therefore failed despite having an exact count/unit contract. The matcher now
accepts zero or more whitespace characters while retaining the complete unit
boundary guard. It does not infer a hierarchy or accept arbitrary neighboring
words. Seven regression cases cover compact/spaced/newline forms and reject
wrong units or longer words. Actual packing-family probes also remove each
ratio recipe deliberately: all eight corrupted fixtures are rejected.

The pre-fix compact path fails; the corrected 7-pallet draw produces 252 inner
cartons from the certified source ratio 540/15. On 10,000 unchanged spaced-unit
renders per repetition, seven repetitions, median latency is
115.182→115.433 ms (**+0.217%, negligible**), with identical 1,465-byte peak
single-call allocation. The new compact path takes 117.089 ms per 10,000 calls.
Evidence: `analysis/synthesis-expansion360-20261010/benchmark_compact_packing.json`.

### Whole printed phone ownership

Fresh source recovery selected phone digits but could leave an immediately
preceding printed `+` outside the owner. Inserting a complete new international
phone then yielded `++94` or `+ +853`. Source recovery now includes that same-
line prefix in the exact occurrence, without crossing captions or line breaks.
Current affected templates and the reusable source plans are both repaired;
generated numbers, company wording and paid call records are retained. Replay
confirms the rendered phone and its target agree. Negative controls cover
unrelated plus signs and line-boundary cases.

### Explicit linked-party context for contact generation

A notify party with `sameAs: consignee` deliberately has no duplicated name or
country in its target. Contact generation previously read only the notify's own
name/country, so an available, explicitly referenced company was described to
the model as unnamed. The actual `654` requests said that company identity and
country were not supplied; the generated email consequently retained the old
company's `ITACEGYPT.NET` domain.

Contact generation now follows the explicit `sameAs` reference solely to supply
the current company's name and country as generation context. Notify-owned
contact paths and symbolic target fields are unchanged. Invalid or unavailable
references fail explicitly. This does not add a shared-domain requirement or
infer references from similar names. The affected request hash changes, so old
contact receipts cannot silently replay; the pilot's contacts are regenerated
against the corrected context.

The targeted contact/campaign/scenario suites pass **224 tests**. The unchanged
named-party path takes 11.645→11.777 µs per call (+0.132 µs, negligible); traced
peak allocation remains 3,723 bytes. Actual before/after requests, falsification
controls and measurements are recorded in the contact-context evidence beside
the batch audit.

### Dependent handling text includes its declared country

An existing source handling instruction prints a loading-port name followed by
its country. The dependent-text renderer formerly supported only the port's
name, leaving the country stale after a route draw. It now supports the exact
declared `.name` and `.country` dependencies with source/target presence and
owned-text guards; unsupported route fields still fail. Atomicity and idempotence
are tested. No arbitrary context replacement or country inference is introduced.

The focused campaign/auxiliary/template suite passes **253 tests**. The existing
name-only path changes 28.197→28.878 µs (+0.681 µs); the no-dependency path is
unchanged at approximately 0.072 µs. Measured peak allocation is unchanged.

## Review judgments: defects versus permitted context

Full source/descendant inspections and the independent production model review
are complementary. An exact source replay is necessary, but cannot detect a
stale source constant that was never made mutable. The pilots therefore also
compare every complete rendered text with its complete target and source.

Confirmed defects include repeated current-shipment equipment, row mass/volume,
package counts, primary-company aliases and directly owned contacts left at their
source values. These are closed at the exact source owner or typed auxiliary
recipe, not by rewriting a correct public target to match a stale repetition.

The following are not automatically defects under the agreed extraction goal:

- A standalone Marks product specification, manufacturer or secondary site is
  excluded context, not a second goods description or party postal target.
- Marks serial/carton ranges are not package totals without an explicit source
  accounting declaration. A serial range is not required to equal a shipment's
  sampled quantity.
- Generated product wording need not prove an exact HS-to-product ontology or
  satisfy inferred container-payload/per-package arithmetic. Printed structured
  quantities, masses and their labels must still agree; explicit shipment totals
  must not disappear from the label.
- Legal suffixes explicitly retained by an identity recipe are not omitted
  company facts merely because the shared generated identity core is shorter.
- A source with seven weight rows and eight listed containers must not acquire
  an invented eighth row. Missing row allocations remain missing.
- A primary postal `CARE OF` recipient is not an `ON BEHALF OF` company-name
  merger. Source-owned third-country postal arrangements remain explicit.

One review incorrectly counted a 13-container table as 12 rows and misstated
`10×960 + 3×640`. Independent decimal summation confirms the full 13-row totals;
the correct samples are retained with exact evidence rejecting the findings.
Every accepted review rejection is bound to the current candidate and review
hash. A genuine repaired finding requires a fresh review, not an old finding
marked resolved.

## Final source and descendant certification

| Preparation group | Accepted sources | Actual accepted descendants | Unresolved holds | Complete inspection / decisions |
| --- | ---: | ---: | ---: | --- |
| 01 | 20 | 40 | 0 | [source review and closures](analysis/synthesis-expansion360-20261010/cohort01-source-review.md), [independent review](analysis/synthesis-expansion360-20261010/cohort01-independent-final-review.md), [strict inspection](analysis/synthesis-expansion360-20261010/cohort01-final-inspection.json) |
| 02 | 20 | 40 | 0 | [full source/descendant report](analysis/synthesis-expansion360-20261010/COHORT02_ACTUAL_PILOT_FINAL.md), [independent review](analysis/synthesis-expansion360-20261010/cohort02-independent-full-review-final.json), [strict inspection](analysis/synthesis-expansion360-20261010/cohort02-inspection-final.json) |
| 03 | 20 | 40 | 0 | [actual descendant review](analysis/synthesis-expansion360-20261010/cohort03-actual-descendant-review.md), [independent review](analysis/synthesis-expansion360-20261010/COHORT03_INDEPENDENT_FULL_REVIEW.md), [final decisions](analysis/synthesis-expansion360-20261010/cohort03-semantic-final.json) |
| **Total** | **60** | **120** | **0** | Every accepted descendant has its own final source/candidate/review/target/geometry binding. |

This is actual review of model-generated samples, not certification of artificial
host fixtures. Complete source OCR/current labels and both descendant text/label
pairs were opened by the source producer and an independent reviewing agent.
Changed passages were reopened; genuinely regenerated pairs were read again.
Final accepted judgments are bound to the current files, and current production
reviews were refreshed after genuine repairs. Exact replay verifies that those
same inspected values are what the installed templates produce.

The final sixty source records provide the following measured layout/accounting
coverage, read from their exact configured source authorities:

- Pages: 20 one-page, 29 two-page, nine three-page and two four-page sources;
  all are within the existing five-page limit.
- Goods: one accounted goods item in every source. Containers: 42 sources have
  one, eight have two, eight have four or more, and two vehicle sources have none.
  The longest tables contain thirteen and fourteen containers.
- Declared package categories: sixteen distinct source categories, including
  pallets, generic packages, bags, cartons, bundles, rolls/reels, vehicles,
  coils, wooden crates, jumbo bags and multi-wall paper bags.
- Supported source fields: gross weight in 52 sources, net weight in 21, volume
  in 32, HS codes in 42, goods handling instructions in 20 and placements in 58.
- Two source instructions are negotiable; 58 are non-negotiable. Eleven sources
  have an explicit notify `sameAs` relationship.

These are source traits, not imposed final campaign proportions. They show that
the additions expand actual layouts and accounting structures without redefining
multiple product lines as multiple goods items. The next coverage-driven campaign
still needs explicit quotas; uniform sampling of these sixty would not, by itself,
provide a balanced negotiability or rare thermal/DG distribution.

The final production model reviews had 24 findings rejected by source/policy
adjudication: nine in group 01, four in group 02 and eleven in group 03. These are
not unresolved errors or silently ignored flags. The reports preserve the exact
finding, source/candidate evidence and rejection reason. Genuine defects found
earlier were repaired and reviewed again. For example, the later, explicit
description-boundary decision excludes the detached ASTUZINC brand/origin after
party/customs text. An older broader audit note cannot override that decision;
the detached text is not incorrectly described as a Marks-column value.

### What was fixed at the template/ownership layer

The per-source reports give complete before/after inventories. The main recurring
classes were:

1. **Repeated shipment facts.** A first equipment/count/description row changed,
   while a header, attached list or later page copy remained at the source value.
   All source-proven copies now share the same mutable fact. Independent row
   allocations, row weights and shipment totals remain distinct owners; unit
   conversions and contained-packing ratios use certified expressions.
2. **Repeated party and postal facts.** Short primary-name aliases, separately
   printed contacts and shared postal occurrences needed complete ownership.
   Shared geography is declared only where exact source evidence establishes it;
   primary A and represented B remain distinct. Source-certified C/O postal
   recipients are not treated as O/B company-name mergers.
3. **Current-shipment documentary references.** Release/handling clauses can
   contain the current B/L, other related B/Ls, container IDs, sampled package
   totals or route names. Their exact owners now cast together where related,
   and independently where not. All three references in a release clause do not
   become one identical B/L merely because one is the current shipment.
4. **Private identifiers and continuations.** Source-private invoice, tax,
   exporter/importer, registration, chassis and booking copies receive typed
   anonymization or an explicitly reviewed whole-field synthetic-only omission.
   Direct-block party contacts stay public; supplementary goods-area consignee
   contacts are not promoted to public contact labels.
5. **False or redundant OCR passages.** Two PDF/layout-confirmed phantom route
   occurrences and specifically approved redundant unit/hierarchy wrappers are
   removed only in descendants through explicit source-owned recipes. Immutable
   source replay still reproduces the original OCR. No unrelated public field is
   removed and no PDF-only value is inserted into a label.

Sources that could not meet the current contract cheaply were replaced before
admission, not forced through relaxed checks. One attempted chemical family
(`630cd159`) coupled UN3266/class 8 alkaline handling with incompatible randomly
drawn chemicals. Its two descendants and their calls are retained in the audit
archive but excluded from the accepted 120; source `74860ac9` replaces it.
This is different from the agreed non-requirement to prove an exact HS-to-product
mapping. Coupled printed DG classifications/handling must remain coherent.

### Final coordinate evidence

| Group | Synthetic lines | Positioned before reflow | Positioned finally | Final coverage | Lost known positions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 01 | 4,188 | 3,416 | 3,520 | 84.05% | 0 |
| 02 | 4,335 | 3,459 | 3,514 | 81.06% | 0 |
| 03 | 3,823 | 3,215 | 3,267 | 85.46% | 0 |
| **Total** | **12,346** | **10,090** | **10,301** | **83.44%** | **0** |

Reflow adds **211** supported positions without changing any plain text or
target. Goods-description coverage is **176/208 lines, 84.62%**; address coverage
is **804/1,167, 68.89%**. Eighty-seven pages accept coherent reflow; 139 have no
eligible reflow. There are zero geometry rejections among final accepted pages.
Unknown lines retain the explicit empty ` ||` marker rather than borrowing an
unproven repeated-text occurrence or inventing a source anchor. These are
accepted missing-coordinate cases, not held text or labels.

Measured Paddle page/region hashes and each source-line occurrence are checked,
including repeated text. No unproved measured-region reuse is admitted. Each
group independently rejects four deliberately corrupted variants: changed text,
off-page geometry, duplicate line identity and inconsistent anchor provenance.
The 120-record geometry audit is separate from the production generator's own
validation, so a generator-side validation omission is not its only guard.

Thirty representative documents have **68 page plots**. Reviewers opened
diagnostic plots spanning long/split product passages, expanded addresses,
two-page O/B layouts, four-page repeated bills, nested packing and vehicles.
Every record is mechanically geometry-audited; the visual inspection is the
explicitly recorded representative subset, not a claim that every page plot was
opened by the root agent.

Inspection galleries:

- [Group 01](../artifacts/kie-synthesis-production/curated-v7-template-expansion360-v1/batch01/positions-reflow-v1/audit/GALLERY.md)
- [Group 02](../artifacts/kie-synthesis-production/curated-v7-template-expansion360-v1/batch02/positions-reflow-v1/audit/GALLERY.md)
- [Group 03](../artifacts/kie-synthesis-production/curated-v7-template-expansion360-v1/batch03/positions-reflow-v1/audit/GALLERY.md)

### Cost, validation and throughput

| Paid work | All-attempt API cost, USD |
| --- | ---: |
| Fresh source-plan preparation, including alternatives | 0.03399245 |
| Group 01 generation, scoped retries and production reviews | 0.04363695 |
| Group 02, including excluded chemical-family attempts | 0.06783499 |
| Group 03 generation, scoped retries and production reviews | 0.04486500 |
| **Entire admission effort** | **0.19032939** |

The approximately **$0.19** includes failed/replaced attempts and paid review,
not merely final accepted generation calls. Cheap GLM-5.3-Flash was used for the
paid generation/review path. Independent agent inspection, CPU replay, geometry
audits and registry admissions incur no additional API charge in these ledgers.
This finite source-admission cost is not a promised bulk-campaign unit price.

The group-02 registry gate also caught a missing shared auxiliary-policy header
before any installation. The local resume writer had preserved its twenty
source entries but reconstructed only `version`, losing the common `policy`
declaration. Restoring the exact shared header and preserving headers in that
writer resolves the cause without changing any sample. A complete before/after
probe verifies all twenty source maps, forty requests/renders/targets and 228
publication/review/inspection artifacts unchanged; production replay and strict
inspection still pass. This took **10.642 s**, **468.59 MiB** peak RSS and **$0**.
The merge guard was not weakened. Evidence:
[header-only closure](../artifacts/kie-synthesis-production/curated-v7-template-expansion360-v1/batch02/audit/auxiliary-policy-header-final.json).

The group-01 dry gate separately detected publication-root spelling: its
publication records `/mnt/d/Projects/DocumentParsing`, while an invocation from
the lowercase alias computed `/mnt/d/projects/documentparsing`. These paths
refer to the same files, but their serialized manifest strings differ. Running
the admission through the exact original uppercase script path preserves the
published manifest rather than rewriting its history or weakening the identity
check. Source/sample bytes, review decisions and paid-call records do not change.

The final full regression suite passes **781 tests in 22.47 s**; targeted Ruff
checks and `git diff --check` pass. The groups' final independent full geometry
checks take approximately **8.8 seconds each**, including imports, at roughly
**983 MiB** peak process RSS. Hot-path before/after measurements for every
production change are reported above or in linked benchmark receipts; the
measured increases are sub-microsecond or negligible, with no material memory
increase. Paid requests are limited to changed contexts rather than regenerating
all accepted company/address/goods wording after a deterministic ownership fix.

### Scope of confidence and the protected existing dataset

All 60 new templates and their 120 exact pilot outputs have completed the stated
component gates, with no remaining semantic or geometry holds. The accepted
components do not need another repair/review wave before registry use. Ordinary
new-sample review remains a production gate because fresh lexical generation is
new content, not a byte-for-byte replay of this pilot.

This task does not rewrite the current 600-real/2,500-synthetic training mixture
or silently add the 120 inspection descendants to it. It preserves the previous
300 case assets and their authorities. A read-only ancillary phone-sign scan
found nine older synthetic descendants of `9ad49fcc` with duplicate international
phone prefixes, plus two real OCR originals with doubled signs. This finding is
recorded in [the exact inventory](analysis/synthesis-expansion360-20261010/protected2500-phone-sign-readonly-inventory.json);
no labels or historical source files were changed here. It is outside the new
60-template admission, not a claim that those older outputs were repaired.
The complete-phone ownership correction prevents that mechanism for the new
fresh source plans; the old family remains an explicitly documented narrow
follow-up for later dataset maintenance.

The next bulk campaign should use the 360-source pool with coverage quotas for
the measured description, size/quantity-distractor, row-list, address and rare
thermal/DG gaps—not equal descendant quotas by default. UAE import/end localities
apply to that next campaign. Its count, distributions, training mixture and
rank-64 experiment remain to be configured; no bulk generation or training was
launched as part of this completed admission task.

## Installed registry and preservation proof

Admission order is a transaction detail, not source quality ranking: preparation
group 03 was ready first, then 02, then 01. Each transaction includes the complete
read-only gates before installation, recoverable backups, installed-case replay,
old-case preservation and a manifest-last commit.

| Group | Registry transition | Installed exact source replays | Installed case assets | New actual descendant replays | Receipt |
| --- | ---: | ---: | ---: | ---: | --- |
| 03 | 300 → 320 | 320 | 1,920 | 40 | [admission](../artifacts/kie-synthesis-production/curated-v7-template-expansion360-v1/batch03/audit/catalog-admission.json) |
| 02 | 320 → 340 | 340 | 2,040 | 40 | [admission](../artifacts/kie-synthesis-production/curated-v7-template-expansion360-v1/batch02/audit/catalog-admission.json) |
| 01 | 340 → 360 | 360 | 2,160 | 40 | [admission](../artifacts/kie-synthesis-production/curated-v7-template-expansion360-v1/batch01/audit/catalog-admission.json) |

The final registry manifest SHA-256 is
`998991dc40c88e0e4c5fad93867b5195de1a5fa9521a3c7dd466a92923c01730`.
The installed [catalog](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/README.md)
contains all sixty additions with their exact capability, source authority,
source/target snapshots, contract, template and measured alignment.

Root's independent [post-installation integrity audit](analysis/synthesis-expansion360-20261010/root-final-integrity.json)
checks all **2,160 case-file hashes**, all **300 original case entries and 1,800
original case hashes**, and all **600 original shared ownership/auxiliary source
entries**. Calibration and selection history remain unchanged. All **120 current
inspection records**, candidate/review/target hashes and both plain/positioned
publication hashes agree. Every new source has exactly two accepted descendants.
All **19 pinned top-level assets** of the protected mixed dataset remain unchanged;
its 600 real + 2,500 synthetic training IDs and 60 validation IDs are unique and
disjoint. This independent audit took **5.836 s**, **82.84 MiB** peak RSS and
**$0**. The full serial admission receipts separately record exact production
source/candidate replay and zero validation overlap by ID, OCR, PDF and B/L alias.

The independent final [all-360/all-120 production replay](../artifacts/kie-synthesis-production/curated-v7-template-expansion360-v1/audit/final-registry360-validation.json)
also passes: **360 source/target replays, 2,160 installed case assets and 120
actual installed-template candidate/publication/strict-inspection/geometry
replays**. All 360 sources have zero overlap with validation by ID, OCR, PDF or
normalized B/L. **2,999 prerequisite pins** remain unchanged. All twelve geometry
corruption controls reject. Source authority is explicit: 340 cases use R16 and
twenty use the frozen corrected snapshot. The replay took **80.763 s internally /
89.25 s end-to-end**, with **785.59 MiB** peak RSS and **zero API calls**.
The exact original uppercase root was used for group 01; no historical
publication or sample was resealed to hide that path-spelling difference.

This closes the requested sixty-template expansion. The catalog is installed
and usable; the separate pilot publications, reviews, rejected attempts and
rollback backups remain available for inspection without changing the current
training mixture.
