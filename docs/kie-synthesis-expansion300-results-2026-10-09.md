# Third 100-template synthesis expansion: implementation and validation

Date: 2026-10-09. **Complete: 100 additional templates admitted, bringing the
shared registry to 300. All 200 pilot descendants are published, reviewed and
geometry-validated. The final suite passes 494 tests; no unresolved review findings
remain.** The active training dataset was not changed.

The selection investigation is in
[the expansion plan](kie-synthesis-expansion300-plan-2026-10-09.md). The active
recipe is
[mpci_bl_curated_v7_template_expansion300.yaml](../configs/synthesis/mpci_bl_curated_v7_template_expansion300.yaml).

## 1. Deliverable and current publication state

The task is to add **100 usable source templates** to the existing 200-template
registry, with **two fully generated, reviewed and positioned descendants per
new template**. The 200 pilot descendants are admission evidence, not an
automatic addition to the active training dataset.

| Component | Current evidence/state |
| --- | --- |
| Exact selection | 100 new source IDs; two descendants per source |
| Source ownership and production preflight | 100/100 passed |
| New-source host-only stress | 1,000/1,000 draws passed, ten per source |
| Generated candidate scope | 200/200 published, two per selected source |
| Final semantic adjudication and plain publication | Complete; 36 final findings across 15 sources adjudicated, zero unresolved |
| Final synthetic geometry audit and plots | 200/200 validated; 20 documents / 35 pages plotted; four falsification controls rejected |
| Catalog admission | Complete: 300 registered sources, all 1,800 case assets hash-verified |
| Active training dataset | Unchanged: 600 real + 1,500 synthetic training, 60 validation |
| Training/GPU work | None launched by this expansion |

The publication, geometry and completed admission receipt links/hashes are
recorded below. All 200 descendants replay through the actual registered assets;
missing or stale evidence was not substituted with an earlier successful review.

All expansion work is isolated under
`artifacts/kie-synthesis-production/curated-v7-template-expansion300-v1/`.
The 100 accepted template assets are now also installed in the common registry,
`artifacts/synthesis-templates/mpci-bl-v7-reviewed/`.

## 2. What the selected sources add

The final selection differs from the initial investigation shortlist. Its
authority is
[final-selection.json](../artifacts/kie-synthesis-production/curated-v7-template-expansion300-v1/audit/final-selection.json),
not the earlier provisional counts. All selected sources are current reviewed
real-training sources; wider-pool eligibility was investigated but was not needed
for this final batch.

| Source trait | Final 100-source selection |
| --- | ---: |
| Historical carrier metadata values | 25 |
| Pages: 1 / 2 / 3 / 4 | 37 / 38 / 22 / 3 |
| No containers / one container / multiple containers | 3 / 66 / 31 |
| At least eight containers | 3 |
| Largest container list | 9 |
| Ambient / frozen / vehicle / chemical DG capabilities | 91 / 5 / 3 / 1 |
| Negotiable / non-negotiable / OCR-unavailable | 4 / 95 / 1 |
| Notify-party `sameAs` | 18 |
| Multiple approved description spans | 7 |
| Explicit transshipment in the new sources | 0 |
| Rebased full historical template assets | 86 |
| Newly authored explicit-binding template assets | 14 |

The 14 explicit-binding assets are deliberately small `{document_id, bindings}`
objects backed by current, hash-pinned source contracts. They are not unvalidated
old compiler metadata with missing fields. The same production ownership,
physical rendering, wording and publication checks apply to both shapes.

The admission probe established zero validation overlap by document ID, exact
OCR hash, PDF identity including aliases, and normalized nonempty B/L number.
All 100 source contract/target/alignment identities passed; six complete assets
per source can be packaged without changing the original OCR or real labels.

The final selection retains 41 sources from the original proposed shortlist and
59 other eligible reviewed training sources. The exact selection was frozen
before publication; difficult or ambiguous candidates were replaced rather than
reducing the target count or weakening admission. Twenty exploratory descendant
candidates outside that selection were archived with checked before/after hashes
under the campaign's `audit/unselected/`; they were not deleted or published.

### Explicit exclusions

Eight specifically recorded cases were left outside this batch rather than
weakening their requirements:

| Source prefix | Reason |
| --- | --- |
| `af151047` | Conflicting source accounting totals |
| `30ef57fe` | Detached product, contact and date dependencies need broader ownership work |
| `654e017c` | Unclosed equipment-dependent handling and a secondary postal block |
| `630cd159` | Unclosed DG/product/country continuations |
| `ebb856c7` | Detached handling geography not yet coherently owned |
| `75e7c809`, `a985b0ee`, `50864bab` | Exporter-country ownership ambiguous between actors |

These are template-admission exclusions, not deletions from the real dataset.
The exact reasons are preserved in
[explicit-selection-exclusions.json](../artifacts/kie-synthesis-production/curated-v7-template-expansion300-v1/audit/explicit-selection-exclusions.json).

## 3. Repairs made at their owning layer

This expansion exposed real gaps in source declarations and several reusable
production paths. Fixes distinguish source-specific certificates from generic
renderer behavior; production code does not branch on the listed source IDs.

### Source ownership, repeated facts and lexical coverage

- Rebased old allocation references using **both allocation-group and row
  coordinates**, resolving equipment identity against current placements. A
  row index from one historical group can no longer accidentally own a row in
  another group.
- Recognized dotted container-number surfaces alongside existing spaces,
  slashes and hyphens, consistently in source proof and rendering.
- Restricted product matches to the approved description byte spans. Identical
  text printed elsewhere cannot acquire description ownership merely because
  it matches a phrase.
- Added an early lexical-owner check: every current address and description
  target must depend on a mutable postal/product owner. A literal old address
  or description is not accepted as a generative implementation.
- Closed repeated private product/package/country/reference/contact occurrences
  by exact owned spans. When a private surface had no legitimate public target,
  its treatment stayed private rather than manufacturing a target field.
- Preserved shared auxiliary policy metadata when writing isolated staging
  files. Admission requires identical top-level shared metadata and additive
  source maps; missing metadata is an error, not a silent merge default.

Representative numeric/source proofs are documented in
[NUMERIC-OWNER-RECOVERY.md](analysis/synthesis-expansion300-plan-20261009/NUMERIC-OWNER-RECOVERY.md).
For example, `3056615e` has three printed weights summing exactly to 43,370 kg;
the repair associates the three row owners with the existing total instead of
guessing ownership from equal values. Not every investigated recovery was
selected for this batch; that report preserves partial/rejected findings too.

### Measurements, cargo physics and operational facts

- Added explicit measure-owner certificates with dimension, source unit,
  baseline, optional container row and optional component-partition status.
  Each certificate must match its printed numeric surfaces and current target.
- Kept source-unit conversion and printed precision together. A kilogram row
  and metric-tonne total now share canonical arithmetic without treating their
  literal numbers as equal. `FTQ` and `CF` are accepted cubic-foot aliases under
  the existing exact conversion machinery.
- Distinguished **components of one total** from independent shipment totals.
  Components must sum to a unique source total, and sampled component amounts
  are distributed on the valid printed precision lattice.
- Rendered explicitly owned DG packing-group Roman numerals from the sampled
  DG fact, with exactly one proven source token. They do not remain attached to
  a different sampled UN/class/group bundle.
- Linked shared temperature instructions to the settings they govern rather
  than letting a repeated instruction remain tied to only one container.
- Corrected seven source capability certificates after broader donor stress
  exposed positive printed measures absent from public labels. Six declare the
  required private donor dimensions; one whole-vehicle source uses its supported
  `source_whole_units` profile. These measures remain private, but participate in
  donor eligibility, mass ordering and equipment-capacity validation.

The capability-only probe passed **140/140 draws** across the seven affected
sources and inventoried all 100 sources with zero uncovered positive private
dimensions. See
[final-private-measure-capability-proposals.json](analysis/synthesis-expansion300-plan-20261009/final-private-measure-capability-proposals.json).
The deliberately source-bound exception is not an unrestricted fallback to old
mass: the source's uniquely owned physical total supports its whole-unit ratio.

### Parties, contacts and generated wording

- Repeated-party identity compares ordered name/address words independently of
  case and punctuation. Address numbers and word order remain significant, so
  separate offices are not merged merely because company names match.
- A company name shared by several roles can legitimately serve offices in
  different sampled locations. The name-generation request expresses one
  identity with those offices; postal fields still have their own geography.
- A symbolic notify party with contact overrides uses its referenced party's
  sampled geography. `sameAs` does not force a separately invented notify
  identity or leave an old-country fax/phone attached to the new party.
- Generated lines are checked against the target application's text-character
  contract. Cached wording that violates a newer policy goes through the
  existing recorded, bounded repair path; it is not silently reused or accepted.
- Scoped wording touchups remove generated schema commentary, redundant country
  aliases and internally contradictory product wording while retaining complete
  postal facts, numbers and source-owned boundaries. The initial touchup receipt
  contains 11 edits; later specific closure receipts are separate, so this is
  not claimed to be a disjoint total of every repair.

Examples in
[wording-touchups.json](../artifacts/kie-synthesis-production/curated-v7-template-expansion300-v1/audit/wording-touchups.json)
include preserving whole berries instead of contradictory sliced wording,
removing sheet wording beside host-declared rolls, and keeping a postcode-only
continuation postcode-only while the main postal block owns the locality and
country. The corrected region is rerendered and reviewed, rather than editing
only its label.

### Final targeted closures before publication

The last full-text review exposed bounded changes, not a need to alter real labels:

- **Notify fax (`d6d020b7`):** the source's exact private fax occurrence is owned
  by the named notify party and now uses that party's sampled country. Both
  descendants passed exact-scope replay, with unchanged public targets. The
  [fax receipt](../artifacts/kie-synthesis-production/curated-v7-template-expansion300-v1/audit/final-notify-fax-proposal.json)
  demonstrates the printed number follows the sampled Ghana/China geography.
- **Exporter country (`7c5b19dd`):** the exporter identifier is repeated in the
  shipper block, establishing that actor's ownership. Country and country code
  therefore follow that shipper; unrelated exporters elsewhere remain separate.
- **Detached carton ranges (`1f1c1d54`):** two old serial ranges each described
  458 cartons. They could contradict new sampled counts/types, so only those
  unlabelled quantity-like suffixes were removed. The VARDHMAN brand and printed
  origin stayed. A locality-dependent issuance-clause occurrence was neutralized
  without changing the separate carrier signature. The
  [three-source closure proof](analysis/synthesis-expansion300-plan-20261009/final-three-source-proposals.json)
  verifies exact changed intervals and unchanged public targets for the source
  declarations above.
- **Translated duplicate country (`7ade3b47`):** generated `TINTANE, MAURITANIE`
  followed by `MAURITANIA` became `TINTANE` followed by `MAURITANIA`. The one
  redundant country translation was removed; street/site/locality, lot number
  and the canonical printed country were preserved. Both repeated-party uses
  share the same corrected owner. See the
  [address touchup](analysis/synthesis-expansion300-plan-20261009/final-address-alias-touchup.json).
- **Roll versus sheet wording (`0ad3c6b5`, both descendants):** descriptions that
  implied delivered cut sheets despite host-declared rolls now explicitly say
  the rolls are intended for conversion to those sheet specifications. Every
  grade, dimension, numeric specification, composition and purpose remains;
  numeric-token equality was checked. The
  [final roll-conversion receipt](analysis/synthesis-expansion300-plan-20261009/final-0ad-roll-conversion-touchups.json)
  contains exact before/after wording and production replay evidence.

These changes were followed by rerendering and current review. They are distinct
from the final rejected findings described next; 36 rejected findings does not
mean 36 unresolved defects were waived.

### Production files touched

| Component | Responsibility of the change |
| --- | --- |
| `scripts/synthesis/rebase_curated_v7.py` | Group-aware allocation identities, dotted equipment surfaces, approved product-span matching |
| `synthesis/curated.py` | Consistent dotted equipment source proof and rendering |
| `synthesis/curated_campaign.py` | Mutable lexical-owner preflight, shared-company wording context, bounded invalid-cache recovery |
| `synthesis/curated_measurements.py` | Explicit cubic-foot aliases |
| `synthesis/curated_physical.py` | Certified units/rows/components, preserved physical constraints, DG packing-group rendering |
| `synthesis/curated_scenarios.py` | Repeated-party signatures and symbolic-notify contact geography |
| `synthesis/curated_wording.py` | Target application text validation before publishing lexical values |

The `synthesis/` entries above are under `src/document_ocr/`. Matching focused
tests cover each changed behavior. Source-specific ownership/capability changes
are in the isolated campaign's staging and audit receipts, not ID-based code.

## 4. Validation results

### Complete rendered-text review and semantic adjudication

Every final source's pair of generated OCR texts and complete targets has a
current review receipt, bound to candidate identities/hashes. Production
publication replay accepted **200/200 samples**, with **zero unresolved findings**.
After corrections and fresh review, the remaining **36 findings across 15 source
families** were individually rejected using current source/rendered evidence and
an explicit reason. The
[final adjudication record](analysis/synthesis-expansion300-plan-20261009/final-semantic-adjudications.json)
records each finding hash, exact candidate hashes and nonempty OCR witnesses.

Examples of correctly rejected findings include a purchase-order reference
excluded from the goods description, a genuinely separate exporter whose country
must not be reassigned to the shipper, unquantified yarn winding form that does
not introduce an extra counted package level, and a review that assigned one
variant's quotation to the other variant. A generated fax-only private contact
can also be correctly absent from this reduced target's phone/email fields.
Private national-registration formats are not being certified as legal identities.

The [plain publication manifest](../artifacts/kie-synthesis-production/curated-v7-template-expansion300-v1/manifest.json)
binds all current source, candidate, target, review and adjudication hashes:

- Manifest SHA-256: `5afaf70e6e0b2f8107115a665cd44e13481ddeb955e74c0d5e45af6497bbf958`.
- Plain dataset SHA-256: `446b357373f296cab92ecb0bd049517e68e025e376325ed9e2fb513ad514df22`.
- Source/variation inspection: [plain rendered gallery](../artifacts/kie-synthesis-production/curated-v7-template-expansion300-v1/samples.md).

### Host-only variability and stress

The latest persisted
[final-scenario-stress.json](../artifacts/kie-synthesis-production/curated-v7-template-expansion300-v1/audit/final-scenario-stress.json)
records **100 sources × 10 draws = 1,000 passing production preflights**, zero
failed sources or variants, in **24.567 seconds**, peak RSS **434.715 MiB**.
This is a deterministic host-fact stress test; it does not claim 1,000 lexical
generations or semantic reviews.

Across these draws there were 551 six-digit HS identities, 20 package categories,
178 origin countries and 180 destination countries. Every source varied its HS
identity, origin country and destination country; 92 sources also varied package
category. Equipment had six observed size/type combinations including deliberately
untyped equipment. Thermal samples exercised 13 setpoints and DG samples four
hazard-category groups. Counts describe observed draws, not guaranteed quotas for
the final 200 published descendants.

The source-format and physics restrictions remain deliberate: a restricted
whole-vehicle source is not forced into cartons merely to increase diversity,
and unprinted equipment categories are not invented as labels.

### Source positional evidence

The exact 100-source coordinate inventory passed all identity and occurrence
checks: **191 pages**, **9,778 nonempty content lines**, **8,473 measured positions
(86.65%)** and **1,305 explicitly unknown positions**. It identified 392 repeated
text groups covering 841 occurrences, with **zero unproved region reuse**.
Missing or ambiguous positions do not borrow anchors from another occurrence.

This source-coordinate linkage is separate from final synthesized geometry. The
independent checker and eight falsification tests are described in
[COORDINATE_AUDIT.md](analysis/synthesis-expansion300-plan-20261009/COORDINATE_AUDIT.md).

### Final synthetic geometry

The independent serialized-receipt audit passed **all 200 samples / 100 sources /
20,428 lines**, including line preservation, source occurrence identities,
page bounds, ordering and coherent page transforms. All **200 positioned records
retain the exact plain OCR and labels**, with **zero previously known anchors lost**.

| Lines | Anchor-only positioned | Final positioned | Final coverage |
| --- | ---: | ---: | ---: |
| All content (20,428) | 16,295 | 17,336 | 84.86% |
| Goods descriptions (1,116) | 189 | 874 | 78.32% |
| Addresses (1,856) | 813 | 1,166 | 62.82% |
| Other content (17,456) | 15,293 | 15,296 | 87.63% |

Reflow adds **1,041 positioned lines**, including **685 goods lines** and **353
address lines**. The remaining 3,092 lines stay explicitly unpositioned; their
text is preserved. The source-conditioned eligibility gates held 150 region
groups for incomplete source anchors and 39 for foreign text inside an owner's
envelope rather than fabricating a plausible-looking location.

Across 382 pages, 179 accepted reflow proposals, 199 had no eligible reflow, and
**four declined the proposed reflow**: two density-floor and two legacy-column-order
decisions. These are **not four invalid samples**. Those pages retain their
previous validated anchors; the final coordinates, including those retained
anchors, passed the independent full serialization checks. A proposal rejection
is not a reason to discard a valid final sample or to relax geometry constraints.

The four falsification controls—text mutation, off-page coordinate, duplicate
line and inconsistent source anchor—were all rejected. Geometry is a plausible,
source-conditioned training signal, not a claim of measurements from a physically
rendered synthetic PDF.

Evidence and inspection:

- [Coverage/preservation report](../artifacts/kie-synthesis-production/curated-v7-template-expansion300-v1/positions-reflow-v1/audit/validation.json): 3.894 s, no API calls.
- [Independent serialized checks and negative controls](../artifacts/kie-synthesis-production/curated-v7-template-expansion300-v1/audit/geometry-independent-probes.json): 20.431 s, peak RSS 565.465 MiB, no API calls.
- [Geometry gallery: 20 documents / 35 pages](../artifacts/kie-synthesis-production/curated-v7-template-expansion300-v1/positions-reflow-v1/audit/GALLERY.md).
- [Independent complete geometry review and actual visual-inspection inventory](analysis/synthesis-expansion300-plan-20261009/GEOMETRY_FINAL_REVIEW.md): every one of those 35 PNGs was opened; all 200 receipts and 20,428 lines independently checked in 1.011 s, peak RSS 51.11 MiB.
- [Positioned rendered gallery](../artifacts/kie-synthesis-production/curated-v7-template-expansion300-v1/positions-reflow-v1/samples.md).
- Positioned dataset SHA-256: `42d0344ab13b785385e4b7c487fa8b9f8c765105016c8a8944d1f20a15890d64`.
- Positioned manifest SHA-256: `7122ef3fbc43f536994666704eac389f799700579bde2fd7940af1cb73a7a94c`.
- Independent-control receipt SHA-256: `7875ea4471da61ab52b5c10e15eb31b2eabd149cef70ecfe36094d81dae9bc44`.

### Tests

The final post-change suite passed **494 tests in 19.38 s**, including production
regression coverage, source/geometry audits, final-adjudication validation and
transactional catalog-admission tests. The 31-test admission subset also passed
independently in 8.11 s. These subsets overlap the 494 total; they are not added
to it. Ruff and `git diff --check` pass.

Relevant falsifications include wrong measure dimensions, invalid partitions,
lost or static lexical owners, repeated-party address-number differences, stale
wording validation, source-coordinate occurrence reuse, malformed corpus-selection
envelopes, each leakage channel, and transactional rollback before manifest commit.
Eleven new admission cases establish that explicitly declined, wholly unchanged
reflow proposals can pass final-geometry validation, while partial edits,
unknown statuses, missing/malformed reasons, duplicate pages, failures on accepted
pages and falsified manifest/audit counts cannot. The production geometry guards
were not weakened or bypassed.

## 5. Existing 200 templates and 1,500 samples are protected

The independent
[regression report](analysis/synthesis-expansion300-plan-20261009/PROTECTED-PUBLICATION-REGRESSION.md)
uses the current published repair receipts—not stale historical wording caches—as
the authority for the already corrected dataset.

- **2,160/2,160** current records pass strict current label validation and exact
  raw/positioned text preservation.
- **1,500/1,500** synthetic inputs and complete targets replay exactly from source
  edits plus authorized later description/package/footer/reference repairs.
- Before admission, all **1,200** existing template assets and **1,211** protected
  file pins remained unchanged. During admission, only the three authorized
  catalog files (`manifest.json`, `ownership.yaml`, `auxiliary.yaml`) changed;
  the other **1,208** protected hashes still match. Every old source's parsed
  metadata and ownership/auxiliary entry is preserved exactly.
- All **1,000** old-template plans match immutable Git HEAD exactly in scenario,
  pre-wording target, physical-render plan and wording-request hashes.

| Measured workload | Immutable HEAD | Expansion worktree |
| --- | ---: | ---: |
| Full publication validation/replay | 12.263 s | 12.181 s |
| 1,000 host plans, preflight and comparison hashes | 17.754 s | 17.613 s |
| Planning throughput | 56.324 plans/s | 56.776 plans/s |
| Complete measured audit | 30.173 s | 29.955 s |
| Peak RSS including publication reconstruction | 820.742 MiB | 820.742 MiB |

These single paired processes ran concurrently, so the 0.8% planning difference
supports **no observed material slowdown**, not a statistically established
speedup. A later isolated current-code check again produced all 1,000 identical
plans in 14.576 s with 445.039 MiB peak RSS; its smaller workload/machine contention
means it should not be presented as a comparable end-to-end speedup.

An additional read-only physical-owner inventory covered 565 occurrences across
the old 200 templates. Of 306 source/measure pairs, 290 are public, 12 are positive
private facts already declared, one is a valid source-bound whole-unit gross
total, and three are printed-zero volumes. **No unexplained positive private
measure omission** was found. The one source-bound exception is `3eaf543b`; the
all-zero cases are `5c5e51a9`, `88bab156` and `d97350b6`.

Historical candidate-cache replay can still reject a changed wording-request
hash. That is correct stale-cache behavior and is distinct from corruption of
the current published samples, which passed complete receipt replay. No hash
guard was weakened to make a historical cache pass.

## 6. Cost: one cumulative campaign ledger

**Cumulative accounted cost: $0.89252107**, from **817** immutable call-attempt
receipts in
`artifacts/kie-synthesis-production/curated-v7-template-expansion300-v1/calls/`.
This includes **$0.87650932 provider-reported** and **$0.01601175 conservatively
reserved for one request with unknown billing**. The reservation is not a
provider-confirmed charge. The single total includes source reviews,
discarded/probed candidates, lexical/contact generation, rendered reviews and
retries; three attempts have no accepted output.

The [cost-accounting receipt](../artifacts/kie-synthesis-production/curated-v7-template-expansion300-v1/audit/cost-accounting.json)
pins every call receipt and records reported usage: 3,711,154 input tokens,
641,147 output tokens including 200,426 reasoning tokens, and zero provider
token-accounting anomalies across the 816 usage-reported attempts. This campaign
cost includes source-admission development/review work; dividing it by 200 is
not an estimate of marginal generation cost at larger scale.

The total is computed **once** as `sum(Decimal(receipt['costUsd']))` over those
call files, the same rule used by the campaign runner. Stage summary `costUsd`
values are cumulative snapshots and must **not** be added together. Review totals,
touchup reports and processing-wave totals must not be added to this ledger sum.
The receipt's call-pin list hashes to
`3318fab667a0baba244ce700365e03df6abbfd92e08671e2917f608f007d9b3e`.
No training cost is part of this expansion. A later provider reconciliation of
the one reserved request should adjust this same total, not create another charge.

## 7. Admission procedure and final completion record

[admit_catalog.py](analysis/synthesis-expansion300-plan-20261009/admit_catalog.py)
is dry-run by default. Its
[operator contract](analysis/synthesis-expansion300-plan-20261009/CATALOG-ADMISSION.md)
requires the exact current 100-source/200-descendant publication and geometry
receipts before any catalog change.

The helper freezes prerequisite hashes before validation; verifies current
reviews/adjudications through production publication replay; rechecks validation
exclusion; packages six assets per new case; and replays all new candidates
against those registered-shape assets in temporary space. It independently
revalidates the protected old dataset and old-template preflight.

Apply takes a registry lock and backs up all existing shared files. It installs
only new case directories and additive ownership/auxiliary maps, validates the
live registered assets, preserves every old source entry and capability, and
commits the new manifest **last, atomically**. Validation failure restores old
shared bytes and moves new assets back to the recoverable transaction payload.
Interrupted transactions fail closed instead of silently resuming unknown state.

The existing 200-source future campaign config remains unchanged; expanding the
common registry does not implicitly change that historical recipe's source scope.
The training mixture and validation files are not edited by admission.

| Final evidence | State |
| --- | --- |
| Plain publication manifest: 100 sources / 200 samples, all reviews adjudicated | Complete; zero unresolved findings |
| Positioned publication manifest: same inputs and labels, 200 receipts | Complete; 200/200 preserved |
| Independent geometry report and four rejected controls | Complete; 200/200 valid final geometries, all controls rejected |
| Inspection gallery and geometry plots | Complete; rendered galleries and 20-document / 35-page plots linked above |
| Catalog dry-run receipt | Passed; 2,462 prerequisite file pins; 200 registered-shape replays |
| Catalog apply receipt: 300 sources, 600 additive case assets | Complete; final live replay passed before atomic manifest commit |
| Live old 1,500-sample replay and old 200-template preflight | 1,500/1,500 exact inputs/targets; 1,000/1,000 planned draws; 2,160 schema/preservation records |
| Final post-change test/lint results | 494 passed in 19.38 s; Ruff, compile checks and diff checks passed |
| Cumulative cost receipt | Complete; one unknown-billing reservation explicitly included |

Completed transaction evidence:

- [Admission dry run](../artifacts/kie-synthesis-production/curated-v7-template-expansion300-v1/audit/catalog-admission-dry-run.json).
- [Admission apply receipt](../artifacts/kie-synthesis-production/curated-v7-template-expansion300-v1/audit/catalog-admission.json).
- [Live shared catalog manifest](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/manifest.json).
- Recoverable original shared files:
  `audit/catalog-admission-transaction/before/` under the campaign directory.
- Final catalog manifest SHA-256:
  `35f32f478418f642b5060cb1338c345fa33f822a9600dea1c2c5428c512a30f4`.

The complete apply process, including repeating preparation and the live
regression gates, took **121.38 s**, peak RSS **778.99 MiB**, with no API calls.
Preparation within that process took 66.06 s. The live old-publication check took
9.05 s and the old-template 1,000-plan preflight 14.26 s. These are validation
workloads, not lexical-generation throughput estimates. A final separate read-only
check confirmed all **300 source directories**, **1,800 case-file hashes**, all
**four shared dependency hashes**, exact old metadata for 200 cases and the
manifest's matching transaction receipt.

The registry is ready for the next separately configured campaign. Existing
two-descendant recipes retain their immutable staging paths for exact publication
replay; a new synthesis recipe should select the desired sources/capabilities
from the 300-source manifest and use shared `cases/`, `ownership.yaml` and
`auxiliary.yaml`, with a fresh output path and seed. Distribution targeting for
the next additional 1,000 samples remains a separate user-approved step.

The pilot's 200 descendants have **not** been joined to training. No model metrics
or training improvement are claimed from template admission alone. The current
600-real + 1,500-synthetic training / 60-validation splits remain byte-identical.
