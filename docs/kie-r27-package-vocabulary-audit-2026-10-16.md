# R27 package vocabulary and free-text fallback audit

## Fresh R28 audit while R27 trains — 2026-10-16

The [consolidated donor/packaging recheck](analysis/r28-packaging-decoupling-20261016/REPORT.md)
updates the earlier investigation without changing training artifacts. Defaults
remain 447 donor-bound out of 453, but independent profile data now cover 29
profiles / 27 package categories / 95 HS6 identities across three files. The
current ordinary pre-physical candidate union is 1,505, not the earlier snapshot's
1,518. The current extraction vocabulary is 48 categories, not the historical 46.

**Important primary-source clarification:** the official NAIC D23A implementation
guide explicitly supports multi-level packaging through up to five GID package
quantity/type entries. PAC is excluded from this profile, but that does not mean
hierarchy is absent. [Exact official pages, verified PDF hash and remaining
SGP/projection questions](analysis/r28-packaging-decoupling-20261016/naic-profile-and-hierarchy.md).
The current declared-accounting extraction policy remains unchanged; richer
filing support and the chosen simplified training target are distinct decisions.

The new pass also documents [real material/level evidence](analysis/r28-packaging-decoupling-20261016/package-evidence.md),
[external compatibility resources and their limits](analysis/r28-packaging-decoupling-20261016/compatibility-resources.md),
and a bounded plan for a complete independent catalogue plus factored targets.
It does not claim a universal HS-to-package registry was found, all 405 codes
were decomposed, or the R28 refactor was implemented.

## R28 scheduled redesign and bounded R27 coverage run — 2026-10-16

The user has explicitly scheduled the larger packaging/sampling redesign for
**R28, after the next training experiment**. This is not an R28 dataset
publication. The current working dataset remains R27 and retains the direct
decoder target shape. Its requested next membership is 6,900 synthetic plus
600 anonymized real records, with the same 60 validation records: add 200 UAE
package-coverage descendants and recoverably archive the exact earlier 300
Egypt-addition descendants, using their current corrected R27 bytes.

### Policy and architectural requirements for R28

**Remove observed-donor dependence, rather than renaming it or merely adding a
few independent profiles.** A source/template supplies presentation, ownership,
line/column structure, field presence and measured coordinate anchors. It must
not ordinarily limit the selectable goods, package family, material or equipment
to a shipment that happened to occur in that source or another donor. Treat a
source as generalizable unless concrete printed constraints demonstrate an
exception. Such exceptions require recorded evidence; vehicle-specific wording
is not permission to make every other template source-bound by default.

The fresh native catalog investigation confirms that this is an actual remaining
dependency, not just unfortunate wording: of 453 default template capability
records, 355 use `observed_train_bundle`, 81 `source_bundle`, ten
`source_whole_units`, one `reviewed_whole_units`, and only six `registry_form`.
The first four paths still obtain numeric/package context from observed or
source-bound shipment bundles. `registry_form` creates its own synthetic physical
observation; the local variable named `donor` in that branch is not an observed
document donor. Its independence is real but its default deployment and form
coverage are narrow. Even this path can still draw printed equipment tares from
train-fitted observations. Nested packing recipes also retain source ratios;
these must be separated from any assertion of complete donor independence.
See the [call-path and registry investigation](analysis/r27-package-vocabulary-20261016/r28-donor-sampling-investigation.md)
for exact eligibility, filters and counts.

The existing 46 flat tokens conflate different concepts. The 405-entry application
list is not 405 equally independent base package shapes and is not a complete
Cartesian product of all materials and shapes. It combines, among other things:

| Independent concept | Example | Extraction consequence |
| --- | --- | --- |
| Declared counting unit / family | set, bale, drum, carton, pallet | Preserve the declared unit and count; a set is not automatically its component-piece count. |
| Material | paper, plastic, steel, wood | An absent material is unknown, not an inferred material from the commodity. |
| Construction or condition | removable head, multi-wall, compressed | Extract only supported distinctions; generic BALE does not mean non-compressed. |
| Dimensions / platform specification | a particular pallet footprint | Do not assign a dimension-specific code from a generic pallet noun. |
| Containment hierarchy | bags on pallets | Separate hierarchy from package type; retain the declared shipment/accounting level and its same-level allocations. |
| Transport equipment | an ISO freight container | Do not confuse the container count with the number of packages inside it. |
| Filing representation | a particular Recommendation 21 application code | Project only when the required distinctions and the actual application's code edition support the mapping. |

This separation is grounded in [Recommendation 21's definitions](https://unece.org/fileadmin/DAM/cefact/recommendations/rec21/rec21rev1_ecetrd195e.pdf),
the separate [CPS packaging-hierarchy segment](https://service.unece.org/trade/untdid/d21b/trsd/trsdcps.htm),
and [GS1's distinct type, material and feature attributes](https://www.gs1.org/standards/gs1-global-data-model-attribute-implementation-guide/16).
It does not authorize replacing our declared-accounting policy with a universal
"always inner" or "always outer" rule.

The recommended R28 design to evaluate is a factored, evidence-conditioned
package representation: family, optional supported material/construction/
dimensions, and an explicit relationship only where the document actually states
multiple levels. Keep the counting unit and allocation ownership unambiguous.
Build a reviewed lossless mapping table for the actual application vocabulary;
do not mechanically split display names at commas and assume that is a verified
ontology. The model should not have to emit mutually inconsistent semantic type
and code. `UN` with `SETS` is specifically rejected, not a fallback to implement.
The exact new target contract and migration remain an R28 design decision.

### Commodity coverage and compatibility research

The pinned UKGT commodity report contains 25,852 raw hierarchy rows, 16,612
ten-digit national-code rows, **5,612 unique HS6 identities, 1,228 HS4 headings
and 96 chapters**. The commercial phrase file covers all 5,612 exactly. Therefore
the main current bottleneck is not a tiny commodity-code registry. It is the
effective sampling intersection: the current default goods candidate pools
outside chemical-DG sampling reach only 1,518 distinct HS6 identities across
453 sources, before physical candidate acceptance. The six default
`registry_form` sources reach 40 HS6
identities. These are eligibility measurements, not proof that every candidate
is physically renderable or has been sampled. National HS extensions and their
printed separators remain governed by the already implemented source-format
policy, not by six-digit-only rendering.

An HS6 identity is a classification bucket, not an individual commercial SKU
or a complete physical-form specification. The audited 7,000-synthetic baseline records
actually print 5,923 HS values across 4,291 documents, representing 1,036 unique
HS6 prefixes, 197 HS4 headings and 42 chapters. All prefixes are in the pinned
registry. Counting those observed products does not replace auditing the
sampler's reachable registry domain or its form/packaging constraints.

Research must distinguish three different assets: a classification vocabulary,
a plausible physical-form compatibility model, and normative DG packaging rules.
HS and CPC classify goods; they do not prescribe one transport package per code.
Recommendation 21 names package forms; it does not supply a global goods-to-pack
crosswalk. GS1 provides useful independent packaging attributes and product-level
structures, but the documentation alone is not an unrestricted exhaustive HS6
to maritime-package dataset. DG packing instructions are a separate, constrained
authority; do not generalize a US non-bulk packing permission into a universal
maritime approval. The [detailed registry research and limitations](analysis/r27-package-vocabulary-20261016/r28-donor-sampling-investigation.md)
records the sources actually checked rather than claiming that no other registry
can exist.

For R28, build a pinned, reviewable many-to-many **goods identity → physical
form → eligible package attributes** catalog, with provenance and explicit
conditional constraints. Then sample compatible quantities, masses, volumes
and equipment jointly, independently of document donors. Templates determine
how those chosen facts are printed, not which observed shipment can donate
them. Category-first coverage draws and commodity-first broad draws should both
use this same compatibility model. Tests must demonstrate supported draws after
removing observed donor rows, cross-layout portability, no silent fallback and
correct labels/repeats/allocations/geometry. Do not label twenty profiles as
complete donor-free coverage of the whole pipeline.

**Additional user requirement for R28:** make the full available commodity-code
registry reachable, rather than perpetuating the current source/donor-conditioned
subset. Report eligible, reachable, attempted and accepted unique HS6 coverage
against all 5,612 available identities, by chapter and physical form. Unsupported
forms or contradictory registry entries need an explicit disposition, not an
invisible filter or a claim that all codes were exercised because the file was
loaded. Full registry reachability is separate from uniform frequency or a
requirement to put every code into the immediate 200-sample R27 campaign.

Apply the same expansion audit to DG: the current assets have 2,934 HMT rows
(2,323 distinct UN identifiers), 3,480 ECICS links, and 64 reviewed flash-point
identities. The measured linked sampling support is much narrower: 386 coherent
chemical-record/HS6 pairs, 384 HMT records and 138 HS6 identities. Compare the
reachable UN/chemical/packing-group/subsidiary-hazard/flash-point combinations
with the complete usable DG registry, expose the narrowing joins and filters,
and expand compatible sampling/rendering coverage. Do not invent missing
identity links, DG facts or packing permissions to claim complete reachability;
record actual data gaps and supported reasons instead. This is an explicit R28
work item, not a request to derail the bounded R27 publication.

One concrete DG expansion path to investigate is to stop making an HS join a
prerequisite for a template that prints **no HS field**: a coherent DG registry
record can supply its own identity/class/packing-group facts and LLM wording
context without printing or inventing an HS code. Templates that do print HS
still need a reviewed compatible link. Both routes require supported physical
form/packing and the correct field-presence policy; not every HMT row is thereby
proved suitable for every maritime template. Measure how much coverage this
separation actually unlocks before adopting it.

### Bounded R27 action, not the R28 refactor

Add `PACKAGE_SET` and generic `PACKAGE_BALE` to the extraction vocabulary while
leaving the authentic 405-code application export unchanged. Their semantics are
known; exact submission-code mapping is not established by that saved export.
Use explicitly reviewed semantic extension metadata with no invented code, not
an implicit Unit/Package mapping or an invented bale compression state. Details
and the existing strict-registry coupling are in the [semantics investigation](analysis/r27-package-vocabulary-20261016/r28-package-semantics-investigation.md).

The user has explicitly confirmed this semantic-category/submission-mapping
separation. The follow-up generic-bale search found exactly two bale entries in
the saved 405 enum: compressed `BL` and non-compressed `BN`. UNECE's older
[Recommendation 21 numeric crosswalk](https://unece.org/sites/default/files/2023-10/rec21rev2_ecetrd211E.pdf)
groups both under numeric class `65`; that is evidence of a common bale family,
not an additional accepted code in our application's detailed enum. `65`, `BAL`,
`BLE` and `BALE` are absent from that saved enum. The generic extraction category
therefore remains explicitly unmapped; no compression assumption is introduced.

The requested synthesis allocation is 60 SET, 60 generic BALE, 20 RACK, and 15
each PARCEL, REEL, SLAB and SACK: 200 UAE descendants. SET is a real/synthetic
recognition gap but has **no current validation examples**; generic BALE has two
validation examples and RACK has one. This campaign must exercise multiple
compatible layouts and reviewed registry-form facts, not obtain its SET coverage
solely by repeating the one vehicle-set source. Physical plausibility envelopes
are explicit synthetic choices, not manufacturer specifications or inferred
donor ratios. Preserve already reviewed package counts and allocations when
converting the existing SET/BALE fallback labels.

Status: investigation documented; implementation, synthesis, archive publication
and final training-readiness evidence are tracked in the continuation below.
Historical certificates in the earlier sections remain evidence of their own
published bytes and must not be rewritten to claim they certify this new run.

### Execution inventory

- [x] Research, donor-path inventory, registry counts and R28 design scheduled;
  no factored-target migration or complete donor-free refactor implemented.
- [x] Settled policy records named SET and unspecified BALE extraction meanings;
  user confirms submission mapping is deferred.
- [x] [Reviewed semantic vocabulary and production support](analysis/r27-package-vocabulary-20261016/coverage200/SEMANTIC-VOCABULARY.md):
  407 entries with 405 exact original entry objects, explicit unmapped extension
  metadata, 48 reviewed renderer categories and 323 passing relevant tests.
- [x] [Independent existing-data and publisher review](analysis/r27-package-vocabulary-20261016/coverage200/independent-publisher-review.md):
  all 11,660 baseline physical rows scanned; 27 transaction/negative-control
  tests pass. Exact intended retained repairs are 17 synthetic, seven real and
  two validation targets; inputs, coordinates, quantities and other facts stay
  unchanged. The two older UAE sidecars need no corresponding target edit.
- [x] Publish and replay source/template authority changes: ten source-target
  repairs (four SET and six BALE), the aggregate SET noun versus fixed one-set
  detail ownership, 48-token template contracts and current configuration pins.
  All 41 guarded files read back exactly; 3,420 protected case/alignment files
  remain unchanged. Live replay passes all 453 sources, 1,359 configured plans
  and six additional SET alias controls (39.99 seconds; 958,624 KiB peak RSS).
  Initial post-publication catalog was `dda15099…`; the subsequent historical
  dependency closure below is complete, with final catalog `bf1573f8…`.
- [x] Generate, review, publish and geometrically validate all 200 UAE samples
  under the [exact campaign plan](analysis/r27-package-vocabulary-20261016/coverage200/PLAN.md).
- [x] Apply the guarded in-place R27 membership/label transaction, archive the
  earlier 300 Egypt rows recoverably and verify all sidecar identities/hashes.
- [x] Publish and validate the 7,500-record training YAML in the rebuilt installed
  image, using CPU/tokenizer-only readiness checks; do not launch training.

### Published R27 package refresh

R27 is now published **in place**, not as another dataset version. Current
manifest SHA-256 is
`43b2d7c486f22ee6095f35fac863c3d727fef5ee0081a50efd4b32686fe28b09`.
The [guarded publication receipt](analysis/r27-package-vocabulary-20261016/coverage200/r27-publication.json)
and independent [final root readback](analysis/r27-package-vocabulary-20261016/coverage200/root-final-readback.json)
bind the exact data. Active membership is **6,900 synthetic + 600 anonymized real
= 7,500 training records**, plus 60 validation records. The two retained UAE
sidecars are unchanged. Physical storage is 11,560 rows / 8,560 unique records,
including 7,900 unique synthetic identities and 3,000 checked mirror copies.
The directory's historical `synthetic7000` name is deliberately retained under
the user's in-place R27 rule.

All 200 new descendants import to UAE and use ten already admitted layouts and
nine reviewed independent physical-form profiles. The following counts are
**documents containing the category**, not assumed component-piece counts:

| Category | New UAE | Final synthetic training | Real training | Validation |
| --- | ---: | ---: | ---: | ---: |
| SET | 60 | 77 | 3 | 0 |
| BALE, compression unspecified | 60 | 60 | 4 | 2 |
| RACK | 20 | 20 | 0 | 1 |
| PARCEL | 15 | 21 | 0 | 0 |
| REEL | 15 | 22 | 4 | 0 |
| SLAB | 15 | 22 | 1 | 0 |
| SACK | 15 | 24 | 1 | 0 |

Explicit non-compressed BALE remains separate: 26 synthetic and one real
training document. Synthetic training now covers 47 named categories; the real
bulk-liquid example brings combined training coverage to all **48 admitted
semantic categories**. Exactly two real free-text holds remain: `BULKS`
(`6ab76190`) and `ITEMS` (`4072d876`). No unsupported subtype, missing count or
invented application code was used to eliminate them. Generic SET and BALE's
submission projection is still explicitly unresolved; the authentic 405-entry
application artifact is unchanged.

Seventeen retained synthetic, seven real and two validation targets received
only the agreed SET/BALE representation change. Their raw OCR, positioned text,
package quantities, allocation facts and all unrelated labels are unchanged.
Ten original-source labels and the affected admitted SET template's current
ownership/contract were aligned too; no original private source text was copied
into anonymized R27. Historical metrics retain their original gold versions:
the two validation package-category repairs must not be silently used to rescore
older models as if their original targets had been identical.

The [historical dependency closure](analysis/r27-package-vocabulary-20261016/coverage200/source-stage/historical-dependency-closure/REPORT.md)
updates only two active metadata files. Seven changed R27 dependency paths and
the original policy path now reference authenticated immutable before-archives
with their **original hashes retained**. The old policy's exact bytes were
recovered and verified, not replaced by today's policy under an old provenance
claim. All 3,764 dependencies verify and all 200 native plan, scenario, target,
blueprint and sample-ID signatures remain unchanged. Final catalog is
`bf1573f83c958bc2443468aeff07f68a6dfa3d1eb309fc5cd60585b037eb22f6`;
current source-authority manifest is
`5c94b6bf19acbb9cf3543a9db64573813454b28eb6f4ca7136a7f02938a9d838`.
All 233 protected R27/generation/geometry artifacts are unchanged by this
metadata-only closure. Three focused controls and lint pass; the complete
closure and native-signature comparison took 58.66 seconds / 711,848 KiB peak
RSS. No historical generation or geometry certificate was falsely repinned.

The archived [300 Egypt additions](analysis/r27-package-vocabulary-20261016/coverage200/archived-egypt300.jsonl)
are their **current corrected R27 row bytes**, SHA-256
`b57afcacbf07b4f79337d9d86dd1383b0d2008c17bd96c06c2a0a3fb53feca8e`.
They are recoverable, not destroyed. This exact requested cohort includes 100
single-container, 100 uniform-multi-container, 40 mixed-container, 40 thermal
and 20 DG descendants; the new 200 are deliberately package-focused ambient
single-container examples. Retirement reduces active synthetic source families
from 413 to 408; three of the five retired-only synthetic families remain in
real training. All 453 admitted templates remain available for future sampling.
This is a disclosed membership/distribution effect, not a template deletion.

### Complete new-sample audit and upstream closure

The [aggregate certification](analysis/r27-package-vocabulary-20261016/coverage200/certification.json),
[detailed sample audit](analysis/r27-package-vocabulary-20261016/coverage200/AUDIT.md)
and [independent root inventory](analysis/r27-package-vocabulary-20261016/coverage200/ROOT-REVIEW.md)
separate the actual review layers: native model review opened every one of the
200 complete rendered-text/target pairs; root additionally opened 35 complete
pairs across all ten layouts. Ten finite wording corrections were reviewed and
rechecked. Eleven policy/cosmetic false positives were rejected explicitly,
including deliberate goods-area consignee-contact omissions and mass/volume
values with no printed unit. Review is not permission to reverse settled policy.

Two confirmed upstream issues were resolved cleanly. Repeated party identities
are checked against full **pre-country-presentation** wording, preserving strict
name/address equality while allowing independently verified printed country
aliases. The guard benchmark improved from about 9,071 to 1,767 microseconds per
call on its fixed workload. Separately, exact country-only postal owners now
receive a country-name-only field instruction instead of the generic
address-hierarchy instruction. Whole/composite postal owners and geographic
validators are unchanged. All five original failure contexts pass the isolated
native generation probe; 366 relevant tests and lint pass. Its full-request
benchmark shows no measurable regression (642.26 / 570.58 microseconds, overlapping
maintenance noise; tracked peak 30,677 / 30,673 bytes).

The [country-only clarification report](analysis/r27-package-vocabulary-20261016/coverage200/COUNTRY-ONLY-FIELD-CLARIFICATION.md)
and [explicit validation-contract rebase](analysis/r27-package-vocabulary-20261016/coverage200/country-only-validation-rebase/closure.json)
preserve original generation history. Exactly **19** accepted wording receipts
were explicitly revalidated against the clarified contract; none was described
as a new model call. All 200 old requests and 730 exact history/proof files were
archived first. Seventy identical model-review prompts reused their existing
calls with model construction forbidden. All 200 text, target and coordinate
payloads remain byte-identical across that provenance-only transition. The old
publication manifests remain archived rather than being rewritten as history.

Independent [geometry certification](analysis/r27-package-vocabulary-20261016/coverage200/geometry/final200/REPORT.md)
replays all 200 source-transfer/reflow/page-augmentation receipts and edit
ancestry: **11,463 / 12,873 lines positioned (89.05%)**, 282 gained and zero
lost. Goods lines are 216/237 and address lines 1,136/1,242. All 553 negative
controls reject; all 18 selected detailed plotted pages were opened with no
geometry findings. There are 237 pages: 147 accepted reflows, 90 with no eligible
reflow, zero rejected. Unknown positions remain explicit, never fabricated or
used to discard text. Replay/plot generation took 12.59 seconds and 516.98 MiB
peak RSS. See the [visual gallery](analysis/r27-package-vocabulary-20261016/coverage200/geometry/final200/GALLERY.md).

The publisher additionally verifies native hash-bound shipment geography, not
the presence of a country label that the OCR never printed. Its initial
mandatory-country assumption would have rejected 48/60 valid SET descendants;
this was caught before writing R27. The [corrected proof boundary](analysis/r27-package-vocabulary-20261016/coverage200/uae-native-candidate-proof-validation.md)
checks actual candidate/source/review membership, object and byte hashes,
decoded target/text, and UAE destination/role geography. Forty-nine publisher
tests pass, including rollback and wrong/stale/missing-proof controls. Extra
proof-validation overhead is 28 ms per 200 fixture rows, a negligible bounded
publication cost. Real R27 publication took 7.38 seconds / 686,468 KiB peak RSS;
the broader independent retained-row/mirror/archive readback took 13.25 seconds
/ 790,440 KiB.

Native generation/contact/review ledger: **204 calls, $0.15862808**, including
the retained output-limit failure and repaired-source re-reviews; 792,994 input
and 79,573 output tokens, including 26,157 reasoning tokens. The isolated
country-only probe adds one call and $0.00072705. **Total: $0.15935513.** No local
GPU, model-weight loading or training was used for these checks.

### Training handoff and installed-image readiness

The new [training config](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic6900_packages_r27_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml)
is ready, SHA-256
`21b2ad8cc92e3a323631bde2bff229888f4b0678d17be08399e0ac6e239a9585`.
It retains the previous R23 run's model/revision, input-only prompt, positioned
inputs, direct compact targets, rank48/alpha32 RSLoRA, EVA512, dropout0.05,
learning rate1e-4, schedule-free AdamW, batch1/accumulation32, seeds, ten epochs,
generation cap3072, best-model loading/export and unlimited checkpoint retention.
A structural comparison found no changes outside data/task pins, names/provenance
and data-dependent schedule counts. All earlier R27 identifier/phone/country/
description/DG/privacy repairs are preserved in the selected data.

With 7,500 rows, `ceil(7500/32) * 10 = 2350` optimizer updates. Five-percent
warmup is rounded up to **118** updates. Evaluation/save interval **587** gives
four scheduled checkpoints near epochs 2.5, 5, 7.5 and 10 (steps 587, 1174, 1761,
2348); the existing separate final evaluation remains enabled. These are the
same proportional warmup/evaluation settings, not a new optimization experiment.

The actual rebuilt trainer image is
`sha256:66ede61992da7d1c70ce35e05009eee789e04d1d85107c4368add9a3280b5157`.
The [installed CPU readiness receipt](analysis/r27-package-vocabulary-20261016/coverage200/installed-cpu-readiness.json),
SHA-256 `f5f3fd6bb0d4cd032965d269dd58596add80ebfd54d6e661b4b5cd9af72285d4`,
passes the actual installed config/task/prompt/data/tokenizer and native target
limit functions. All **348 host/installed source pins match**; all 7,500 training
and 60 validation rows pass with **zero source/target truncations or exclusions**.
The container uses `runc`, no network, no GPU devices, read-only workspace/cache
and the exact cached tokenizer revision. It loads no model weights and starts
no training.

| Cohort | Records | Maximum input tokens | Maximum complete target tokens |
| --- | ---: | ---: | ---: |
| Synthetic train | 6,900 | 8,077 | 1,649 |
| Anonymized real train | 600 | 7,838 | 3,145 |
| Validation | 60 | 5,698 | 1,334 |

All fit the unchanged teacher-forced limits **19,200 / 5,500**. Three real
training targets (3,084 / 3,132 / 3,145 tokens) exceed the independent generation
cap3072 but are retained in full for training. No validation target exceeds that
cap. This length check does not claim a model can never generate a repetition
loop or reach its output limit.

The full CPU check took **37.26 seconds / 1,278,992 KiB peak RSS**. Twelve targeted
helper controls and lint pass. Two earlier audit-only launch failures remain
archived: an unnecessary synthesis/provider-module import was removed from the
training-module-origin list while retaining its installed source-byte check;
and the offline tokenizer copy explicitly disables `token_env` alongside its
local-cache path and `local_files_only=True`. The real YAML's `HF_TOKEN` policy,
production loader and dependency groups are unchanged. This credential-free
offline probe deliberately does not validate Hub authentication. See the
[execution plan and attempt boundaries](analysis/r27-package-vocabulary-20261016/coverage200/CPU-READINESS-PLAN.md).

Launch from the repository root (the explicit Docker-config environment avoids
the previously observed credential-helper failure):

```bash
mpci_docker_config="$(mktemp -d /tmp/mpci-docker-config.XXXXXX)"
DOCKER_CONFIG="$mpci_docker_config" docker --config "$mpci_docker_config" compose --profile training run --rm --build kie-trainer train \
  --config configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic6900_packages_r27_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml \
  --project-root /workspace
```

**Training has not been launched.** The larger donor-independent, full-HS/DG
reachability and factored-package R28 redesign remains scheduled work, not a
property claimed for this bounded R27 run.

## Current support and synthesis-coverage audit — 2026-10-16

This later read-only audit distinguishes application-code metadata, enabled
extraction categories, rendering capability and actual training coverage. It
does not change R27, templates, task constraints, sampling or policy. The earlier
repair history below remains evidence of its own phase, not a claim that every
application entry is enabled in the model.

The [complete machine-readable support inventory](analysis/r27-package-vocabulary-20261016/support-coverage-audit.json)
pins the unchanged R27 manifest `39e13a0d…`, raw application export, derived
registry, both task contracts, 453-template catalog, render aliases and physical
profiles. It records all 46 category/code names and per-cohort coverage, all 359
application entries not admitted by the task, and the retained fallback counts.
All eleven R27 payload hashes and the before/after manifest agree.

| Layer | Count | Meaning |
| --- | ---: | --- |
| Saved application enum and complete token/code crosswalk | 405 | One-to-one code/name/token metadata; not 405 supported sampling categories. |
| Current extraction vocabulary | 46 | Exact admitted package-category tokens, all mapped to valid saved application codes. |
| Reviewed renderer alias categories | 46 | Same enabled subset, with explicit singular/plural wording. |
| Categories in synthetic training | 44 | Current 7,000 records. |
| Categories in real training | 43 | Current 600 anonymized records. |
| Categories in combined training | 45 | Union, not the sum of the preceding two counts. |
| Categories in validation | 15 | One category, RACK, is absent from both training cohorts. |
| Categories in admitted source-template labels | 37 | 452 categorized package rows plus one SETS fallback source, across 453 templates. |
| Donor-independent registry-form profile categories | 20 | Eleven common plus nine rare; source-bundle paths provide additional sampling support. |

The 405 codes cover global transport presentations, not 405 equally common coarse
families. Materials, construction, closure/head details, pallet dimensions,
handling systems and transport form create many distinct entries. Grouping exact
English display names by the portion before the first comma gives 44 intermediate
bulk-container entries, 24 composite-packaging entries, 22 bags and 14 each for
drums, boxes and crates. These illustrative groups are not an empirical maritime
frequency distribution. The full original list is retained in the pinned registry.

### Confirmed current coverage gaps

Counts below are declared package rows; multiple categories can occur in the same
document. Historical UAE sidecars overlap current records and are not extra
independent training examples.

| Category/presentation | Synthetic | Real train | Validation | Finding |
| --- | ---: | ---: | ---: | --- |
| SETS, free text | 19 | 3 | 0 | Present in data, but no named SET category/alias. One admitted source owns all nineteen synthetic descendants. |
| RACK / RK | 0 | 0 | 1 | Enabled vocabulary and renderer, no training supervision or load profile. Do not use the validation source as a donor. |
| BULK LIQUID / VL | 0 | 1 | 0 | Enabled, but no synthetic supervision or compiled source. Real row 61 is a tanker B/L for 350 metric tonnes of palm oil, with no declared package count. Do not invent one or relabel flexitanks as unpackaged bulk. |
| PARCEL / PC | 6 | 0 | 0 | Sparse, one admitted source. |
| REEL / RL | 7 | 4 | 0 | Sparse, four admitted sources. |
| SLAB / SB | 7 | 1 | 0 | Sparse; existing descendants do not imply an admitted source or a registry-form profile. |
| SACK / SA | 10 | 1 | 0 | Sparse, one admitted source. |
| SUPER BULK BAG / 43 | 21 | 1 | 0 | Limited supervision. |
| UNIT / UN | 24 | 9 | 1 | Limited supervision. |
| FIBRE DRUM / 1G | 26 | 1 | 0 | Limited supervision. |
| NON-COMPRESSED BALE / BN | 27 | 1 | 0 | Limited supervision; the real source explicitly prints non-compressed. |

BLOCK, FIBREBOARD BOX, POWDER BULK, GRAIN BULK, FLEXIBAG, FLEXITANK and OCTABIN
each have twenty current synthetic package rows. These are thinly covered, not
missing types. No categorized current target lies outside the admitted 46-token
task vocabulary. This is a label/vocabulary check, not proof that no unlabelled
raw-text packaging modifier was ever missed.

### SET: recognition gap versus code-version compatibility

The user's requested semantic direction is clear: recognize a printed set as a
set, preserve its declared count and support synthesis, rather than treating it
as unknowable or silently converting it to constituent pieces/vehicles.

However, `SX` is absent from the **original saved application enum**, not merely
our 46-token subset. There is relevant version evidence that the previous answer
did not capture: older UNECE Recommendation 21 tables and the semantic vocabulary
list SX as Set, whereas government-mirrored later Annex lists mark that row with
`X`. UNECE's change-indicator legend defines X as marked for deletion. See the
[UNECE legend, section III.8](https://unece.org/sites/default/files/2023-10/rec21rev4_ecetrd309.pdf)
and the [government-hosted Revision 11 Annex](https://ana.gob.pa/w_ana/images/ANA_pdf/siga_portel/Rec21.pdf?csrt=1464199935874947681).
The SX rows were verified through indexed document text; direct retrieval of
the official spreadsheet/PDF was blocked. This is evidence of edition/deletion
status, not a complete downloaded latest-list conformance test or current deployed
MPCI acceptance test. Do not silently amend a pinned raw form export to claim
acceptance of SX, nor infer that SET is semantically unsupported because a wire
code is missing. SET in Recommendation 20 is separately a unit-of-count code.

The existing nineteen synthetic and three real labels need a deliberate named-SET
category contract if that policy is adopted. For the admitted `fdbc7d94` DG-vehicle
template, migrate the exact package binding paths and target together; keep the
three-set bundle and allocations unchanged. One source set contains both a car
and motorbike, so three sets must not become three vehicles. Existing source-
whole-unit sampling can retain that bundle without a new generic physical profile.
Wider set-product variability would additionally need reviewed profiles defining
the contents/measures of one declared set. Recognition support and downstream
submission compatibility must be checked separately before publication.

### How to expand synthesis correctly

The existing production controls already separate package-category selection
(`package_weights`), physical product form (`load_profiles` / `profile_weights`)
and wording (`package_mode: source|registry`). Registry-form sampling chooses
category, compatible profile and registry HS identities before quantities,
equipment and LLM wording. Its twenty profiles are not the whole sampler:
source-specific/observed bundles cover other categories, and DG/thermal goods
have their own compatibility paths.

Priorities are: (1) resolve SET semantics and application-code compatibility;
(2) add training support for RACK and appropriate bulk-liquid presentation;
(3) reinforce sparse categories across multiple eligible layouts; (4) selectively
admit additional useful form entries, such as explicit compressed bales, kits,
specific bags, jerricans or aluminium drums, when source evidence or reviewed
compatible physical profiles justify them. The 359 non-admitted entries are an
inventory, not 359 proven dataset defects or a requirement for equal-frequency
synthesis. Category-specific physical facts must not be fabricated to fill a quota.

For each new category/profile: validate the code contract, explicit render aliases,
goods/package/equipment compatibility and per-declared-unit accounting; reuse
compatible template ownership where supported, rebasing only genuinely specific
printed facts; test three descendants per selected template/profile and review
rendered text, labels, all repetitions, totals/allocations and geometry before
larger generation. A successful enum edit alone is not end-to-end coverage.

No label, template, registry, schema/configuration, input or coordinate was changed
by this support audit. The current data remain on R27; no synthesis was launched.

## Earlier package-repair and hold inventory

Date: 2026-10-16. Scope: package category coverage and source-supported resolution,
not a new inner/outer accounting or goods-description repair. The eight confident
R27 target changes were published separately in the root agent's consolidated
in-place repair. The active template/current-source authority changes are now
published and independently read back: all **413 source replays and 1,239 native
shipment plans pass**. Historical source authorities remain unchanged. The final
promotion also incorporates the separately certified phone/privacy repairs and
a narrowly grounded country-ownership correction discovered by full preflight.
No training or GPU inference was run. The package-only probes were deterministic;
the additional three country-owner wording/review descendants cost $0.00287.

**Post-expansion readback:** after both authorized groups were admitted, the
[453-source check](analysis/r27-package-vocabulary-20261016/post-expansion-453-readback.json)
again verifies every manifest-pinned source target and all 46 supported category
tokens against the independent 405-code MPCI export. Thirty-seven categories occur
in source labels; none is unsupported. The only template fallback remains the
already documented `fdbc7d94` three-sets declaration; the forty additions introduce
no new fallback family. This read-only check took 1.00 second. The 413-source
publication counts below describe the earlier package-repair phase, not the
current final template count.

## Findings

The apparent fallback coverage gap has two distinct causes: genuine underspecified
package wording, and an unnecessarily narrow 44-category task vocabulary. The
saved MPCI application vocabulary contains **405 package codes**. Every one of the
44 currently admitted task categories, every observed category in R27 and every
existing reviewed package-rendering category maps to an exact entry in that
application vocabulary. **No supported category outside the MPCI vocabulary was
found.**

The audit verifies the categorical registry's application-code set against the
separately saved raw form export, not just against its own derived token names.
This is evidence about the locally pinned MPCI form revision, not a claim about
an uninspected current external deployment.

Authorities:

- [Pinned 405-entry MPCI package registry](../artifacts/mpci-ai-schema/categorical-registry-followup/package-category-registry.json).
- [Raw application enum export](../artifacts/mpci-ai-schema/categorical-registry-followup/package-registry.raw.json).
- [Form, draft and submission fallback behavior](../artifacts/mpci-ai-schema/categorical-registry-followup/package-fallback-behavior.md).
- [Declared shipment-unit policy](kie-real-baseline-label-policy-2026-10-05.md#shipment-packaging-and-allocations-2026-10-09).

All 11,660 physical rows across the active R27 shards and retained UAE sidecars
were scanned. Sidecars overlap the active cohort; these are not 11,660 independent
training documents. All 413 admitted template targets were also scanned.

| Cohort | Records | Fallback package rows before | Confident corrections | Retained review holds |
|---|---:|---:|---:|---:|
| Synthetic training | 7,000 | 25 | 6 | 19 |
| Anonymized real training | 600 | 9 | 0 | 9 |
| Validation | 60 | 4 | 2 | 2 |
| Previous UAE sidecar | 2,500 | 0 | 0 | 0 |
| Overlapping old-UAE addition | 1,500 | 0 | 0 | 0 |
| Active training plus validation | 7,660 | **38** | **8** | **30** |

Every observed fallback occurs in one package row in one document. The exact
document IDs, row numbers, source-family identities, printed evidence, decisions
and before/after objects are in the [complete inventory](analysis/r27-package-vocabulary-20261016/inventory.json).

## Confident resolutions

| Printed type | Occurrences | Resolution | Evidence and limit |
|---|---:|---|---|
| `PARCEL` | 6 synthetic | `PACKAGE_PARCEL`, MPCI `PC` | Exact registry meaning. Missing only from the task's admitted subset, not from MPCI. |
| `RACKS` | 1 validation | `PACKAGE_RACK`, MPCI `RK` | Each of four container rows states five packages of five racks of steel casing; the same-level total is 20. Counts and allocations remain unchanged. |
| `PLASTIC PALLET` | 1 validation | `PACKAGE_PALLET`, MPCI `PX` | The declaration explicitly states `39 PALLETS` / `SAY THIRTY-NINE PALLETS`; the product block clarifies plastic construction. The registry's plastic-pallet SRS categories require dimensions not supplied here. Generic pallet is supported without inventing them. |

Validation row 21, `5222fa98`, is the rack case. Validation row 57, `45ed05eb`,
is the plastic-pallet case. Their inputs and positioned coordinates do not need
editing. Only `typeOfPackages` is replaced by `typeCategory`; package quantities,
container identities, allocations, goods text and every other target stay intact.

The six parcels belong to admitted source `c7060e22`. This source is intentionally
unchanged on its separately reviewed Egyptian country interpretation. Its package
correction must not become a pretext to reopen that independent decision.

## Genuine unresolved types: do not erase uncertainty through categorization

| Printed type | Synthetic | Real | Validation | Why no confident narrower category is available |
|---|---:|---:|---:|---|
| `SETS` | 19 | 3 | 0 | No set category exists in the pinned MPCI vocabulary. The source unit is a set, not necessarily one piece or generic package. Some descendants contain cars, but that does not authorize a universal `SETS` to vehicle/piece mapping. |
| `BALES` / `BALE(S)` | 0 | 4 | 2 | MPCI offers compressed `BL` and non-compressed `BN`, not generic bale. None of these six OCRs establishes compression. Commodity knowledge is not printed subtype evidence. |
| `BULKS` | 0 | 1 | 0 | Real row 233 declares `13 BULKS` of used refrigeration compressors, water pumps and electrical cabinets. It does not establish any particular liquid, granular, powder, scrap or unpackaged category. |
| `ITEMS` | 0 | 1 | 0 | Real row 453 prints `Total Items: 34`, but its rider does not identify the package form. A carrier's generic total heading does not establish pieces, units or packages as a narrower physical category. |

Real review-hold rows are 64, 151, 163, 211, 233, 391, 435, 453 and 457.
Validation review-hold rows are 24 and 35. All 19 synthetic holds belong to
`fdbc7d94`, the source that declares three sets of electric passenger cars.
These records remain unchanged pending a deliberate user decision; the audit does
not replace an unknown material/subtype with `PACKAGE_PACKAGE` merely to lower
the fallback count.

The existing 143-versus-144-bale source conflict in real row 391 also remains
unchanged: this pass must not manufacture a total or redistribute its printed
48/48/48 allocations while investigating the category.

The decoder is a partial extraction contract, not a complete MPCI filing. The
saved form permits supplementary free text but requires a categorical code at
submission. Consequently these holds identify unresolved submission mapping,
not permission to invent unsupported code values in extraction gold.

## Upstream changes and publication scope

The admitted-template scan finds only two fallback source families: `c7060e22`
(`PARCEL`, confident correction) and `fdbc7d94` (`SETS`, retained hold). No admitted
source carries the validation-only rack or plastic-pallet samples. Those validation
sources must not be compiled for training.

The implementation extends the reviewed package alias table with exact
`PARCEL`/`PARCELS` and `RACK`/`RACKS` forms. Alias validation now checks the explicitly
configured task subset against reviewed aliases and the pinned application codes;
it does not require every possible reviewed alias to be admitted by every frozen
historical task. Unknown configured categories, missing application codes and
mismatched code assignments still fail. This preserves old 44-category contracts
without retroactively expanding their labels or rewriting historical results.

The active task vocabulary expands from 44 to 46 with `PACKAGE_PARCEL` and
`PACKAGE_RACK`. The target model/schema and field descriptions are unchanged;
only task constraints and the constrained prompt-schema enum need new pins.
Future extraction/compilation receives the explicitly admitted vocabulary. The
existing generic fallback sampler remains available only for genuinely unresolved
printed units; no fuzzy matching or source-ID runtime special case is added.

For the parcel template, the published rebase changes its target package field and
renames the two exact `package_nouns` binding paths to `typeCategory`. Both source
occurrences, byte offsets, source text, positions and the rest of the source target
remain intact. Its contract target hash changes accordingly. The current
`country-normalization-20261016/real704` source-authority row must receive the same
package change and the catalog case must point there. The historical `real677`
authority and completed campaign outputs remain preserved.

Publication artifacts:

- [Eight exact, hash-guarded R27 target patches](analysis/r27-package-vocabulary-20261016/reviewed-patches.json).
- [Current source-authority patch and historical-preservation rule](analysis/r27-package-vocabulary-20261016/source-authority-patches.json).
- [Staged parcel case assets](analysis/r27-package-vocabulary-20261016/staged-case/).
- [Staged direct-target task constraints](analysis/r27-package-vocabulary-20261016/r27-task-constraints.json), [staged prompt schema](analysis/r27-package-vocabulary-20261016/r27-prompt-schema.json), and [staged template task constraints](analysis/r27-package-vocabulary-20261016/template-task-constraints.json).

The active publication refreshes the current authority/catalog manifests and all
29 synthesis configurations using the current registry task pin. R27 data and its
manifest are owned by the separate consolidated dataset publication. Completed
historical campaign contracts, source authorities and scores are not rewritten.

## Validation and measured cost

The [fresh validation receipt](analysis/r27-package-vocabulary-20261016/validation.json)
records three actual `ScenarioCatalog` draws from the updated parcel source with
registry DG identities (UN 1818, 2244 and 2610). All retain the named parcel category,
the source-bundle physical profile, total 2 and allocations 1/1. These are new
deterministic shipment scenarios, not paid wording generation.

Separately, three replays load the staged target/contract/historical bindings through
the real owned-blueprint and auxiliary paths, then render through the production
package count/alias machinery. All produce byte-identical source text with correct
category targets, both parcel occurrences and unchanged allocations. No coordinate
recalculation is necessary for a target-only change.

The initial targeted regression suite passed **154 tests**. The wider run passed
**439 tests in 16.00 s**, covering all reviewed aliases, named and fallback
source-bundle sampling, scenario/registry sampling, templates, physical rendering
and allocation conservation. Ruff passes.

Startup-only alias validation, 100 repetitions: previous 44-category code 0.03632 s;
updated code with the same 44-category task 0.03480 s; expanded 46-category task
0.03470 s. Existing carton-alias selection, 100,000 calls: 0.01888 s before and
0.01860 s after. These sub-millisecond differences are measurement noise, not a
claimed speedup; the unchanged hot path has no measured regression. The full fresh
registry-fit plus three-scenario probe took 5.41 s and peaked at 459,056 KiB RSS,
including fitting the existing source/registry corpus. No new per-document model
call or buffering path was introduced.

## Complete current-source promotion and preservation proof

The [guarded promotion plan](analysis/r27-package-vocabulary-20261016/current-source-promotion/plan.json)
contains exact before/after hashes and recoverable byte backups for 45 active
files. The [final readback](analysis/r27-package-vocabulary-20261016/current-source-promotion-readback.json)
proves that every promoted file equals its reviewed staged bytes, all 413 native
owned-blueprint source replays are unchanged byte-for-byte, and all 3,853 inherited
current-source dependency hashes are valid. All **1,387 protected historical and
alignment files** match their pre-promotion pins. No R27 shard was edited by this
source/catalog promotion.

Target-bearing source edits are restricted to three admitted training families:
`c7060e22`'s parcel category and `551f3a34`/`a47ae6af`'s separately printed phone
lists. The two phone families also receive the independently reviewed private
reference/address bindings, not new extraction labels. Three current source704
validation rows receive the same already-approved target-only repairs as R27:
the rack, generic pallet and four-number delivery contact. Validation sources
remain excluded from template admission. All original OCR, positioned input,
alignments, non-target source assets and unrelated case assets remain unchanged.

The existing `23a49094` source-authority override metadata is repinned to the
current authority and auxiliary files. Its source contract and required-absence
guard are preserved. The historical `real677` authority is not edited to make a
current repair pass. The parcel case now points to mutable current `real704`.

Final pinned active state:

| Artifact | SHA-256 |
|---|---|
| Catalog manifest | `365b452252dc0b5ee4db9f08cd91df2141b56150a4895638955667d5a1053b42` |
| Current real704 manifest | `73b7dcfa0c72d30ce00c06c730b524055de27540df08042c9952a55e61737b50` |
| Template task constraints, 46 package categories | `2a0370f7684c03af44a3bfbb1199be2fd9b353079a18b0a89e6f03a1b9002ea0` |

An intermediate publication attempt was rejected by exact-byte readback: the
patch emitter had produced context-free insertion hunks, which `apply_patch`
appended instead of locating. Seven files differed from the reviewed proposal;
the remaining 38 were exact. All seven were restored to the unchanged staged
bytes using contextual patches, the catalog manifest was committed last, and
the emitter now supplies context. The [failure evidence and recovery backups](analysis/r27-package-vocabulary-20261016/current-source-promotion/application-context-error.json)
remain preserved. No certification was issued for the rejected intermediate
state. Final byte verification, native source replay and full planning all pass.

## Country-owner integration gap found by full native planning

The full-catalog test deliberately exercises shipment planning and preflight,
not just source no-op replay. It found a pre-existing metadata mismatch after
country-label normalization: a party's normalized target country could be present
in the complete owned address as a valid code or alias, but the old builder linked
the country only when its full normalized name occurred literally. For example,
source `33546e11` prints `GIZA(ET)` against the existing reviewed `ETHIOPIA` country
label. This pass **does not geocode Giza or change that reviewed country label**.
It repairs the missing country-to-postal ownership link so subsequent sampled
addresses and country targets change together.

The implementation uses the already loaded, pinned country registry and the
existing postal country-span matcher. It adds only a missing country path to the
same party's complete owned postal region when that region contains an exact,
eligible witness for the existing target country. It never changes source text,
target values or geometry. Already reviewed country owners, including `c7060e22`'s
Egyptian `ARE` interpretation, are retained. Email/URL tokens, another party's
country, wrong-country words, substrings and partial address regions do not supply
the witness. This is not fuzzy country inference or a relaxed preflight.

One reviewed source, `c2180108`, prints the OCR typo `NETHERLANDSS` while its current
country target is `NETHERLANDS`. Rather than make the country registry fuzzy, this
one explicit source occurrence receives a normal reviewed ownership binding. Its
case assets, source text and targets do not change.

| Native full-catalog planning stage | Passing plans | Failing plans | Time | Peak RSS |
|---|---:|---:|---:|---:|
| Counterfactual old country metadata behavior | 1,209 | 30 across 10 sources | 35.28 s | 2,946,460 KiB |
| General exact-alias postal ownership correction | 1,236 | 3, the reviewed typo source | 37.43 s | 2,946,460 KiB |
| Final staged proposal, including typo binding | 1,239 | 0 | 35.01 s | 2,946,460 KiB |
| Published current catalog readback | **1,239** | **0** | **24.64 s** | **2,946,460 KiB** |

All runs use the same 413 sources and three native plans each, with **zero model
calls and no GPU use**. The elapsed-time differences include ordinary cache/run
variation and differing amounts of failed work; no kernel speedup is claimed.
There is no measured peak-memory increase or material planning slowdown.
The [before inventory](analysis/r27-package-vocabulary-20261016/country-owner-before-fix-native-preflight.json),
[staged inventory](analysis/r27-package-vocabulary-20261016/proposed-source-native-preflight.json)
and [published live inventory](analysis/r27-package-vocabulary-20261016/current-source-native-preflight.json)
retain every source/variant result.

## Fresh synthesis, geometry and final regression validation

The two phone/privacy families passed their separately recorded [six-descendant
certification](analysis/r27-contact-completeness-20261016/final-certification.json)
before promotion. It includes full rendered/target review, native publication,
positions and independent geometry checks: 569/654 nonempty lines have coordinates
(87.0%) across six descendants/twelve pages.

The country-owner correction also received **three fresh native descendants** of
`33546e11`, not merely patched existing samples. All source text, generated raw
texts and complete targets were inspected. Sampled country pairs include
Brunei/Lithuania, Solomon Islands/Republic of Congo and American Samoa/Peru;
printed aliases such as `BN`, `LT`, `AS`, `PE` and `PER` agree with their normalized
country labels and generated addresses. The standard postal-correction stage
resolved one initial exact-locality wording mismatch before re-render/review.
The final independent rendered review reports **zero findings**.

The [fresh native certification](analysis/r27-package-vocabulary-20261016/country-owner-fresh-three-certification.json)
pins publication, candidate review and positioned dataset receipts. Independent
geometry confirms **226/260 nonempty lines (86.9%)** with coordinates, six coherent
page transforms, two accepted reflow pages and no rejected reflows. Text, target
and coordinate-suffix preservation checks pass. Missing anchors remain explicit
rather than invented. These inspection descendants are not added to training.
All generation, standard postal correction and final review cost **$0.00286675**.
The completed probe retains its frozen pre-promotion provenance; it is not
retroactively repinned to the newly published active task contract.

The final combined package, country, ownership, publication, physical rendering,
sampling, source-allocation and promotion-guard suite passes **775 tests in
33.71 seconds** ([observed command/result receipt](analysis/r27-package-vocabulary-20261016/final-regression-validation.json)).
Ruff passes. Eight promotion-specific tests check explicit hash
repins, tightly scoped target edits, preservation of unrelated shared-YAML bytes,
and rejection of metadata/source-inventory expansion. No GPU or training run was
started, no held fallback label was guessed, and no existing historical campaign
output was rewritten.
