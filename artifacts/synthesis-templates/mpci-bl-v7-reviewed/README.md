# MPCI V7 reviewed synthesis templates

This is the reusable **360-source catalog**, not a generation-run directory.

## Mixed ambient equipment admission — 2026-10-12

The catalog now records 85 sources certified for **both uniform and mixed GP
size/height inventories**. All 85 produced two ordinary production descendants
(170 total), one per mode, passing complete semantic review, exact authority
replay, independent printed-ID/type/allocation ownership and geometry checks.
`manifest.json.mixedEquipmentIntegration` pins that publication, audit,
current sampling registries and exact admitted source list.

One shared-scalar source remains uniform-only; six sources lacking printed
per-ID type ownership are not admitted to mixed sampling. No per-ID categories
or geometry are guessed. Source container cardinality is unchanged, and
thermal/DG/mixed-family equipment is outside this ambient-GP capability.

See the [production integration report](../../../docs/kie-mixed-equipment-production-integration-2026-10-12.md)
and [paired production configuration](../../../configs/synthesis/mpci_bl_curated_v7_mixed_inventory_validation.yaml).
Mixed mode is opt-in through `sampling.equipment_inventory`; ordinary default
uniform behavior remains supported. The 170 inspection descendants were not
merged into training, and the planned 2,500-document campaign was not launched.

The dated 2026-10-11 integration metadata below is historical evidence, not a
current alias/profile hash authority. Use the latest mixed-equipment admission
pins when preparing a new campaign.

## Current 360-source pool — 2026-10-10

Three additive cohorts of 20 sources are now admitted. Their 120 actual pilot
descendants passed individual source/render/label inspection, independent
semantic review, production publication checks and coordinate validation before
admission. All 360 installed source/target contracts replay exactly; all 120
pilot candidates also replay against the installed catalog. Admission preserves
the original 300 case assets and source declarations. These pilot samples have
**not** been added to the active 3,100-record training dataset.

Admission receipts, in committed order:

- [300 → 320: cohort 03](../../kie-synthesis-production/curated-v7-template-expansion360-v1/batch03/audit/catalog-admission.json).
- [320 → 340: cohort 02](../../kie-synthesis-production/curated-v7-template-expansion360-v1/batch02/audit/catalog-admission.json).
- [340 → 360: cohort 01](../../kie-synthesis-production/curated-v7-template-expansion360-v1/batch01/audit/catalog-admission.json).
- [Final source/candidate replay and preservation proof](../../kie-synthesis-production/curated-v7-template-expansion360-v1/audit/final-registry360-validation.json).
- [Expansion report](../../../docs/kie-synthesis-expansion360-results-2026-10-10.md).

`manifest.json.cases[documentId].sourceDataset` records the authority for each
new case; where absent, use `manifest.json.dependencies.sourceDataset.path`.
The original 300 sources and cohorts 02/03 use the reviewed R16 real dataset.
Cohort 01 uses a frozen isolated source-authority snapshot with three explicitly
reviewed target corrections. The original real dataset remains unchanged. A
campaign must use its cases' designated authority; inspection snapshots are not
a fallback around dependency checks.

The 300-source recipes and initial 100-source relocation notes below describe
earlier cohorts. They do not automatically select the expanded 360-source pool.

## Historical 300-source maintenance and initial admission

The initial admission evidence is the 200-sample expansion campaign:
[full report](../../../docs/kie-synthesis-expansion100-2026-10-07.md).

## Current party/date/route maintenance — 2026-10-10

The [published repair](../../../docs/kie-party-date-route-repair-2026-10-10.md)
updates 44 source cases and their ownership/dependency declarations. For A acting
on behalf of B, targets select primary printed A with only A-owned postal/contact
data; B is separate audit/source-only generation context unless another printed
role independently supports it. Genuine C/O recipients remain postal routing.
Every repeated occurrence of a declared related identity must share its mutable
owner; an uncovered repeat is rejected before paid generation.

Numeric date labels use visible format evidence, otherwise DD-MM; numeric order
ambiguity does not justify dropping a supported date. Three source capabilities
no longer synthesize delivery labels whose OCR occurrence was omitted. Source
text/positions remain exact; all 300 current source/target replays pass.

The current 300-source recipes are
[coverage1000](../../../configs/synthesis/mpci_bl_curated_v7_coverage1000.yaml)
and [template-expansion300](../../../configs/synthesis/mpci_bl_curated_v7_template_expansion300.yaml).
They use the repaired R16 source authority, current declarations and repinned
task constraints. Older recipes/campaign receipts below remain historical
evidence; they are not an instruction to bypass current dependency checks.

Actual fresh synthesis tested ten policy-bearing sources, with complete rendered
reviews, direct inspection and production coordinate validation. See the
[hash-bound inventory](../../../docs/analysis/party-date-route-repair-20261010/UPSTREAM-INVENTORY.md).
The current mixed dataset is also repaired; existing completed training-run
snapshots are deliberately unchanged.

## Current field-boundary maintenance — 2026-10-10

The catalog includes the source-proven package-heading, repeated-total,
legal-clause, description-boundary and duplicate-country repairs, plus mutable
identity owners for formerly fixed composite names. Optional identity-kind
certificates distinguish people, organizations, and organization/proprietor
composites. VAT spellings share a draw only through an explicitly certified
`canonical_identity`; independent identifiers remain independent.

The generation API now exposes semantic field aliases, retains existing internal
keys, and gives each requested field a concise ownership description. Final text
review receives the certified identity components as well as rendered OCR and
targets. Existing postal/accounting holds and bounded follow-ups remain active;
uncertain values are not silently stripped or published.

The two corrected source targets are mirrored into the canonical real dataset.
`manifest.json` pins the current case files, shared declarations and source
dataset; historical campaign receipts are retained as history. The running mixed
training dataset is not rewritten by catalog maintenance. Full details and the
repaired sibling dataset are in the [repair report](../../../docs/kie-synthesis-field-boundary-repair-2026-10-10.md).

## Contents

```text
mpci-bl-v7-reviewed/
  manifest.json               # case hashes, capabilities, shared dependencies
  ownership.yaml             # 360 current-field ownership/format declarations
  auxiliary.yaml             # 360 source-only dependency/printing declarations
  selection-history.yaml     # admission and rejected-candidate history
  position-calibration.json   # hash-pinned source spacing/density priors
  cases/
    doc_<source hash>/
      contract.json          # current source/target rebinding contract
      template.json          # compiled source layout/bindings used by rendering
      source.txt             # exact current real OCR snapshot
      source-positioned.txt  # corresponding measured-position input snapshot
      target.json            # exact current real target snapshot
      alignment.json         # measured source-line/region correspondence
```

At the initial 100-source relocation, `contract.json` and `template.json` were
**moved**, not regenerated. The shared
declarations were relocated and restricted to the 100 admitted IDs; every active
entry remains semantically identical. Thirteen excluded candidates remain in the
original campaign with their review evidence and are not members of this catalog.
The original unfiltered declarations are preserved in that campaign's
`audit/template-relocation/`.

## Runtime entry point and dependencies

The original 100-source executable recipe remains
[`configs/synthesis/mpci_bl_curated_v7_expansion100.yaml`](../../../configs/synthesis/mpci_bl_curated_v7_expansion100.yaml).
Its `source_contracts` and `historical_catalog` both point to `cases/`; its
`ownership` and `auxiliary` paths point here. There is no fallback to old
campaign-local template directories. Its source capabilities/topologies are
also indexed per source in `manifest.json`, for reuse and comparison.

The completed earlier 200-source recipe is
[`mpci_bl_curated_v7_expansion1000.yaml`](../../../configs/synthesis/mpci_bl_curated_v7_expansion1000.yaml):
three new descendants per original source and seven per newly admitted source,
with a per-call batch limit of two. Its new seed/output preserve all existing campaigns. See the
[run report and validation](../../../docs/kie-synthesis-expansion1000-2026-10-08.md),
including reusable ownership/auxiliary fixes, explicit equipment rendering and
full-text reviews of all 1,000 accepted descendants. The resulting training
mixture has 600 real and 1,500 synthetic records. The subsequent
[old500 back-check and upstream repair](../../../docs/kie-synthesis-followup-integrity-2026-10-08.md)
updated the repeated exporter-ID binding and repaired the affected current
publications in place, retaining exact pre-edit archives. Actual printed private
and public gross totals now constrain generated wording. Real source snapshots
and all real training/validation records remain unchanged.

The catalog is **not a standalone replacement for the application or shared
data dependencies**. `manifest.json.dependencies` records exact paths and hashes:

- the current 600-train / 60-validation real dataset; runtime uses it to check
  source identity and fit train-only physical, equipment and vessel support;
- reduced-V7 task constraints;
- the common country, location, port, commodity and DG registries;
- Paddle page/region geometry used for coordinate-envelope checks.

The current 100-source campaign additionally enables joint elastic page reflow.
`position-calibration.json` contains the measured spacing range, clearance and
density floor, with hashes of the 80 fitting-source alignments and the source
dataset manifest. These priors are shared across samples, not refitted to each
generated description. The 20 diagnostic sources were excluded from fitting;
they are not an untouched semantic evaluation set.

The campaign pins a DejaVu Sans font file by path and SHA-256 for width estimation.
The font must be present at the configured location (or the path explicitly
changed to the identical file). There is no silent font substitution. The font
is only a geometric surrogate, not a claim to reproduce the source PDF font.
See [production reflow and audit](../../../docs/kie-synthesis-position-reflow-investigation-2026-10-07.md#production-integration-and-200-sample-regeneration).

These shared dependencies are not copied into every template. Per-case source
views are immutable inspection snapshots; the configured real dataset remains
the runtime source authority. A changed source dataset must pass the existing
source/target/hash checks; snapshots are not used as a silent fallback.

Generated candidates, LLM wording/contact caches, billed-call receipts,
adjudications, published datasets and galleries stay in
`artifacts/kie-synthesis-production/curated-v7-expansion100-v1/`. They are not
template assets and do not belong in new generation campaigns. Cached wording
is needed to replay those exact old outputs, not to create new variations.

## Adding compatible templates

### Party instruction policy and sampling (2026-10-08)

The active source labels and all 500 published descendants now always include
`negotiability`: the actual consignee's order instruction means `negotiable`;
an OCR-readable named consignee without it means `non_negotiable`; unavailable
consignee instruction means explicit `null`. A notify-only repeated identity
does not supply the missing consignee occurrence. Conditional form
captions and copy/document titles do not establish an order instruction.
Every emitted notify entry has `sameAs`: the printed referenced role, or `null`
for an independently printed party. Matching company details alone do not
establish a reference. A missing notify block stays absent, not an empty party.

Sampling preserves these source instructions. Mutable company/address regions
beside `TO ORDER OF` are still generated normally; a bare order instruction has
no company region to invent. Instructions outside owned mutable spans stay fixed.
Source/descendant policy agreement is checked during rendering and publication.

To control document proportions, add an optional top-level policy to a **new
campaign configuration**, alongside `variants_per_source`:

```yaml
variants_per_source: 2
template_sampling:
  samples: 1000
  negotiable_fraction: 0.20
  notify_reference_fraction: 0.25
```

This allocates exactly 1,000 documents, 200 negotiable and 250 with at least one
notify reference. Integer quotas round half-up. Both fractions are optional;
unspecified dimensions follow available source diversity. With both supplied,
the feasible joint distribution nearest independence is selected. Quotas are
spread over eligible families in deterministic seed order. Impossible requested
combinations fail before generation. It never manufactures order wording or
notify references to satisfy a quota.

Unknown sources remain unknown in their descendants. They may share the
non-affirmative sampling pool, but are never relabeled non-negotiable to meet
a quota. `negotiable_fraction` measures affirmative negotiables across the
entire requested sample count, not only sources with known instructions.

Without `template_sampling`, every selected source retains the previous fixed
`variants_per_source` count. With it, that value caps shipments per generation,
contact and review call; it no longer limits a source's total allocation.
Use a new output directory and seed; do not change the quota of a published run.

The original 100-family pool had **one negotiable family**. Its existing five
descendants remain negotiable. Increasing its quota increases prevalence, not
layout diversity; admit more negotiable sources before expecting broad template
coverage. Existing campaign distributions were not changed by this migration.

Migration receipts, exact before copies, dataset/pipeline tests and full replay:
[party-policy audit](../../../docs/analysis/party-instruction-policy-20261008/REPORT.md).

### Admission steps

1. Prepare and review a candidate outside this admitted catalog. Establish the
   complete source/target contract, owned spans, auxiliary dependencies and
   sampling capabilities/topology.
2. Validate baseline replay, sampled physical accounting, generated text/labels
   and coordinate behavior before admission. Document known coordinate gaps.
3. Add a distinct `cases/doc_<id>/` directory and its exact source snapshots.
   Add that ID's declarations to both shared YAML files and its capability to
   the chosen campaign configuration; select the ID explicitly.
4. Refresh the hash inventory and capability metadata in `manifest.json`.
   Run exact replay and source-split checks. Do not mark a draft as admitted
   merely because its JSON compiles.
5. Use a new campaign output directory and seed for a new synthesis batch.
   Templates are shared; generated outputs and their reviews are run-specific.

## Relocation validation

All **200 existing candidates** were recomputed before and after relocation.
Entire candidate objects—including text, labels, sampled facts and edit
proofs—are identical. No API calls were made. Real train/validation and published
datasets are unchanged. The hash inventory covers all 600 per-case files.

Before/after replay: **23.87 / 23.64 seconds**, peak RSS **432.8 / 434.4 MiB**.
No renderer algorithm changed. Exact movement and validation receipts are in
`artifacts/kie-synthesis-production/curated-v7-expansion100-v1/audit/template-relocation/`.

Coordinate coverage is separate from text/label acceptance. The initial
anchor-transfer release had 76.9% line coverage; joint reflow subsequently
improved it. Use each campaign's current positional manifest and audit rather
than treating that historical coverage as the present implementation.

## Second 100-source admission — 2026-10-08

At this admission the catalog reached 200 sources. The additional 100 produced 200
reviewed descendants with audited positions. They add 11 negotiable families,
bringing the pool to 12, and preserve 29 explicit `sameAs: consignee` families.
All new sources are from the current real training split. Validation IDs, OCR
duplicates and shared B/L identifiers were excluded. Existing case files and
old ownership/auxiliary entries are unchanged.

See [the admission report](../../../docs/kie-synthesis-expansion200-2026-10-08.md)
for content decisions, sampling stress tests, coordinate gaps, cost and replay
evidence. The new campaign's artifacts are in
`artifacts/kie-synthesis-production/curated-v7-expansion200-v1/`.
Its original configuration intentionally retains its staging paths to preserve
the immutable publication receipt. For new campaigns, use this shared catalog's
`cases/`, `ownership.yaml`, `auxiliary.yaml`, and the per-case capabilities in
`manifest.json`. Replay through these shared assets reproduced all 700 published
samples exactly (the previous 500 plus these 200).

Per-source OCR/label snapshots remain local ignored data, as before. The compact
manifest, shared declarations, calibration and this guide are versioned; keeping
those files alone does not replace preserving/transferring the `cases/` assets.

## Third 100-source admission — 2026-10-09

The catalog now contains **300 admitted sources**. All 100 new templates passed
current source/target ownership checks and ten host-only stress draws each, then
generated two complete descendants each. Every final rendered OCR/target pair
has a current model review and explicit adjudication where needed; **200/200
published samples have zero unresolved findings**. Existing description-block,
declared-package/allocation, casing and party-instruction policies are retained.

All 200 final geometry receipts passed independent exact text/label preservation,
source occurrence, line/page, normalized-coordinate and transform checks. All
35 diagnostic PNGs across 20 documents were actually inspected. Four optional
reflow proposals were declined with explicit reasons and full preservation of
prior anchors, not partially applied. Overall positional coverage is 84.86%,
goods-description coverage 78.32%; unsupported positions remain explicit unknowns.

See [the complete results and admission report](../../../docs/kie-synthesis-expansion300-results-2026-10-09.md)
and the manifest's latest `admissions` event. New source snapshots live in
`cases/` here; the 200 descendants, call receipts, reviews and inspection galleries
remain in `artifacts/kie-synthesis-production/curated-v7-template-expansion300-v1/`.
The isolated two-descendant recipe retains its staging paths for exact historical
replay. A new production recipe should select sources and capabilities from this
manifest and use the shared `cases/`, `ownership.yaml` and `auxiliary.yaml` paths,
with a fresh output directory and seed.

Admission revalidated all 1,500 existing synthetic inputs/targets, 2,160 current
schema/position-preservation records and 1,000 old-template plans. All 1,200 old
case assets and old source declarations remain unchanged. The current training
mixture remains **600 real + 1,500 synthetic / 60 validation**; these 200 pilot
descendants were not added to it. The final suite passes 494 tests. No training
or GPU workload was launched by this admission.

## Multi-contact ownership correction — 2026-10-10

The shared ownership inventory now covers every name in the existing combined
`contactName` labels for sources `147a2626`, `0ba3ad4f` and `85b6388a`. These are
source occurrence/field routing corrections, not new labels or changed source
text. The latter two reuse historical notify-contact roles and printed
continuation markers; the explicitly reviewed `Y\nEHIA FARID` occurrence is one
name. Current synthesis routes each contact component to its own owned slot
instead of repeating the complete semicolon list in every slot. Missing
component ownership fails source readiness before paid generation.

All 300 source blueprints and 1,000 exact host plans pass after this correction.
The ownership/manifest before-images and exact old/new hashes are preserved in
`docs/analysis/synthesis-next1000-production-20261009/contact-ownership/`.
No case assets, source labels, source OCR, auxiliary declarations or existing
dataset files were changed. The old 1,500 dataset has 28 descendants from four
other families requiring only contact-slot input deduplication; those repairs
are staged separately for the new combined dataset, preserving the old copy.
