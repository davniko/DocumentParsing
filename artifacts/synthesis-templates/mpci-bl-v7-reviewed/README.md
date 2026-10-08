# MPCI V7 reviewed synthesis templates

This is the reusable **200-source catalog**, not a generation-run directory.
The initial admission evidence is the 200-sample expansion campaign:
[full report](../../../docs/kie-synthesis-expansion100-2026-10-07.md).

## Contents

```text
mpci-bl-v7-reviewed/
  manifest.json               # case hashes, capabilities, shared dependencies
  ownership.yaml             # 200 current-field ownership/format declarations
  auxiliary.yaml             # 200 source-only dependency/printing declarations
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

`contract.json` and `template.json` were **moved**, not regenerated. The shared
declarations were relocated and restricted to the 100 admitted IDs; every active
entry remains semantically identical. Thirteen excluded candidates remain in the
original campaign with their review evidence and are not members of this catalog.
The original unfiltered declarations are preserved in that campaign's
`audit/template-relocation/`.

## Runtime entry point and dependencies

The current executable recipe remains
[`configs/synthesis/mpci_bl_curated_v7_expansion100.yaml`](../../../configs/synthesis/mpci_bl_curated_v7_expansion100.yaml).
Its `source_contracts` and `historical_catalog` both point to `cases/`; its
`ownership` and `auxiliary` paths point here. There is no fallback to old
campaign-local template directories. Its source capabilities/topologies are
also indexed per source in `manifest.json`, for reuse and comparison.

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

The catalog now contains 200 admitted sources. The additional 100 produced 200
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
