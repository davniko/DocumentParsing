# MPCI V7 reviewed synthesis templates

This is the reusable **100-source catalog**, not a generation-run directory.
The initial admission evidence is the 200-sample expansion campaign:
[full report](../../../docs/kie-synthesis-expansion100-2026-10-07.md).

## Contents

```text
mpci-bl-v7-reviewed/
  manifest.json               # case hashes, capabilities, shared dependencies
  ownership.yaml             # 100 current-field ownership/format declarations
  auxiliary.yaml             # 100 source-only dependency/printing declarations
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

Coordinate coverage is separate from text/label acceptance. The current release
has 76.9% line coverage overall, but much weaker coverage on expanded goods
descriptions; see the follow-up analysis in the full report before scaling a
position-focused training experiment.
