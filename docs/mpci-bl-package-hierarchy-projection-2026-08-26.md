# MPCI B/L package-hierarchy training projection

Date: 2026-08-26

## Boundary

The immutable extraction label retains every OCR-grounded package fact. That is the correct
provenance representation, but it is not automatically the best target for a 270M KIE model. The
MPCI form supports repeated package rows under a goods item, yet the researched form contract has
no parent/child field that distinguishes an outer pallet/skid from an inner carton, bag, drum, or
other direct goods package.

The new projection therefore creates a separate dataset. It never mutates the 1,157-record source.
It uses exact, pinned role aliases after a narrowly defined normalization:

- pallet/skid/explicit container-abbreviation facts are outer transport metadata;
- generic `PACKAGE`/`PKG` facts are aggregate metadata when a more specific fact is present; and
- every other printed type is a possible direct-goods package, not an inferred ontology class.

All source-ordered direct-goods package facts remain in the training target. This includes multiple
legitimate direct types under the same outer pallet level, such as drums and tinplate containers.
Their outer/generic companions and the complete original targets are retained in
`package-metadata.jsonl`. If an allocation referred to a removed level, the projection keeps only
container membership; it does not transfer an outer quantity to a retained package.

If a multi-package group contains an untyped quantity, the whole document is held unless an exact,
source-pinned review establishes its semantic role. A reviewed fact remains untyped—the override
may retain its role, but it cannot invent a package type. The only reviewed override is the printed
quantity `1` row-linked to used semi-trailer cargo beside `27 PCS` spare parts. `SEMI TRAILER` is
the cargo description, while `UNPACKED AND UNPROTECTED` is a handling statement, so neither is
used as the missing type.

## Reproduce

```bash
TMPDIR=/tmp UV_CACHE_DIR=/tmp/documentparsing-uv-cache \
uv run --frozen python -m document_ocr.training.package_projection \
  --config configs/transforms/mpci_bl_combined1157_task_facing_packages_v2.yaml
```

The output manifest is published last under
`artifacts/kie-training/datasets/mpci-bl-combined1157-task-facing-packages-v2/`.

## Published real-data audit

The exact counts, manifest hashes, task-constraints identity, and benchmark are recorded in the
immutable generated `summary.json`, `manifest.json`, and `REPORT.md` beside the projected dataset.
