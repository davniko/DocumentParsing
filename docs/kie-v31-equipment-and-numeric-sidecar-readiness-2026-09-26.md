# v31/v32 equipment and cargo-token resolution; numeric-sidecar handoff

## What is already resolved

The v31 catalog
`artifacts/kie-synthesis-production/template-base/catalogs/mpci-bl-production-template-catalog1507-v31-equipment-grounding`
is an immutable derivative of v30 (commit SHA-256
`0811603dfd62e06e97f62bf63fd34b33832e2fe254bd721b40713ea134e5363c`).
In `doc_643f38c66fb77285fd27e6e4941c9cfe12f3f51953ba5a9a6fcf18ad5bec0dce`,
the second per-flexitank weight row is corrected from OCR `DFSU5710788`
to the PDF-printed and already labelled `FCIU5710788`, **only in a
synthetic-template source derivative**. The real OCR and label remain unchanged.
Both occurrences of this container ID now share one target-backed binding.
The v30 production guard rejected the source-only ID; the v31 guard and joint
cargo source contract pass. A generated replacement ID was observed in both
printed positions. The full v31 receipt and replay are under
`artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground012_equipment/`.

The `GEN` on Helvetia source
`doc_0dbc75a020661f698a938c6882159a11ea5a398c95b01f7b3d8a2ac6a648ef3d`
is a **cargo/commodity category** (*general cargo*), not container type,
ISO size/type, or transport mode. An independent
[bill of lading](https://docs.ftgs.us/applications/co/42758-otherDocs-ttclub_bill_of_lading___s00406676__002_.pdf)
prints `GEN` for both `20GP` and `40HC`, and a
[DSV bill of lading](https://www.akib.org.tr/files/downloads/2023/04/e01d5c415ad64de01d5c415ad64de01d5c41.pdf)
explicitly prints `COMMODITY TYPE: GEN (General Cargo)`. The pinned MPCI
`natureOfCargo.cargoTypeClassificationCode` enum does **not** admit UN/CEFACT
general-cargo code `12`, and the current training target has no literal `GEN`
field. Do not add it to a container label or coerce it to a different cargo
code. The current 13 training records from this lineage need no `GEN` label
edit. The v32 derivative
`artifacts/kie-synthesis-production/template-base/catalogs/mpci-bl-production-template-catalog1507-v32-cargo-token-grounding`
removes the misclassified source-only `equipment_mode` binding and leaves the
printed `GEN` literal in the OCR/template. Source/label bytes are unchanged;
the recertified template passes production source loading, exact source
round-trip, and cargo preflight. Detailed evidence is in
`artifacts/kie-training/analysis/schema-label-contract-audit-20260924/repairs/ground012_equipment/GEN-CLASSIFICATION.md`.

## Numeric sidecar: future synthesis prerequisite, not current-data mutation

Training consumes already materialized JSONL rows. The missing **full current
numeric sidecar does not invalidate those rows by itself**. A new synthesis
campaign must, however, pin a committed numeric-contract run covering every
selected template. The production loader at
`src/document_ocr/synthesis/template_compiler/complete_pipeline.py`
(`_load_numeric_contracts`) rejects a stale source, latest target, or effective
template hash, incomplete numeric binding keys, missing selected sources, and
failed numeric source replay/row ownership. No old broad run is a complete
current-v32 sidecar; the six-source v30 run only covers its six unchanged
sources.

Read-only full-catalog inventory, pinned to the v31 commit above; v32 changes
only a nonnumeric `GEN` slot, but a new sidecar must still use the v32 pin:

| Category | Sources |
| --- | ---: |
| Total | 1,507 |
| With numeric bindings (5,131 keys) | 1,051 |
| Without numeric bindings | 456 |
| Exact old pins | 6 |
| Source/target/key-compatible numeric contracts passing identity replay | 894 |
| Compatible empty contracts | 391 |
| Numeric sources requiring substantive review | 151 |
| Empty-contract rows to create/review | 65 |

The 151 numeric reviews break down into 75 without an old row, 30 with changed
binding keys, 23 with changed source, and 23 with changed target. The v31
flexitank source is one of the source-changed cases: its old numeric reason
still names the OCR-wrong ID and must be corrected before repinning. A
separate full-catalog `load_source` pass rejected 38 cases on source-label
integrity (26 equipment count/identity, 12 party evidence). These overlap the
numeric counts and must be fixed or explicitly omitted from the next selected
manifest. The old `GEN` joint-sampler rejection is resolved in v32 and was
not included in the 38.

The exact source IDs, mechanical classifications, and reproducible script
live in
`artifacts/kie-synthesis-production/analysis/numeric-sidecar-readiness-20260926/`.
Mechanical portability **is not semantic approval**: check source evidence,
units, arithmetic, and ownership before mass repinning. Then commit one
sidecar with exact selected-manifest coverage, validate through the real
production loader and varied-scenario preflight, and only then launch new
bulk synthesis. The sidecar is not published yet.

The working train JSONL currently hashes to
`127e33fda4affdcda1120b41f5144277d43367ce9c2d431819cb97361fc591a0`.
The historical `ground010_clean` training config still pins the earlier
`70a764...` hash; update the config deliberately before launching a new
training run against the current working dataset. This mismatch is separate
from numeric-sidecar readiness.
