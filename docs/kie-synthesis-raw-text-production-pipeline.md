# Certified synthetic raw-text production pipeline

The terminal bill-of-lading synthesis path is one fail-closed workflow:

```text
inventory renderer -> independent certification -> exact-evidence correction
                   -> fresh certification -> complete-cohort publication
```

The production entry point is `document-kie-synthesis run-raw-text-pipeline`. It composes the
existing stage runners; it does not duplicate their renderer, audit, correction, or publication
logic. Every component run is independently staged and committed. Every handoff pins both the
upstream commit receipt and transaction SHA-256.

The current 100-document configuration is:

```text
configs/synthesis/mpci_bl_raw_text_pipeline100_additional_maritime_v8_glm53.yaml
```

Its inventory input is a separately pinned configuration. Certification and correction are
templates because their exact case scopes and upstream commit hashes do not exist until the prior
stage commits. The orchestrator creates deterministic YAML child configurations in
`generated-configs/`, shards each provider stage to at most 50 documents, and records every child
run and every per-case transition under `lineage/`.

## Safe operator sequence

Validate the static schema without compiling documents or loading a provider secret:

```bash
uv run --frozen document-kie-synthesis validate-raw-text-pipeline-config \
  --config configs/synthesis/mpci_bl_raw_text_pipeline100_additional_maritime_v8_glm53.yaml \
  --project-root .
```

Validate every immutable parent, component run, candidate transition, and lineage edge without
loading a provider secret or making a provider request:

```bash
uv run --frozen document-kie-synthesis preflight-raw-text-pipeline \
  --config configs/synthesis/mpci_bl_raw_text_pipeline100_additional_maritime_v8_glm53.yaml \
  --project-root .
```

For a fresh or inventory-resume run, preflight also compiles the complete new inventory scope. For
a post-inventory continuation such as v8, it replays the committed pipeline state and reports zero
compiled inventory documents. In both cases its result states `providerSecretsLoaded: false` and
`providerRequests: 0`.

Only the following command is provider-backed:

```bash
uv run --frozen document-kie-synthesis run-raw-text-pipeline \
  --config configs/synthesis/mpci_bl_raw_text_pipeline100_additional_maritime_v8_glm53.yaml \
  --project-root .
```

## Completion and restart contract

- Inventory retries contain only cases that did not become `training_ready` in an earlier round.
- A provider failure or structurally rejected audit is retried only within the configured bound.
- Transient correction failures retry the same audited candidate only up to
  `max_attempts_per_round`; they do not consume a successful correction round.
- An actionable audit may enter correction; the correction can change only cited physical lines.
- A correction candidate is never accepted directly. It must pass a new, read-only certification
  against `source-contract.json`.
- A repeated candidate hash terminates that case instead of cycling.
- Publication runs only when the certified-source union is exactly the original unique cohort.
- An exhausted case is quarantined while other independent cases continue within their own bounds.
  If any case remains exhausted, the parent commits the exact certified and blocked categories but
  publishes no training records and is never labeled complete.
- A new top-level run may continue an inventory-blocked parent through
  `inventory_resume_run`. The parent commit and transaction are pinned, every committed child and
  per-case result is revalidated, behavior-sensitive implementation hashes must match, and the new
  inventory config must name exactly the unresolved cases in original cohort order. Ready
  candidates are reused byte-for-byte; certification still covers the complete cohort.
- A new top-level run may continue a post-inventory blocked parent through
  `pipeline_resume_run`. The continuation replays every inventory, certification, and correction
  result against its committed artifacts and input hashes. It carries forward certified cases,
  accepted correction-round counts, candidate hashes, and unresolved stage ownership; only
  unresolved cases may receive new requests. Successful corrections are always recertified, and
  chained continuations must preserve the prior case history as an exact prefix.
- `inventory_resume_run` and `pipeline_resume_run` are mutually exclusive so that one run has one
  unambiguous continuation authority.
- Inventory and certification already checkpoint each paid case. Correction checkpoints embed and
  deterministically replay the exact output, stage receipts, request contract, host audit, and
  input identity before a resumed run can reuse it.

Re-running the same top-level command reuses matching committed child runs and valid per-case
checkpoints. Changed configuration, prompts, input commits, or behavior-affecting implementation
hashes change the transaction identity and are rejected under the old run name.

## Calibration quality gate

Pipeline certification is a screening result, not sufficient evidence for a larger launch. The v8
100-document calibration ended with 87 machine-certified and 13 pipeline-quarantined cases, but a
pinned manual review found false positives in the machine-certified set. Consequently, v8 remains
blocked, has no publication child, and must not be used as a training-data source or as approval to
scale the run.

The immutable outcome analyzer validates the parent and all referenced component receipts, replays
the current deterministic host audit for every candidate, verifies the exact deterministic manual
sample, and partitions cases without publishing training records:

```bash
uv run --frozen python tools/analyze_raw_text_pipeline_outcome.py \
  --project-root . \
  --pipeline-root artifacts/kie-synthesis/mpci-bl-raw-text-pipeline100-additional-maritime-v8-glm53 \
  --manual-audit configs/synthesis/mpci_bl_raw_text_pipeline100_additional_maritime_v8_manual_audit_v1.json \
  --output-parent artifacts/kie-synthesis \
  --run-id mpci-bl-raw-text-pipeline100-additional-maritime-v8-quality-audit-v2
```

Its four disjoint dispositions are `confirmed`, `manual-quarantine`, `provisional`, and
`pipeline-quarantine`. `machine-certified/manifest.json` is explicitly not an acceptance manifest;
neither it nor `confirmed/manifest.json` is a training publication. A larger provider-backed run is
gated on resolving the observed certification blind spots and repeating this calibration.

## Published artifacts

The parent run contains the original top-level and inventory configs, generated child configs,
component-run references, complete per-case histories, transaction provenance, a summary, and a
human-readable report. The publication child contains the canonical schema-v5 training JSONL and
lineage JSONL. A successful parent summary must report the configured document count as both
`certifiedDocuments` and the publication count, with `trainingRecordsPublished: true`.

The earlier standalone inventory and analysis tools remain useful for diagnostics. They are not a
publication path and cannot bypass independent certification or the complete-cohort gate.
