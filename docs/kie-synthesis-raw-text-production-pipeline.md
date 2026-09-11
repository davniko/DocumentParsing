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

There are two deliberately separate contract eras. The following 100-document configuration is
the immutable historical schema-v1/legacy-certification run:

```text
configs/synthesis/mpci_bl_raw_text_pipeline100_additional_maritime_v8_glm53.yaml
```

It remains useful for reproducible diagnosis, but it is blocked and is not a current launch
configuration. A new contract-v3 launch must use top-level `schema_version: 2`, task
`bill_of_lading_synthetic_raw_text_pipeline_v2`, and matching audit contract 3 in certification,
correction, and publication. It must start in a new run namespace; the legacy v8 certifications
cannot be promoted or silently resumed into the new authority.

The first authorized schema-v2/contract-v3 100-document calibration used the immutable v9
configuration:

```text
configs/synthesis/mpci_bl_raw_text_pipeline100_additional_maritime_v9_contract_v3_glm53.yaml
```

That run is also complete and blocked. It is a calibration record, not a configuration to rerun or
an approval to scale. The outcome is documented below.

Every inventory input is a separately pinned configuration. Certification and correction are
templates because their exact case scopes and upstream commit hashes do not exist until the prior
stage commits. The orchestrator creates deterministic YAML child configurations in
`generated-configs/`, shards each provider stage to at most 50 documents, and records every child
run and every per-case transition under `lineage/`.

## Safe operator sequence

The v8 commands below validate or replay only the historical blocked lineage. They are retained so
that its evidence remains reproducible; they are not authorization to execute another legacy run.

Validate its static schema without compiling documents or loading a provider secret:

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

The following command is provider-backed. Do not run it for v8; the historical run is complete and
blocked:

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

## Contract-v3 implementation gate

Contract 3 makes deterministic evidence the primary correctness boundary instead of treating an
empty model finding list as proof. Before a semantic audit can run, the host compiles and evaluates
an immutable invariant envelope covering all eight audit dimensions. A deterministic rejection
short-circuits provider use. When every finding has a complete compiler-owned repair set, correction
is one atomic host-authored replacement followed by a full fresh audit; incomplete or unowned
repairs are quarantined. The model remains a read-only residual semantic screen, and production
certification still requires independent low/high unanimous-clean passes.

New top-level configurations must satisfy all of these coupled requirements:

- top-level schema/task v2 and `certification.audit_contract_version: 3`;
- pinned contract-v3 invariant inputs and the semantic-auditor-v3 prompt;
- `correction.workflow.required_audit_contract_version: 3` and
  `publication.required_audit_contract_version: 3`;
- exactly one deterministic correction round and attempt; and
- complete-cohort certification before any publication.

The provider-backed v9 calibration used inventory configuration
`configs/synthesis/mpci_bl_raw_text_inventory100_additional_maritime_v65_certification_v3_production.yaml`.
It pins the same contract-v3 hybrid compiler, customs/party registry, transport-capacity policy,
and reference lineage exercised by the provider-free implementation calibration. Replay the
development calibration without loading a secret or editing candidate bytes:

```bash
uv run --frozen python tools/run_raw_text_certification_benchmark.py deterministic \
  --project-root . \
  --config configs/synthesis/mpci_bl_raw_text_certification_benchmark15_deterministic_v10.json
```

The immutable report is
[`mpci-bl-raw-text-certification-benchmark15-deterministic-v10/REPORT.md`](../artifacts/kie-synthesis/mpci-bl-raw-text-certification-benchmark15-deterministic-v10/REPORT.md).
It records 15/15 correct dispositions, all 9 defect cases caught, all 6 controls clean, all 31
reviewed issues matched with 1.0 structured-finding precision and recall across all eight
dimensions, zero candidate-byte edits, zero provider requests, and zero provider cost. This is a
development calibration over known reviewed cases, not an untouched holdout and not a provider
canary.

The same replay was also run from the production image with the repository mounted read-only and
networking disabled. A current operator should rebuild that image after any behavior-affecting
source or dependency change, verify its `/opt/document-ocr/src` hashes against the host, and repeat
the network-disabled replay before considering a paid run.

## Contract-v3 v9 100-document calibration

The authorized provider-backed v9 run completed the full inventory stage but correctly stopped at
the all-or-nothing publication gate:

- 100/100 inventory candidates reached `training_ready` after three bounded inventory rounds;
- 51 were rejected before certification provider use by deterministic contract-v3 invariants;
- of the remaining 49, 43 were rejected by the independent semantic audit and 6 were
  machine-certified;
- exhaustive manual review of all 6 machine-certified cases confirmed 4 and quarantined 2 false
  positives that retained a `maersk.com` carrier-policy reference after changing the carrier;
- no correction was authorized for the semantic/unowned findings; and
- the parent remains blocked with no publication child and no training records.

The exact parent is
[`mpci-bl-raw-text-pipeline100-additional-maritime-v9-contract-v3-glm53`](../artifacts/kie-synthesis/mpci-bl-raw-text-pipeline100-additional-maritime-v9-contract-v3-glm53/REPORT.md).
The immutable outcome analysis validates all parent/component receipts, replays host checks,
preserves four disjoint review dispositions, and emits case workbooks, reproducible CSV tables, and
35 Matplotlib/Seaborn figures:

```bash
uv run --frozen python tools/analyze_raw_text_pipeline_outcome.py \
  --project-root . \
  --pipeline-root artifacts/kie-synthesis/mpci-bl-raw-text-pipeline100-additional-maritime-v9-contract-v3-glm53 \
  --manual-audit configs/synthesis/mpci_bl_raw_text_pipeline100_additional_maritime_v9_manual_audit_v1.json \
  --output-parent artifacts/kie-synthesis \
  --run-id mpci-bl-raw-text-pipeline100-v9-contract-v3-outcome-eda-v3
```

Its immutable
[`REPORT.md`](../artifacts/kie-synthesis/mpci-bl-raw-text-pipeline100-v9-contract-v3-outcome-eda-v3/REPORT.md)
records the 4.0% manually accepted cohort yield, direct 1.6%-9.8% Wilson interval, exact
$0.505416050 lineage cost, finding taxonomy, label/raw-text exploration, and review boundaries.
Scaling remains blocked. The next admissible engineering step is to add provider-free retirement
of carrier-owned policy URLs/references, prove it on positive and negative regression cases, and
then run a separately authorized fresh canary before considering a larger cohort.

## Historical v8 calibration record

Pipeline certification is a screening result, not sufficient evidence for a larger launch. The
legacy v8 100-document calibration ended with 87 machine-certified and 13 pipeline-quarantined
cases, but a pinned manual review found false positives in the machine-certified set. Consequently,
v8 remains blocked, has no publication child, and must not be used as a training-data source or as
approval to scale the run. The later contract-v3 calibration does not retroactively certify any v8
candidate.

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
neither it nor `confirmed/manifest.json` is a training publication. The observed blind-spot classes
are now represented in the contract-v3 compiler and invariant audit. Scaling is nevertheless gated
on a fresh bounded provider canary because the deterministic development benchmark does not prove
residual semantic-auditor behavior on unseen production cases.

## Published artifacts

The parent run contains the original top-level and inventory configs, generated child configs,
component-run references, complete per-case histories, transaction provenance, a summary, and a
human-readable report. The publication child contains the canonical schema-v5 training JSONL and
lineage JSONL. A successful parent summary must report the configured document count as both
`certifiedDocuments` and the publication count, with `trainingRecordsPublished: true`.

The earlier standalone inventory and analysis tools remain useful for diagnostics. They are not a
publication path and cannot bypass independent certification or the complete-cohort gate.
