# MPCI B/L next-training preparation

Date: 2026-08-26

## Outcome

The next training input is one immutable, unsplit, current-schema JSONL containing 1,157
OCR-conditioned Bill-of-Lading/Sea-Waybill labels. Runtime training configuration now performs a
deterministic 1,057/100 train/validation partition from seed 424 before tokenization or model load.
Outer/generic package hierarchy remains available in sidecars, while only task-facing package
facts appear in the decoder target. Readable package category tokens are used only when an exact,
reviewed registry decision exists.

Training config:

`configs/training/t5gemma2_270m_lora.mpci_bl_combined1157_task_facing.yaml`

## Additional 50-document cohort

The completed 50-document run initially contained 29 accepted, 19 held, and two excluded
documents. Every held document was inspected against page-ordered raw OCR and, where needed, the
PDF/raster only for layout relationships. Narrow deterministic grounding defects were repaired and
regression-tested rather than bypassed. The final cohort contains:

| Outcome | Documents |
|---|---:|
| Promoted | 48 |
| Excluded | 2 |
| Still held | 0 |

The exclusions are evidence-based: one is a non-maritime accompanying document and the other is a
nine-page file containing three independent transport-document units. The finalized cohort is at:

`artifacts/kie-labels/pydantic-ai/consolidated/mpci-bl-relation-single-source-v4-next50-adjudicated-v1`

It was merged with the current labeled corpus at:

`artifacts/kie-labels/pydantic-ai/consolidated/mpci-bl-relation-single-source-v4-main730-description-clean-v1`

That selection manifest still records 35 historical held items and 21 exclusions for provenance;
only its 674 validated records are carried into training. They were combined with 483 aligned
legacy records to publish 1,157 current-policy records before the package projection.

## Paired comparison of the two completed 487-ish-document experiments

The exact runs are:

- minimal JSON/schema baseline: `t5gemma2-270m-lora-mpci-bl-combined487-v2`;
- semantic/relation-explicit experiment: `t5gemma2-270m-lora-mpci-bl-relation-explicit-v3-v1`.

Both evaluate the same 60 document IDs with byte-identical raw OCR. The baseline has 427 training
documents; the relation run has 423 after four deterministic exclusions. This is therefore a
strong paired validation comparison, but not a pure one-factor ablation: prompt semantics, target
topology, four training records, sequence limits, and micro-batch size changed together.

### Evaluation progression

| Epoch | Minimal F1 | Relation F1 | Delta | Relation-link F1 | Category F1 |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.7447 | 0.6791 | -0.0656 | 0.5211 | 0.6618 |
| 10 | 0.7334 | 0.7734 | +0.0400 | 0.6727 | 0.6963 |
| 15 | 0.7649 | 0.7798 | +0.0149 | 0.7163 | 0.7194 |
| 20 | 0.7909 | 0.7953 | +0.0044 | 0.7028 | 0.7534 |
| 25 | 0.7940 | 0.7964 | +0.0025 | 0.7057 | 0.7397 |

### Final generated evaluation

| Metric | Minimal | Relation | Delta |
|---|---:|---:|---:|
| Field/value precision | 0.8551 | 0.8588 | +0.0037 |
| Field/value recall | 0.7399 | 0.7425 | +0.0026 |
| Field/value F1 | 0.7934 | 0.7964 | +0.0030 |
| JSON validity | 0.9000 | 0.9333 | +0.0333 |
| Schema validity | 0.8000 | 0.8167 | +0.0167 |
| Whole-document exact match | 0.0167 | 0.0167 | 0.0000 |

The paired 10,000-bootstrap 95% interval for the F1 delta is `[-0.0473, +0.0492]`; the aggregate
gain is not statistically demonstrated on 60 documents. The useful intervention-specific signal
is cargo packaging:

- package-type F1 improved from 0.592 to 0.765;
- package-quantity F1 improved from 0.627 to 0.694;
- package-type F1 among stable JSON-valid documents improved from 0.699 to 0.864.

Container identifiers regressed on the full set because of four invalid-output loops and repeated
check-digit/identifier errors. Additional cargo information also regressed. Epoch 20 is the better
multi-objective relation checkpoint: it is within roughly 0.001 overall F1 of epoch 25 but has
better validity, category F1, generation length, and evaluation runtime. The configured selector
nevertheless continues to use overall field/value F1 to remain comparable with prior runs.

The relation experiment took 4.18 trainer hours versus 2.18 hours for the baseline. Mean source
length grew by 39%, and scheduled generation dominated the difference. The relation-named dataset
did **not** put auxiliary GLM-OCR table text into the prompt; table augmentation remains untested.

Authoritative analysis:

- `artifacts/kie-training/analysis/t5gemma2-270m-lora-mpci-bl-relation-explicit-v3-v1-vs-combined487-v2-paired-audit-v1/REPORT.md`
- `artifacts/kie-training/analysis/t5gemma2-270m-lora-mpci-bl-relation-explicit-v3-v1-vs-combined487-v2-mpl-seaborn-supplement-v1/VISUAL_SUPPLEMENT.md`

The supplement uses Matplotlib and Seaborn, renders all 88 compared fields across paginated plots,
and applies constrained/tight layout. It contains 14 PNGs and does not use top-N field filtering.
The earlier dataset EDA renderer used Pillow because Matplotlib/Seaborn were not installed in the
locked environment; those preferred libraries are now in the dedicated analysis dependency group.

## Non-destructive package target projection

The source 1,157 labels retain every OCR-grounded package fact. The training projection audited all
204 multi-package groups in 190 documents:

| Result | Groups/documents |
|---|---:|
| Groups projected to task-facing facts | 155 |
| Multi-package groups retained as legitimate direct facts | 49 |
| Ambiguous groups held fail-closed | 0 |
| Reviewed quantity-only role decisions | 1 |
| Documents changed | 142 |
| Outer/generic package facts moved to metadata | 189 |
| Allocation groups reduced to container membership | 103 |

Pallet/skid/explicit transport-container levels and generic aggregate `PACKAGE`/`PKG` levels move
to metadata only when a more specific direct-goods package is printed. Multiple true direct-goods
facts remain source ordered—for example, drums and tinplate containers under one outer pallet
level. The transform never moves an outer quantity onto an inner package or invents an allocation.
Eighteen quantity-only facts remain quantity-only. In the sole previously held document, raw OCR
row-links quantity `1` to used semi-trailer cargo and `27 PCS` to spare parts. The reviewed role
decision retains both facts but leaves the first type absent; it does not turn `SEMI TRAILER`,
`VEHICLE`, or the handling statement `UNPACKED AND UNPROTECTED` into a package type.

The complete pre-projection targets, removed package facts, original allocations, and decision
lineage remain in:

`artifacts/kie-training/datasets/mpci-bl-combined1157-task-facing-packages-v2/package-metadata.jsonl`

The second non-destructive projection maps exact reviewed printed types to readable registry-backed
tokens. Of 1,405 typed occurrences, 1,348 use 47 observed category tokens. The remaining 57
occurrences across 27 genuinely ambiguous abbreviations/descriptions stay as printed raw text.
All 1,970 container descriptions remain printed; no container category was inferred. Full source
targets and category decisions remain in sidecars.

Final unsplit dataset:

`artifacts/kie-training/datasets/mpci-bl-combined1157-task-facing-package-categories-v2/records.jsonl`

## Runtime split and next config

Training YAML supports exactly one of:

1. existing pinned `dataset.splits`; or
2. one pinned `dataset.source` plus `dataset.partition`.

The new mode accepts either an exact record count or a fraction with explicit half-up rounding. It
rejects duplicate IDs/input hashes and null targets, keeps source order within each fold, and records
all resolved memberships and hashes in the dataset report. Resume fails if the partition report
changes.

The next config uses seed 424 and an exact 100-document validation set. It resolves to 1,057 train
and 100 validation documents with no coverage skips. With micro-batch one and gradient accumulation
24, there are 45 optimizer steps per epoch; evaluation/checkpoint interval 225 is exactly every five
epochs. Evaluation also runs once before training with the LoRA adapter disabled.

The remaining defaults match the prior relation run: 25 epochs, BF16/TF32, SDPA, non-reentrant
gradient checkpointing, fused AdamW (`lr=1e-4`, betas 0.9/0.999, epsilon `1e-8`), cosine schedule,
5% warmup, weight decay 0.01, gradient clip 1.0, rank-32 RS-LoRA with alpha 64/dropout 0.05, and no
early stopping. The LoRA regex matches 252 modules: all q/k/v/o attention and gate/up/down MLP
projections across all 18 encoder-text and all 18 decoder layers.

The pinned tokenizer measured maxima of 12,350 source and 3,651 target tokens, including terminal
EOS. Configured source and target/generation ceilings are 13,312 and 4,096, leaving 7.8% and 12.2%
headroom while staying well below the model's 32,768-position encoder/decoder limits. Dynamic
padding means the higher guards do not enlarge ordinary training batches. No source or target is
truncated.

## Prediction identity correction

The historical relation run's aggregate metrics were correct, but 59 of 60 published prediction
rows had the wrong `document_id`: Transformers 5.15 length-grouped evaluation produced predictions
in sampler order while publication zipped source-order IDs. Future publication records the exact
native sampler permutation without replacing or re-sorting it. It fails if the permutation is not
complete and unique. A real Trainer probe paired all IDs correctly; measured wrapper overhead was
2.98 microseconds per complete 60-row sampler iteration.

## Validation and resource probes

- 393 integrated labeling/training tests passed in the base preparation pass; 36 focused
  projection/config tests passed after the final reviewed-role correction.
- Ruff and mypy pass on all touched source files.
- Strict config validation and all 1,157 schema/dual-view validations pass.
- Runtime partition inspection: 0.63 seconds internal, 1,828 records/second, 80,524 KiB peak RSS
  for the host process including startup.
- Tokenizer preparation: 65.71 seconds internal, 1,599,040 KiB peak RSS, zero truncation.
- Longest-source optimizer probe: 12,350/1,106 source/target tokens, 5.46/6.21 GiB peak
  allocated/reserved, 5.82 seconds.
- Longest-target optimizer probe: 6,965/3,651 source/target tokens, 12.68/14.73 GiB peak
  allocated/reserved, 4.02 seconds.
- Package projection: 3.72 seconds, 141,308 KiB peak RSS, byte-identical rerun.
- Category projection: 3.40 seconds, 131,572 KiB peak RSS, byte-identical rerun.

The optimizer probes use the exact pinned model, BF16/LoRA target set, fused optimizer, SDPA, and
gradient checkpointing. They publish no run, checkpoint, or MLflow state.

## Operator commands

Revalidate without loading the tokenizer or model:

```bash
TMPDIR=/tmp UV_CACHE_DIR=/tmp/documentparsing-uv-cache \
uv run --frozen document-kie-train validate-config \
  --config configs/training/t5gemma2_270m_lora.mpci_bl_combined1157_task_facing.yaml \
  --project-root .

TMPDIR=/tmp UV_CACHE_DIR=/tmp/documentparsing-uv-cache \
uv run --frozen document-kie-train inspect-dataset \
  --config configs/training/t5gemma2_270m_lora.mpci_bl_combined1157_task_facing.yaml \
  --project-root .
```

Start MLflow and the actual training run only when desired:

```bash
docker compose --profile training up -d --wait mlflow-server

docker compose --profile training run --rm --build kie-trainer train \
  --config configs/training/t5gemma2_270m_lora.mpci_bl_combined1157_task_facing.yaml \
  --project-root /workspace
```
