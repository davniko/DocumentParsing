# MPCI B/L error mechanisms and table-input experiment

Date: 2026-08-27

## Scope

This note binds two related pieces of work:

1. a conservative mechanism audit of the 100 unconstrained checkpoint-900
   validation predictions from `t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-v2`;
2. a first raw-OCR-plus-GLM-OCR-table training arm over the 483 documents for
   which a parsed table view already exists.

No constrained decoding is used in the measured validation predictions. The
error audit does not call unsupported output a hallucination unless source
review can prove that conclusion.

## What the model gets wrong

The exact leaf comparison contains 1,293 errors across 99 of the 100 validation
documents. The main result is that arbitrary invention and malformed field names
are not the dominant failure modes.

| Mechanism | Leaf errors | Fraction |
|---|---:|---:|
| Omission | 464 | 35.9% |
| OCR-grounded over-extraction | 140 | 10.8% |
| Relation or structure error | 127 | 9.8% |
| Incomplete or shortened value | 108 | 8.4% |
| Correct value at wrong list position | 101 | 7.8% |
| Contaminated or over-complete value | 96 | 7.4% |
| OCR-grounded wrong numeric selection | 67 | 5.2% |
| Categorical semantic error | 55 | 4.3% |
| OCR-grounded wrong text selection | 29 | 2.2% |
| Wrong field assignment | 28 | 2.2% |
| Near-copy corruption | 27 | 2.1% |
| Unsupported or wrong numeric candidate | 22 | 1.7% |
| Unsupported text candidate | 16 | 1.2% |
| Date selection/normalization | 5 | 0.4% |
| Numeric scale | 4 | 0.3% |
| List alignment mismatch | 4 | 0.3% |

The 28 wrong-field leaves form 14 one-to-one add/omit pairs. The 38 unsupported
candidates are only a 2.9% upper bound on hallucination: normalized substring
matching can miss punctuation, unit conversion, OCR variants, and label defects.
The candidate-level table must be reviewed before assigning that stronger label.

Output-contract failures are similarly concentrated away from field naming:

- 2 malformed JSON predictions;
- 1 extra/wrong key and 0 missing required keys;
- 36 schema issues involving values or cross-object relationships.

Typical observed failures include shortened party addresses, a marks value copied
into a goods description, the right container allocation placed under the wrong
group/list index, allocation totals that do not reconcile with the emitted
package levels, and an OCR-grounded row total selected instead of the labeled
row quantity. Near-copy failures include character-level B/L, seal, URL, and
address corruptions, but they are a small minority.

The complete auditable publication is under:

`artifacts/kie-training/analysis/t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-v2-error-mechanisms-v1/`

It contains the classified JSONL, summary/field/candidate CSVs, a Markdown
report, and eight plots generated with matplotlib and seaborn.

## Why the table arm is a bounded 483-document experiment

The existing immutable GLM-OCR table publication covers 487 older documents;
483 of them remain in the current 1,157-document task-facing corpus. The other
674 current documents do not yet have table-recognition output. Supplying an
empty table marker to those 674 rows would confound table availability with
document cohort and silently teach a missing-view pattern, so the first table
arm is restricted to the 483 completely covered rows.

The materialized dataset is:

`artifacts/kie-training/datasets/mpci-bl-combined483-task-facing-table-input-v1/`

It binds the current task-facing target to each older table view by document ID
and raw-text SHA-256. `modelInputText` contains two explicitly delimited views:

- `<raw_ocr_text>` is the only value truth boundary;
- `<glm_ocr_table_view>` is auxiliary row/column and grouping evidence.

The prompt forbids emitting a value that occurs only in the table view. This
allows GLM-OCR's table mode to restore row structure without letting a second OCR
pass silently overwrite the audited raw text.

## Exact token and batch validation

The Compose execution path tokenized all 483 records with the pinned T5Gemma2
tokenizer revision:

| Split | Rows | Maximum input | Input p95 | Maximum target | Truncated |
|---|---:|---:|---:|---:|---:|
| Train | 423 | 11,505 | 8,443 | 3,651 | 0 |
| Validation | 60 | 12,538 | 8,947 | 1,721 | 0 |

The training config therefore uses source/target caps of 13,312/4,096 and keeps
overflow fail-closed.

A first forward, backward, optimizer-step, and adapter-publication probe was run
with physical batch 2 on the two longest **encoder inputs**. It completed, but it
did not include the longest decoder target and therefore was not a valid
worst-case capacity proof:

- 21,241 non-padding input tokens;
- 252.7 seconds and 84.1 input tokens/second for this deliberately worst-case step;
- 1.52 GiB model initialization allocation;
- 17.00 GiB additional peak training allocation reported by Transformers;
- 23,719 MiB peak driver usage observed on the 24,564 MiB GPU.

The full run disproved the original conclusion. It reached four optimizer steps
and then failed with CUDA OOM. The source-only length sampler groups on
`input_length`; it does not account for decoder target length. The omitted
target-long record has 10,069 encoder tokens and 3,651 target tokens. T5Gemma2's
decoder merges target self-attention with cross-attention keys, so both axes
materially affect activation cost.

A fresh two-record probe containing the maximum encoder record (11,505 / 2,688
input/target tokens) and maximum target record (10,069 / 3,651) reproduced the
batch-2 OOM immediately. The exact same records completed at batch 1 with
accumulation 2 in 13.53 seconds at 1,595 input tokens/second.

An apples-to-apples first-update benchmark then consumed the exact same 24
records and 195,549 non-padding input tokens as the failed run:

| Physical batch / accumulation | Runtime | Input tokens/s | Peak driver memory |
|---|---:|---:|---:|
| 2 / 12 | 745.43 s | 262.3 | 25,504 MB |
| 1 / 24 | 30.65 s | 6,380.9 | 14,283 MB |

Batch 1 was therefore 24.3 times faster on the actual first optimizer update,
not merely safer. A median-length two-record control showed the expected
opposite result (batch 2: 8.09 seconds; batch 1 with accumulation 2: 15.25
seconds), confirming a VRAM-pressure crossover rather than a general batching
overhead. A variable token-budget batcher could exploit that crossover, but it
would also require variable accumulation and exact loss weighting to preserve
the optimizer objective. The current Trainer does not provide that contract.
The clean runnable configuration consequently uses physical batch 1 and
accumulation 24, preserving effective optimizer batch 24.

The table view also makes the encoder workload materially longer than the
otherwise matched relation-explicit dataset: mean input length rises from 4,561
to 5,731 tokens (+25.7%), p95 from 6,293 to 8,443 (+34.2%), and maximum from
7,991 to 11,505 (+44.0%). No input or label is truncated to recover memory.

The benchmark publication is:

`artifacts/kie-training/benchmarks/benchmark-t5gemma2-270m-lora-mpci-bl-relation-v3-table-input-b2-r1/`

The corrective benchmark publications are:

- `artifacts/kie-training/benchmarks/benchmark-t5gemma2-270m-lora-mpci-bl-table-worst-pair-b1ga2-r1/`
- `artifacts/kie-training/benchmarks/benchmark-t5gemma2-270m-lora-mpci-bl-table-median2-b2-r1/`
- `artifacts/kie-training/benchmarks/benchmark-t5gemma2-270m-lora-mpci-bl-table-median2-b1ga2-r1/`
- `artifacts/kie-training/benchmarks/benchmark-t5gemma2-270m-lora-mpci-bl-table-first-update-b1ga24-r1/`

## Configurations

The runnable training arm is:

`configs/training/t5gemma2_270m_lora.mpci_bl_relation_v3_table_input.yaml`

The remaining 674 documents have a separate table-recognition config:

`configs/glm_ocr.table.mpci_bl_current674.yaml`

Its CPU preparation probe validated all 674 documents and 1,580 retained page
rasters. No table model was started. Once that table extraction is published, a
future materialization can construct a fully covered 1,157-document table arm;
the current 483-row run should remain an immutable first pilot. Comparing it
directly with the 1,157-row run cannot isolate the effect of tables because the
cohort size differs; a causal table ablation would pair this config with the same
423/60 rows, targets, seed, and hyperparameters while reading `joinedRawText`
through the non-table relation-v3 prompt.

## Validation evidence

- 50 targeted tests passed across the mechanism analysis, materializer,
  table-view pipeline, and training configuration.
- Ruff passed on all touched source/test files.
- Mypy passed on the three changed executable source tools/modules.
- Training config validation, exact dataset inspection, and Compose tokenizer
  preparation all passed.
- The longest-pair batch-2 CUDA training probe completed and published a valid
  run manifest.
