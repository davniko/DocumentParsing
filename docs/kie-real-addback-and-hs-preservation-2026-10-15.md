# Real data addback and HS presentation correction

The next experiment trains on the **same 7,000 synthetic documents plus 600 real
documents**, retaining direct JSON targets, rank 48 and the preceding optimizer
and model settings. Its [configuration](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic7000_hs_preserved_r23_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml)
uses a [new real-label dataset revision](../data/curated/mpci-bl-real600-synthetic7000-hs-preserved-r23-positions-v1/README.md).
The synthetic shard and all historical dataset/run files remain unchanged.

## Form contract

The saved [MPCI form schema](../artifacts/mpci-ai-schema/mpci-cuscar-canonical.schema.json)
defines customs identifier values as printable ASCII, `U+0020–U+007E`. This
includes dots, spaces, commas, hyphens and colons. Its
[discovery provenance](../artifacts/mpci-ai-schema/README.md) identifies the
inspected application revision; this is a repository form-schema check, not a
live-deployment check. The historical AI adapter's digits-only normalization
is not a form-field restriction.

The extraction contract retains the narrower observed format: 6–20 ASCII digits
with those separators. The real source printing `HS CODE: 5205:22` established
the missing colon case. The schema, field description, source-shaped rendering
and direct literal grounding now support it. Active synthesis contract hashes
and 29 matching config pins were refreshed; 413 template entries, 2,478 template
assets and 386 historical/unrelated configs remain unchanged.

## Exact label changes

| Split | Existing HS values | Values corrected | Documents corrected |
|---|---:|---:|---:|
| Real training | 597 | 108 | 81 |
| Real validation | 51 | 14 | 13 |

All 122 replacements were reviewed with their source lines and exact character
spans. They preserve all digits, leading zeros, identifier membership and field
ownership. Every other target field, input text and coordinate remains unchanged.
Repeated complete codes retain their first owned presentation; a short code is
not matched inside a longer dotted code. Four wrapped aliases remain unchanged
because complete inline copies already support their targets. Hyphen-separated
lists remain multiple existing identifiers rather than one combined code.

The [complete plan](analysis/real-hs-presentation-20261015/plan.json) inventories
all 648 real HS values. Its
[review receipt](analysis/real-hs-presentation-20261015/review.json) approves the
122 bounded presentation changes. The new validation retains the same 60
identities and inputs, but its corrected HS labels change the reference hash;
historical predictions must be rescored on this reference for fair comparisons.

## Training settings

Rank 48, alpha 32, rsLoRA/EVA, learning rate 0.0001, ScheduleFree AdamW,
microbatch 1, accumulation 32, ten epochs, seed 424, input-only positioned text,
compact direct JSON, generation settings and sequence limits are unchanged.
The 7,600-row loader has 238 updates per epoch and 2,380 updates in total.
Warmup is 119 updates, the first five percent. Evaluation and checkpointing use
595-update intervals, approximately epochs 2.5, 5 and 7.5, plus the final epoch-10
evaluation. All checkpoints are retained; best-field-F1 loading/export is enabled.

## Validation and measured cost

Independent readback verifies six protected input files, all artifact hashes,
all digit sequences, exact repair paths and unchanged non-HS data. Publication
took 2.94 seconds at 800.41 MiB peak RSS; readback took 0.54 seconds at 87.43 MiB.

The rebuilt Compose image passed config validation, dataset inspection and full
tokenization. All 7,600 training and 60 validation records remain, with zero
filtering or truncation. Maximum source/target lengths are 8,073/3,146 training
tokens and 5,698/1,334 validation tokens. Cold tokenizer preparation took
49.39 seconds; the cache replay took 15.71 seconds. No model weights were loaded.

The combined targeted suite passed **122 tests** in 12.36 seconds; Ruff passed.
There were ten existing multiprocess fork deprecation warnings. The contract
refresh passed 45 offline sampling/rendering/target-validation probes, including
colon and mixed-separator cases. Direct grounding's measured colon-aware search
adds 5.95 microseconds per 5,297-character OCR search; the training hot path and
optimizer are unchanged. Native Trainer argument calculation passed independently.

Evidence: [native checks](analysis/real-hs-presentation-20261015/native-readiness.json),
[argument validation](analysis/real-hs-presentation-20261015/arguments-validation.json),
[active synthesis refresh](analysis/real-hs-presentation-20261015/contract-refresh.json).
No training, LLM labeling or synthesis calls were launched.
