# MPCI B/L dual-cargo, PDF review, categorical, and training handoff

Date: 2026-08-22  
Status: dataset and runnable training config published; no API labeling call or GPU training run made

## Outcome

The next 270M experiment is prepared end to end:

- PydanticAI produces both the existing normal B/L label and the relation-explicit cargo label in
  one evidence-rich annotation.
- An auxiliary layout escalation sends a hash-verified, requested-page PDF rather than page images.
  The PDF may resolve layout only; raw GLM-OCR text remains the sole factual source for labels.
- Package categories use proven platform registry meanings and an exact reviewed source-key map.
- Container categories are deliberately absent. Platform research proved the accepted code set but
  found no stable semantic name per generated code, so printed container type text/code is retained
  for deterministic downstream resolution.
- The 487-row source clone produced 483 valid relation-explicit examples, with 423 train and 60
  validation rows. Four declared relation-ambiguous documents remain excluded.
- A hash-pinned training YAML was cloned from the current combined487 baseline and validated against
  the exact schema, prompt, data, tokenizer, and token ceilings.

## PDF auxiliary-review contract

OpenAI Responses accepts PDFs as file inputs. For vision-capable models, OpenAI supplies both PDF
text and page images to the model. PydanticAI exposes this through
`BinaryContent(data=..., media_type="application/pdf")` and maps it to a Responses `input_file`.

That broader provider visibility does **not** change the label truth boundary. The PDF helper prompt
forbids introducing, correcting, or normalizing values from the PDF. It may return only raw-OCR-
anchored observations about headings, row/column grouping, continuation, and document boundaries.

To avoid sending irrelevant pages, the runner constructs an in-memory PDF containing only the
explicitly requested source pages. It verifies the original source SHA-256 and page count before
construction and records:

- original PDF SHA-256;
- original source page numbers;
- derived attachment SHA-256 and byte count;
- MIME type; and
- construction method (`pypdf_strict`, explicit malformed-PDF recovery, or explicit empty-password
  decryption).

Ollama remains text-only in the portable configuration because its OpenAI-compatible interface does
not establish an equivalent PDF-file contract for arbitrary local models. There is no automatic
provider fallback.

Real-corpus probe over all 487 source PDFs:

| Measure | Result |
|---|---:|
| Successful page-scoped attachments | 487 / 487 |
| Strict PDFs | 474 |
| Empty-password owner-protected PDFs | 6 |
| Explicit recovery for malformed xref/dictionary structures | 7 |
| Requested source pages | 774 |
| Construction throughput | 54.96 documents/s |
| Peak RSS | 101,760 KiB |
| Largest attachment | 5,386,091 bytes |
| Configured/API ceiling | 50,000,000 bytes |

The selected first/last-page benchmark reduced bytes to 87.76% of complete source-PDF bytes. The
savings are modest because most documents are one or two pages, but page scoping prevents a long
document from sending every page for one local ambiguity.

## Dual annotation contract

`BillOfLadingDualCargoAnnotation` contains:

- `normalLabel`: the normal semantic-v2-compatible document/cargo representation;
- `relationExplicitLabel`: the semantic-v3 cargo graph (`cargoGroups`, `cargoPackages`, and
  `cargoAllocationGroups`);
- normal field/value OCR evidence;
- relation-specific OCR evidence for every allocation group; and
- warnings/review metadata.

The validator requires identical non-cargo facts and a lossless, source-ordered relationship between
the normal goods/package/container facts and the relation-explicit graph. Agent drafts cannot emit
package or container categories; categorical mapping is a separate frozen transform. The training
publication uses `relationExplicitLabel` as `target` and retains `normalLabel` as `normalTarget` in
agent-produced rows, avoiding a decoder target that redundantly emits both views.

Primary configs:

- `configs/labeling_agents/mpci_bl_dual_cargo_v3_luna_cost50.yaml`
- `configs/labeling_agents/mpci_bl_dual_cargo_v3_ollama.example.yaml`

No paid Luna request has been made by this preparation pass.

## Categorical result

The platform follow-up is under
`artifacts/mpci-ai-schema/categorical-registry-followup/` and is pinned to application revision
`72f9aff735c2581f2ccff74e7fde34c2aea9b265` (contained by tag `v1.278.0`). Repository evidence does
not prove that this exact revision was deployed to the target tenant/data period; that remains a
release-provenance caveat, not an untracked assumption.

### Package types

The platform registry contains 405 unique two-character application codes with one stable readable
token per code. All 405 passed the reported backend-validation probe. The experiment exposes only
the 31 tokens actually assigned in this dataset in its prompt schema; it does not place all 405
unsupervised categories into every prompt.

Exact source-key review results:

| Measure | Count |
|---|---:|
| Source variants | 99 |
| Package occurrences | 658 |
| Safely mapped variants | 88 |
| Safely mapped occurrences | 636 (96.66%) |
| Verbatim unresolved variants | 11 |
| Verbatim unresolved occurrences | 22 (3.34%) |

The unresolved values are `BALES`, `Bale(s)`, `SETS`, `SET`, `PACK`, `PL`, `RO/RO`, `BULK`,
`CARBOUY`, `JERRY CAN`, and `MODULES`. Each could select zero or multiple registry meanings, so the
target retains its printed `typeDescription` instead of manufacturing a category.

The reviewed mapping source and immutable expanded assignment are:

- `configs/transforms/mpci_bl_semantic_v3_category_review_v1.yaml`;
- `artifacts/kie-training/reviews/mpci-bl-semantic-v3-relation-explicit-table-view-v1/category-assignments.json`
  (SHA-256 `e36652e068839aa329a76a186bfe9ffe5e0c2a61fc45e61cde7c5367e59ce4a3`).

### Container types

The platform generates 56,100 accepted four-character submission values, but it has no stable,
nonlocalized semantic identity for each value. The 19 readable UI type-group labels are deprecated
and are not emitted into CUSCAR. Creating model tokens from those labels would therefore be false
supervision.

All 105 observed container source variants (905 occurrences) remain uncategorized. Their printed
description is retained; if the prior label contains only a printed code, that exact code is copied
to relation-v3 `typeDescription`. The frozen container registry is intentionally empty, the prompt
schema omits container `typeCategory`, and the projection manifest contains no container token map.
Known platform compatibility mappings such as `40HC -> 40GP`, `40HQ -> 40GP`, and `40RH -> 40RE`
remain downstream resolver behavior, not learned labels.

## Published semantic-v3 dataset

Dataset root:

```text
artifacts/kie-training/datasets/mpci-bl-semantic-v3-relation-explicit-table-view-v1/
```

Published files include hash-pinned train/validation JSONL, four declared exclusions, lineage,
task constraints, token-length audit, training YAML, and a manifest written last.

| Measure | Result |
|---|---:|
| Source documents | 487 |
| Published documents | 483 |
| Train / validation | 423 / 60 |
| Package category vocabulary in prompt | 31 |
| Container category vocabulary in prompt | 0 |
| Source tokens p50 / p95 / p99 / max | 4,363 / 6,318 / 7,086 / 8,298 |
| Target tokens p50 / p95 / p99 / max | 569 / 983 / 1,980 / 3,663 |
| Configured source / target ceilings | 9,216 / 4,096 |

The auxiliary GLM-OCR table view remains attached under `auxiliaryViews` but is not rendered by the
training task. This experiment tests semantic instructions, package categorization, and explicit
cargo relations without simultaneously changing the model input.

## Runnable training config

The transform-generated config is:

```text
artifacts/kie-training/datasets/
  mpci-bl-semantic-v3-relation-explicit-table-view-v1/training.yaml
```

SHA-256: `16435475061a3dd9f66f2b88db33a9accea8b4f72d2b0f03d39fd22f592c2402`.

It preserves the current combined487 baseline model revision, optimizer, Adam betas, cosine
scheduler, 25 epochs, LoRA rank/alpha/dropout/RSLora settings, and encoder/decoder attention+MLP
projection target regex. It changes only the task/prompt/data bindings and capacity required by the
new targets:

- max source/target lengths: 9,216 / 4,096;
- per-device train batch: 1;
- gradient accumulation: 24 (effective batch remains 24);
- eval batch: 2;
- base-model eval on start: enabled;
- eval/checkpoint interval: 90 optimizer steps, exactly every five epochs for 423 rows; and
- final evaluation: enabled.

The conservative microbatch is configuration-safe but has not been GPU-benchmarked at these larger
sequence ceilings. The tokenizer/cache preparation gate passes; run a GPU smoke/occupancy probe
before committing to the complete 25-epoch experiment. No automatic truncation or batch-size
fallback is enabled.

## Reproduction and training commands

Re-publish the exact source-key assignment (idempotent when bytes match), audit, and transform:

```bash
uv run --frozen document-kie-semantic-v3 categories \
  --config configs/transforms/mpci_bl_semantic_v3_relation_explicit_table_view.yaml \
  --review-config configs/transforms/mpci_bl_semantic_v3_category_review_v1.yaml
uv run --frozen document-kie-semantic-v3 audit \
  --config configs/transforms/mpci_bl_semantic_v3_relation_explicit_table_view.yaml
uv run --frozen document-kie-semantic-v3 transform \
  --config configs/transforms/mpci_bl_semantic_v3_relation_explicit_table_view.yaml
```

CPU-only gates:

```bash
uv run --frozen document-kie-train validate-config \
  --config artifacts/kie-training/datasets/mpci-bl-semantic-v3-relation-explicit-table-view-v1/training.yaml
uv run --frozen document-kie-train inspect-dataset \
  --config artifacts/kie-training/datasets/mpci-bl-semantic-v3-relation-explicit-table-view-v1/training.yaml
```

Tokenizer/cache preparation and the eventual operator-started GPU run:

```bash
docker compose --profile training run --rm kie-tools prepare-dataset \
  --config artifacts/kie-training/datasets/mpci-bl-semantic-v3-relation-explicit-table-view-v1/training.yaml \
  --project-root /workspace

docker compose --profile training run --rm kie-trainer train \
  --config artifacts/kie-training/datasets/mpci-bl-semantic-v3-relation-explicit-table-view-v1/training.yaml \
  --project-root /workspace
```

## Validation evidence

- PDF subset unit/regression tests cover deterministic identity, source drift, empty-password
  decryption, and malformed-xref recovery.
- Semantic-v3 tests cover lossless fallback behavior, exact registry/assignment coverage, and prompt
  removal of category fields whose frozen vocabulary is empty.
- Real transform: 23.20 seconds wall time, 576,024 KiB peak RSS, 483 valid outputs.
- Dataset inspection: 483 records at 1,597 records/s; all hashes, targets, and prompt/task contracts
  passed.
- Rebuilt Compose training image validates the same config/schema/prompt hashes. Containerized
  tokenizer preparation completed in 36.87 seconds with cache identity
  `9ebe27604ca43f1194012ef7145efc38db3adfc88e634db158eed4817928e8ab`, 0 source
  truncations, and exact train/validation maxima within the configured ceilings.
- Repository tests: 640 non-training-group tests pass; the focused changed-path suite passes 94/94.
  Three host-only optional tests require the `train` dependency group (`torch`/`datasets`) and were
  not run in the lean host environment; the intended rebuilt training container passed its own
  build-time environment verifier and runtime gates.
- Static checks: Ruff and mypy pass on the changed labeling, semantic-v3, and training modules.

## Official references

- OpenAI, [File inputs and PDF behavior](https://developers.openai.com/api/docs/guides/file-inputs).
- PydanticAI, [Image, audio, video, and document input](https://pydantic.dev/docs/ai/core-concepts/input/).
- PydanticAI, [OpenAI provider](https://pydantic.dev/docs/ai/models/openai/).
- UNECE, [Recommendation 21 package/cargo code lists](https://unece.org/code-list-recommendations).
- BIC, [ISO container size/type code reference](https://www.bic-code.org/size-type-code/).
