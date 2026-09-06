# Synthetic raw-OCR rewrite optimization probe

Date: 2026-09-04

## Executive result

The experiment validates a substantially cheaper execution architecture, but it does **not** yet
validate an end-to-end production rewrite. No synthetic training record was published.

- A 50-document, zero-API audit found 2,387 changed text leaves. The original source value was an
  exact raw-OCR substring for 1,522 leaves (63.8%). This is discovery coverage, not automatic write
  authority.
- On ten previously audited difficult documents, the enriched compiler resolved or deliberately
  preserved 250/649 work items (38.5%), delegated 393 (60.6%), and blocked 6 (0.9%).
- Removing context-only blank span edges reduced the residual window from 746 to 660 lines across
  the ten cases (-11.5%). The final windows contain 660/1,513 source lines (43.6%).
- One GLM-5.3-Flash case completed one editor and one independent reviewer request. The declared
  semantic contract, transactional host audit, and reviewer all passed.
- Against the old full-context execution on the same document, input tokens fell 69.3%, reasoning
  tokens fell 74.9%, visible output fell 57.6%, and model-call latency fell 72.9%.
- The retained full-context reference exposed three unchanged source-only identifier lines in that
  apparently successful case. A ten-case audit found 59 such lines. The current implementation now
  withholds `quality_validated` whenever this reference-gap condition occurs.

The architecture is therefore promising and technically proven, while the semantic plan is still
incomplete for raw-only and repeated auxiliary slots.

## Questions investigated

### Would materializing OCR as Markdown and giving the model filesystem tools help?

It helps human review and reproducibility, but it does not intrinsically make remote inference
cheaper. A hosted model cannot read a local file without the selected bytes being returned through
a tool call; those tool results then enter provider-visible context. Search/read/edit loops also add
round trips and repeatedly expose context.

PydanticAI's filesystem capability supplies bounded reads, writes, search, and optimistic hashes.
Code Mode can keep intermediate tool results out of subsequent model history. Both are useful for
general coding tasks, but neither removes the transmission cost for this single terminal rewrite.
Giving this task a generic filesystem or code-execution capability would also enlarge the action
surface unnecessarily.

The selected interface is consequently narrower:

1. The host materializes immutable source, intermediate, work-item, span, diff, and result files for
   operators.
2. The host locates and owns every writable line range.
3. The model receives only typed semantic work items and those authorized OCR spans.
4. The model returns one provider-native, strict JSON-schema result.
5. The host maps opaque span IDs back to immutable ranges and applies one transaction.

No filesystem tool, code-execution tool, free-form edit tool, correction loop, graph, or fuzzy write
locator is exposed to the model.

### Can a meaningful fraction be rewritten without an LLM?

Yes, but the safe fraction is smaller than the literal-discovery upper bound.

| Discovery class | Leaves | Meaning |
|---|---:|---|
| Exact unique, consistent target | 865 | Strongest deterministic candidate once field ownership is proven |
| Exact repeated, consistent target | 499 | Potential replace-all candidate only when every occurrence has the same semantic ownership |
| Exact conflicting | 158 | Same printed source maps to different target roles; global replacement is unsafe |
| Flexible-whitespace match | 118 | Useful locator, not automatic write authority |
| Non-string or absent source | 527 | Requires typed rendering, projection, or insertion logic |
| Not a literal surface | 220 | Requires better typed localization or a residual model |

The difference between 63.8% literal discoverability and 38.5% compiler resolution is deliberate.
For example, `BARCELONA` occurs in port-of-loading, prepaid-at, and place-of-issue slots, but those
roles map to three different synthetic values. A blind replace-all would corrupt the document.

## Implemented probe flow

### 1. Immutable input validation

The run pins the source corpus, the 50 synthetic targets, the earlier audited atomic run, and both
prompts by SHA-256. Existing committed runs are validated before reuse. A run publishes through the
repository's staged immutable-artifact protocol.

### 2. Semantic-delta compilation

The compiler rebuilds the reviewed source/target workspace and projects changed label leaves into
work items. Existing deterministic, line-bound replacements are reapplied first. Exact literals,
whitespace-normalized literals, parsed numeric surfaces, party-name aliases, and same-object sibling
evidence are discovery mechanisms. Fuzzy matching never grants write authority.

### 3. Host-owned residual spans

Only non-overlapping line intervals containing proven work-item evidence are emitted. Page markers
cannot be crossed or edited. Context-only blank lines are trimmed from span edges; internal blank
lines and semantic blank slots remain protected by the transaction validator.

### 4. One-shot constrained editor

The editor receives:

- compact work-item IDs, paths, actions, source values, and target values;
- the authorized spans and their current lines;
- target occurrence requirements;
- explicit compound-party and raw-auxiliary identity requirements, when present.

It returns every span exactly once with exactly one output entry per retained input line. PydanticAI
uses provider-native JSON Schema, not a function tool. The model never chooses a line number and
cannot write outside the host-owned intervals.

### 5. Transactional host validation

The host rejects the entire patch if it changes page order, line count, blank-line topology,
structural prefixes, inline slot topology, newline convention, unrelated punctuation, target-label
hash, required occurrences, operational constraints, or other existing rewrite invariants. There is
no partial commit and no silent fallback.

### 6. Compact independent review

The reviewer receives the work items, before/after residual spans, and host-audit booleans. It
returns a strict pass/revise receipt. A pass requires all review dimensions to pass and no finding;
a revise verdict requires a failed dimension and exact rewritten-span evidence.

### 7. Audited-reference gap guard

This experiment has a retained full-context rewrite for each difficult case. The new guard locates
lines that remain byte-identical to source even though the retained audited reference changed them.
Such a gap prevents `quality_validated`. Exact equality with the reference is not required—valid
case/style alternatives remain possible—but unchanged source-specific material cannot pass unseen.

## Worked case

Document `doc_1249175a6240081bbc038acab925f1b703431a9c359fdfbcad5249d8ea96cc61`
contains 92 OCR lines and 28 compiled work items.

- Six items were already deterministic.
- Twenty-two items were residual.
- Eight host-owned spans retained 41/92 lines.
- The residual contract distinguished the three meanings of `BARCELONA`: port of loading became
  Surabaya, prepaid-at became Depok/Indonesia, and place of issue became Pangkalan
  Brandan/Indonesia.
- It also rewrote shipper/consignee/notify/carrier identities, vessel/voyage, port of discharge,
  package quantity/type, and gross-weight surface while preserving the document's layout.
- The model returned all eight spans in one native-schema response. The host committed all eight,
  and the deterministic audit passed every boolean invariant.
- The independent reviewer returned pass with zero findings.

The reference comparison then found three out-of-contract source lines: ACID, importer taxation,
and exporter identifiers. They were not label fields, were not declared auxiliary requirements,
and lay outside the residual spans. This case is now classified as incomplete rather than silently
accepted.

## Live request evidence

### Successful bounded proof

Provider/model: OpenRouter / `z-ai/glm-5.3-flash`, downstream `NextBit`.

| Stage | Requests | Input | Reasoning | Visible output | Total output | Time | Provider cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Residual editor | 1 | 2,764 | 496 | 441 | 937 | 13.51 s | $0.00087542 |
| Residual reviewer | 1 | 2,524 | 134 | 32 | 166 | 5.97 s | $0.00045392 |
| Total | 2 | 5,288 | 630 | 473 | 1,103 | 19.48 s | $0.00132934 |

The editor request comprised 1,647 system-prompt characters, 6,248 user-payload characters, and a
1,887-character native output schema. The reviewer comprised 910 system characters, 6,462 payload
characters, and a 1,234-character schema. The reviewer is 47.7% of total input tokens; reducing it
to changed lines rather than full before/after spans is a measurable future optimization, but must
be quality-tested before adoption.

### Same-document baseline comparison

| Metric | Full-context baseline | Hybrid proof | Reduction |
|---|---:|---:|---:|
| Input tokens | 17,199 | 5,288 | 69.25% |
| Reasoning tokens | 2,511 | 630 | 74.91% |
| Visible output tokens | 1,115 | 473 | 57.58% |
| Total output tokens | 3,626 | 1,103 | 69.58% |
| Model-call time | 71.99 s | 19.48 s | 72.94% |
| Actual provider cost | $0.002196425 | $0.001329340 | 39.48% |

The successful fallback endpoint cost twice the discounted DeepInfra rate. At the original
$0.075/M input, $0.015/M cache-read, and $0.25/M output rates, the same hybrid tokens would cost
approximately $0.00066467/document, or $0.665/1,000 documents for this particular case. That is a
case-specific conditional estimate, not a production forecast, because source-only coverage is not
yet complete.

## OpenRouter/PydanticAI routing findings

Three zero-token failures were retained rather than overwritten:

1. GLM-5.3-Flash rejected disabled reasoning; `minimal` is its lowest accepted setting on the tested
   route.
2. PydanticAI output-function mode required forced function-tool choice. Of the four discounted
   endpoints, only DeepInfra advertised that exact capability, so the other endpoints were not
   valid fallbacks under `require_parameters: true`.
3. Switching to `NativeOutput` initially hit PydanticAI's generic OpenRouter profile, which did not
   know that the `z-ai/*` model supports native schema output. A narrow OpenRouter gateway profile
   override was added. `require_parameters: true` remains the downstream compatibility guard.

The original $0.075/$0.25 maximum-price ceiling still admitted only DeepInfra for strict structured
output. It returned an upstream shared-pool 429. Expanding the ceiling to $0.15/$0.50 admitted other
strict-schema GLM endpoints, and OpenRouter completed through NextBit. This was an availability and
capability-filter interaction, not a retry bug.

## Completeness audit across the ten hard cases

There were 63 baseline-reference changes outside the initial compiled spans. Four were places where
the compiler had already made a different, intentional jurisdictional rewrite. The remaining 59
lines were still byte-identical to source and form the actual uncovered set.

| Missing semantic class | Lines | Examples |
|---|---:|---|
| Auxiliary identities, references, and contacts | 30 | booking/service references, customs/ACID values, tax/exporter IDs, linked B/L references, fax/telephone values |
| Repeated aggregates and equipment facts | 18 | total package/item/container counts, duplicate gross/net weights, volume, repeated equipment counts |
| Reefer/cargo/boilerplate-dependent flavor | 8 | temperature/ventilation statements, cargo state, carrier-branded liability text |
| Duplicate or auxiliary dates | 3 | dates printed outside the labeled issue/on-board slots |

This is the central correctness finding. A label-only delta is insufficient to anonymize and
semantically reconcile a raw document. The earlier full-context model noticed many of these slots
opportunistically; the compact flow needs them represented explicitly rather than relying on model
intuition.

## Required next implementation before production

### Extend the synthetic semantic plan

Add a non-training `auxiliarySlots` contract alongside the task label. It must cover every
source-specific printed slot even when the model is not trained to output it:

- booking, customs, tax, exporter, service-contract, linked-document, and contact identifiers;
- duplicate counts, package totals, gross/net weights, volumes, and equipment summaries;
- operational reefer/ventilation/free-time statements;
- repeated dates and legal/agency relationships;
- source-specific carrier names inside boilerplate.

This is not permission to hard-code observed aliases. Slot discovery should be built from the
corpus's template/line evidence inventory, typed value grammars, source-label evidence, and
cross-document variability. Every discovered slot needs provenance and an explicit generation or
preservation policy.

### Generate auxiliary values coherently

- Shape-preserving identifiers and contacts can be deterministic and collision-safe.
- Aggregate quantities, weights, volumes, and equipment counts must derive from the synthetic cargo
  and container plan rather than be independently sampled.
- Reefer wording must project the target container/temperature state.
- Natural legal or operational wording can remain a compact model residual after its slot and
  intended semantics are host-defined.

### Complete typed deterministic renderers

Prioritize fields with strong ownership and formatting contracts: dates, scalar measures, contact
values, identifiers, container/seal numbers, and repeated aggregates. Exact multi-occurrence
replacement is safe only after all occurrences are assigned to the same semantic source and target.
Conflicting cross-role surfaces stay model residuals.

### Harden validation

Before publication, require:

1. every changed label leaf to be rendered or explicitly non-printing;
2. every source-specific auxiliary slot to be regenerated or coherently preserved;
3. every repeated aggregate to reconcile with the target semantic plan;
4. no stale source identity or identifier to survive;
5. all structural/formatting invariants to pass;
6. an independent residual review until held-out evidence justifies risk-scoped sampling.

## Decision

The generic Markdown/filesystem-agent branch should not be the default. The host-owned hybrid
compiler is the better direction: it materially reduces navigation, context, reasoning, latency,
and cost while making write authority narrower and more auditable.

It is not yet ready for a 10k–50k production synthesis run. The next gate is full auxiliary-slot
coverage, followed by an offline 50-document reference audit and a small live held-out test. The
current run is evidence for the architecture, not training data.

## Artifacts and implementation

- Live artifact/report: `artifacts/kie-synthesis/mpci-bl-raw-text-hybrid-compiler50-glm53-flash-poc1-v1-r6/`
- Earlier no-token failure artifacts: sibling runs `r1` through `r4`
- Successful editor/local-rejection diagnostic: sibling run `r5`
- Probe implementation: `src/document_ocr/synthesis/raw_text_hybrid_probe.py`
- Strict config models: `src/document_ocr/synthesis/config.py`
- CLI commands: `src/document_ocr/synthesis/cli.py`
- Probe config: `configs/synthesis/mpci_bl_raw_text_hybrid_compiler50_glm53_flash_poc1.yaml`
- Prompts: `prompts/synthesis/mpci_bl_raw_text_hybrid_residual_{editor,reviewer}_v1.md`
- Tests: `tests/test_synthesis_raw_text_hybrid_probe.py`

The live artifact retains source, deterministic intermediate, work items, residual spans, exact
provider-visible messages, response text/reasoning parts, usage receipts, final text, unified diff,
review, plots, transaction manifest, and commit manifest. No API key is persisted.

## Validation performed

- Ruff: pass.
- Mypy on config, CLI, implementation, and targeted test: pass.
- Targeted pytest suite: 128 passed; one unrelated `pydantic_graph` event-loop deprecation warning.
- Saved GLM editor-response replay after blank-edge correction: 8/8 replacements committed, zero
  failed host checks, zero residual candidates.
- Live native-schema editor + reviewer: two successful responses, both `finish_reason=stop`.
- No current corpus, label dataset, training dataset, or training config was mutated.

## Authoritative references

- [PydanticAI capabilities](https://pydantic.dev/docs/ai/capabilities/overview/)
- [PydanticAI filesystem capability](https://pydantic.dev/docs/ai/harness/filesystem/)
- [PydanticAI Code Mode](https://pydantic.dev/docs/ai/harness/code-mode/)
- [PydanticAI toolsets](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/)
- [PydanticAI output modes](https://pydantic.dev/docs/ai/core-concepts/output/)
- [OpenRouter structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs)
- [OpenRouter provider routing](https://openrouter.ai/docs/guides/routing/provider-selection)
- [OpenRouter GLM-5.3-Flash endpoint catalog](https://openrouter.ai/api/v1/models/z-ai/glm-5.3-flash/endpoints)
- [OpenRouter reasoning controls](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
