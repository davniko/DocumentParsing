# Synthetic raw-OCR rendering: deterministic proof boundary

Date: 2026-09-05

## Executive finding

The proposition that every current synthetic label can be inserted into every selected raw-OCR
template through deterministic substitution is **false**.

Two different claims must be kept separate:

1. **Given a complete, correct set of mutable byte spans, can rendering preserve the source
   document's formatting exactly?** Yes. This is proven for all 50 pinned OCR sources.
2. **Can the current task label and OCR alone identify every mutable fact and guarantee that each
   current synthetic target fits the source's printed structure?** No. The 50-document census
   found both incomplete semantic inventories and incompatible target/template pairs.

No synthetic training rows were published. This is intentional: a byte-safe renderer is not, by
itself, proof of a semantically correct synthetic document.

## Reproducible 50-document result

The final pinned run is in
`artifacts/kie-synthesis-analysis/mpci-bl-raw-text-template-proof50-v3/`.

| Gate | Result |
|---|---:|
| Source SHA-256 valid | 50 / 50 |
| Byte-identical source round trip | 50 / 50 |
| All-slot sentinel isolation | 50 / 50 |
| Printed target/source topology compatible | 34 / 50 |
| Complete mutable semantic inventory | 0 / 50 |
| Certified synthetic render | 0 / 50 |

The compiler processed 2,395 accepted source-label facts. It located unambiguous OCR spans for
2,205 (92.1%). The remaining evidence includes repeated or normalized surfaces whose semantic
ownership cannot be inferred safely by a global text replacement.

The 16 current topology failures comprise:

- seven documents whose target adds or removes one or more temperature setpoints;
- eight documents whose target changes cargo additional-information cardinality; and
- two dangerous-goods shape changes, overlapping one of the additional-information cases.

These differences require either a topology-compatible target/template pairing or an explicitly
compiled structural transformation. The byte renderer must not guess which one was intended.

## What is mathematically preserved without a model

Each compiled template is an ordered stream of exact UTF-8 source slices and typed slots. The
template records:

- the source payload SHA-256 and byte length;
- each slot's exact byte start/end and captured-source SHA-256;
- the semantic target path(s) owning the slot;
- its evidence origin and render policy; and
- its line-ending, edge-whitespace, case, and optional identifier-shape envelope.

Compilation rejects out-of-bounds, overlapping, source-drifted, and UTF-8-splitting spans.
Rendering requires exactly one value per slot and copies every literal byte directly from the
pinned source. Rendering with source values must reconstruct the source byte-for-byte before a
template can leave the compiler boundary. A sentinel pass then mutates every slot and verifies
that page markers, line endings, and all literal regions remain unchanged.

This is stronger than asking a model to “preserve formatting”: the model has no ability to change
literal bytes at all.

Measured on the 50 documents:

- compiler latency: 0.378 ms median / 0.651 ms P95;
- source-round-trip render latency: 0.278 ms median / 0.496 ms P95;
- a separate 10,000-render loop sustained 963.9 documents/second with 0.807 MiB peak Python
  allocation under `tracemalloc`.

The latter benchmark includes Pydantic validation, hashing, and proof-object construction. It does
not include semantic span discovery or linguistic generation.

## Why the labels are insufficient as templates

The KIE label is intentionally a task-facing projection, not a lossless model of every printed
shipment fact. The raw documents also contain:

- exporter/importer, tax, VAT, customs, booking, invoice, batch, and service-contract values;
- contact details and signing-agent or affiliate identities;
- printed container/equipment assertions not represented by a container label;
- tare, gross/net, volume, package aggregate, and total relationships;
- free-time, reefer, ventilation, and route-dependent clauses; and
- duplicated values whose role depends on their section rather than their spelling.

For example, one audited source and target both have `containers: null`, while the OCR prints
`(1X40HQ)` twice. A label-only substitution cannot discover that assertion. Another source prints
gross, tare, and total weights while the task label retains only gross weight; a correct synthetic
total must be derived rather than copied or independently invented.

These are not renderer bugs. They show that the source needs a complete, source-bound rendering
contract in addition to the training label.

## Correct end-to-end architecture

### 1. Compile each source template once

Build a complete semantic inventory of all shipment-dependent surfaces. Every inventory entry is
classified as exactly one of:

- `target_binding`: supplied directly by a synthetic task-label field;
- `derived_binding`: computed deterministically from target facts;
- `auxiliary_binding`: synthesized for source-only flavor or private information; or
- `intentional_literal`: reviewed boilerplate that is safe and semantically independent.

The compiler must fail closed if any source-specific value lacks a disposition.

### 2. Keep target generation topology-aware

Template selection happens before the final semantic plan. By default, the target preserves all
printed field presence and collection cardinality. Feature prevalence is controlled by selecting
a matching template cohort—not by adding a reefer, dangerous-goods, or auxiliary slot to an
incompatible document after selection.

Structural variants are allowed only as separately compiled/tested template branches.

### 3. Render deterministic domains locally

Identifiers, dates, quantities, measurements, container/seal numbers, checksums, repeated
bindings, and derived totals use typed deterministic renderers. `Decimal` arithmetic is required
for printed financial/weight precision and equality checks.

### 4. Use a model only where semantics are genuinely linguistic

A bounded PydanticAI call may propose either:

- unresolved source span ownership during one-time template compilation; or
- a natural-language slot value for a party/cargo/legal-flavor block whose structured facts are
  already fixed.

The model never receives authority to rewrite the complete document. Provider-native constrained
output enforces the response shape; host validators resolve every proposal to exact source bytes,
check the semantic owner and format envelope, and reject an incomplete proposal. Once audited,
the compiled template is reusable without further model calls for rendering.

PydanticAI's `NativeOutput`/`ToolOutput` mechanisms constrain structure but do not prove
cross-field or source-text semantics; output validators remain necessary. Its evaluation guidance
also distinguishes fast deterministic checks from model-based evaluators. See:

- <https://pydantic.dev/docs/ai/core-concepts/output/>
- <https://pydantic.dev/docs/ai/core-concepts/agent/>
- <https://pydantic.dev/docs/ai/evals/evals/>

### 5. Certify each compiled template

Publication requires all of the following:

- byte-exact source round trip and source hash;
- disjoint UTF-8-safe spans and unchanged literal bytes;
- sentinel mutation of every slot;
- complete target-leaf and repeated-occurrence traceability;
- complete auxiliary inventory and zero retained private/source identifiers;
- topology compatibility;
- exact date/number/identifier surface envelopes;
- deterministic unit, total, allocation, temperature, and DG invariants;
- repeated and concurrent deterministic rendering;
- property tests over every renderer domain; and
- one independent human/gold audit of the compiled source template.

The independent audit is a one-time compilation cost, not a per-descendant generation step.

## Defensible quality claim

After the missing semantic compilation and topology-safe regeneration are complete, the strongest
honest claim is:

> All 50 sampled source documents round-trip byte-identically, and all 50 synthetic renders pass
> the declared provenance, topology, formatting, leakage, and semantic invariants plus an
> independent audit.

It is not technically defensible to claim that every future arbitrary OCR document and arbitrary
target are universally transformable. New template families must pass the same compile-time
contract or be rejected.

## Current blocker to a 50/50 synthetic publication

The existing 50 target/template pairs were produced before topology became part of the generation
contract. Sixteen must be regenerated or re-paired. All 50 then require complete source-only
semantic inventory compilation. Until both conditions are met, producing “50 passed” output would
repeat the earlier false-pass defect and violate the project's fail-closed quality standard.

The next implementation unit is therefore a topology-first template compiler and target-planning
gate, followed by bounded model assistance only for unresolved natural-language/auxiliary spans.
