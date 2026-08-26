# MPCI platform categorical-registry follow-up

Use this document as the complete task prompt for a coding/research agent that has read access to
the MPCI platform repository. This is an evidence-recovery task, not a request to guess a taxonomy
or change application behavior.

## Objective

Recover and prove the exact package-type and container-size/type controlled-value contracts used by
the MPCI/CUSCAR application. The result must let the DocumentParsing project construct readable,
learnable model categories with a thin deterministic mapping to the application's actual wire
codes.

The current dataset publication is intentionally blocked because the supplied platform analysis
identifies the registry owners but does not contain their values. Current labels mix apparent ISO
codes and carrier/application aliases, including `45G1`, `40GP`, `HC40`, and `40HQ`; do not treat a
four-character string as canonical merely because its shape looks plausible.

## Repositories and immutable inputs

In the DocumentParsing repository, read all of:

- `artifacts/mpci-ai-schema/README.md`
- `artifacts/mpci-ai-schema/selects-and-code-lists.md`
- `artifacts/mpci-ai-schema/field-catalog.md`
- `artifacts/mpci-ai-schema/ai-genie-normalization-and-mapping.md`
- `artifacts/mpci-ai-schema/schema-layer-matrix.md`
- `artifacts/mpci-ai-schema/open-questions-and-conflicts.md`
- `artifacts/mpci-ai-schema/mpci-cuscar-canonical.schema.json`
- `docs/mpci-bl-next-experiment-preparation-2026-08-22.md`
- `src/document_ocr/semantic_v3/transform.py`
- `configs/transforms/mpci_bl_semantic_v3_relation_explicit_table_view.yaml`
- `artifacts/kie-training/datasets/mpci-bl-semantic-v3-relation-explicit-table-view-readiness-v2/category-registry.schema.json`
- `artifacts/kie-training/datasets/mpci-bl-semantic-v3-relation-explicit-table-view-readiness-v2/category-assignments.schema.json`
- `artifacts/kie-training/datasets/mpci-bl-semantic-v3-relation-explicit-table-view-readiness-v2/package-category-review.jsonl`
- `artifacts/kie-training/datasets/mpci-bl-semantic-v3-relation-explicit-table-view-readiness-v2/container-category-review.jsonl`
- `artifacts/kie-training/datasets/mpci-bl-semantic-v3-relation-explicit-table-view-readiness-v2/summary.json`

The prior platform investigation inspected:

- revision: `72f9aff735c2581f2ccff74e7fde34c2aea9b265`
- branch: `feature/CHIC-1647-ai-parsing-error-slack-message`
- expected registry package: `@cargox/shared/schemas/edifact/util-cuscar`
- expected source family: `libs/shared/schemas/edifact/util-cuscar/src/lib/entities/*.ts`
- option projection: `libs/shared/schemas/edifact/util-cuscar/src/lib/form-options.ts`
- expected container source: `container-size-types.ts`

Work from a detached checkout of that exact revision so the result is reproducible. Separately
determine whether that revision was deployed for the application/data period in scope; an inspected
feature-branch revision is not deployment evidence.

## Fixed ML boundary

These decisions are already made and are not questions for this investigation:

- T5Gemma 2 270M will emit readable semantic category tokens, not opaque numeric/wire identifiers.
- A deterministic projector will map each category token to exactly one proven application code.
- Package and container categories are separate registries.
- Exact source strings are reviewed against those registries in a later, separate task. Do not
  silently perform the 99 package and 105 container source-key assignments here.
- Printed container descriptions remain available even when a category is assigned.
- Underspecified text such as `40` or `20'` must remain unresolved rather than receive a guessed
  type.
- Countries, localities, and ports remain as printed model output. ISO-2 and UN/LOCODE resolution is
  downstream and outside this task.
- UI-localized labels are presentation text and must not become stable model tokens.

The current 487-document audit contains 658 package occurrences across 99 exact source variants and
905 container occurrences across 105 exact source variants. Container evidence is 812 text-only,
37 text-plus-code, and 56 code-only occurrences. The observed `typeCode` inventory currently mixes:

```text
20DC  20DV  20GP  40GP  40HC  40HQ  40RH  45G1  DV20  HC40
```

Every one of these values must be classified using actual platform behavior as canonical, accepted
compatibility alias, rejected, or unresolved.

## Investigation method

Trace each registry end to end:

```text
source constant/export
  -> form option value
  -> shared Zod acceptance
  -> form/autosave persistence
  -> CUSCAR converter acceptance
  -> emitted EDIFACT value
```

Use executable probes against the real exported package wherever possible. Record exact commands,
package/lockfile versions, source paths and symbols, and SHA-256 hashes. Do not infer runtime
acceptance solely from TypeScript types or UI options.

For each canonical code, establish:

1. the unchanged raw exported object;
2. its stable nonlocalized semantic name or enum key;
3. the user-visible English meaning, clearly separated from localized display text;
4. whether the value is shown in the UI;
5. whether shared Zod validation accepts it;
6. whether the converter accepts it;
7. the exact value emitted to EDIFACT;
8. whether it is deprecated, restricted, or filtered by operator, filing mode, document type, or
   feature flag; and
9. aliases accepted at any boundary, with precedence and normalized output.

Also test the existing ai-genie normalizers. A value accepted only by ai-genie is a compatibility
alias, not a canonical registry value. Keep canonical values and aliases disjoint.

## Questions that must be answered

### Package types

- What is the complete deployed package-code registry, its order, and its authoritative source?
- Are all stored/submitted values exactly two-character UNECE Recommendation 21 codes?
- Does the application registry equal its upstream UNECE source, or is it a curated subset/version?
- Can `typeOfPackages` free text exist without `packageTypeDescriptionCode` in a partial draft?
- Can that free-text-only row pass the full submission/converter boundary?
- May code and free text coexist, and if so, which controls emitted EDI semantics?
- Are aliases, deprecated codes, duplicate display meanings, or multiple legal codes for one
  concept present?

### Container size/type

- What is the complete deployed container size/type registry, its order, and source?
- Are all stored/submitted canonical values exact ISO 6346 four-character codes?
- Is the registry a full ISO/BIC list, a versioned subset, or an application-specific extension?
- Which of the ten observed values are canonical codes versus aliases or invalid values?
- Are shorthand families such as GP, HC, RF, OT, FR, TK, and BU accepted only by ai-genie, or also
  by form validation/conversion?
- Can `equipmentDescription` without `containerCode` survive a partial draft?
- Can it pass full submission, and what reaches EDIFACT if the code is absent?
- Are old decomposed length/height/type fields still accepted or emitted, and with what precedence
  relative to `containerCode`?

### Other controlled values

Audit the already-described smaller categoricals for contradictions rather than redoing the full
schema study: negotiability, freight payment, mass/volume/temperature units, IMDG hazard class,
dangerous-goods packing group, and path-implied party/route/text/measurement qualifiers. Resolve in
particular what application packing-group code `4` means; the ML schema currently has only the
readable I/II/III danger levels.

Identify any additional document-derived categorical field for which the current ML target would
need a readable category plus deterministic projection. Do not include system/user/context-only
fields merely because the form exposes them.

## Required deliverables

Write the results under a new `categorical-registry-followup/` directory in the platform research
artifact area. Do not overwrite the original investigation. Produce all of the following.

### 1. Raw exports

- `package-registry.raw.json`
- `container-registry.raw.json`

These must preserve the platform's exported objects unchanged and include, in an enclosing metadata
object or companion evidence file, the export symbol, exact source path, source revision, source-file
SHA-256, workspace package version, and lockfile/package-resolution evidence.

### 2. DocumentParsing category registries

- `package-category-registry.json`
- `container-category-registry.json`

Each must validate against the supplied `category-registry.schema.json`:

```json
{
  "schemaVersion": 1,
  "registryKind": "package",
  "sourceAuthority": "...",
  "sourceRevision": "40 lowercase hex characters",
  "sourcePath": "...",
  "sourceSha256": "64 lowercase hex characters",
  "entries": [
    {
      "categoryToken": "READABLE_STABLE_TOKEN",
      "applicationCode": "PROVEN_CODE",
      "displayName": "Canonical English meaning"
    }
  ]
}
```

`categoryToken` must be derived from a stable nonlocalized semantic source, not generated from a
translated UI label. If the platform has no stable semantics sufficient to create a one-to-one
token/code mapping, do not invent one: publish a blocking analysis instead.

### 3. Runtime conformance matrix

Create `registry-runtime-conformance.jsonl`, one row per canonical code, with at least:

```text
registryKind, applicationCode, sourceKey, canonicalEnglishMeaning,
uiPresent, zodAccepted, converterAccepted, emittedEdiValue,
modeOrOperatorRestrictions, deprecated, sourceEvidence
```

### 4. Alias evidence

Create `container-alias-resolution.jsonl`. It must cover at minimum all ten observed values and
record:

```text
observedValue, classification, canonicalApplicationCode,
aiGenieAccepted, zodAccepted, converterAccepted, precedence, sourceEvidence
```

Allowed classifications are `canonical`, `compatibility_alias`, `rejected`, and `unresolved`.

Create the equivalent `package-alias-resolution.jsonl` if the platform exposes package aliases or
normalizers.

### 5. Boundary behavior reports

- `package-fallback-behavior.md`
- `container-fallback-behavior.md`
- `categorical-crosswalk-audit.md`

The fallback reports must distinguish nullable/deep-partial draft acceptance from complete
submission validity. The crosswalk audit must resolve packing-group code `4` and list any remaining
controlled-value uncertainty.

### 6. Evidence manifest

Create `evidence-manifest.json` containing:

- inspected commit and branch/tag;
- deployed-build evidence or an explicit `unresolved` deployment status;
- every source and generated artifact SHA-256;
- exact extraction/probe/test commands and their exit status;
- package-manager/lockfile versions;
- counts and ordering hashes for both registries;
- explicit unresolved items; and
- a statement that no application runtime source or behavior was changed.

Also write `README.md` with the concise findings, decisions unlocked, and remaining blockers.

## Acceptance gates

The work is complete only when all of these are true:

- Registry values are extracted from the platform source and mechanically compared across UI/Zod,
  persistence, converter, and EDIFACT boundaries.
- Every mismatch is explained; no mismatch is silently normalized away.
- Package canonical codes are proven two-character values and container canonical codes are proven
  four-character values, or the contrary is explicitly evidenced as a blocking contract conflict.
- Canonical codes and compatibility aliases are disjoint.
- `applicationCode` and `categoryToken` are each unique within a registry.
- Every token has one stable, nonlocalized semantic meaning and maps to exactly one application code.
- Draft fallback behavior and submission behavior are tested separately.
- Registry filtering/restrictions are identified.
- Every claim cites a source file/symbol/line or executable probe.
- Generated JSON validates against the supplied schema.
- Any unresolved uncertainty remains explicit and leaves downstream semantic-v3 publication
  fail-closed.

Do not use the public UNECE or BIC lists as substitutes for the deployed application snapshot. They
may be used only as independent cross-checks, cited separately from platform proof.

## Final handback

Report:

1. registry counts and hashes;
2. whether the inspected revision is deployment-relevant;
3. canonical-versus-alias results for all observed container codes;
4. package/container draft and submission fallback behavior;
5. the resolution of packing-group code `4`;
6. exact tests/probes run and their outcomes;
7. paths and SHA-256 values for every deliverable; and
8. any blocker that still prevents building the semantic-v3 category mappings.

Stop rather than guess if repository access, deployed-build provenance, runtime dependencies, or
authoritative values cannot be established.
