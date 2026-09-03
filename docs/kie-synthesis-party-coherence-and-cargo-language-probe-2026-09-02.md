# Party coherence and cargo-language probe

Date: 2026-09-02

## Decisions proved in this pass

### Notify-party semantics

The task schema remains intentionally asymmetric:

- an explicitly printed `SAME AS CONSIGNEE` or `SAME AS SHIPPER` phrase is represented by the
  compact `sameAs` relation;
- a fully printed notify block remains a concrete notify-party object, even when its values equal
  the consignee or shipper;
- the MPCI projector expands `sameAs` by copying the referenced party identity/location and then
  applying any explicitly printed notify contact override;
- no boolean and no duplicated party payload is added to the model-facing relation target.

The route sampler previously broke the second case by assigning repeated concrete notify blocks a
new locality. It now detects a conservative exact source identity match (same name plus at least one
corroborating exact address/city/country field), retains a concrete target, reuses the referenced
party's generated geography, and serializes `concreteIdentitySource` for the later party-identity
stage. It never converts a concrete block into `sameAs`.

Real-data audit over the pinned 100-document route selection:

| Check | Result |
|---|---:|
| Explicit printed `sameAs` relations | 21/21 unchanged |
| Concrete repeated notify/consignee identities | 47 |
| Concrete repeated identities kept geographically coherent | 47/47 |
| Divergences | 0 |

The route stage remained performance-neutral within run noise:

| Run | Elapsed | Peak RSS |
|---|---:|---:|
| Before | 4.636 s | 445.219 MiB |
| After | 4.501 s | 443.812 MiB |

### Linguistic-generation boundary

Only target values whose labels retain printed language are generated before raw-text patching.

| Separate linguistic generation | Final raw-text patching only |
|---|---|
| cargo `description` | printed package-type spelling |
| `additionalInformation` leaves already present in the template | printed container/equipment spelling |
| substantive `marksAndNumbers` | HS-code formatting/surface |
| `handlingInstructions` | formatting of dates, identifiers, measures, and other structured facts |
| party identities and contacts (separate party stage) | placement, punctuation, casing, and surrounding boilerplate |

Generic marks such as `N/M` and `NO MARKS` are neither regenerated nor anonymized: they are copied
verbatim. Substantive marks are generated as new fictional values.

## Five-case constrained probe

Artifact:
`artifacts/kie-synthesis/mpci-bl-cargo-language-luna-high-probe5-v1/`

Every case was an independent GPT-5.6 Luna High request through provider-native strict JSON Schema.
There was no shared conversation, semantic repair call, reviewer, raw OCR, image, or PDF. Complete
model-visible requests, provider transcripts, response IDs, outputs, validation, usage, and pricing
receipts were retained.

| Case | Coverage | Generated target-facing text |
|---|---|---|
| 1 | ordinary cargo + substantive mark | `POLYURETHANES`; `VELORA` |
| 2 | two cargo groups + two marks | transformer and filling/labelling machinery descriptions; `RZ8 47126`, `LK5 80317` |
| 3 | frozen cargo + handling | `FROZEN COD`; `KEEP FROZEN AT -20°C` |
| 4 | DG + flashpoint semantics + auxiliary text + three marks | `HEXAMETHYLENEIMINE`; `LIQUID CHEMICAL`; three new marks |
| 5 | ordinary cargo + auxiliary measure + generic mark | paper packing-container description; `76.8 CBM`; literal `N/M` |

Aggregate:

| Metric | Result |
|---|---:|
| Independent requests | 5 |
| Provider schema-valid outputs | 5/5 |
| Automatic semantic repair requests | 0 |
| Current deterministic validation passes | 5/5 |
| Input tokens | 7,177 |
| Output tokens | 1,034 |
| Reasoning tokens | 638 |
| Visible output tokens | 396 |
| Estimated API cost | USD 0.0030343 |
| Concurrent wall time | 7.422 s |
| Mean per-case latency | 5.478 s |
| Peak process RSS | 127.051 MiB |

The immutable run originally recorded 4/5 deterministic passes because the first validator ignored
three-letter semantic words and therefore failed the correct `FROZEN COD` response while retaining
only the longer taxonomic terms. The validator now keeps meaningful three-letter words; a regression
test covers this exact case, and all five retained outputs pass current validation without another
paid request.

## Quality observations for the next review

The five outputs are structurally correct, source-topology preserving, and semantically consistent.
Three quality leads should be resolved or measured rather than hidden:

1. An HS leaf description such as `Of polyurethanes` is weak upstream identity evidence. The output
   `POLYURETHANES` is correct but underspecified relative to the supplied heading. Fragmentary HS
   leaves may need a deterministic leaf-plus-heading semantic seed.
2. `LIQUID CHEMICAL` is consistent auxiliary text for hexamethyleneimine but is less informative
   than the formulation-style source slot. A larger probe should measure whether additional-
   information slots need a typed semantic purpose (formulation, grade, packaging statement,
   measure, or origin) before production integration.
3. The linguistic model correctly avoided inventing package surfaces, but its upstream structured
   seeds exposed package/goods compatibility defects: frozen cod was paired with `PACKAGE_ROLL`,
   and liquid hexamethyleneimine was paired with `PACKAGE_BAG`. The language stage must not repair
   or hide an authoritative structured-plan error. Package-category sampling therefore needs to be
   conditioned on the newly sampled goods/physical form before a larger end-to-end synthesis run.

These findings do not change the categorical printed-surface boundary: package, container, and HS surfaces
correctly do not appear as independently generated output fields.
