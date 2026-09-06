# Hybrid raw-text synthesis: GLM vs Luna 50-document audit

Date: 2026-09-04

## Executive verdict

The compiler-first rewrite reduced model-visible context and made cost acceptable, but the
current flow is **not training-ready**. The two arms processed the same 50 pinned documents:

| Outcome | GLM-5.3-Flash | Luna Low |
|---|---:|---:|
| Declared `quality_validated` | 20 | 15 |
| Declared `needs_review` | 3 | 1 |
| `call_failed` | 27 | 34 |
| Manually confirmed training-ready | **0** | **0** |

The strict transactional guards correctly prevented partial model failures from becoming
training rows. The problem is that `quality_validated` currently means only that the *declared
residual contract* passed. It does not prove that the entire rewritten OCR is semantically
coherent or fully anonymized.

An independent full-text inspection of every declared pass and reviewer hold found:

- GLM: 14/20 passes have a direct target/dependency contradiction; the other 6/20 retain
  source-only auxiliary or private data needing replacement.
- Luna: 12/15 passes have a direct contradiction; the other 3/15 retain auxiliary/private data.
- All four `needs_review` decisions are meaningful and correct.
- Neither arm published training records.

## Experimental controls

Both arms used the same:

- 50 source OCR texts, source labels, and synthetic target labels;
- deterministic compiler and target-integrity policy;
- changed-line-only residual editing contract;
- provider-native structured output;
- host-owned line IDs and transactional postconditions;
- compact independent reviewer;
- concurrency of eight cases/requests; and
- disabled training publication.

Only the model/provider settings changed:

- GLM: `z-ai/glm-5.3-flash` through OpenRouter, minimal reasoning, DeepInfra then NextBit.
- Luna: `gpt-5.6-luna` through the OpenAI Responses API, low reasoning.

The pinned configs and complete requests/responses/receipts are retained in each committed run.

## One-document comparison

On the same pinned document, both low-reasoning models completed an editor and reviewer call:

| Arm | Provider-stage wall time | Requests | Input | Reasoning | Visible | Cost basis | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| GLM-5.3-Flash | 19.5 s | 2 | 5,288 | 630 | 473 | $0.001329 actual | Declared pass under the earlier contract |
| Luna Low | 16.8 s | 2 | 4,840 | 1,104 | 497 | $0.002889 estimated | Held by the newer auxiliary-gap guard |
| Luna High | 31.1 s | 1 | 2,511 | 4,096 | 0 | $0.005417 estimated | Call failed |

The GLM and Luna outcomes above are not directly comparable as final-quality verdicts because the
auxiliary-gap guard changed between the earlier GLM canary and the Luna canary. The timing and token
measurements remain useful. Luna High exhausted its entire 4,096-token reasoning allowance without
producing visible structured output.

## Fixed 50-document benchmark

| Arm | Pass | Review | Failed | Runner wall | Runner docs/h | Requests | Input | Reasoning | Visible | Total cost | Cost/1,000 attempted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GLM-5.3-Flash | 20 | 3 | 27 | 182.9 s | 984.0 | 74 | 379,817 | 57,900 | 38,425 | $0.104106 actual | $2.082 actual |
| Luna Low | 15 | 1 | 34 | 100.5 s | 1,791.7 | 75 | 368,417 | 27,399 | 37,376 | $0.151413 estimated | $3.028 estimated |

The GLM cost is the downstream provider-reported cost. Luna did not return a billing amount, so its
cost is calculated from the pinned pricing snapshot in the config. The runner throughput values are
useful for this execution, but should not be treated as a hardware-independent production SLA.

### Paired outcomes

| GLM status | Luna status | Documents |
|---|---|---:|
| `call_failed` | `call_failed` | 24 |
| `quality_validated` | `quality_validated` | 12 |
| `quality_validated` | `call_failed` | 7 |
| `call_failed` | `quality_validated` | 3 |
| `needs_review` | `call_failed` | 3 |
| `quality_validated` | `needs_review` | 1 |

The low paired agreement and the manual audit show that pipeline status is dominated by the
contract and stochastic structured-output behavior, not just semantic rewrite quality.

### Provider fallback behavior

DeepInfra returned an upstream `429 engine_overloaded` for all 74 attempted GLM requests. The
application-level route then tried NextBit, which carried all 74 billed GLM calls and prevented the
transport outage from becoming the terminal cause of any document failure. DeepInfra's attempts
had zero model tokens and zero billed cost, but added about 5.27 seconds each.

This validates the explicit fallback mechanism, while also showing that keeping a persistently
overloaded route first adds avoidable latency.

### Model-stage measurements

Zero-token provider transport attempts are excluded from these means:

| Arm | Stage | Billed calls | Successful structured stage | Mean input | Mean reasoning | Mean visible | Mean cost | Mean latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GLM | Editor | 50 | 24 | 5,711 | 1,115 | 744 | $0.001770 | 27.90 s |
| GLM | Reviewer | 24 | 23 | 3,927 | 89 | 51 | $0.000651 | 5.22 s |
| Luna | Editor | 50 | 25 | 5,430 | 377 | 702 | $0.002381 | 11.74 s |
| Luna | Reviewer | 25 | 16 | 3,877 | 342 | 91 | $0.001295 | 6.14 s |

GLM was cheaper per attempted document and produced more declared passes. Luna was faster in this
run. Neither result establishes production quality because both failed the full-text audit.

## Failure taxonomy

GLM's 27 terminal failures comprise:

- 21 host-postcondition failures: seven omitted target literals, four role-bound party occurrence
  mismatches, two date/HS rendering errors, two compound-party errors, and one each for auxiliary
  principal, placeholder, repeated auxiliary identity, jurisdiction, formatting, and anchored
  scalar handling;
- five editor structured-output failures; and
- one reviewer structured-output failure.

Luna's 34 terminal failures comprise:

- 19 host-postcondition failures: eight party occurrence mismatches, seven omitted target literals,
  two stale jurisdiction surfaces, one duplicate line ID, and one anchored-scalar failure;
- six editor structured-output failures; and
- nine reviewer structured-output failures.

The failures were contained: the host applied edits atomically and published no partial result.
Provider-native JSON Schema alone does not express all local invariants, and models can still
produce locally invalid values. One representative Luna response returned a valid JSON object but
attempted empty replacement lines, violating the line-edit schema and topology contract. A
representative GLM response consumed the 8,192-token output allowance and ended with truncated JSON.

## Higher-reasoning Luna probes

Higher reasoning/output budgets were tested only on two representative failure pairs to limit
spend:

| Configuration | Documents | Pass | Review | Failed | Requests | Input | Reasoning | Visible | Estimated cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Medium editor + Medium reviewer | 2 | 0 | 1 | 1 | 3 | 21,254 | 4,316 | 2,719 | $0.012693 |
| High editor + Medium reviewer | 2 | 0 | 0 | 2 | 3 | 21,275 | 10,398 | 2,600 | $0.019853 |
| Low editor + Medium reviewer | 2 | 0 | 0 | 2 | 2 | 11,531 | 816 | 1,315 | $0.004863 |

There is therefore no measured “confidently successful higher-reasoning” cost to extrapolate: no
escalated document passed. Higher reasoning increased spend and did not fix omitted repeated
occurrences, stale facts outside the residual contract, or reviewer-schema failures. Broadly
raising reasoning would be an unjustified cost regression.

## Root cause

The dominant correctness problem is a **coverage boundary in the host contract**:

1. Work items are primarily derived from changed source-label versus target-label leaves.
2. The model sees only selected residual spans around those work items.
3. Source-only values not represented in either label—tax IDs, booking references, named agents,
   operational totals, route boilerplate, product batch details, and similar flavor—are often not
   materialized as work items.
4. The compact reviewer sees the same restricted span set, so it cannot reject stale data elsewhere
   in the document.
5. `residualCandidates` are currently diagnostic only and are not part of `core_passed`.

This explains why every declared pass can satisfy its visible contract while still failing the
actual end-to-end requirement. It also explains why more model reasoning cannot solve the problem:
the missing facts are not in the model's authorized task.

There are additional ownership defects in repeated/nested data. For example, one document's
equipment work item associated all detailed container rows with the same category, so both models
rendered twelve 20-foot containers although the target contained ten 40-foot high-cube and two
20-foot containers. That error originates in host evidence/work-item construction.

## Complete manual audit of non-failed outputs

Legend:

- **Contradiction**: a final-text value directly conflicts with the target or a dependent target
  fact.
- **Auxiliary/privacy gap**: core target rendering appears coherent, but original source-only
  identifiers, contacts, identities, or operational flavor remain.
- **Technical failure**: no candidate output was accepted for that arm.

| Document | GLM status and manual finding | Luna status and manual finding | Ready |
|---|---|---|---:|
| `doc_0951955d` | **Review / contradiction:** refrigeration and `-3 C` remain although target deactivates it; stale `2600 CARTONS`, volume and IDs also remain. | **Technical failure.** | No |
| `doc_0bdef3ca` | **Pass / contradiction:** old container ID and `1 PLT` totals remain beside target container and two packages. | **Technical failure.** | No |
| `doc_0dcc89ac` | **Pass / contradiction:** all 12 detail rows become 20-foot although target is ten 40HC plus two 20-foot. | **Pass / contradiction:** same equipment error; delivery-agent phone is malformed. | No |
| `doc_0eeaecee` | **Pass / contradiction:** stale `1040 BAGS` versus 83 target packages. | **Pass / contradiction:** same aggregate; cargo prose is inserted into a customs-reference slot. | No |
| `doc_11ac17a4` | **Pass / contradiction:** written total 143 versus 1,995 target sacks. | **Pass / contradiction:** same. | No |
| `doc_1249175a` | **Pass / contradiction:** target discharge Alexandria but operational clause retains Port Said West; source IDs remain. | **Pass / contradiction:** same. | No |
| `doc_1c3797a5` | **Pass / auxiliary/privacy gap:** original customs, tax, VAT and registration values remain. | **Technical failure.** | No |
| `doc_1cbeb131` | **Pass / auxiliary/privacy gap:** allocations appear complete, but tax/fax/customs/invoice and party data remain. | **Review / contradiction:** final three-bag allocation omitted; same private data remains. | No |
| `doc_24c81b10` | **Pass / contradiction:** stale 3,016-package aggregate and secondary gross weight. | **Pass / contradiction:** same. | No |
| `doc_38da8dbf` | **Pass / contradiction:** Long Beach terminal and Egyptian/carrier-agent clauses remain against the new route/identities. | **Technical failure.** | No |
| `doc_3e47ddcd` | **Pass / auxiliary/privacy gap:** booking, service contract, VAT/customs/exporter IDs, contacts and external references remain. | **Technical failure.** | No |
| `doc_4c4e0752` | **Pass / auxiliary/privacy gap:** duplicated locality; customs/signing-agent flavor remains; VAT slot deleted rather than synthesized. | **Pass / auxiliary/privacy gap:** original VAT, customs reference and signing agent remain. | No |
| `doc_52f97e4d` | **Pass / contradiction:** issue/signature block remains Ningbo, China instead of Port Macquarie; IDs remain. | **Pass / contradiction:** same. | No |
| `doc_56e593ae` | **Pass / contradiction:** cargo says five units while total remains four. | **Technical failure.** | No |
| `doc_69e32613` | **Review / contradiction:** `SAY SIX PACKAGE(S)` remains against 198 pallets. | **Technical failure.** | No |
| `doc_6ba2f406` | **Technical failure.** | **Pass / auxiliary/privacy gap:** booking, customs/export IDs, contacts, PI and operational references remain. | No |
| `doc_7383087e` | **Pass / contradiction:** target gross 23,180.8 but total remains 22,993.8. | **Pass / contradiction:** same. | No |
| `doc_751f068c` | **Pass / contradiction:** exporter country remains Hong Kong despite Indian target; IDs and fax remain. | **Technical failure.** | No |
| `doc_8c58655a` | **Pass / auxiliary/privacy gap:** exporter country correctly becomes Ukraine, but source contact/tax/customs/booking data remain. | **Pass / contradiction:** exporter country remains China against Ukrainian target; source data also remain. | No |
| `doc_9493b545` | **Pass / auxiliary/privacy gap:** exporter/customs/import IDs and old phone/fax remain. | **Pass / auxiliary/privacy gap:** same. | No |
| `doc_a95a03d3` | **Technical failure.** | **Pass / contradiction:** Italy exporter/agent data remain against Indonesian target. | No |
| `doc_c471b894` | **Pass / contradiction:** target has package levels 1,562 and 31,184, while aggregate/detail rows retain 88/36/52 and old weights. | **Pass / contradiction:** aggregate only renders 1,562 and old detail quantities/weights remain. | No |
| `doc_cf208c27` | **Pass / contradiction:** total gross remains 2,122 versus target 3,893.9 kg. | **Pass / contradiction:** same. | No |
| `doc_dd69725e` | **Review / contradiction:** 1,644 boxes versus 135 pallets, conflicting gross values, and reefer boilerplate against ambient target. | **Technical failure.** | No |
| `doc_f3e40729` | **Technical failure.** | **Pass / contradiction:** customs quantity remains 1,317 versus 622 cartons; old agent/service/export/import values remain. | No |
| `doc_fbfcab7a` | **Pass / contradiction:** exporter country remains USA against Omani shipper; customs/registration/invoice party data remain. | **Technical failure.** | No |

Full document IDs and machine-readable measurements are retained in the analysis CSV files.

## Required next correction boundary

Another broad model run is not justified until the host contract is repaired. The next pass should:

1. Build a complete source-only inventory for identifiers, contacts, auxiliary identities,
   jurisdictions, operational totals, volume, free-time clauses, and route-dependent boilerplate.
2. Generate target-consistent replacements for every retained flavor slot; do not delete the slot
   or use placeholders.
3. Bind repeated/nested facts to indexed container, cargo, and party objects rather than global
   literal matches.
4. Make unchanged high-entropy source values and stale source-dependent values deterministic
   postcondition failures, with reviewed allowlisting only for genuine boilerplate.
5. Give the reviewer the full source/final text plus the inventory, or replace reviewer work with
   deterministic checks wherever possible.
6. Re-run a targeted regression cohort containing the defects above before comparing models again.

Only after that cohort is clean should reasoning levels or models be compared at 50-document scale.

## Artifacts

- Quantitative report: `artifacts/kie-synthesis-analysis/mpci-bl-raw-text-hybrid50-glm-vs-luna-v2/REPORT.md`
- Per-arm/per-document measurements: `.../data/cases.csv`
- Provider-stage measurements: `.../data/stages.csv`
- Paired measurements: `.../data/paired.csv`
- Conservative retained-identifier review queue: `.../data/retained-identifier-review.csv`
- Matplotlib/Seaborn plots: `.../plots/`
- GLM committed run: `artifacts/kie-synthesis/mpci-bl-raw-text-hybrid50-glm53-flash-v7/`
- Luna committed run: `artifacts/kie-synthesis/mpci-bl-raw-text-hybrid50-luna-low-v6/`

