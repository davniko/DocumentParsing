# Mixed-equipment synthesis: contained pilot and regression audit

Date: 2026-10-11. Scope: experiment only; no production enablement, training launch, dataset replacement, or template-registry edit.

## Decision

**The contained mixed-GP equipment experiment succeeds:** five existing templates produced three genuinely mixed descendants each, with correct per-container labels, printed equipment aliases, scoped summary counts, allocations and equipment-dependent tares. Fifteen homogeneous matched controls establish that the mixed assignment does not change unrelated labels or rendered text. Position processing introduces no additional coverage loss or geometric defect.

This is a positive finding about **mixing GP length/height categories on independently owned container rows**. It is not a demonstration of reefer/tank/open-top mixtures. It also does not make the current production registry-form sampler mixed-capable: an additional offline test identifies its explicit inventory-contract gate, described below. The generated samples remain inspection artifacts, not training additions.

The full-text audit also finds inherited auxiliary tax/reference artifacts in three families. Those appear identically in controls and do not contradict the reduced extraction labels, but they prevent a complete anonymization/coherence certificate. They are recorded separately rather than called equipment defects or silently approved.

## Inspection artifacts

- [Original source OCR + all 15 mixed renders + complete labels + positioned inputs](analysis/mixed-equipment-pilot-20261011/SOURCES_AND_VARIANTS.md).
- [Independent per-sample certification inventory](analysis/mixed-equipment-pilot-20261011/INDEPENDENT_CERTIFICATION.md), with [hash-bound JSON](analysis/mixed-equipment-pilot-20261011/INDEPENDENT_CERTIFICATION.json).
- [Matched regression, source-integrity, cost and benchmark report](analysis/mixed-equipment-pilot-20261011/matched-regression-audit.json).
- [Geometry audit report](analysis/mixed-equipment-pilot-20261011/GEOMETRY-AUDIT.md), [JSON evidence](analysis/mixed-equipment-pilot-20261011/geometry-audit.json) and [six matched plots](analysis/mixed-equipment-pilot-20261011/geometry-plots/).
- [Mixed inspection dataset](analysis/mixed-equipment-pilot-20261011/campaign/positions-reflow-v1/dataset.jsonl), [homogeneous control dataset](analysis/mixed-equipment-pilot-20261011/control/positions-reflow-v1/dataset.jsonl).

All experiment code, receipts, provider calls, wording, contacts, reviews, adjudications, plain renders and position receipts are together under `docs/analysis/mixed-equipment-pilot-20261011/`. The two arms deliberately reuse the same sample IDs in different directories; never concatenate them into training.

## 1. Source selection and tested mixtures

Sources come from the current reviewed training set and registered template pool, not validation. Their source OCR/target hashes must match their current contracts. All are ambient GP cases with at least two separately identified, independently owned equipment rows. The experiment preserves their row cardinality, identifier/quantity ownership and allocation visibility.

Abbreviations below: 20GP = 20-foot standard-height general purpose; 40GP = 40-foot standard-height general purpose; 40HC = 40-foot high-cube general purpose; 45HC = 45-foot high-cube general purpose. **HC is a height distinction, not a different semantic `typeCategory`; every row here remains `GENERAL_PURPOSE`.**

| Source / training row | Source shape | Variant 1 | Variant 2 | Variant 3 |
| --- | --- | --- | --- | --- |
| `4799921c` / 29 | 5 rows; two-page attachment; summaries for two source subsets | 3 × 20GP + 2 × 40HC | 2 × 40GP + 3 × 40HC | 1 × 40HC + 4 × 45HC |
| `b60ce9db` / 146 | 5 rows; one-page table; whole-inventory summary | 3 × 20GP + 2 × 40HC | 3 × 40GP + 2 × 40HC | 2 × 40HC + 3 × 45HC |
| `06744b8f` / 202 | 2 rows; separate type cells; count-only summary | 1 × 20GP + 1 × 40HC | 1 × 40GP + 1 × 40HC | 1 × 40HC + 1 × 45HC |
| `0bd3ce7b` / 329 | 3 rows; repeated subgroup receipts across pages | 1 × 20GP + 2 × 40HC | 2 × 40GP + 1 × 40HC | 2 × 40HC + 1 × 45HC |
| `e745f5ba` / 551 | 2 rows; singleton receipts; aggregate tare; negotiable consignee | 1 × 20GP + 1 × 40HC | 1 × 40GP + 1 × 40HC | 1 × 40HC + 1 × 45HC |

This covers 17 independently owned source rows and **51 descendant container rows per arm**. It includes both the requested five-container 2+3/3+2 pattern and two-/three-container cases. Table order is retained; different types can be interleaved instead of requiring one contiguous category group.

Exact source IDs and full source content are in the gallery and independent inventory. The source layouts include repeated product wording, attachment pages, printed per-row weights, explicit per-container package allocations, whole-fleet totals, partial fleet summaries and repeated references.

## 2. Experiment architecture and actual flow

The prototype composes the existing campaign rather than building a new lexical or rendering pipeline:

1. The existing R16 scenario catalog samples commodity identities, packages, quantities, allocation counts, routes, locality/company context, vessel/voyage, identifiers and dates.
2. A contained adapter proposes two GP size/height categories for the existing independently owned rows. It preserves container count, equipment identifiers, seals, commodity facts, package counts and allocation ownership.
3. The adapter checks each row's implied gross weight and volume against the selected equipment capacity, using the existing public/private physical ledger and row shares. A non-fitting assignment is not admitted. If no requested mixture fits, the candidate is rejected; there is no conversion to a homogeneous sample to hide the failure.
4. The existing candidate admission and physical planner enforce numeric print precision, totals, row allocations and tares before wording is requested. One preflight candidate was rejected because its physical volume rounded to zero; normal bounded sampling found an admissible replacement.
5. The existing GLM-5.3-Flash native structured-output flow generates company/address/product wording and realistic contacts. Goods/package quantities and equipment remain host-controlled. Targets use the existing uppercase normalization; rendering retains configured style variability.
6. The contained alias projector selects a compatible registered alias per canonical pair, assigns it to each owned type cell, and recomputes each summary from **its own source subset**. A summary is a derived multiset, not a scalar equality constraint that forces all member containers to share one category.
7. Existing full-text review, deterministic target/render validation, publication into the experiment directory, source-conditioned reflow and coherent page transforms run on the actual candidates.
8. Matched controls reuse the accepted base shipment and the same validated wording/contacts, but with its homogeneous equipment. Reuse provenance retains the original generated request hash and the separately revalidated control contract hash. The controls receive fresh whole-text reviews; no model output is represented as having been generated from a prompt it never saw.

### Important containment boundaries

- No production code is patched or monkeypatched. Prototype subclasses and checkers live only in the experiment directory.
- Existing R16 host package realization is preserved in **both** arms. The newer independent package-alias pass is not introduced into this equipment comparison.
- The assignment search deliberately covers a small finite domain: two mandatory distinct categories, `2^N - 2` possible assignments, at most 64. The tested largest `N` is five. Larger domains fail explicitly, not silently truncate. Production integration should use a count/composition-first bounded allocator rather than extend an exponential enumeration.
- Shared scalar equipment owners cannot receive contradictory row categories. The independent checker falsifies this case even though all five selected sources have singleton equipment owners.
- The pilot does not alter source PDFs, OCR inputs, current labels, compiled templates, registry ownership declarations or training configs.

## 3. Regression evidence: what was checked, and what passed

| Component | Measured result |
| --- | --- |
| Actual mixed candidates | 15/15 pass independent per-ID and scoped equipment/count checks |
| Matched homogeneous controls | 15/15 pass the same actual-byte checks |
| Equipment/count spans | 162 checked across both arms |
| Container identifier spans | 102 checked across both arms |
| Non-equipment published target equality | 15/15 matched pairs exactly equal |
| Unrelated rendered-owner equality | 15/15 matched pairs exactly equal |
| Allowed text differences | Equipment-owned spans and equipment-dependent row/aggregate tares only |
| Ordinary target/schema/render/proof/geography validation | All 30 final candidates pass |
| Full semantic reading | All five source OCR/targets and all 15 mixed OCR/targets; controls independently checked and non-equipment content proven equal |
| Whole-text LLM review | All 30 final candidates, with hash-bound findings and adjudications |
| Source/production integrity | All 2,052 pre-pinned files unchanged after the experiment |
| Relevant regression/experimental tests | **388 passed in 21.85 s** |

The pinned integrity set covers production synthesis modules, synthesis configurations, sampling registry JSON, registered template/alignment/contract/target files, ownership declarations and the current real dataset's train/validation/configuration files. It does not pretend to be a pre/post hash certificate for every historical synthetic artifact. This experiment's write paths are restricted to its own directory and the present report.

### Independent equipment checker

The checker is separate from the sampler/physical planner/receipt projector/alias renderer. It:

- compiles row ownership from pinned source bindings;
- distinguishes shared scalar equality, whole-fleet summaries, subset summaries and count-only receipts;
- verifies source bytes, target visibility and equipment identifiers;
- replays all exact edits and independently calculates their output offsets, including prior multibyte Unicode changes;
- reads the actual rendered type/count/ID spans rather than trusting intended replacement values;
- parses their registered canonical meanings and compares them with the actual per-ID labels and scoped count multisets.

It shares the repository's canonical equipment vocabulary, so it is not an independent ISO standards implementation. The scope/byte/projection check is independent of the code that produced the candidate.

### Falsification, not only passing examples

The tests include **500 independently constructed mixed render/label combinations on the five real source blueprints**, and deliberately reject:

- missing rows, duplicate IDs, invalid categories and added fallback-description fields;
- a promised mixed inventory that silently becomes homogeneous;
- wrong source identity, stale source/target contracts and unresolved owner scopes;
- changed or unreplayed raw text, shifted/reordered/duplicate edits and stale quoted bytes;
- mismatched printed equipment versus target category or identifier;
- malformed summaries and incomplete equipment terms;
- unequal categories assigned to a shared scalar owner;
- impossible payloads and assignment domains exceeding the declared search bound.

The independent geometry audit additionally rejects five corruptions: wrong source owner, wrong edit owner, unreplayed rendered text, stale source edit and changed source bytes. These are finite, concrete contract tests; they are not a claim to have tested every conceivable semantic interpretation.

## 4. Geometry and coordinate audit

Each arm has 15 documents / 30 pages / 1,626 eligible nonblank lines:

| Position measure | Mixed | Homogeneous control |
| --- | --- | --- |
| Positioned lines | 1,325 / 1,626, **81.49%** | 1,325 / 1,626, **81.49%** |
| Unknown positions, explicitly ` ||` | 301 | 301 |
| Equipment-owner lines positioned | 33 / 63, **52.38%** | 33 / 63, **52.38%** |
| Container-ID lines positioned | 21 / 51, **41.18%** | 21 / 51, **41.18%** |
| Newly introduced generated-block overlaps | 0 | 0 |
| Newly introduced generated-centroid collisions | 0 | 0 |
| Reflow rejection pages | 0 | 0 |

All 1,626 line coordinates, all source anchor coordinates and all 30 page-node inventories are identical between matched arms. Equipment changes remain in the same source-owned cells and preserve the line structure in this pilot; text is not relocated to an arbitrary new part of the page.

Known coordinates retain valid page association, bounds, transformations and source/edit ownership. Raw text and labels are exactly preserved by positional enrichment. The 120 source-source rectangle overlaps in each arm are inherited measured source boxes; none is a newly introduced overlap. This is a coordinate/anchor geometry audit, not an exact synthetic PDF typesetting certificate.

Unknown equipment lines concentrate in `b60ce9db` (15), `0bd3ce7b` (9), `4799921c` (3) and `06744b8f` (3). `e745f5ba` has complete equipment-line coverage. The four incomplete sources need improved source GLM/Paddle alignment if fuller table-coordinate coverage is desired. This experiment correctly preserves unavailable positions rather than fabricate location evidence. Mixed equipment itself does **not** cause the gaps.

Six side-by-side plots were generated and opened for visual inspection. They show equivalent layouts for controls/mixed samples, including retained unknown equipment rows and source anchored table cells. Examples: [five-container summary](analysis/mixed-equipment-pilot-20261011/geometry-plots/syn_full_v7_43ccf0358cc6d2b26c39a2d2.png), [two independently positioned equipment rows](analysis/mixed-equipment-pilot-20261011/geometry-plots/syn_full_v7_3de8ed470c36158726e883cd.png).

## 5. Findings during generation/review and their disposition

### Resolved before final candidate audit

- An initial assignment enumeration privileged 1+(N-1) compositions. It was replaced in the contained adapter by seeded shuffling of the entire admitted finite assignment domain; the final five-row cases include both 2+3 and 3+2. Changed generation contexts were regenerated, and all attempt costs remain in the ledger.
- Existing bounded lexical validation caught invented package-fill wording in one goods description. Final wording no longer introduces those host-owned facts.
- Existing postal validation caught geography wording errors before publication. Bounded postal repair resolved them; one final `SANA'A` versus supplied `SANAA` disagreement required an explicit pilot wording adjudication. The receipt records the original value and decision. The exact normalized locality appears in both arms. This is not a hardcoded production geography rule or evidence that no future lexical request will require review.

### Equipment review false positives

- A model interpreted registered `45G1` as 45-foot equipment and suggested a wrong label. The actual code means the registry's 40HC pair; the label and actual per-ID proof were right.
- A model claimed all five labels in `b60ce9db` variant 3 were 45HC. Actual text/labels contain three 45HC and two 40HC, with the correct identifiers and summary.

Both were rejected against direct text, labels and registered semantics, without changing good data. Repeated import-reference occurrences likewise retain matching generated values where the source intentionally prints the same value twice; duplication of that printout is not a label/count defect.

### Inherited auxiliary issues: real findings, not mixed regressions

| Family | Inherited text | Effect and disposition |
| --- | --- | --- |
| `4799921c`, all three variants in both arms | Original `TAX CARD:728-559-838`; regenerated importer footer uses a different ID | Source identity remains and the duplicated importer identity no longer agrees. Outside reduced targets; strict auxiliary coherence/anonymization hold. |
| `0bd3ce7b`, all three variants in both arms | `EGYPT IMPORTER` caption with UAE recipient; original exporter/VAT IDs and external-reference footer | Stale auxiliary nationality/identity values. Outside reduced targets; strict auxiliary coherence/anonymization hold. |
| `e745f5ba`, all three variants in both arms | Original export refs, registration/tax number, L/C reference and `BOROUGE` order caption | Source-specific identities/references survive new generated parties. Outside reduced targets; strict auxiliary anonymization hold. |
| `b60ce9db`, all three variants in both arms | Original personal email prefix `amr.helmy`, with a newly generated domain | Partial identity carryover warning, not an unchanged complete email or a target/render disagreement. |

Thus **9/15 mixed samples and their 9 controls** are held under a strict auxiliary anonymization/coherence gate. The remaining b60 prefix warnings are separately visible. All 15 mixed samples still pass the scoped equipment/target audit. Carrier logos, carrier names and legal boilerplate are not equated with generated shipper/consignee identities or indiscriminately anonymized.

No current dataset was repaired on the strength of this small experiment. The full-text audit's scope must not be misread as establishing how many historical descendants share these auxiliary artifacts. Before admitting these pilot families under a strict full-anonymization requirement, add complete auxiliary ownership for those fields, derive repeated IDs from one owner, and rerender/review the affected families. Do not use a blacklist that deletes arbitrary numbers from goods/address/identifier targets.

## 6. Production integration prerequisites, confirmed by offline probes

### A. Current registry-form path needs a real inventory contract

Switching the same five source capabilities to `physical_profile: registry_form` and the currently integrated load-profile sampling config yields **0/15 admitted plans**. Every attempt stops before paid calls at `curated_scenarios.py` with:

```text
mixed-equipment source requires its separate inventory contract
```

This is an intentional source-level guard. It runs before the contained post-base-sampling mixed adapter can run. Deleting that guard or temporarily relabelling the source as homogeneous would not be a correct integration.

The root fix is to make registry-form sampling accept a compiled equipment inventory contract: independent/equality-constrained row groups, receipt scopes, per-row allocation shares and capacities. Sample the category composition/assignment before deriving the profile quantity interval; bound quantity against every assigned row, not one representative homogeneous capacity. Derive both targets and aliases/receipts from this same inventory. No such production change was authorized or made in this experiment.

### B. New package-alias wrapper has an independent ownership conflict

A preflight with the newer package-alias pass finds `e745f5ba`'s adjacent quantity binding targets both the shipment total and the first placement. Those happen to agree in its original source, but correctly diverge in a generated shipment. The existing host physical planner knows the printed row context; the alias wrapper instead sees two different labelled quantities and raises a count-owner conflict.

The experiment preserves the baseline package renderer in both arms instead of disguising that exception as a mixed-equipment defect. The clean follow-up is for package noun/count projection to consume the physical occurrence's authoritative row/total context. Do not add a source-ID exception or arbitrarily choose one of the two quantities.

### C. Minimal configurable production interface

The tested contract suggests a narrow extension, not a second general synthesis framework:

- **Cohort eligibility:** at least two compatible equipment owner groups; source-certified cardinality and receipt scopes. A source with one shared scalar equipment category remains uniform unless its rendering contract is deliberately recompiled.
- **Mixed-document quota:** explicit batch-level count/fraction, measured after admission; do not confuse a probability draw with an achieved quota.
- **Equipment domain:** canonical pair weights and compatibility rules from the existing registry. Ambient GP mixtures first; thermal/DG behavior remains governed by its separate existing requirements.
- **Composition:** configurable exact counts or bounded weighted composition, followed by seeded group assignment and per-row capacity checks. Preserve source cardinality and allocations.
- **Aliases:** current registered pair-compatible aliases, projected to the owned cells and derived summaries. Explicit size/height support must agree with each row's labels.
- **Failure:** an unrepresentable requested mixture is a surfaced source/candidate incompatibility, never a silently uniform fallback.
- **Regression guards:** retain the matched-control target/text/geometry comparison and independent actual-byte checker when integrating with registry profiles and package-alias ownership.

These are identified, bounded next implementation tasks. The mixed rendering contract is experimentally demonstrated; registry-form composition and the package-wrapper ownership issue are not certified by pretending they used the same path.

## 7. Cost, throughput and memory

Actual GLM-5.3-Flash/OpenRouter ledger totals, including discarded attempts and bounded repairs:

| Work | USD |
| --- | ---: |
| Mixed generation, contacts, repair and full-text reviews | 0.02214465 |
| Fresh full-text control reviews; wording/contacts reused | 0.00829990 |
| **Total experiment** | **0.03044455** |

That is approximately **3.04 US cents total**, or $0.00203 per mixed sample **including its matched control review**. It is not a whole-dataset cost projection. Extra offline independent tests/audits incurred no API cost.

Warm physical-plan + alias-projection microbenchmark: 12 rounds × 15 cases per arm, alternating arm order; imports/catalog construction, LLM calls and output persistence excluded.

| Measure | Mixed | Homogeneous control |
| --- | ---: | ---: |
| Renders measured | 180 | 180 |
| Median per sample | 0.865 ms | 0.836 ms |
| Mean per sample | 0.845 ms | 0.826 ms |
| p95 per sample | 1.248 ms | 1.175 ms |
| Peak traced allocations over 15 sequential renders | 54,992 bytes | 57,154 bytes |

Mixed median overhead is **0.029 ms / 3.49%**, negligible relative to generation/review latency. Whole audit process maximum RSS was 1,030,776 KiB, including the two catalog instances used for the control experiment; this is not mixed rendering's incremental allocation footprint. The standalone independent equipment checker measured median 0.240 ms / p95 0.357 ms, 500 checks in 0.121 s, peak traced allocations 54,214 bytes.

Position stages took 1.645 s (mixed) and 0.737 s (controls) on separate invocations. Do not interpret those as an algorithmic regression: cold/warm cache state differs. The actual paired geometry is identical, and the warmed benchmark above controls arm order for the equipment-specific work.

## 8. Reproduction and handoff

Read-only/offline validation and gallery regeneration:

```bash
PYTHONPATH=src TMPDIR=/tmp .venv/bin/pytest -q \
  tests/test_curated_registry_sampling.py \
  tests/test_curated_equipment_aliases.py \
  tests/test_package_aliases.py \
  tests/test_curated_campaign.py \
  docs/analysis/mixed-equipment-pilot-20261011/test_validate_equipment.py \
  docs/analysis/mixed-equipment-pilot-20261011/test_pilot.py

PYTHONPATH=src TMPDIR=/tmp .venv/bin/python \
  docs/analysis/mixed-equipment-pilot-20261011/audit_pilot.py

PYTHONPATH=src TMPDIR=/tmp .venv/bin/python \
  docs/analysis/mixed-equipment-pilot-20261011/audit_geometry.py
```

The generation CLI and exact experiment config remain in the experiment directory, but rerunning paid stages or admitting these samples to training is a separate action. No training command or production synthesis config is changed by this deliverable.

**Recommended next step:** integrate the compiled per-owner mixed inventory into registry-form sampling and physical package/alias projection, then rerun this matched-control test on that real production path before generating the replacement 2,500-sample cohort. Preserve the explicit inherited auxiliary findings and the honest source-coordinate gaps rather than attributing them to the mixed assignment.
