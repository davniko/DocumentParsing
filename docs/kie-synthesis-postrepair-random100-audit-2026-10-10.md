# Post-repair synthesis verification and second random 100-sample audit

Date: 2026-10-10. This pass concerns the dataset, source/template casting and
synthesis path. Training-result analysis is deliberately deferred.

## Subsequent repair and date-policy correction

The [party/date/delivery repair](kie-party-date-route-repair-2026-10-10.md)
has now been published and validated. Its adopted policy supersedes this audit's
date adjudication below: ambiguous numeric order defaults to **DD-MM**, not an
absent date. Source `0d102efa` and all nine current descendants now include their
supported issue dates; the earlier omission was a policy error, not a correctly
rejected review finding. The original observations below remain chronological
evidence and must not be used as current acceptance rules.

That subsequent repair also resolves primary/represented-company targets and
the six unsupported real delivery occurrences, including the three active
families and their 21 current descendants. See its exact applied census rather
than treating the historical held/pass table here as current dataset status.

## Outcome

The [previously enumerated repair](kie-synthesis-field-boundary-repair-2026-10-10.md)
is implemented in the dataset, source contracts, ownership declarations and
generation path. This pass additionally ran actual fresh synthesis; it did not
rely only on replaying saved wording.

The new random audit is complete: **50 new source families, 100 current
descendants, two per family**, with complete original OCR and complete rendered
OCR/labels opened for every pair. Final dispositions are **91 scoped passes,
four confirmed casting/anonymization defects, and five review holds**. These
statuses are final audit decisions, not applied repairs. Two initial date-error
allegations were explicitly withdrawn after checking the actual date policy.

The fresh upstream test confirms the repaired mechanisms execute, but also
finds an adjacent generated package-count contradiction. Its candidate is held,
not included in training. Thus the previous repair is complete within its
enumerated scope; this audit does **not** certify every future generated document
or every inherited private reference as clean.

Key evidence:

- [Complete 50-source/100-descendant markdown inventory](analysis/synthesis-postrepair-random100-20261010/INVENTORY.md).
- [Frozen random draw and hashes](analysis/synthesis-postrepair-random100-20261010/draw.json),
  [final machine-readable decisions](analysis/synthesis-postrepair-random100-20261010/manual-inventory.json),
  [explicit adjudications](analysis/synthesis-postrepair-random100-20261010/adjudications.json).
- [Fresh synthesis execution, individual decisions, cost and geometry](analysis/synthesis-postrepair-random100-20261010/upstream-synthesis/REPORT.md).
- [Confirmed-family census and actual identifier-function counterexample](analysis/synthesis-postrepair-random100-20261010/family-closure.json).

## Which dataset was inspected

The inspected current repaired publication is
`data/curated/mpci-bl-real600-synthetic2500-v7-positions-boundary-repaired-v1`:
600 real training documents, 2,500 synthetic training documents, and 60 real
validation documents. This random pass selects synthetic documents only.

The previously running/completed experiment's predecessor,
`mpci-bl-real600-synthetic2500-v7-positions-v1`, remains untouched. The earlier
repair was published as a sibling dataset with a matching `boundaries` training
configuration; this audit does not retroactively change the data consumed by an
already completed training run.

The earlier completed repair changed 151 synthetic records across 41 source
families and two real training labels. It included the older 500-sample cohort,
not just the latest 1,000. The source registry has 300 current families; the
previous report documents its exact changes, validation and publication hashes.

## Random selection and actual inspection method

The population contains 300 source families and 2,500 current synthetic records.
The previous random audit covered 50 families, and the final changed-record
repair review covered 41. Their union is 74 families. This draw excludes **all
74**, leaving 226 eligible families. Therefore none of the 50 sources here
overlaps either previous review set.

Selection used one saved random seed, `9654580258061073925`, with no redraws:
uniform 50-family selection without replacement, followed by two uniformly
selected, distinct current descendants per source. It is a family-stratified
sample, not a document-weighted estimate of the whole dataset's error rate.
The selected families comprise 20 original-100, 16 expansion-200 and 14
expansion-300 sources.

Root directly inspected sources 01–10; four review assistants directly inspected
11–20, 21–30, 31–40 and 41–50. Each read the entire original OCR/source target and
both entire descendant OCR/targets, including continuation pages. Root then
adjudicated the flagged cases using current policies, contracts and targeted
function/data probes. This is actual Codex text/semantic inspection, not a claim
of 100 human reviews, 100 independently trained model reviewers or a regex-only
scan. The random audit incurred no additional provider API calls.

Each source's complete inspection card contains the original source text, source
target and both full rendered descendant texts and targets. Receipts identify
the reviewer, actual-read completion, retained-field checks, exact findings,
recommended action and final disposition. Source, source-target, row, raw-input,
positioned-input and card hashes bind these decisions to specific versions.

Policies used:

- [Baseline labeling decisions](kie-real-baseline-label-policy-2026-10-05.md).
- [Main description passage and field ownership](description-boundary-repair-2026-10-09.md).
- [Declared package accounting and allocations](package-accounting-repair-2026-10-09.md).
- Current schema field descriptions, including ambiguous-date handling.

No new rejection criterion was introduced for neighboring HS product classes,
container physical payload limits, deliverable postcodes, shared email/website
domains, or unchanged permitted carrier/legal wording. Address punctuation was
not revised in this pass. Product capacities remain distinguishable from
shipment accounting.

## Final random-audit dispositions

| Outcome | Documents | Meaning |
| --- | ---: | --- |
| Pass, no active finding | 16 | Inspected retained fields/casting and anonymization review have no active finding |
| Pass, secondary follow-up | 75 | No blocking retained-field/casting issue; specific non-target private/context residues remain recorded |
| Confirmed defect | 4 | Two retained original principal identities; two split exporter co-references |
| Review needed | 5 | Two name/service boundaries, one goods-topology question, two OCR-occurrence/route-role questions |
| Total | 100 | Exactly two descendants from each of 50 new sources |

**82 documents have secondary findings**, overlapping the categories above.
This is not 82 incorrect extraction labels. Nor does a scoped pass with such a
finding certify complete source anonymization. Four confirmed defects are also
not four invented or mismatched goods/address labels: their retained targets
agree with their text, but their source casting/anonymization is faulty.

No definite description transcription, declared-package total/allocation,
thermal-setting or DG-label mismatch was established in these 100 current
records. The goods-topology hold is separate and must not be counted as a
description-copying pass for that unresolved component.

### Confirmed defect 1: one original principal company remains literal

[Source 39, both selected descendants](analysis/synthesis-postrepair-random100-20261010/cards/39-e5a6a403.md),
family `doc_e5a6a40300b6e50a58dd6db5e6c1c9da7f58d71b15ab3dc683561b711eb99325`.

The source shipper identity is a principal plus represented company. The current
contract's expression is:

```text
LKW-HANDEL M.NOCK O/B {v006}
```

`v006` generates the represented/notify organization, but the principal remains
literal. For example, the new label is:

```text
LKW-HANDEL M.NOCK O/B MESURADO IMPORT GROUP
```

The generated postal address, route and other organization are new. The same
principal is printed in the input, sometimes with render-style casing variation.
Therefore input/target agreement holds, but primary-company anonymization does
not. A check that merely finds some generated name variable in the composite
target cannot establish ownership of all independent identity components.

The closure census confirms **all six current family descendants** retain this
principal, not only the two random draws. It compares the exact source principal
with target identity and OCR, allowing only whitespace/casing variation in the
printed occurrence. This is not a fuzzy company-name search over all documents.

Required subsequent correction: introduce a separate generated principal-name
owner and cast it consistently into the shipper input and target. Preserve the
O/B framing and independently shared represented/notify organization. Validate
component-level generation coverage, not just variable presence in the overall
name path. No correction was applied during this read-only audit.

### Confirmed defect 2: repeated exporter registration becomes two numbers

[Source 48, both selected descendants](analysis/synthesis-postrepair-random100-20261010/cards/48-9839db70.md),
family `doc_9839db703d177b5af7fbb9b454f5c1932c2e64c81ad5958c18e4a620e2d08363`.

The original OCR repeats one registration:

```text
CN-01-91330782MA28ECL35C
...
USCI:91330782MA28ECL35C
```

Both auxiliary declarations explicitly use
`identifier: source_reference_91330782MA28ECL35C`, yet one new descendant prints:

```text
EXPORTER REGISTRATION: 93576159711546
USCI:821178581748
```

Actual source PDF inspection confirms the repeated original identity. Actual
`_render_recipe` inspection and an execution probe confirm the cause: without a
certified `canonical_identity`, the generator normalizes the **whole original
spelling**. The first becomes `CN0191330782MA28ECL35C`, the second
`91330782MA28ECL35C`. This changes both random namespace and numeric length.
Matching recipe names alone are not used as the generated identity certificate.

Using the current production function, the read-only probe produced different
numbers, `92049236103920` and `264778110111`. A control with the existing explicit
`canonical_identity: 91330782MA28ECL35C` produced `264778110111` for both.
The live contracts were not altered to obtain the control.

The closure census confirms **all nine current family descendants** split this
same registration. Their retained extraction targets remain correct; the defect
is same-owner coherence in non-target rendered text. Route loading, discharge
and delivery fields in this family are correctly printed and are not defects.

Required subsequent correction: certify the shared identity in the two existing
auxiliary owners and update only their current descendant occurrences. Do not
strip arbitrary prefixes globally or merge nearby numbers by digit similarity.
The current mechanism already supports an explicit certificate; this diagnosis
does not require inventing a new identifier subsystem.

### Review holds: five documents, three narrow questions

1. **Two generated delivery-agent identity continuations — source 12.**
   [Complete pairs](analysis/synthesis-postrepair-random100-20261010/cards/12-88bab156.md).
   The second generated lines are `PORT AGENTS AND LOGISTICS HANDLING` and
   `TERMINAL HANDLING AND CUSTOMS CLEARANCE`, included in `name`. They read as
   services rather than clearly named identities; the source example had two
   GMAC-named lines. OCR/label agreement and package accounting are sound.
   Do not certify a field-semantic boundary solely from exact copying or
   automatically erase an uncaptioned continuation. Resolve only this identity
   wording/boundary, not the whole shipment. The family contains ten current
   descendants; the two selected cases are held, not all ten declared wrong.

2. **One newly quantified product list — source 17.**
   [Complete pairs](analysis/synthesis-postrepair-random100-20261010/cards/17-81877adf.md).
   The generated variants declare 60 nylon, 50 polypropylene and 45 polyester
   cartons. They sum correctly to 155. The description target copies the text
   correctly. However, independently accounted products are a goods-topology
   signal under the policy, and the lexical model introduced this split without
   host-provided per-product quantities. Decide whether to remove those invented
   accounting assertions through an approved wording correction or explicitly
   adjudicate the goods grouping. Do not turn correct arithmetic into a pass
   for an unresolved one-goods topology. Only one of the two selected records
   has this issue; the family contains nine descendants.

3. **Two delivery roles with omitted OCR occurrences — source 47.**
   [Complete pairs](analysis/synthesis-postrepair-random100-20261010/cards/47-291f67b3.md)
   and [actually inspected original PDF](../data/corpora/blc-swb-remaining-max15-23947b3edb6b/files/blc/2024/2024-05-02_3522dc0b-a36b-48db-aa7c-7959d6d85255.pdf).
   The PDF really has a filled Place of Delivery box beside Port of Discharge,
   both `ELSOKHNA, EGYPT`. GLM omitted the delivery caption and repeated value.
   The remaining equal strings/positions belong to discharge and freight
   payment. The label's delivery value is historically true, but its particular
   role is not represented by an OCR occurrence. Do not state that the source
   lacks delivery information, and do not treat equal words elsewhere as proof
   that the role survives OCR. This is a narrow grounding decision, not a
   defective discharge/payment label. The family contains nine descendants;
   this audit holds the selected two pending the occurrence-level decision.

### Historical date adjudication — superseded by the DD-MM repair

**Correction:** the decision recorded in this section used the wrong policy.
It is preserved as the original adjudication history, not a current conclusion.
The supported date additions described in the subsequent repair are applied.

Source 08's `03/06/2025` issue date initially looked like an omitted label.
The actual `issueDate` description says numeric day/month order must be
established by format evidence or unambiguous same-convention dates, never
country, issuer or language. It expressly requires absence when both orders
remain possible. The complete source has no resolving format evidence.

Therefore both descendants **pass** on this point. The suggested
`2025-06-03` addition was withdrawn and never applied. All nine family descendants
share that deliberately unresolved date. The raw initial observations remain
preserved, and the final decisions explicitly record their withdrawal. This is
adjudication of an incorrect review claim, not a dataset regression or repair.

### Secondary residues and what they affect

The 82 documents with secondary findings contain identified or unclassified
non-target context requiring separate ownership/anonymization review:

- Original tax/exporter/importer registrations and private booking, invoice,
  purchase-order, customer-code, service-contract and shipping-order references.
- Secondary invoice/correspondence organizations, personal signatories and CPF
  numbers, or a retained personal email local part under a new domain.
- Country-specific customs/warehouse/agency captions after primary geography
  changes, and retained company names in auxiliary order captions.
- A few uncaptioned footer strings whose roles are not established; they are
  flagged for classification, not guessed into extracted B/L/reference fields.

They are individually quoted in the inventory, with ownership-based suggested
actions. Most are excluded from the retained targets. They still matter for
source anonymization, unwanted memorization and coherent future casting; they
are not automatically physical or retained-label inconsistencies. Allowed
carrier/issuer/legal constants are explicitly distinguished and not blanket
removed. No whole-dataset prevalence estimate is inferred from these 82 flags.

## Fresh upstream synthesis verification

Earlier validation included fresh component requests and deterministic renders,
but not the combined new campaign requested here. This pass therefore executed
the **actual production Campaign** on 13 targeted repair-witness families,
new seed `2026101017`, concurrency four, using GLM-5.3-Flash. Fresh test families
deliberately include earlier repaired sources; they are separate from the random
audit's no-overlap requirement.

Executed stages: current blueprint/host preflight, registry sampling, structured
lexical generation, company contact generation, bounded wording repair, postal
checking/correction, exact rendering and target construction, rendered-text
model review, and production coordinate transfer/reflow/augmentation with
independent receipt validation. All 13 complete fresh outputs were also manually
inspected. Test outputs remain inspection-only.

Observed repaired mechanisms include package nouns versus headings, repeated
totals versus stale row counts, preservation of legal clause 14, main-product
start boundaries, all missing composite-name components, personal versus
organization identities, postal ENVNT exclusion, duplicate-country handling,
and certified VAT co-reference. The current thermal witness drew cartons, not
bags; earlier BAG counterfactuals are separate evidence, not a falsely claimed
fresh BAG draw.

Actual provider usage: **30 calls, $0.01709345 total** (about 1.7 US cents),
78,333 input tokens, 10,687 output tokens including 4,341 reasoning tokens.
This includes fresh generation, reviews and both bounded repair calls. No
failed transport calls or unreported extra generation occurred. The per-fresh-
candidate figure is about $0.001315; it is not a scalable quality guarantee.

### What the fresh test did not pass

Five candidates were plain scoped passes and two passed with recorded private-
context follow-up. Six have separate holds, not six proven bad extraction labels:

- **One confirmed new package-count contradiction:** host total three packages;
  two generated product lines claim `3 X PACKAGE` and `1 X PACKAGE`. Unlike
  retail capacities or container payload realism, these are conflicting shipment
  unit declarations. The model reviewer caught it. The deterministic accounting
  guard did not cover this embedded product-line phrasing.
- One unsupported `9.391 KG OF ...` phrase generated without supplied product/
  shipment mass. It is not proven greater than the host gross weight; its
  accounting meaning is unresolved. The rendered reviewer did not flag the mass;
  manual inspection did. It remains a hold, not an invented overload claim.
- One total `PACKED IN 1898 CARTONS` repeated inside both individual container
  descriptions. The real split is 936 + 962; global versus row-local reading is
  ambiguous, not proof of 3,796 cartons.
- Two fixed secondary private-identity blocks and one Egypt-specific importer
  caption after a geography change.

The normal publication validation was invoked and blocked; no adjudication
manifest was fabricated to force publication. No fresh candidate was added to
training. These counterexamples are recorded rather than hidden by rerunning
until a model happens to produce clean wording. The next correction should
target these specific generation/owner issues, not introduce broad removal of
product quantities, capacities or arbitrary numeric text.

## Mechanical integrity, geometry and falsification

### Current random 100 records

All 100 selected rows:

- Validate under the actual reduced training schema/canonicalizer loaded through
  the repaired production training configuration and task constraints.
- Equal their current `train.jsonl` mirrors exactly.
- Preserve raw text exactly in positioned input, apart from the intended
  coordinate suffixes; raw/positioned hashes are valid.
- Keep known normalized coordinates within 0–1000 and retain explicit unknowns.

The sample covers **194 pages, 9,717 content lines, 8,035 positioned lines
(82.69%)**. The mechanical pass took 2.119 seconds, peak RSS 983.36 MiB including
training-stack imports. This is not a visual collision/layout certificate for
every existing synthetic page.

### Fresh production geometry path

All 13 fresh candidates, including semantic holds, were separately exercised
through the production geometry path in an explicitly inspection-only directory:

- 31 pages, 1,493 / 1,728 lines positioned (86.40%). Joint reflow adds 41 useful
  positions over the measured-anchor transfer baseline of 1,452.
- Ten accepted joint reflows, 21 pages without eligible reflow; no geometric
  validation failures. Missing-source-anchor/foreign-text groups remain unknown
  instead of receiving unsupported coordinates.
- Independent receipt validation checks preserved text, box/centroid agreement,
  transformed source provenance, bounds, order and spatial conflicts.
- **39/39 negative controls rejected:** changed coordinates, unrelated inserted
  text and stale source-proof hashes, independently tested per candidate.
- 10.80 seconds; peak RSS 983.36 MiB. Semantic holds are not overridden by good
  geometry.

### Regression tests and immutable inputs

- **588/588 curated production synthesis tests passed in 18.15 seconds.**
- **19/19 audit integrity/adjudication tests passed**, including stale hashes,
  unrelated text insertion, off-page coordinates, removed-field leakage,
  missing/duplicate/wrong/unread review coverage, missing/stale adjudications,
  preservation of raw review decisions and rejection of an unresolved flag
  silently promoted to pass.
- Ruff checks pass for the audit utility and its tests.
- All nine draw/publication pins and all **1,842 protected files** remain
  hash-identical; the final complete protected-file check took 4.12 seconds.

No production source-code hot path was changed during this audit, so no
before/after speedup is claimed. Reported times/memory measure actual audit/test
workloads. No GPU training/inference benchmark or training launch was performed.

## What is complete, and the bounded next work

Complete: checking previous implementation, actual fresh upstream synthesis,
full 50-source/100-descendant random inspection, explicit final adjudication,
exact family closure for both confirmed mechanisms, integrity/geometry tests,
and the certification inventory. The audit is not waiting on credit or more
generic review rounds.

Remaining work is concrete and separate from this completed audit:

1. Correct the primary-principal owner in one family and its six descendants.
2. Apply the already-supported certified exporter alias in one family and its
   nine descendants.
3. Resolve the five selected narrow semantic/grounding holds; inspect their
   relevant family descendants only for the corresponding mechanism.
4. Target generated shipment-count assertions in product wording, using the
   fresh counterexample, without stripping legitimate capacities/specifications.
5. If complete auxiliary anonymization is required, inventory the quoted
   non-target residues by source owner and replace repeated private facts
   coherently. This is not a reason to relabel all otherwise sound targets.

No new repairs, regeneration, training inclusion or filtering were silently
performed in this audit. The 15 confirmed family records are exact known scope,
not a claim that there are only 15 such defects in all 300 families. Further
scope expansion should use the identified owner/component and co-reference
mechanisms, not assume error-rate extrapolation or reopen all descriptions.
