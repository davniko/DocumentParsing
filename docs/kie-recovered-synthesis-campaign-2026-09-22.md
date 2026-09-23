# Recovered-template synthesis campaign — completed 23 September 2026

## Outcome and entry points

The third synthesis population contains **10,000 accepted documents**. Its ledger
settled at **$6.92543453**, including failed attempts, retries and replacements:
2,989 requests, no outstanding reservations or uncertain charges. This is below
the user's **$8 cap for this campaign only**. It is not a new guarantee that every
future run will fit $6, and excludes earlier compilation/recovery and historical
dataset spending.

The final training union contains **30,967 records**: 29,910 synthetic plus the
original 1,057 real training documents. The same 100 validation documents remain
byte-for-byte unchanged. Training has **not** been launched.

- [Interactive EDA: 34 plots](../artifacts/kie-synthesis-production/campaigns/recovered-20260922/eda/index.html)
- [Final dataset manifest](../artifacts/kie-training/datasets/mpci-bl-real1057-synthetic29910-recovered-v5-v1/manifest.json)
- [Training configuration](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real1057_synthetic29910_recovered_v5_e5_v1.yaml)
- [Recovery methods, queue and continuation instructions](kie-template-recovery-handoff-2026-09-22.md)

Below, `C` denotes
`artifacts/kie-synthesis-production/campaigns/recovered-20260922`.

## Accepted populations and provenance

| Population | Accepted documents | Final immutable snapshot under C |
| --- | ---: | --- |
| Egypt-heavy | 9,980 | `egypt-tariff-final-v7/records.jsonl` |
| Earlier diversified | 9,930 | `diversified-tariff-final-v7/records.jsonl` |
| Recovered-template generation | 10,000 | `recovered10000-final-v2/records.jsonl` |
| Combined synthetic | **29,910** | Included in final training union |
| Original real train | **1,057** | Included once, with target serialization reordered |

The earlier 20 and 70 exclusions remain excluded; no excluded record was silently
reintroduced. Original source documents, generated populations, paid attempts,
correction receipts and training histories remain intact.

Every final synthetic record has an auditable parent and exact correction chain.
The final training manifest pins all inputs, outputs and audits; `lineage.jsonl`
records sample membership. All records use the existing relation-explicit v5
schema. No exporter/private-physics metadata was added to the training labels.

There are no exact duplicate training texts, no exact train/validation text
overlap and no synthetic validation-source overlap. Seven repeated target payloads
belong to the original real training set (different input texts); they were
retained. The synthetic union has no duplicate target payloads.

## What the new data contains

| Measure | New 10,000 | Combined 29,910 synthetic |
| --- | ---: | ---: |
| Distinct source templates | 1,100 | 1,306 |
| Dangerous-goods documents | 1,000 (10%) | 3,000 (10.03%) |
| Temperature-controlled documents | 1,500 (15%) | 4,560 (15.25%) |
| Transshipment documents | 161 | 279 |
| Distinct observed shipper countries | 53 | 77 |
| Distinct observed consignee countries | 129 | 137 |
| Shipper city/country pairs | 4,528 | 7,803 |
| Consignee city/country pairs | 4,781 | 8,034 |
| HS6 codes | 1,554 | 2,075 |
| DG UN numbers | 467 | 751 |
| Package categories | 31 | 46 |
| Distinct observed loading/discharge surface pairs | 7,668 | 14,621 |

Country aliases are normalized **only for EDA**; labels retain their source
formatting. Missing values are not counted as countries or package categories.
Port surface counts do not imply that differently spelled ports are physically
different ports. Description uniqueness includes generated product references;
HS6 and UN diversity are the stronger evidence of commodity variety.

Egypt is the explicitly labelled consignee country in 209 new documents (2.09%),
versus 8,228 in the combined synthetic set (27.51%). The older Egypt-heavy
population remains intentionally included, so the union is not geographically
uniform.

The new equipment-size observations are 62.44% 40-foot high cube, 20.41% 20-foot,
8.54% standard 40-foot, 2.11% 45-foot high cube and 6.49% unlabelled. The combined
set retains 9.02% 45-foot high cube because the earlier diversified set oversampled
that category. The EDA exposes this rather than describing the union as perfectly
balanced.

### Selection and sampling

- Structural selection blends 80% real-training structural frequencies and 20%
  uniform supported structural bins. This improves rare-structure coverage
  without pretending every source supports every topology.
- Origins use 75% observed exporter support and 25% registry exploration.
  Routes, destination countries and party localities are sampled together.
- Equipment-size priors are 27% 20-foot, 10% standard 40-foot, 60% 40-foot high
  cube and 3% 45-foot high cube. Actual marginals are conditioned on goods,
  packing, source contracts and physical capacity.
- Goods/DG use pinned registry support, including 2,934 HMT and 3,480 ECICS
  records, with joint goods/package/equipment compatibility. They are not
  restricted to the small set of source UN numbers.
- Carrier identity remains fixed. Transshipments and row/group structures are
  sampled only where the source template supports them.
- Missing public size/type/unit labels stay absent. Audited private physical
  context supports capacity calculations without inventing visible labels.
- HS6 identities come from the sampled goods. Additional digits and separators
  follow the template's supported HS format.
- Party locality supervision uses party-owned visible evidence. No new
  requirement was introduced to infer an unseen city from an address.

### Genuine variation, not lightly modified copies

The independent pre-curation replay found 382,989 changed leaves out of 581,712
(65.84%), averaging 38.30 changes per new document. Among observed leaves:

| Field | Changed from the source |
| --- | ---: |
| Shipper name/address | 100% / 100% |
| Consignee name/address | 99.23% / 100% |
| Container numbers and seals | 100% |
| Cargo descriptions | 100% |
| HS codes | 99.50% |
| Package quantities | 88.34% |

Zero silent source restorations were accepted. Fixed carriers, supported
topologies and legitimately retained categorical/reference values are not
required to mutate. These rates describe generated values, not extraction-model
accuracy.

## Review findings and repairs

### Template corrections, retained samples

Four source templates were corrected as explicit derivatives:

1. **c8a72… — redundant global temperature.** A global scalar could contradict
   independently sampled container settings. It now refers to the individually
   printed settings; all actual container temperatures remain.
2. **a915a… — duplicated address fragment.** Removed only the unowned redundant
   AZARITA fragment; retained the complete bound party address.
3. **103c6… — description/mark and packing equation.** Separated the physical
   description and repeated-mark owners, connected their dependency, and bound
   the count in the explicit mass-times-count equation to its package quantity.
4. **dfbf63… — commodity code misclassified as a generic reference.** The source
   explicitly prints `03035510`, but its HS label was omitted. Restored the
   existing-schema HS field and direct tariff binding from unambiguous source
   evidence. Generated records use the sampled goods' HS6 and retain the
   supported two-digit suffix format.

All four source repairs passed three route/goods/render variants each:
**12/12**, in 15.13 seconds, with 473,648 KiB peak resident memory and no model
calls.

The final new population contains 152 explicit presentation/count corrections
and 118 tariff corrections (270 distinct corrected rows). The historical
correction chains affect 183 distinct Egypt-heavy and 179 distinct diversified
rows. Each correction has exact before/after evidence; unrelated bytes and
label values are checked. Tariff corrections intentionally update the existing
HS label to agree with the goods and printed code.

### What was not promoted

Two templates, **07590a…** and **1e7b16…**, still have source-only carrying
temperature statements without a complete, unambiguous physical sampling
contract. They remain in review, not deleted or declared intrinsically
unrecoverable. Their 70 candidates were replaced, not silently accepted.

The original 10,000-candidate execution produced 9,912 passes and 88 failures:
50 host rejections, 19 generation rejections and 19 provider content-filter
failures. Removing the 70 review-template candidates left 9,842 retained records.
The 158 replacements preserved the final DG, temperature and transshipment
counts. Paid failures remain in the ledger; there was no ledger reset.

### Durable production changes

- Compiler certification and generation preflight reject explicit tariff
  captions treated as unrelated random references.
- Explicit mass-times-count equations require owned package quantities.
- Generation preflight checks ownership of source-only carrying-temperature
  settings instead of letting them bypass goods sampling.
- Private sampled equipment tares now survive generation/export/publication.
  The loader requires exact expected keys, finite positive values and pinned
  receipts. Five paid records that exposed the missing context were recovered;
  no source tare was used as a fallback.
- Explicit human linguistic adjudications remain source/scenario/checkpoint
  pinned and pass ordinary validators. They cannot bypass contracts.
- Contact prefixes, country aliases/codes, postal fragments and party-owned
  locality dependencies retain the fixes exercised by the historical audits.

The repaired next-run input is `C/next-run-template-inputs.json`, which pins
`ready-template-base1110-v2` and its numeric/lexical contracts. **Generate fresh
sample and route plans** against those pins; do not reuse the old 1,112-candidate
snapshot's plan hashes.

The original 1,510-template inventory is now explicitly accounted for:
**1,110 ready candidates, 334 review, 66 validation-excluded**. The 334 include
332 previously paused cases and the two new temperature review cases. The new
accepted population actually uses 1,100 templates. Ready-pool membership is not
a guarantee that every possible future linguistic candidate passes acceptance.

## Validation performed

| Check | Result |
| --- | --- |
| New records independently replayed through native host contracts | 10,000 |
| Independently checked numeric contracts | 32,234 |
| Silent source restorations | 0 |
| New population duplicate texts, targets or container identities | 0 |
| Historical parent audits | 9,980 + 9,930; zero failures/warnings |
| Historical committed artifacts re-hashed | 14,308 + 19,453 |
| Full final-text semantic screen | 29,910 records; zero findings |
| Synthetic exact validation-source overlap | 0 |
| Synthesis tests | 2,966 passed; 11 dependency warnings |
| Focused training/order/alignment tests | 11 passed |
| Changed production modules | Ruff and mypy passed |
| Model/GPU training launched | No |

The new parent replay took 590.56 seconds for 9,842 records plus 22.10 seconds
for 158 replacements, including provenance checks. Historical parent audits
took 68.54 and 82.42 seconds, with approximately 1.84 and 1.46 GiB peak RSS.
The full synthesis suite took 174 seconds. The source-ownership guards measured
approximately 213.94 microseconds per template, once per source rather than
once per generated field.

Acceptance is layered: native generation/publication replay proves the parent
records; exact edit and target receipts prove the final corrected derivatives;
the final semantic screen checks the actual published text. The report does
not incorrectly claim that a changed derivative has its parent's original
render hash.

Manual review was purposive and targeted, not a human read of every one of the
29,910 documents. The automated coverage is full-record. There are no unresolved
**detected** acceptance findings in the published set; this is not a guarantee
against every possible semantic error, legal-flavor imperfection, or a promise
of a particular downstream F1 score. Shared carrier layouts are allowed:
this is exact-source-ID exclusion, not unseen-layout evaluation.

## EDA

The 34-plot gallery compares real training data, each synthetic generation and
the combined synthetic population. It includes carrier/template reuse,
geography/localities/ports, country and category concentration, HS6/chapters/code
lengths, UN/hazard classes, package and equipment categories, group/allocation
counts, field presence, DG/thermal/transshipment coverage, numerical
distributions, text/target lengths, audit coverage and actual generated-value
variation. Plot labels distinguish descriptive distributions from quality
evidence. Numeric source tables are in `C/eda/summary.json`.

## Training handoff

The copied baseline retains five epochs, batch size one, gradient accumulation
32, learning rate 1e-4, LoRA rank 32, BF16/SDPA and reentrant gradient
checkpointing. Evaluation/checkpoint cadence remains 173 optimizer steps and
start evaluation remains disabled. With 30,967 examples, the configured
single-device run has approximately **4,840 optimizer updates**.

`cargoAllocationGroups` is now the final field inside `documentPatch` in the
training serialization and model-facing schema. This changes presentation order,
not field semantics. The validation source file remains byte-for-byte unchanged;
runtime canonicalization applies the same ordering.

Exact pinned-tokenizer measurements, including the actual input prompt/schema
and target EOS, found:

| Split | Maximum input tokens | Maximum target tokens |
| --- | ---: | ---: |
| Train | 19,194 | 8,753 |
| Validation | 12,637 | 1,490 |

The configuration uses **19,200 input / 8,960 target** ceilings with explicit
overflow errors and dynamic padding. One input exceeded the previous 18,432
ceiling; 37 targets exceeded the previous 6,144 ceiling. No records were
truncated or dropped to satisfy the old settings. The ceilings are the measured
maxima rounded up to a 256-token boundary, not a global fixed padding length.
Exact histograms are in `C/exact-input-lengths.json` and
`C/exact-target-lengths.json`.

CPU-only dataset preparation passed through the rebuilt actual Compose image:
30,967 train and 100 validation records, zero truncation, 114.76 seconds.
The complete runtime receipt is `C/training-preparation.json`. Training input
lengths are 5,188 / 8,523 / 10,557 tokens at p50/p95/p99; target lengths are
632 / 1,227 / 2,170. The extreme lengths are a small tail, not typical examples.
No model weights were loaded for that check. **GPU peak memory and training
throughput for the long-tail examples have not been benchmarked**; the data
preparation proof is not a GPU-capacity guarantee.

From the repository root, launch training yourself:

```bash
docker compose --profile training run --rm --build kie-trainer train \
  --config configs/training/production/t5gemma2_270m_lora.mpci_bl_real1057_synthetic29910_recovered_v5_e5_v1.yaml \
  --project-root /workspace
```

Original training outputs/configurations were not overwritten. Outstanding
template recovery is documented separately and is not required to use this
published training union.
