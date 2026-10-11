# Coverage-targeted 1,000-document synthesis and dataset expansion

Date: 2026-10-10. Campaign configuration:
[mpci_bl_curated_v7_coverage1000.yaml](../configs/synthesis/mpci_bl_curated_v7_coverage1000.yaml).

This report records the approved generation, correction, final-pair semantic
review, geometry validation and protected dataset merge. It is not a training
result or a claim that a particular future model score is guaranteed.

## Completed result

**All 1,000 requested new samples are accepted and published. There are zero
unresolved review findings or held documents.** The new training dataset is
committed at
[`mpci-bl-real600-synthetic2500-v7-positions-v1`](../data/curated/mpci-bl-real600-synthetic2500-v7-positions-v1/README.md):
**3,100 training records (600 real + 2,500 synthetic) and the unchanged 60
validation records**. The old dataset remains intact.

The [independent final-1,000 audit](analysis/synthesis-next1000-production-20261009/quality/final-audit.json)
and [independent combined-dataset audit](analysis/synthesis-next1000-production-20261009/quality/combined-audit.json)
both pass with zero failures. Each final rendered text/target pair has a current
complete model review. These are accepted training samples, not just a pilot or
an uncommitted proposed merge. No GPU training or model-weight loading was started.

Inspection outputs:

- [Original OCR followed by every new rendered variant](../artifacts/kie-synthesis-production/curated-v7-coverage1000-v1/samples.md).
- [Position-enriched variants](../artifacts/kie-synthesis-production/curated-v7-coverage1000-v1/positions-reflow-v1/samples.md).
- [Coordinate plot gallery](../artifacts/kie-synthesis-production/curated-v7-coverage1000-v1/positions-reflow-v1/audit/GALLERY.md).
- [Independent visual review of all 45 plotted pages](analysis/synthesis-next1000-production-20261009/quality/GEOMETRY_VISUAL_REVIEW.md).

## Scope and acceptance contract

The requested expansion uses all 300 admitted templates: two descendants from
each original template, two from each second-cohort template and six from each
newest template. That is exactly 200 + 200 + 600 = 1,000 new documents. The
separate 200 newest-template admission examples are inspection evidence and
are not added implicitly. The committed mixture is 600 real + 2,500 synthetic
training documents, with the same 60 real validation documents.

Acceptance requires the following, on the final current outputs rather than
an earlier intermediate version:

1. Complete source/scenario/wording/contact replay, current target schema,
   casing, source ownership, negotiability and notify-reference inheritance.
2. Full rendered OCR and labels reviewed by the configured model for every
   new sample. Every reported finding must be resolved against the current
   sample and review hashes. Actual corrections require a fresh review.
3. Independent byte-edit reconstruction, approved-description reconstruction,
   physical row sums and exact coverage checks.
4. Source-anchor lineage, geometry, coordinate serialization, text retention,
   baseline comparison and visual inspection of selected plotted pages.
5. Protection of the old dataset and exact preservation of validation; train
   and validation separation through document, OCR, B/L and PDF identities.

Detailed reproducible commands and validation contracts are in the
[campaign quality README](analysis/synthesis-next1000-production-20261009/quality/README.md).

## Coverage and sampling decisions

The source quotas preserve layout diversity while placing more weight on the
newest cohort. Existing compatible sources are deliberately used for plastic
bags and steel drums rather than introducing unsupported package categories.
Source topologies remain the authority for container-list length, split
description/address regions, allocation availability and transshipment layout.
Countries, ports, localities, commodity identities and joint loads are sampled
from the pinned registries and eligible physical bundles.

The final inventory contains 181 origin countries, 181 destination
countries and **561 distinct HS6 identities**. The preflight had 560 identities;
the six required private-measure redraws account for that difference. It includes 296 multi-container documents,
24 with at least eight containers, 84 requiring volume aggregation from
container rows, 60 with multiple approved description regions, 130 with split
postal regions, 44 thermal documents, 12 dangerous-goods documents and 12
transshipment documents. These are overlapping traits, not additive groups.
Instruction inheritance gives 48 negotiable, 938 non-negotiable and 14
OCR-unavailable values, plus 216 notify-reference documents.

The final package inventory spans **24 categories**, including 24 plastic-bag
documents and 10 steel-drum documents. The container-count distribution is
22 with none, 682 with one, 108 with two, 80 with three, 24 with four, 34 with
five, 18 with six, eight with seven, 10 with eight, 12 with nine and two with
13. See the [final coverage inventory](../artifacts/kie-synthesis-production/curated-v7-coverage1000-v1/audit/final-coverage-inventory.json).

New description lengths are median 236, mean 253.932 and maximum 1,211
characters, versus median 477, mean 492.345 and maximum 1,638 for the inherited
1,500. This is a measured distribution difference: the new source-style prompt
does not force every concise source into an expanded assortment. Long/list
descriptions and the approved separated product regions remain represented.

The lexical instruction was tightened to match the source's descriptive
presentation: concise product identity, specification passage or product list.
It no longer requests an assortment universally. Every sampled commodity
identity remains represented; there are no character or line-count quotas and
no requirement to teach exact HS-to-product classification.

## Problems found and root-level corrections

### Physical source-grid admission

A sampled volume that cannot be distributed at the printed precision over the
source's actual positive rows is rejected before API generation, through the
existing bounded sampling loop. The motivating example had six positive
integer-volume rows but only two cubic metres. The next admissible draw was
accepted. There is no invented zero row or silent precision change.

The preflight comparison preserved the other 999 accepted plans. Runtime was
53.900 versus 54.779 seconds (+1.63%); peak RSS was unchanged at 820.742 MiB.
See the [physical-admission report](analysis/synthesis-next1000-coverage-20261009/sampling/physical-candidate-fix/REPORT.md).

### Repeated multi-person contacts

Combined contact-name labels were being printed in full at each individually
owned source occurrence. The repair maps each occurrence to its own generated
person while preserving the combined target label. Every component must have
source ownership before generation; unsupported cardinality fails explicitly.
Three source contracts needed ownership corrections, with shared registry pins
updated and the supersession recorded.

A targeted historical search identified 28 affected old synthetic inputs from
four families. Only the new mixture receives the exact 56 raw and 56 positioned
occurrence replacements. The old dataset is not overwritten; targets,
coordinates and all other content remain unchanged. An independent proof and
mutation tests enforce that narrow exception. See the
[contact repair report](analysis/synthesis-next1000-production-20261009/contact-ownership/REPORT.md).

### Postal correction completeness

The original correction checker inspected the complete address but did not
enforce the specific generated region that owned its country. Its blanket
country instruction also contradicted sources that intentionally omitted
country. A shared strict country-owner check now participates both in rendering
and correction detection/post-checks. Correction context identifies complete
assemblies, required localities, country presence or omission, canonical
spelling and exact country-owner keys. Name and ISO-code duplication is not
introduced as a second country occurrence.

The dry inventory found exactly the same 54 residual documents as rendering;
eight of those had been invisible to the former whole-address-only checker.
The source contracts for those eight were correct and were not changed to
accommodate bad generated placement. See the
[postal correction diagnosis](analysis/synthesis-next1000-production-20261009/postal-correction/REPORT.md).

### Goods wording and internal-key leakage

A read-only inventory scanned all 1,000 wording records, 300 families and
1,066 goods regions. Eighteen exact region edits across 16 samples removed
unsupported generated pack-fill/accounting claims, mismatched package names
and two standalone internal `product_1` keys. These edits change the printed
wording authority; final labels are then reconstructed from that wording,
never independently edited to disagree with it. Before-images and exact
replacements are preserved in the
[applied wording receipt](analysis/synthesis-next1000-production-20261009/quality/applied-wording-edits.json).

The new key-leakage guard checks complete lines against the actual request's
field keys, case-insensitively. It preserves embedded commercial model codes,
prefix matches and unrelated identifiers. Its measured incremental cost is
1.02 ms per 1,000 wording records. The existing full-pair review separately
checks every packing and transport claim against host-controlled facts.

These synthetic corrections do not change the real-description policy: retain
the approved main product passage and genuine continuations, including embedded
product qualifiers, while excluding separately owned Marks, headings,
declarations and detached accounting/tracking passages.

## Validation evidence

The final expanded curated and campaign-quality suite passes **561 tests
in 27.02 seconds**, including the separate inherited-equipment mutation tests.
The earlier 548-test subset passed in 19.92 seconds; these are different suite
sizes, not a same-workload performance comparison. Tests include negative controls for changed targets,
unowned text edits, stale reviews, description-boundary crossing, incorrect
physical sums, identity leakage, coordinate movement and unauthorized changes
to inherited rows. Final publication, semantic-review resolution,
geometry results and total provider costs are recorded below.

## Final-pair review: root causes, not target-only patches

The initial complete rendered-text/target review covered all 300 sources and
1,000 descendants, producing 210 findings with no failed review requests. Each
finding was inspected against the source contract, current rendered text,
sampled shipment and relevant physical context. Findings included true defects,
false role assumptions, observations that explicitly reported no error and
requests for cosmetic or national-registration changes outside the agreed
extraction objective. None of those categories is treated as an automatic pass.

The substantive source corrections use existing declarative ownership,
auxiliary-date and physical-measure mechanisms:

| Defect | Repair at its authority | Preserved information |
| --- | --- | --- |
| Old package nouns beside new sampled counts | Bind the existing printed noun to the declared package category, including worded totals | Original count ownership, container membership and source layout |
| Repeated source product wording after commodity replacement | Bind the existing repeated product occurrence to the current generated product | Separate Marks and fixed documentary context |
| Stale CIF destination or party-locality footer | Bind the exact shipment-owned location to its sampled role | Independent exporter/agent identities are not equated to shipper |
| Duplicate old city/country suffix | Clear only the certified duplicate occurrence | Complete generated postal address and other party facts |
| Old equipment alias beside new equipment | Bind the existing alias to the sampled equipment owner | Printed container count, STC/FCL captions and other bytes |
| Composite fixed principal plus generated represented company | Correct the existing name expression and generated identity role | Both printed identities and the source adjudicated party policy |
| Stale formal issue date | Shift the existing private date by the public shipment-date offset | Original date format and source relative timing |
| Printed weight/volume excluded from public targets stayed fixed | Declare the exact private numeric owners and require complete physical donors before sampling | The real/public target schema remains unchanged; missing donor facts cannot be guessed |
| Unsupported generated packing or totals | Edit the source-authoritative generated wording, then derive the label again | Product identities/specifications and host package/allocation facts |

Old candidate/review snapshots, exact lexical before-images and protected-pin
supersessions remain in the campaign's `audit/` directory. Corrections invalidate
the affected review: final acceptance requires a fresh review of the changed
family. A historical `fixed` decision cannot approve a replacement candidate.

The independent registered exporter is not assumed to be the shipper. For
example, exporter registration `84899112` identifies the source's notify actor;
its country follows that actor, not an unrelated shipper country. Detached
Marks, opaque references and conditional carrier clauses remain valid negative
context and are not silently recast as route or product facts. Conversely,
printed gross weight, volume, actual issue dates and actual CIF destinations are
shipment facts and cannot be dismissed as harmless flavor text.

The finite inherited-input search found three earlier equipment-alias defects
in addition to the 28 contact-occurrence defects. The new mixture receives
those **31 input-only repairs**; every inherited target and coordinate suffix
is unchanged. The old dataset itself stays immutable. See the
[equipment repair proof](analysis/synthesis-next1000-production-20261009/equipment-ownership/README.md)
and the [merge contract](analysis/synthesis-next1000-production-20261009/quality/README.md).

### Final review decisions

After root/wording corrections and fresh reviews, the current 300 source review
receipts cover all 1,000 final samples and contain **134 findings**. Every one
has an individual, hash-bound adjudication with literal evidence from its
current rendered OCR. These remaining findings were rejected because their
proposed change conflicted with the established source ownership or policy,
or they explicitly described no actual error. This was not a blanket rejection
of the original 210 findings: genuine contradictions were repaired first and
the affected families reviewed again.

The [final adjudication inventory](analysis/synthesis-next1000-production-20261009/quality/final-adjudication-inventory.json)
has zero unresolved findings. The independent publication replay verifies the
current review/candidate/wording/source hashes, all exact source quotas and all
1,000 current renderings, rather than accepting earlier successful snapshots.

## Positional synthesis and geometry audit

All 1,000 positioned outputs were checked against their exact plain outputs,
source anchors and serialized transformation receipts. Labels and raw text
remain identical; position information is appended only to the appropriate
content lines. The source audit also checks repeated text ownership rather than
assigning every repeated string to its first OCR occurrence.

| Line class | Content lines | Anchor-only coordinates | Final coordinates | Final coverage |
| --- | ---: | ---: | ---: | ---: |
| All content | 99,270 | 81,556 | 84,381 | 85.00% |
| Goods descriptions | 2,865 | 1,184 | 2,174 | 75.88% |
| Party addresses | 9,191 | 4,269 | 6,079 | 66.14% |
| Other content | 87,214 | 76,103 | 76,128 | 87.29% |

There are **2,825 additional positioned lines**, with zero lost text, changed
targets or lost previously known coordinates. Across 1,892 pages, 808 reflows
were accepted, 1,081 had no eligible reflow and three proposals were declined
for density or pre-existing column order. Those three pages preserve their
entire exact baseline placement; they are not held documents or partially
edited pages.

All 4,323 generated reflow centres pass bounds, order and coherent-transform
checks. The 14,889 remaining unknown positions retain explicit ` ||` markers;
incomplete anchors or foreign content inside an owner envelope do not authorize
invented coordinates. This is deliberate missing position information, not
missing OCR or target data.

An independent reviewer actually opened **all 45 plotted PNGs from 24 selected
documents**, comparing measured source geometry, baseline placement, reflow and
the final page transform. Dense container lists, goods/address expansion and
declined proposals were included. No geometry defect was found. All four
mutation controls (text change, off-page position, duplicate line, incoherent
anchor) were rejected by the independent validator.

## Dataset preservation and final training readiness

The mixture was built only after the final acceptance audit passed, then audited
independently again. All 1,000 additions equal their accepted positioned rows.
Of the old 2,100 training rows, 2,069 remain byte-identical; the only exceptions
are the 31 precisely proven contact/equipment input-only repairs described
above. **All 2,100 inherited targets, all coordinate suffixes and all 60
validation rows are unchanged.** The old source files and manifest hashes were
rechecked before and after publication. Their original metadata is archived
verbatim in the new dataset rather than rewritten as a new review claim.

All 3,160 records pass the current schema/casing policy and exact raw/positioned
text checks. There are zero train/validation collisions by document ID,
synthetic ancestry, raw or positioned OCR, normalized B/L identity or PDF
content identity. The separate 200 newest-template admission examples are
excluded.

The new [rank-48 training configuration](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic2500_descblocks_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml)
retains the last run's settings: rank 48, alpha 32, input-only positional input,
compact JSON, 10 epochs, microbatch 1, accumulation 32 and ScheduleFree AdamW.
Dataset paths, record counts, hashes, run identity and MLflow tags are updated.
The larger dataset gives **97 updates per epoch / 970 total**, with **49
optimizer warmup updates** (rounded-up 5%). Eval/save runs every **243 updates**,
approximately 2.5 epochs, with final evaluation enabled, all intermediate
checkpoints retained and best-checkpoint loading unchanged.
The installed Transformers final-step callback was inspected independently:
it evaluates and saves at the final update even though 970 is not divisible by
243. Planned evaluation/checkpoint steps are **243, 486, 729 and 970**.

An independent configuration/mixture sanity check found only the intended
identity/data/interval/warmup differences from the previous rank-48 config and
verified 10 manifest files plus 2,533 dependency hashes with zero mismatches
in 8.548 seconds. The shared task-constraints directory still contains `1500`
in its historical name; its schema content is dataset-size independent and its
current hash is verified, so no new schema copy is needed.

The real `validate-config` and `inspect-dataset` CLI paths pass both locally
and inside the existing training Docker image, with identical resolved schema,
prompt, configuration and dataset hashes. The image identity is
`sha256:1e790f3868836ce277ed2fb0585660e111bcfe5c5363660e126a6fbfeacc933e`.
The Docker checks used `kie-tools`, `--no-deps` and `--pull never`; no training,
GPU access or model loading was requested. See the
[Docker preflight receipt](analysis/synthesis-next1000-production-20261009/quality/docker-preflight.json).
The
[CPU-only tokenizer preflight](analysis/synthesis-next1000-production-20261009/quality/training-input-preflight.json)
uses the saved model tokenizer and the production prompt/target normalization
and `_tokenize_batch` implementation on all 3,160 rows:

| Split | Records | Maximum source tokens | Maximum target tokens, including EOS | Truncated or training-overflow rows |
| --- | ---: | ---: | ---: | ---: |
| Training | 3,100 | 8,066 | 3,139 | 0 |
| Validation | 60 | 5,698 | 1,348 | 0 |

Source/target training limits remain 19,200/5,500. The unchanged 3,072-token
evaluation generation limit covers every validation target. Three training
targets exceed that evaluation-only limit; they are nevertheless fully retained
for teacher-forced training. No GPU training/model startup benchmark was run here.

To start the next experiment using the already validated local image:

```bash
docker compose --profile training run --rm --pull never kie-trainer train \
  --config configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic2500_descblocks_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml \
  --project-root /workspace
```

Omitting `--build` deliberately reuses the current image and avoids contacting
the registries whose credential helper failed earlier. Synthesis-only source
changes do not require a new trainer image; the exact current training config
and dataset were validated inside this image.

## Measured execution and cost

These are separate CPU stage timings, not an asserted end-to-end speedup:

| Completed stage | Seconds | Peak RSS, where measured |
| --- | ---: | ---: |
| Final 1,000 current render/replay readiness | 27.735 | 448.34 MiB |
| Coordinate synthesis | 32.539 | — |
| Anchor-only comparison | 27.726 | — |
| Geometry comparison and 45 plots | 13.115 | — |
| Independent final-1,000 audit | 39.503 | 820.74 MiB |
| Protected mixture publication | 53.246 | 982.85 MiB |
| Independent combined audit | 41.533 | 979.11 MiB |
| Production CLI dataset inspection | 1.853 | — |
| Final CPU-only tokenizer preflight | 5.001 | 614.68 MiB |

Ruff and `git diff --check` pass on the final changes. The final test command was:

```bash
.venv/bin/pytest -q tests/test_curated* \
  docs/analysis/synthesis-next1000-production-20261009/quality/test_*.py
```

The provider ledger totals **$0.93432576 for the 1,000 accepted documents**, or
**$0.00093433 each** (about 0.0934 US cents). All 1,425 request receipts have
provider-reported costs, including the two failed/retry attempts. There is no
unreported reserved cost. This total includes this campaign's generation,
contact generation, repairs and all rendered-pair reviews; it excludes earlier
template-admission campaigns and Codex subscription cost.

| Provider stage | Cost, USD |
| --- | ---: |
| Company contacts, including recovery | 0.01102025 |
| Lexical generation | 0.32990125 |
| Postal correction | 0.03036945 |
| Full rendered-text/target reviews | 0.54596335 |
| Wording validation repairs | 0.01707146 |
| Total | **0.93432576** |

The [complete provider receipt ledger](../artifacts/kie-synthesis-production/curated-v7-coverage1000-v1/audit/final-cost-ledger.json)
contains every request hash, stage, failure status and reported charge. Local
audits and plotting made no API requests.

## Coverage limits that remain deliberate

The campaign increases existing certified layouts and compatible categories;
it does not add unsupported IBC/paper-bag layouts or create new cross-page
description structures. A historically mixed `1 × 20 DR + 3 × 40 HC` source
without complete per-container equipment labels retains that omission. Its two
current descendants were checked against the available printed capacity
multiset; that does not certify arbitrary future draws from this source without
a corresponding mixed-equipment admission rule. No missing size/type labels are
invented. Synthetic coordinates remain useful coarse layout estimates, not
measurements of a newly rendered PDF.
