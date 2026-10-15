# Full HS-code formats, DG field coverage and historical VIA repairs

Date: 2026-10-14. Repairs published in place; registry now has **363 admitted
templates** (360 existing + three DG additions). Execution evidence:
[audit directory](analysis/hs-dg-format-repair-20261014/).

## Scope and publication boundary

This repair covers the complete R20 synthetic pool: **5,000 active records plus
2,500 previous UAE records**. The latter remain a sidecar, not an automatic
addition to training. The active training file remains 600 real + 5,000 synthetic;
validation remains the same 60 real records. No training was launched.

The work separates two concerns:

1. Existing synthetic inputs/labels: restore source-shaped full customs identifiers
   and correct four historical composite shipper labels.
2. Future synthesis: repair the active sampling/rendering contract and admit
   independently reviewed DG templates that actually print the missing fields.

Do not infer that an existing DG sample has acquired packing-group or flash-point
supervision merely because the sampler now supports those fields. Existing
samples which do not print those declarations retain their original field set.

Before-state dataset files, all 360 case assets, shared ownership/auxiliary
files and the promoted source authority are preserved under
[before/](analysis/hs-dg-format-repair-20261014/before/). Per-record edits and
before/after pins make this a reproducible scoped repair, not a rewrite of the
whole extraction target.

## Diagnosis: HS6 identity was incorrectly serving as the complete label

The pre-repair pool had **6,181 HS-code target values, all six digits**, despite
source templates printing longer national/customs identifiers. The sampler
used the genuine commodity HS6 as the final target, and an old renderer path
collapsed different national extensions to their shared HS6 identity. Fixing
only a target schema regex would not fix this data loss.

The correct separation is:

- Commodity identity/LLM context: the genuine sampled HS6 and its registry wording.
- Printed customs identifier: that prefix plus deterministic fictional suffix
  digits, at the source's complete digit width and separator positions.
- Extraction target: all printed digits and the source-shaped presentation.

For example, genuine `610910` with a source `8528.59.00.00.11` can render
`6109.10.29.65.43`; the target is also `6109.10.29.65.43`. The suffix is a
fictional extraction example, not a valid national tariff claim. It is never
passed to the wording model as though it were an authoritative classification.

The user clarified that punctuation should remain if accepted by the form.
The saved [canonical MPCI form schema](../artifacts/mpci-ai-schema/mpci-cuscar-canonical.schema.json)
accepts printable ASCII customs identifiers. Its separate, historical
[AI-adapter mapping](../artifacts/mpci-ai-schema/ai-genie-normalization-and-mapping.md)
normalizes to digits, caps width at 18 and caps identifier count. Those adapter
restrictions are not form-field restrictions. The current extraction contract
accepts **6–20 ASCII digits**, with printed dots, spaces, commas and ASCII
hyphens. Future adapter integration must align with this changed contract.

Multiple complete codes remain multiple labels, including different extensions
of one HS6. Repeated aliases of the same complete digit sequence share one
generated identifier; the label uses its first owned presentation. Captions,
other identifiers and several unrelated codes cannot be smuggled into a single
owner via a broad digit-extraction regex.

### Two source-level defects found by complete replay

- Source `069cc830`: source presentations were 8, 12 and 9 digits. The 9-digit
  cargo-classification reference was incorrectly aliased to the 12-digit target.
  It now has its own target slot and exact owner; the source has one commodity
  identity with three distinct printed extensions, not three unrelated products.
- Source `c1b20a71`: 8- and 10-digit owner paths were swapped. Ownership now maps
  each complete printed value to its actual slot; a separately printed NCM alias
  participates in first-presentation selection. Its 6-digit code remains distinct.

These are binding/topology fixes, not country-specific tariff rules.

## Exact existing-dataset changes

**3,687 distinct synthetic documents were corrected through 7,187 owned HS text
edits.** No global find-and-replace was used. Original production proof edits
reconstruct each historical source, locate its owned HS occurrences through
intervening privacy repairs, and authorize only those text ranges.

| Cohort | Documents | HS-repaired documents |
| --- | ---: | ---: |
| Active synthetic set | 5,000 | 2,390 |
| Its current UAE subset, already included above | 2,500 | 1,284 |
| Previous UAE sidecar | 2,500 | 1,297 |
| Distinct synthetic total | 7,500 | 3,687 |

There are now **6,446 complete HS-code values**, including 265 restored distinct
extensions/explicit fields:

| Digits | Before | After |
| --- | ---: | ---: |
| 6 | 6,181 | 1,287 |
| 7 | 0 | 27 |
| 8 | 0 | 2,240 |
| 9 | 0 | 15 |
| 10 | 0 | 1,961 |
| 11 | 0 | 12 |
| 12 | 0 | 904 |

Actual admitted sources currently span 6–12 digits; the implementation and tests
cover all widths 6–20. Longer support is not a claim that the present pool has
13–20-digit positive examples.

The same repaired records are synchronized in `synthetic.jsonl`,
`replacement-synthetic.jsonl`, `previous-uae-synthetic.jsonl` and `train.jsonl`.
Real 600 and validation 60 records are unchanged. Per-line coordinate suffixes,
physical line counts, unrelated text, descriptions, package counts, equipment,
allocations, dates and all other label values remain exactly unchanged, except
the explicitly approved four shipper corrections below.

Coordinates are source-conditioned layout anchors, not measured glyph centres
of freshly typeset synthetic text. Restoring the source HS width does not
require discarding those anchors; it does not justify claiming newly measured
synthetic geometry either.

## Four old UAE VIA shipper labels

Source `ca74e1e4` produced four historical targets of the form `A VIA B` in one
shipper name. The current active 5,000 contained none of these four defects.
The sidecar now labels **primary printed company A only**, as agreed. Ambiguous
shared address/country/contact ownership is omitted rather than assigned to A
without support. B and the original shared block remain in
[audit metadata](analysis/hs-dg-format-repair-20261014/via/repair-receipt.json).
No forwarding-agent/NVOCC role is inferred from `VIA` alone.

The source's explicit organization identity declaration already controls new
synthetic casting. Two fresh company/address positives passed semantic review;
all four composite negative controls were rejected. The historical compiled
target is retained as source-replay evidence, not reused as an approved real
annotation. This distinction prevents destroying compiler provenance while
ensuring descendants have a single generated party identity and its own address.

## DG gap: registry facts existed, printable source slots did not

The four previously admitted DG templates printed UN number/class but **none
printed packing groups, subsidiary hazards or flash points**. Their absence in
synthetic labels was therefore not evidence that the DG registry lacked groups
or subsidiaries: it reflected source/template presentation coverage.

The HMT authority has 2,934 records; existing ECICS links yield 468 eligible
linked records, of which 384 satisfy the chemical packaging profile. These
include PG I/II/III and one/two subsidiary hazards. Generic non-bulk chemical
sampling does not silently admit explosives, gases or radioactive profiles.

Two new reviewed sources add the required presentations:

- `97ded689`: separately owned `UN Number … IMDG Class … PG …` declaration and
  PSN, with repeated-page ownership and privacy aliases.
- `32ebac37`: compact primary/subsidiary-class/UN/PG declaration with repeated
  UN data per container. The old singular subsidiary label is migrated to the
  current indexed array; each atomic class retains its own owner.

Both produced two full descendants. Every declaration matched its sampled HMT
tuple; full model review had zero findings. Source coordinates and repeated
text regions were independently audited. The four samples had **463/531
positioned content lines (87.2%)**, no geometry rejection, and no unproved
measured-region reuse across their 14 repeated source-text groups.

### Authoritative flash-point registry

The [CDC/NIOSH Pocket Guide](https://www.cdc.gov/niosh/npg/default.html) is an
authoritative public source for chemical properties. Its
[field definitions](https://www.cdc.gov/niosh/npg/pgintrod.html) specify CAS/DOT
identity fields and Fahrenheit closed-cup flash points unless marked open-cup.
DOT number alone is not a unique substance key, so the join uses **both exact
CAS and the same DOT/UN identity**, through pinned ECICS/HMT links.

The compiled, pinned subset is
[registry.json](../artifacts/sampling-registries/niosh-flashpoints-20261014/registry.json).
The acquisition inspected 103 candidate pages and accepted **64 exact-identity
scalar closed-cup records**. The other 39 were rejected for qualified/conflicting
identity, non-scalar/range/unknown properties or open-cup values. Raw HTML,
source URL, SHA-256, original literal, compilation receipt and identity pins are
retained. All 64 admitted properties were independently re-read from their raw
cards and checked against CAS, UN, literal and hashes.

Examples: acetone CAS `67-64-1`, UN `1090`, `0°F`; acetaldehyde CAS `75-07-0`,
UN `1089`, `-36°F`. Fahrenheit-to-Celsius conversion is explicit. When a source
prints integer Celsius, the renderer rounds to that source precision and sets
the label to the **printed** value, retaining the unrounded authoritative value
and rounding receipt. It never leaves a rounded OCR value paired with an
unrounded target.

Production configuration uses a hash/count-pinned `sampling.flash_points` file.
A flash-bearing source without this authority fails before paid generation.
Non-flash templates retain their previous field sets and can run without the
optional property pin. No LLM is asked to guess flash points, and no generic
mixture/NOS value is inferred from an unrelated pure compound.

The flash-bearing source `a1087462` is newly audited from a recovered typed
checkpoint, not treated as historically certified. It contains compact `PGII`
and per-drum accounting; these require exact PG-prefix formatting and an
explicit source-bundle/fixed-quantity contract. Its two fresh descendants passed
complete manual and model review with zero findings: Nitroethane UN `2842`,
PG III, **28°C** (NIOSH 82°F rounded at source precision), and Aniline UN `1547`,
PG II, **70°C** (NIOSH 158°F). Both printed repeated declarations consistently.
Geometry validated **136/196 positioned content lines (69.4%)**; the measured
source starts at 72% anchor coverage. Missing repeated/ambiguous anchors remain
explicitly unpositioned, not guessed or borrowed from the other occurrence.
All four geometry corruption controls were rejected. Together with the two
PG/subsidiary sources, these three new sources are admitted in the live registry.
The six inspection descendants are not silently inserted into training.

Every one of the 64 admitted chemical identities also passed **two complete
source-specific scenario/physical preflights (128 plans)**. These cover PG I/II/III
(6/42/16 identities) and 20 negative-Celsius presentations. The complete-domain
probe took 0.505 seconds without paid calls; see
[domain-probe.json](analysis/hs-dg-format-repair-20261014/dg/flash/domain-probe.json).

These are extraction-training fixtures, not instructions or certified values
for operational carriage of dangerous goods.

## Package categories and uncompiled layouts

Ten named package categories occurring only in real training have vocabulary
and aliases, but **zero matching production load profiles**:
`PACKAGE_BLOCK`, `PACKAGE_BOX_FIBREBOARD`, `PACKAGE_BULK_LIQUID`,
`PACKAGE_BULK_SOLID_FINE_PARTICLES_POWDERS`,
`PACKAGE_BULK_SOLID_GRANULAR_PARTICLES_GRAINS`, `PACKAGE_DRUM_FIBRE`,
`PACKAGE_FLEXIBAG`, `PACKAGE_FLEXITANK`, `PACKAGE_OCTABIN`,
`PACKAGE_UNPACKED_OR_UNPACKAGED_SINGLE_UNIT`.

This is not solved merely by increasing their sampling weights. Additional
physical load profiles and compatible printable owners are needed for the
next augmentation. The new flash source supplies an explicit fibre-drum
source-bundle presentation; this does not add a general fibre-drum load profile.
A bulk-powder source is already admitted; the other nine presentation categories
were not in the admitted set before this work. Free-text `typeOfPackages` appears
in nine real-training and four validation records (e.g. `SETS`, `BALES`, `RACKS`),
but none of the nine training source layouts was admitted. Thus that gap is also
source/presentation coverage, not a proven omission in a generated label. Likewise,
the 240 uncompiled real layouts are known source-coverage work, not evidence
that their existing labels are all defective. No fabricated category coverage
was added to the current dataset in this repair.

## Validation: what each PASS establishes

| Gate | Scope and meaning |
| --- | --- |
| Exact existing-record replay | All 7,500 synthetic inputs/labels equal before-state plus only declared HS edits and four party decisions |
| Mirrors/schema/hashes | Active 5,000, UAE subset, old sidecar and train synchronize; direct target contract and unchanged real60 validation pass |
| Source render controls | 360 sources × 2 full-code controls; unchanged source replay and all non-HS facts preserved |
| Real production preflight | 360 sources × 2 sampled shipment scenarios, including ambient/thermal/DG/mixed equipment families |
| Fresh HS synthesis | Six templates × two complete model-generated descendants; genuine HS6 context and exact full printed targets checked |
| Independent geometry | Source ownership, coherent transforms, printed coordinate receipts and off-page/duplicate/text/anchor mutation rejection |
| Fresh DG synthesis | New PG/subsidiary and flash presentations reviewed through the complete production path, not isolated strings only |
| Authority verification | All 64 raw NIOSH property records checked independently against exact identity and pins |
| Negative tests | Wrong digits, added separators, unrelated text, package mutations, missing codes and coordinate changes are rejected |

The 12 fresh HS samples produced eight review observations: old invoice dates,
an explicitly printed third-party transit instruction, caption typos, incidental
fill arithmetic and a conflicting product adjective. They were adjudicated
against the user's extraction objective: all target fields matched their printed
owners, and no generated aggregate declaration was left unlabelled. The raw
observations and explicit rejection reasons are preserved. This is **not** a
claim that every review returned zero flags or that physical realism is certified.

The final comprehensive relevant test run passed **1,127 tests in 39.77 seconds**,
including compact PG, exact property sampling and current/historical render
replay. Current destination hashes and actual rebuilt
Compose trainer checks are recorded in the final receipts. The trainer validation
loads no model and starts no training; tokenization is CPU-only.

## Measured overhead and cost

- Existing-pool HS repair: **47.13 seconds**, **$0 API**, with independent replay
  validation about **12.37 seconds**, peak RSS about **994 MiB**.
- Isolated added full-HS sampling stage: approximately **24.4 microseconds per
  scenario**, no extra network requests.
- Final PG/subsidiary scenario + render/preflight probe: 4.608 → 4.793 ms,
  +0.185 ms (+4.0%, negligible beside model latency); measured RSS 436.2 MiB,
  existing draws/facts unchanged.
- Fresh HS generation/review/postal correction: **$0.01154718**.
- VIA semantic probe: **$0.00100060**.
- PG/subsidiary production probe: **$0.00592460**.
- Flash compilation/download: **$0 model API**; fresh flash synthesis and review
  accounting is **$0.01488925**, including a conservative $0.012965 upper-bound
  reservation for a review request with a connection error. Its actual remote
  billed usage is unknown, so this is not presented as a confirmed charge.
  Together with the other probes, total accounted API cost is **$0.03336163**
  (an upper-bound accounting total). Provider choices/retries were explicit,
  not silent fallbacks.

## Final receipts and reusable entry points

- [Dataset exact validation](analysis/hs-dg-format-repair-20261014/dataset-validation.json)
- [HS per-record edits](analysis/hs-dg-format-repair-20261014/hs-edits.json)
- [Fresh production synthesis and labels](analysis/hs-dg-format-repair-20261014/fresh-synthesis/)
- [Independent geometry](analysis/hs-dg-format-repair-20261014/geometry-validation.json)
- [DG samples, sources and audit](analysis/hs-dg-format-repair-20261014/dg/)
- [Registry publication](analysis/hs-dg-format-repair-20261014/registry-publication.json)
- [Completion certificate](analysis/hs-dg-format-repair-20261014/completion-certificate.json)
- [Rebuilt-container validation](analysis/hs-dg-format-repair-20261014/runtime-validation.json)
- [Current training config](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic5000_private_ids_direct_r20_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml)
- [Reusable DG property validation config](../configs/synthesis/mpci_bl_curated_v7_dg_properties_validation.yaml)

Historical campaign artifacts retain their original pins. Old augmentation
selection proposals need refreshed row/file pins when assembled against the
repaired sidecar; their historical hashes must not be bypassed or treated as
current. No synthetic-only dataset assembly or additional bulk synthesis was
performed in this repair.

Final runtime confirmation: the rebuilt Compose image passes `validate-config`,
`inspect-dataset` and tokenizer-only `prepare-dataset`. All 5,600 training and 60
validation records are retained: **zero source truncations and zero target-length
exclusions**. Maximum training source/target lengths are 8,073/3,146 tokens;
validation maxima are 5,698/1,334. Training hyperparameters are unchanged; only
the updated task-constraint and training-file hashes are repinned.
