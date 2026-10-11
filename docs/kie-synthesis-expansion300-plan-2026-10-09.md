# Selecting the third 100 synthesis source templates

Date: 2026-10-09. **Completed: 100 additional templates admitted; shared catalog
now contains 300 sources.**

The initial investigation proposed the next 100 sources for the current
200-template catalog. The subsequently authorized preparation rebased and
validated the final 100-source selection, published 200 fully reviewed descendants
with positions, and committed the additive catalog admission after live replay.
The full implementation, repair, cost, geometry and test evidence is in the
[completion report](kie-synthesis-expansion300-results-2026-10-09.md).
The protected training inputs, real labels and existing synthetic labels remain
unchanged. No training or GPU workload was launched by this work.

The inventory measurements below describe the original investigation, not the
later implementation. Final admission requires current-contract replay, complete
rendered-text/target review with finding adjudication, and independent geometry
checks, all now passed for the final selection. Replacements for complex shortlist sources are
recorded in the campaign audit rather than silently weakening those gates.

## Recommendation

Use the [explicit 100-source shortlist](analysis/synthesis-expansion300-plan-20261009/proposed-selection.csv)
as the starting work list. It reuses current reviewed training labels and
exact-matching historical templates rather than recompiling sources from scratch.
There are **62 ordinary rebases, 28 binding reconciliations and 10 sources with
known dependency work**. Each row records the actual remaining issue, source
and target hashes, traits, and OCR/PDF paths.

Keep the [20 ranked wider-pool alternatives](analysis/synthesis-expansion300-plan-20261009/proposed-selection.json)
available. These are genuine alternatives, not rejected because they are outside
the current training set. If a dependency-heavy source requires disproportionate
work, replace it before paid generation rather than broadening this admission
into another repair project.

This is a recommendation for the fastest path **with the existing campaign
wiring**. It is not a claim that every individual training source is easier to
rebase than every wider-pool source. Historical compiler complexity and current
binding conflicts, not coordinate availability, determine the comparison.

## Evidence and reproduction

Read-only inventory command:

```bash
.venv/bin/python docs/analysis/synthesis-expansion300-plan-20261009/inventory.py
```

Outputs:

- [Current-training source inventory](analysis/synthesis-expansion300-plan-20261009/inventory.json): all 400 unused sources.
- [Wider-pool inventory](analysis/synthesis-expansion300-plan-20261009/external-inventory.json): all 1,258 PDFs outside the current 660.
- [Proposed selection and scenario stress receipts](analysis/synthesis-expansion300-plan-20261009/proposed-selection.json).
- [Current, post-repair coverage](analysis/synthesis-expansion300-plan-20261009/current-coverage.json).
- [Wider-pool alignment experiment](analysis/synthesis-expansion300-plan-20261009/external-alignment-probe.json).

Protected inputs are SHA-256 checked before and after the investigation,
including the running experiment's mixed training/validation files. The mixed
dataset's 600 real targets were also checked to equal the current reviewed real
targets. Historical source hashes, current source hashes, reviewed description
span receipts and saved ownership declarations were used for the rebase probes.

The final inventory took 15.66 seconds for the current-training pool and 12.19
seconds for the wider pool. Eight extra draws per shortlisted source took 0.22
seconds; all 800 passed. Peak RSS for the inventory process was about 821 MiB.
These are CPU diagnostics, not timing projections for full source adjudication
or lexical generation. **API cost for this investigation: $0.**

## 1. The available sources

| Pool | Sources | What is already available |
|---|---:|---|
| Current reviewed real training | 600 | Current labels, original OCR, Paddle output, persisted line alignments |
| Already admitted synthesis sources | 200 | Current owned contracts, declarations, geometry and prior generated-sample reviews |
| Unused reviewed real training | 400 | Current labels and geometry; old compiled template availability varies |
| Wider filtered PDFs outside the current 660 | 1,258 | Saved GLM OCR/old labels and completed Paddle extraction |

The wider pool **does not need another Paddle OCR run**. The extra work there is
reconciling its historical labels/bindings to current policy and making those
sources available to the campaign independently of the existing real-training
fit set. Line alignment is a small deterministic preparation step.

### Current-training inventory

Of the 400 unused sources:

- 227 have a historical template whose source hash matches current OCR: 165
  from v38b and 62 from v34.
- 13 have historical templates but changed OCR, so their old offsets cannot be
  reused unchanged.
- 160 have neither of these historical templates.
- 92 share a historical layout-proxy bucket with validation; 13 have previous
  exclusion decisions. Those two counts describe distinct inventory flags.

After the conservative exclusions, 123 exact-source historical candidates
remain. Of those, 86 pass ownership compilation with current description
receipts and mechanically proposed lexical spans; 37 require binding review.
Of the 123, 101 pass both current scenario draws. One of those 101, `015c845e`,
still has the previously documented shared-address/party-identity alias issue
and is omitted from the proposed 100.

Passing compilation here means disjoint, replayable region ownership under the
specified declarations. It does not substitute for checking whether an old
source-only fact has the right dependency owner. We explicitly reloaded the
previous admission findings, which identify dependency work in 10 shortlisted
sources despite their mechanical compilation passing.

### Wider-pool inventory

The 1,258 PDFs comprise 679 with a v38b historical template, 165 with v34 only,
and 414 with neither. After exact-source and conservative validation screens,
**673** retain an exact-matching historical case.

Within those, **75** meet the conservative low-complexity screen:

- one historically labelled goods row and one package row;
- a nonempty description and no additional-information values;
- at most one cargo-text binding;
- zero residual-agent bindings in the historical compiler metadata;
- no thermal or DG label topology;
- no validation exclusion.

These 75 span 22 carrier families. This is a mechanical rebasing-effort screen,
not an assertion that their old descriptions or package accounting already
follow the newly corrected policy. The inventory reads the historical
template's own `source-label.json`, rather than substituting the earlier,
unprojected label snapshot.

The 20 recorded alternatives additionally have containers and canonical
historical package categories. Ranking prioritizes new carrier-family metadata
then fewer bindings. Examples are `2c383fa7` and `bc767285` (Diamond Line,
40/42 bindings), `85386dc8` (GDT Logistic,43), and `d29b0e2a` (Sidra Line,62).
The list also includes `5f8251b7`, a 45-word historical description suitable for
reviewing longer wording without immediately starting a high-complexity source.

### Validation exclusion contract

The proposed 100 have zero overlap with current validation by document ID,
exact OCR hash, PDF identity or normalized B/L identifier, and none is already
in the admitted 200. The wider inventory also checks document aliases mapped to
the same PDF. Historical validation layout-proxy matches are additionally
excluded, preserving the prior conservative admission practice.

A layout proxy is not proof of shipment duplication, and sharing a carrier's
form is not inherently label leakage. That extra proxy screen is conservative;
it is not being represented as a new user requirement or a defect in those
sources.

## 2. Projected rebasing work

| Work tier | Sources | Specific work |
|---|---:|---|
| A: ordinary rebase | 62 | Reuse reviewed targets/description spans, exact historical offsets and saved declarations; finalize auxiliary ownership, route/capability declarations and full-path preflight. |
| B: binding reconciliation | 28 | Resolve reported overlapping wrappers or mismatched historical identities, then rerun exact ownership compilation. Existing current numeric/postal/product owners often supersede older broader regions. |
| C: dependent printed facts | 10 | Close the named dependency groups below before generation; passing basic compilation/draws does not discharge them. |

All 100 use historical templates: 61 v38b and 39 v34. No from-scratch template
compilation or source-OCR replacement is proposed for this fast-path list.

Representative tier-B issues:

- `0262ae40`: old `15 PALLET` wrappers overlap the new declared-pallet numeric
  owner; a country wrapper also overlaps the complete postal owner. This needs
  region reconciliation, not another address generation campaign.
- `08ec8a5f`: an old complete destination-office address wrapper overlaps the
  current owned address. Its multi-wall-bag sampling capability is a separate
  coverage question.
- `90b0c6a0`: a mechanically proposed whole lexical span consumes another
  field's region. Review the exact region boundaries; do not accept the quote
  merely because its words occur in OCR.
- `c5634568`: a product-description span intersects an older mutable numeric
  wrapper in the banana-puree packing passage. Preserve the agreed complete
  description and jointly model its embedded package/mass facts.
- `0bdef3ca`: the historical allocation identity's source surface does not
  establish its target identity under the current reconciliation check.
  Rebind using the actual current equipment row, not its old array index.

### The ten known dependency sources

| Source | Remaining dependency | Intended bounded resolution |
|---|---|---|
| `11c21de2` | Printed vessel IMO would stay attached to a newly sampled vessel | Remove the exact source-only IMO value/caption in synthetic descendants, or sample a proven vessel/IMO bundle. Preserve real source text. |
| `11da8a8c` | Public handling sentence names Damietta | Co-own/render the entire sentence from its sampled discharge-port owner, updating OCR and target together. |
| `c08ce779` | Public handling sentence names the final CFS/destination | Same dependent-text treatment, with the correct final-destination owner. |
| `2418b6d9` | Carton serial ranges encode old counts; a packing sentence names the old product/type | Regenerate/derive supported count-dependent text, or omit detached source-only ranges; keep all surviving product/package mentions coherent. |
| `383ed9b9` | Repeated bulk-form rows can contradict a newly sampled packed commodity | Retain an explicit compatible bulk physical domain, or co-own every row's sampled packaging wording. Do not leave static bulk rows beside cartons. |
| `85b6388a` | Two people share one public contact-name field but separate printed positions | Use the existing split-contact rendering contract with both positions owned. |
| `978a3990` | Final destination and a detached named-bank address have different owners | Bind the destination to its proper route node; treat the entire bank/address block coherently rather than replacing only its city. |
| `a1c2da82` | Joint-release condition lists companion B/Ls plus the primary B/L suffix | Model the whole release-reference group so OCR and public handling agree and the primary B/L reference follows the sampled bill. This is the strongest replacement candidate if closure is disproportionate. |
| `a2c343f9` | Destination-agent office caption and postal locality repeat | Assign both to that party's sampled locality; remove only obsolete duplicate postal tails, not independent office captions. |
| `cf5d1861` | Registration-office locality is not the loading port or postal city | Use a coherent private registration locality/caption; do not alias Brussels to the port or assume it is the generated street locality. |

These are existing saved findings, not newly asserted defects in current real
gold. Their status must be checked against actual source regions when rebasing.
The low-work strategy is to resolve reusable declaration patterns in groups,
then instantiate exact source spans. It is not to add source-ID branches to
production Python.

## 3. Variability supplied by the proposed 100

| Trait | Proposed sources |
|---|---:|
| Carrier-family metadata values | 38 |
| Pages: 1 / 2 / 3 / 4 | 21 / 56 / 22 / 1 |
| Single container | 53 |
| Multiple containers | 42 |
| No labelled containers | 5 |
| At least eight containers | 6 |
| Longest container list | 14 |
| Source-shaped descriptions of at most ten words | 77 |
| Descriptions longer than twenty words | 5 |
| Reviewed descriptions with multiple spans | 10 |
| Membership-only placement topology | 3 |
| All equipment size/type attributes absent | 6 |
| Negotiable / non-negotiable / OCR-unavailable | 5 / 92 / 3 |
| Notify `sameAs` | 13 |
| Explicit transshipment | 2 |
| Provisional scenario families: ambient / frozen / vehicle / chemical DG | 92 / 4 / 2 / 2 |

The 100 have 80 distinct historical layout proxies. Of these, 56 are absent
from the mapped 199 existing templates; the remaining existing template lacks
this old proxy metadata. Therefore at least 55 of those proxies are new to the
existing set. These are historical grouping keys, not a claim of 100 different
physical form designs. Likewise provisional family assignment is sampler
diagnostic metadata until source capability review is completed.

Useful long-list candidates include `5718f96f` (9 containers), `aaa232a2`
(9,unequal declared allocations), `f36409a0` (13), and `af151047` (14).
The three membership-only sources are `69209200`, `f256749b`, and `b2617901`
(the last has eight containers).

All selected source-alignment artifacts were identity checked. Their aggregate
known-coordinate fraction is 83.27%, with a minimum individual fraction of
24%; admission must inspect the actual product/party/table anchors rather than
equating that whole-document fraction to geometry usability.

## 4. Coverage must use the post-repair labels

The earlier [model-weakness/feasibility audit](mpci-training-results-and-next-experiments.md)
predates the declared-package repairs. Its counts are not the authority for the
next campaign. The current files give the following document-level support:

| Category | Real training | Current synthetic | Validation |
|---|---:|---:|---:|
| Plain paper bag | 0 | 0 | 0 |
| Multi-wall paper bag | 1 | 0 | 0 |
| Plastic bag | 1 | 0 | 0 |
| Steel drum | 2 | 0 | 1 |
| IBC | 7 | 0 | 1 |
| Unpackaged single unit | 5 | 0 | 1 |

In particular, **the old paper-bag validation gap is no longer present** under
the corrected declared-package targets. Material-specific bag coverage can
still be useful generally, but should not be presented as a current validation
failure requiring a quota.

| Shape | Real training | Current synthetic | Validation |
|---|---:|---:|---:|
| Membership-only placements | 21 | 51 | 4 |
| Mixed known/unknown placement quantities | 7 | 0 | 1 |
| Entirely absent equipment attributes | 28 | 44 | 3 |
| At least eight containers | 26 | 24 | 3 |
| Fallback package wording | 9 | 0 | 4 |
| Package row without quantity | 8 | 0 | 0 |
| No package row | 8 | 0 | 1 |

Counts overlap and must not be summed as distinct failing documents. They
describe coverage, not new model performance measurements while training runs.

### Template admission is not a category quota

The current unrestricted scenario draws for the two proposed IBC sources,
`37c96354` and `920a4297`, produced other package types. The steel-drum and
multi-wall-bag sources also produced generic packages. This is permitted
variability, but does not fill the specifically requested rare-category cells.

For the next 1,000 samples:

1. Define compatible per-source category/physical domains and explicit coverage
   targets. Verify actual produced categories, not just source-template labels.
2. Use existing compatible chemical-DG templates for steel drums where their
   full render paths validate; the earlier feasibility work already demonstrated
   the available category constraint.
3. IBC and nonthermal grain/milk-powder bag coverage need suitable identity and
   physical domains. Adding these templates alone does not bypass current
   registry-pool restrictions.
4. Consider admitting `45cff3aa` separately for unpackaged single units. It has
   current reviewed labels but no matching v38b/v34 template. Existing
   `source_whole_units` support is promising; it is a targeted coverage admission,
   not one of the least-work historical rebases.
5. Mixed quantities, absent packages and fallback package wording remain
   separately identified sampler-shape requirements. Do not silently invent
   counts/categories to get those sources through preflight.
6. Use the many short source descriptions for source-shaped lexical depth,
   without character truncation. Template count alone does not prevent the
   generator from writing unnecessarily long descriptions.

## 5. Wider-pool alignment experiment

The existing production line aligner was exercised in memory on all 20 wider
alternatives using the current alignment policy and hash-verified Paddle
`lines.parquet`/`pages.parquet`.

- 20/20 parsed and aligned successfully.
- 1,492/1,882 content lines received measured coordinates (79.28%).
- Unknown lines remained explicitly unknown; no coordinates were fabricated.
- Exact original-text preservation and region-claim collision checks passed.
- Alignment computation: 0.127 seconds across all twenty documents.
- One-time input loading and hash checks: 2.02 seconds; total 2.15 seconds.
- Peak RSS: 269 MiB. No GPU/API calls; no production alignments published.

This confirms the user's correction: alignment is not a substantial reason to
prefer training sources. The remaining outside-pool distinction is current
source-label/contract review and campaign source registration.

The current campaign loads both selected templates and sampler fit support from
its configured training JSONL and checks exact train-fit target identity. An
outside source cannot simply be named in that configuration today. If using
wider alternatives, introduce an explicit reviewed support-source pool with
validation exclusion and deliberate fit inclusion, rather than silently adding
old labels to real training or bypassing the existing source-identity guard.

## 6. Admission sequence for the next pass

1. Stage the proposed 100 separately from the admitted catalog. Reuse exact
   historical templates and current reviewed description-span receipts.
2. Close tier-B overlaps and tier-C dependencies in the declarations. Run
   baseline/source replay, complete ownership and route/physical preflight.
   Replace expensive blocked candidates before paying for generation.
3. Check per-source category constraints against the intended coverage cells;
   keep description-boundary and declared-package policy unchanged.
4. Produce two complete descendants per source, including casing, notify/
   negotiability policy and the current positional reflow/transforms.
5. Review complete rendered OCR and targets, particularly description passage
   boundaries, embedded numbers, declared package levels, container rows,
   source-only dependent facts and optional-field restraint. Resolve findings
   against exact candidate hashes.
6. Run geometry checks/negative controls and representative plots, concentrating
   on long container lists, descriptions and postal reflow. Preserve text even
   where positions legitimately remain unknown.
7. Admit only completed sources, replay the old catalog if shared code changes,
   then configure the separate 1,000-document campaign from measured coverage.

The previous 100-template/200-descendant admission reported about $0.216 in API
spend for its documented generation/review campaign. That is a useful historical
benchmark, not a fixed quote for new source preparation or human/engineering
work. This investigation's source inventory and scenario/alignment probes were
entirely local and free of API charges.
