# Compiled-template recovery: continuation and synthesis handoff

## Current decision and scope

On 22 September the user paused recovery of the remaining difficult templates
and authorized the next synthesis campaign using the approximately 1,100 already
reconciled candidates. Do not resume the unresolved queue merely because a new
synthesis run is being prepared. Finalize/validate the chosen pool, synthesize
10,000 new documents, audit all three synthetic populations, merge accepted
records, produce EDA, and put cargo relations last in the training JSON/schema.
Training itself remains user-launched.

This document is the durable entry point for resuming template recovery later.
It distinguishes evidence levels; it is not blanket quality approval.

### 23 September update: use the repaired, frozen handoff

The next planning input is now
`artifacts/kie-synthesis-production/campaigns/recovered-20260922/next-run-template-inputs.json`.
It pins `ready-template-base1110-v2`, its numeric contracts and lexical contracts.
Do not launch another run against the original `frozen-catalog1112` snapshot or
reuse its sample/route-plan hashes. That snapshot remains the provenance parent
of the paid campaign, not the corrected next-run base.

The new base has **1,110 templates**, exact source round trips, and four explicitly
reviewed template repairs that passed **12/12** route/goods/format replay variants:

- `c8a72…`: replace the redundant global scalar temperature with a reference to
  the individually printed container settings; retain every individual setting.
- `a915a…`: remove only the duplicated, unowned AZARITA neighborhood fragment;
  retain the complete bound source address.
- `103c6…`: separate the product-description and repeated-mark owners, connect
  their dependency, and bind the inline multiplication count to its package.
- `dfbf63…`: restore the omitted, explicitly printed `03035510` HS label and
  direct tariff binding. Its former generic reference generator was wrong.
  This is a source-evidence-backed label derivative, not an inferred label.

Two additional sources (`07590a…`, `1e7b16…`) remain explicit review cases because
their source-only carrying-temperature statements lack a complete physical
sampling contract. Their originals are retained. The **332 previously paused
recovery cases remain paused**, and the **66 validation exclusions remain separate**.
Thus the original 1,510 inventory is accounted for as 1,110 ready, 334 review,
and 66 validation-excluded. Read `ready-template-base1110-v2/review.json` for
the two additional dispositions; do not call them intrinsically unrecoverable.

Compiler/pre-generation guards now reject random-reference treatment of explicit
tariff codes and unowned counts in explicit mass-times-count equations. The
generation preflight also checks source-only carrying-temperature ownership.
Private sampled tare values are exported and restored explicitly across the
generation/publication boundary, with exact contract coverage and receipt hashes.
No missing private context is replaced with a source value.

The campaign report documents dataset curation separately: template repair,
generation acceptance, full-record replay, and final derivative acceptance are
different evidence levels. Never promote a failed candidate merely because its
template appears in the ready catalog.

## Exact inventory at handoff

The original immutable catalog contains **1,510** templates. The last complete
recovery queue (`recovery-intervention-queue-v83.json`) accounts for:

- **1,105** sources passing all three offline route/cargo/render-plan variants.
- **339** unresolved sources, partitioned by primary intervention.
- **66** exact validation-source exclusions. These are not failed templates.

The equipment64 sweep subsequently accepted **seven additional, non-overlapping
sources**, yielding a **1,112-candidate cumulative tally**, **332 unresolved**, and
66 validation-excluded. The 1,105 did not all receive the newer actual-generation,
whole-document model and manual acceptance used for the seven. Do not describe
the tally as 1,112 uniformly end-to-end certified sources. A fresh frozen-pool
preflight is required after shared-code changes.

The seven accepted equipment sources have prefixes `124917`, `17440`, `1eb691`,
`455bec`, `4a8aad`, `db56`, `f997`. Their committed recovery bundle is separate from
the production catalog. Merge by exact source ID and pins, not by filename order.

## Authoritative locations

Paths below are relative to the repository root. Define `R` as
`artifacts/kie-synthesis-production/analysis/catalog-recovery-20260921` when reading
this document; it is notation, not an instruction to overwrite an environment variable.

| Purpose | Location |
| --- | --- |
| Original immutable catalog | `artifacts/kie-synthesis-production/template-base/catalogs/mpci-bl-production-template-catalog1510-v6` |
| Reconciled working view | `R/recovered-catalog-view/cases/<source-id>/` |
| Working-view source/repair lineage | `R/recovered-view-lineage.json` |
| Re-pinned numeric contracts | `R/recovered-numeric-contracts.jsonl` |
| Reviewed lexical contracts | `R/recovered-cargo-lexical-contracts.json` |
| Last full static inventory | `R/recovered-inventory-v83.json` |
| Last full three-variant replay | `R/recovered-preflight-v83.json` plus four shard reports/logs |
| Exact unresolved queue and source pins | `R/recovery-intervention-queue-v83.json` |
| Queue explanation | `R/RECOVERY-INTERVENTIONS-v83.md` |
| Equipment batch dispositions | `R/equipment-batch-v87/batch-closure.json` |
| Equipment methods/results | `R/equipment-batch-v87/REPORT.md` |
| Seven accepted recovery candidates | `R/equipment-batch-v87/reviewed-recovery-bundle/` |
| Equipment candidates not generally accepted | `R/equipment-batch-v87/replay-view/` |
| Full-generation equipment probe | `R/equipment-batch-v87/complete-acceptance/093505f1f9246294a0fbfb64b367fa4e8b50072b499f908542b627b4e25eb994/` |
| Final independent equipment reviews | `R/equipment-batch-v87/semantic-reviews-final22/report.json` |
| Recovery API accounting | `R/review-spending.sqlite3`, cap **$10** |
| Cached source-pinned GLM reviews | `R/review-json-cache/` and `R/review-tool-failures/` |
| Corrected diversified dataset | `artifacts/kie-synthesis-production/reviewed-datasets/mpci-bl-diversified-corrected9930-v4` |
| Reviewed Egypt-heavy dataset | `artifacts/kie-synthesis-production/reviewed-datasets/mpci-bl-egypt-reviewed9980-v1` |

Working views are not immutable publications. `_COMMIT.json`, exact file hashes,
source IDs, derivative lineage and execution receipts determine authority.
Historical reports remain evidence of their snapshot, not current population counts.

## The original failure mechanism

Source certification established exact source reconstruction, aligned spans,
declared bindings and critic acceptance. Broader synthesis then varied routes,
parties, goods, package levels and equipment jointly. Some source-only shipment
facts did not have complete dependency contracts for that mutation. Others had
sound evidence but unsupported sampling capabilities. A third class had genuine
source contradictions or labels for information absent from the OCR.

The original 742-template quote was an intermediate selection, not a deleted
half-catalog. The prior diversified publication ultimately used 681 sources.
Never confuse “excluded from one sampling policy” with intrinsically bad data.

## Normal recovery procedure

1. **Freeze evidence.** Pin source bytes, original label, certified template,
   numeric/lexical contracts, config, implementation and failure receipt.
   Check exact validation exclusion before any trial.
2. **Read the source in context.** Inspect all occurrences, headings, continuation
   pages, party roles, row order, units and relationships. Matching text alone
   does not prove two occurrences have the same semantic owner.
3. **Separate intervention types.** Is this wrong annotation, missing rendering
   grammar, missing numeric/physical relation, missing sampler capability, an
   empty configured support intersection, or ambiguous/contradictory source data?
   Do not fix an empty domain with repeated random retries.
4. **Resolve the general contract.** Add shared production behavior only after a
   real repro identifies the cause. Use existing relation/owner primitives where
   they exactly fit. Never add a document-ID branch to a hot path to conceal it.
5. **Build an evidence-backed candidate transaction.** Include source, label,
   template, catalog row, numeric/lexical contracts and lineage together.
   Original files remain immutable. A corrected derivative is allowed only when
   the document supplies unambiguous evidence; ambiguous cases remain review.
6. **Validate source fidelity and ownership.** Strict schemas, exact span/text
   alignment, byte round-trip, host realization/coherence/carrier checks, and
   numeric identity replay. Numeric ownership includes units, dimensions,
   rounding precision, row/subtotal/aggregate scope and packing level.
7. **Exercise mutation.** Run route-first sampling, structured proposal, joint
   goods/equipment/package sampling, lexical preparation, auxiliary derivation
   and initial render plan across at least three independent variants. A source
   round-trip alone cannot demonstrate safe mutation.
8. **Run actual complete generation/rendering.** Use sample IDs, not source IDs,
   for independently generated scenarios. Validate changes survive to accepted
   targets and rendered documents. Keep source-only internal facts out of the
   extraction schema. Test checkpoint/resume as well as fresh execution.
9. **Independent semantic review and human adjudication.** Check repeated
   identities, numeric equations, addresses/localities, route ownership, cargo
   descriptions, quantities, HS presentation, DG and temperatures. A model
   verdict is advisory; require exact evidence and reject unsupported objections.
10. **Publish only proven candidates.** Commit immutable artifacts, coverage,
    approvals, exclusions and costs. Merge by source identity with provenance,
    then re-pin downstream configs. Do not silently promote proposals or merely
    mechanical passes.

Useful tools in `R`: `repair_common.py`, the bounded `recover_*.py` scripts,
`recheck_excluded.py`, `parallel_preflight.py`, `build_equipment_batch_v87.py`,
`equipment_acceptance_pilot_v87.py`, `review_equipment_batch_v87.py`,
`collect_equipment_acceptance_v87.py`, and `close_equipment_batch_v87.py`.
Read their CLI and pinned inputs before reuse; many are intentionally snapshot-
specific one-time tools, not general production entry points.

`recheck_excluded.py --all-static --variants 3` still starts from the supplied
inventory's passing static checks. To prove exact selected-pool coverage, supply
an explicit selected inventory and independently verify all requested IDs and
variants appear. Do not infer full catalog coverage from a passing summary.
Its saved previews deliberately retain pending linguistic/residual fields and
are labelled **partial previews, not accepted training data**.

## Safety and semantic rules established with the user

- No source-copy fallback when generation or representation fails. No silent
  party/cargo restoration and no unseen extraction-field enrichment.
- Carrier and source-supported topology stay bound to the template; repeated
  template use is intended. Every new sample has an independent sample ID.
- Private equipment size/type is allowed when unprinted, for compatibility and
  capacity only. Partial printed observations stay partial in text/labels.
- Audited private measurement units are allowed where units are absent. This
  does not authorize guessing an unknown numeric dimension or adding units to
  text/labels.
- Source-only exporter, contacts, marks, counts and totals need semantic owners
  too; they do not become unwanted target-schema fields.
- Cities must have textual role-owned evidence. Do not geocode street addresses
  or train the model to infer an unprinted city/country.
- Cargo origin, shipper domicile and loading country can differ. Do not erase
  legitimate distinctions to satisfy an overbroad geography screen.
- Nested pallets/cartons/pieces are distinct levels. Partial rows are not a
  complete inventory; never invent missing rows or allocate residual totals as
  source facts.
- HS6 registry identity plus deterministic synthetic suffix digits is permitted;
  preserve each slot's printed digit length and separators. Suffixes are not
  claimed to be legally valid national tariff codes.
- DG tuples come from the supported registry and physical constraints, not
  arbitrary invented UN numbers. Thermal/DG sampling remains template-capable.
- Legal/flavor text is lower priority; concrete shipment contradictions are not.
- Retain all API spending, including failures, retries and uncertain reservations.
  Recovery cap is $10. A fresh 10k synthesis has its own **$6** cap. The historical
  $7 exception applied to one run only and must not be inherited.

## Attempted speedup: method and actual result

The proposed broad sweep grouped the 64 equipment cases into intervention
families: inventory/anonymous ownership (24), source-only physical measurements
(14), partial/opaque equipment (14), tare support/ownership (7), and representation/
tank grammar (5). These were primary diagnoses, not promises of full recovery.

The approach combined independent diagnostics, reusable core fixes, exact
candidate transactions, an all64 assembler, three-variant offline replay and a
bounded complete-generation/semantic review. Parallel local replay uses four
process shards; provider calls use PydanticAI concurrency up to 16. Existing Codex
workers completed the initial bounded implementation tasks. The user then asked
to avoid further Codex subagents because of usage consumption; subsequent model
work used **OpenRouter GLM-5.3-Flash**, normally high reasoning, with strict output
validation, response caching and persistent accounting.

Provider caveat: Phala rejected the tested native structured-output requests.
The later equipment probes used the explicitly verified Parasail native profile
in `R/provider-native-v80.json`. Do not assume advertised support is sufficient;
probe transport/schema behavior before expanding a new endpoint. The production
generation provider is a separately pinned config, not implicitly this reviewer.

Measured equipment sweep:

| Stage | Result |
| --- | --- |
| Accounted sources | 64, exact coverage |
| Changed candidate/numeric transactions | 32 |
| All-three offline passes | 14 initially → 26 after proven shared fixes |
| Actual generation admitted | 25; one invalid source-ID case held out |
| Actual generation/rendering passes | 20 initially → 22 after one explicit retry pass |
| Independent semantic passes | 8 |
| Final manual acceptance | **7**, with 57 still held |
| Tests | 619 targeted passed; source Ruff and targeted mypy passed |
| Offline replay | 192 variants in 30.39 seconds, about 464 MiB peak RSS |
| Actual 25-document probe | 169.73 seconds; retry pass 69.14 seconds |
| Final 22-document semantic review | 91.83 seconds, using matching cached reviews |
| Additional API cost | $0.137614 settled; $0.149243 including additional exposure |
| Recorded work window | Approximately 68 minutes; exact task start was not instrumented |

Improvements included exact receipt grammar, bounded immutable parsing caches,
anonymous/mixed inventory, independent sampled tare contracts, physical row
ownership, per-unit mass floors, nested allocation equality and dependent party
marks/warehouse context. Nested allocation tests improved from 93/200 inconsistent
draws to 0/200 while preserving changed quantities and partial row coverage.

**The speedup did not demonstrate 64 fully accepted repairs in one pass.** The
first-error classification hid later failures. Full semantic review found stale
package counts, inconsistent exporter identities/countries, phone-prefix/address
projection corruption, unsupported generated port captions, and gross values
that ignored an explicit net-plus-tare equation. Even a model-approved case had
competing original-count declarations, caught manually. More reviewer calls
alone do not supply missing executable contracts or resolve ambiguous evidence.

For future recovery, collect independent failures across the entire dependency
graph before batching; group by complete intervention signature, not first error.
Pair a repair proposal with executable source proof, generated mutation and
independent rendered review. Stop promoting at the weakest unfinished boundary.

## Remaining recovery queue (paused)

At v83 the primary cohorts were: equipment64, route49, mass35, scenario30,
goods-support29, packing27, source-evidence24, numeric22, DG/thermal20, cargo-
language18, surface10, party/geography8, diagnosis3. Seven equipment approvals
reduce the total unresolved from 339 to **332**; the equipment cohort retains57.
Within those57: 39 offline/source holds, three generation holds, 15 semantic holds.

Examples of work still required: observed row/aggregate physical ownership,
opaque carrier abbreviations, empty joint goods/equipment domains, explicit area
versus volume, source-proved tare-inclusive gross, multi-level package equations,
complete route/transport occurrence ownership, vehicle scenarios, DG properties,
fixed/segmented cargo language and source contradictions. Some cases are
recoverable capabilities, others are ambiguous evidence. Do not call the entire
remaining set irrecoverable or delete it.

## Synthesis and dataset follow-through

1. Freeze the 1,105 offline candidates plus the seven accepted additions and
   validate current code against this pool; preserve exact exclusions.
2. Compare real train-only distributions and the two existing synthetic sets.
   Keep 10% DG and 15% thermal quotas; improve supported rare categories without
   pretending source templates support unprinted structures. Retain route/locality,
   full registry goods and joint package/equipment sampling.
3. Generate the next10k with unique sample IDs and the campaign ledger. The
   initial $6 cap was subsequently raised explicitly to $8 for this campaign
   only; the completed run settled at $6.92543453. Preserve paid work on resume
   and fail explicitly; do not launch training.
4. Audit each population record-by-record with automated evidence/semantic checks,
   manually adjudicate flags, produce documented corrective derivatives, and
   exclude genuinely unresolvable records. Verify schema, hashes, IDs, label/text
   agreement, numerical consistency, diversity and validation leakage.
5. Merge only accepted synthetic records; add the 1,057 real training records
   once, and preserve the same 100 real validation documents. The historical
   “first100k” wording means the Egypt-heavy **10k** run in this approximately30k
   request, not an additional100k population.
6. Reorder relations to the final field of `documentPatch` in serialized training
   targets and the model-facing prompt schema. JSON object order is a training
   presentation contract here, not a change to label meanings or values.
7. Produce a combined EDA with per-cohort comparisons, source/layout/carrier
   reuse, geography/routes, goods/HS/UN, package/equipment, numerical and sequence
   lengths, actual mutation rates, repair/exclusion counts and costs. Report
   observed coverage and limitations rather than claim universal zero errors.

The older datasets already have correction snapshots, but their existence does
not substitute for a fresh audit under the final rules. Preserve all original
paid run histories, real data, validation identity and existing training outputs.
