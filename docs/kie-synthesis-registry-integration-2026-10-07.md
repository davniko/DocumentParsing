# Registry-driven V7 synthesis integration

## Diagnosis and scope

The active V7 pilot narrowed ordinary and frozen goods to exact HS codes found
in current training donors (`explore_within_heading: false`). Chilled goods
were unconditionally donor-only. Its thermal wording context then instructed
the generator to retain the donor commodity. Chemical DG already used the
HMT/ECICS registry. This is not equivalent to the earlier registry-driven flow.

The earlier `template_compiler/cargo_scenarios.py` selected registry identities
conditional on joint package/equipment/measurement support. It did **not**
authorize arbitrary combinations of every HS code and every package.

This pass restores that distinction: registry identities provide goods
diversity; train-only observations provide physical support; source contracts
provide layout, printed-field presence and accounting topology. Current real
train/validation records and previously published synthetic datasets stay
unchanged. Casing, company-conditioned contacts, full goods wording and bounded
position generation remain on the actual campaign path.

## Baseline

Measured before edits, same 24-source configuration, 30 deterministic variants
per source, seed from `mpci_bl_curated_v7_full_pilot24_contacts.yaml`:

| Family | Draws | Distinct HS6 |
|---|---:|---:|
| Ambient | 450 | 166 |
| Frozen | 30 | 1 |
| Chilled | 90 | 3 |
| Chemical DG | 30 | 27 |
| DG vehicles | 30 | 1 |
| Vehicles | 90 | 3 |

Registry/catalog loading: 4.15 seconds. Sampling 720 shipments: 0.375 seconds.
These counts measure sampling, not generated/accepted full text.

## Thermal authority and limits

HS wording distinguishes explicitly frozen/chilled commodity identities, not
all operational carrying requirements. Fresh produce can need commodity-specific
settings. The retained earlier frozen synthesis range is -24 to -18 C; its
interpretation is a synthetic training envelope, not shipping advice.
[Maersk's reefer explanation](https://www.maersk.com/tr-tr/logistics-explained/transportation-and-freight/2025/03/06/reefer-containers)
supports frozen carriage at or below -18 C.

[CMA CGM's conservation guide](https://www.cma-cgm.com/assets/public/page-complex-documents-apl/GuideConservation.pdf)
distinguishes commodity-specific temperatures and fresh-air settings, including
closed ventilation for refrigerated meat and frozen cargo. Registry-defined
chilled synthesis uses an explicit conservative temperature policy, not an
apple/garlic donor's unrelated temperature. Observed produce settings remain
bound to that commodity and are recorded as empirical extensions.

## Validation contract

- Registry files move without byte changes; pinned hashes and counts must pass.
- Registry membership and package/equipment compatibility must be checked for
  every host draw; receipts distinguish commodity and physical-load authority.
- Draws must demonstrate HS identities absent from training, including thermal
  identities; source-only exact HS sampling must not silently remain active.
- Temperatures, ventilation and DG tuples must agree across targets and all
  owned printed occurrences. No additions to absent label fields.
- Exact allocation sums, load bounds, native structured output, target casing,
  edit replay, semantic review and publication gates remain active.
- Run the full CLI through publication and position enrichment; separately
  measure coordinate coverage rather than implying every line is positioned.
- Record errors, corrective attempts and all paid calls; preserve originals.

## Implemented flow and configuration

Active configuration:
[`mpci_bl_curated_v7_registry_pilot72.yaml`](../configs/synthesis/mpci_bl_curated_v7_registry_pilot72.yaml).
The actual campaign CLI, not a separate prototype, ran generation, rendering,
review, publication and coordinate enrichment.

1. Load hash-pinned authorities from `artifacts/registries/`.
2. Select train-only physical support compatible with the source's printed
   package/equipment/measurement topology. This supplies joint load statistics,
   not the only permissible goods identity.
3. Sample goods from the compatible registry domain. Ordinary HS6 identities
   span supported headings; frozen/chilled food identities span their explicit
   registry profiles. Chemical DG draws a linked HMT/ECICS record.
4. Sample package counts, allowed equipment and loads together; preserve exact
   allocation sums, container capacity bounds and required measurement fields.
5. Derive thermal/DG facts from the sampled identity/profile. Reuse observed
   fresh-produce settings only for that exact empirical commodity extension.
6. Generate commercial product wording, parties and company-conditioned
   contacts. Chemical DG and tightly constrained whole-vehicle identities are
   host-owned exact registry phrases; unrestricted chemical formulation changes
   are not lexical decoration. Long ordinary/thermal descriptions still use
   the complete source-description context and jointly generated fragments.
7. Render changed facts into owned source regions and derive current V7 labels.
   Apply configured target casing. Review full rendered text against the labels
   and sampled facts, resolve findings, then publish the complete scope.
8. Produce a separate positioned dataset from source anchors, measured local
   space and coherent page transformations. Preserve every word and target.

### Goods domains and deliberate boundaries

- Ordinary goods: 5,612 HS commercial phrases are available as the overall
  registry, filtered by the earlier ambient chapter policy and compatible
  donor headings. The new sampler does **not** claim every HS code can fit any
  source package, container, mass or volume.
- Frozen: **82 registry HS6 identities**, plus applicable exact observed produce
  extensions. Configured setpoint lattice: **-24 through -18 C, step 0.5 C**;
  the generic frozen-food profile uses closed fresh-air ventilation.
- Chilled: **73 registry HS6 identities** in meat/fish chapters, specifically
  choosing non-live chilled forms. Default setpoint **0 C**, closed fresh-air
  ventilation. The profile is configurable within its supported food envelope.
  Fresh apples/other empirical produce can retain their own paired temperature
  and ventilation. This avoids assigning a garlic donor's settings to tuna.
- Chemical DG: **386 eligible regulatory-record/HS tuples**, from the existing
  HMT/ECICS maritime-eligible non-bulk domain. UN number, proper shipping name,
  exact class, packing group and subsidiary hazards stay one regulatory tuple.
  This is not unrestricted support for every DG class or legal packing approval.
- Vehicle and DG-vehicle templates retain their explicit whole-unit domains.
  One liquid-fuel DG vehicle domain legitimately remains HS870323/UN3166;
  randomly adding electric/gas vehicles would require different declarations.
- DG/thermal *presence* remains source-dependent. This pass varies the goods
  inside compatible templates; it does not inject new unprinted sections into
  arbitrary ordinary documents.

The obsolete `explore_within_heading` option was removed from code and the
three curated configurations that supplied it. Extra/obsolete fields raise
validation errors rather than silently restoring donor-only sampling.

### Casing and prior integrations

The main path retains full-description generation, scoped edits, generated
company contacts and position generation. Target casing is configured through
`casing.target` (`uppercase` or `preserve`); render casing through
`casing.render_styles`. This pilot uses uppercase targets and a deterministic
36 uppercase / 36 title-style render split. Casing normalization is limited to
the existing human-readable target policy; identifiers, emails and enum/unit
formats are not blindly uppercased.

## Shared registry home

[`artifacts/registries/README.md`](../artifacts/registries/README.md) describes:

- `sources/`: downloaded authorities and provenance.
- `compiled/`: versioned normalized registry snapshots.
- `phrases/`: HS commercial-wording registry.

**360 files, 157,979,106 bytes** were moved without byte changes. All hashes were
checked before and after, and checked again after implementation. The exact
old/new mapping is in `artifacts/registries/relocation-receipt.json`.
**166 code/config/test/script files** had literal paths mechanically updated.
No compatibility symlinks or duplicate registry trees were left behind.
Historical experiment receipts retain their original paths and hashes; the
relocation receipt locates their assets without falsifying old provenance.

Some historical training constraint references also contained registry paths
and were mechanically updated. This does not change their vocabulary contents,
but historical runs pinned to the *old file bytes* remain historical snapshots.
The active r16 task-constraints pin was verified on the actual campaign startup.
No training dataset, label or OCR was modified by this pass.

## Offline diversity and physical validation

The same 720-draw source/seed workload used for the baseline now produces:

| Family | Draws | Before: distinct HS6 | After: distinct HS6 | After: HS6 absent from training |
|---|---:|---:|---:|---:|
| Ambient | 450 | 166 | 350 | 254 |
| Frozen | 30 | 1 | 29 | 28 |
| Chilled | 90 | 3 | 53 | 52 |
| Chemical DG | 30 | 27 | 27 | 26 |
| Vehicles | 90 | 3 | 3 | 2 |
| DG vehicles | 30 | 1 | 1 | 0 |

Chemical DG was already registry-driven; unchanged diversity is expected there.
A larger **2,400-draw** stress test passed registry membership, public-field
presence, allocation sums, reefer/setpoint/ventilation coherence, DG tuple
checks and non-mutation of training records:

| Family | Draws | Distinct HS6 | Absent from training |
|---|---:|---:|---:|
| Ambient | 1,500 | 636 | 478 |
| Frozen | 100 | 62 | 59 |
| Chilled | 300 | 73 | 72 |
| Chemical DG | 100 | 55 | 51 |
| Vehicles | 300 | 3 | 2 |
| DG vehicles | 100 | 1 | 0 |

Evidence and runnable audit:
[`sampling-720-final.json`](analysis/synthesis-registry-integration-20261007/sampling-720-final.json),
[`sampling-2400.json`](analysis/synthesis-registry-integration-20261007/sampling-2400.json),
[`audit_registry_sampling.py`](../scripts/synthesis/audit_registry_sampling.py).
The earlier `sampling-720.json` is a **superseded diagnostic** from before the
negative thermal-wording correction described below.

## Full 72-sample pilot

**24 sources × 3 variations = 72 published samples**, all accepted by final
replay/schema/physical checks and fresh full-text reviews, with zero unresolved
findings and no findings waived by adjudication.

| Family | Samples | Distinct HS6 | Distinct final descriptions |
|---|---:|---:|---:|
| Ordinary | 45 | 54 | 45 |
| Chilled | 9 | 9 | 9 |
| Frozen | 3 | 3 | 3 |
| Chemical DG | 3 | 3 | 3 |
| Vehicles | 9 | 3 | 3 |
| DG vehicles | 3 | 1 | 3 |

Overall: **73 distinct sampled HS6**, **11 package categories**, **5 equipment
size/type pairs**, **57 origin countries**, **61 destination countries**, and
**72 distinct country pairs**. Egypt is one destination, not the sole destination.
Some templates support several HS identities inside the same accounting group.

Examples include frozen abalone (HS030783, -20.5 C), chilled southern bluefin
tuna (HS030236, 0 C), and exact registry chemical identities lithium hydroxide
(UN2680), antimony pentafluoride (UN1732), and diallylamine (UN2359). All thermal
containers are refrigerated. Setpoints/ventilation propagate only into fields
and text regions the source supports; absent public fields are not invented.

Inspection outputs:

- [Original OCR followed by all rendered variations](../artifacts/kie-synthesis-production/curated-v7-registry-pilot72-v1/samples.md).
- [Plain dataset](../artifacts/kie-synthesis-production/curated-v7-registry-pilot72-v1/dataset.jsonl).
- [Positioned variations](../artifacts/kie-synthesis-production/curated-v7-registry-pilot72-v1/positions-v2/samples.md).
- [Positioned dataset](../artifacts/kie-synthesis-production/curated-v7-registry-pilot72-v1/positions-v2/dataset.jsonl).
- [Independent publication audit](analysis/synthesis-registry-integration-20261007/publication-audit.json).

## Defects found and resolved during validation

These were real failed attempts, not hidden successes:

1. **Negated thermal words:** the shared HS classifier treated “not frozen” or
   “whether or not frozen” as positive evidence of frozen goods. Corrected the
   classifier and extended its existing tests. Eleven inappropriate frozen
   candidates disappeared before the accepted diversity test.
2. **Composite thermal ownership:** source `0951955d…` bound a fixed refrigerated
   sentence prefix to an entire handling instruction containing the temperature.
   The temperature already had a separate numeric owner. Made the prefix an
   explicit source constant, retaining the separate setpoint binding. New
   sampled temperatures now render without duplicating or losing the statement.
3. **Generated wording validation:** one initial response copied shipment totals
   into product prose. Added one bounded, costed, recorded correction after
   wording validation, with failure still surfaced if correction fails.
4. **Postal correction prompt:** it constructed region-specific requirements
   but did not actually send them in its compact request. Fixed that omission.
   The failed-region correction pass resolved the remaining locality issues;
   one sample also needed a recorded manual removal of repeated company/country
   wording. This does not certify real-world postal deliverability.
5. **Full-text review:** seven initial findings across five samples exposed
   literal escaped newlines, invented per-carton weight/packing, unrelated
   mobility-aid wording instead of mineral machinery, and geography wording.
   Corrected the true text defects, rerendered, and reviewed again. The county
   wording was removed; postcode-to-city deliverability remains outside the
   agreed extraction-training objective. All original calls and corrections
   remain recorded. The scoped product correction also needed manual removal
   of fresh extraneous shipment/drafting clauses; it was not blindly accepted.
6. **Independent inspection after the first review:** DG expansion mixed solid
   lithium hydroxide with a solution, although the registry assigns solution
   UN2679 rather than UN2680. A second chemical included an unlicensed generated
   solution. A frozen-food description invented a per-carton count. The initial
   reviewer missed these. The initial publication is explicitly preserved under
   `superseded-initial-publication/`, marked **do not train**, not silently
   overwritten or represented as the final accepted result.
7. **Root correction for DG wording:** chemical identities now come directly
   from the sampled proper shipping name. The wording model cannot add another
   chemical/formulation. Parties and routes remain generated; chemical diversity
   still comes from the 386-tuple registry domain. Added tests proving the
   chemical product field is absent from the free-wording request and remains
   host-owned during label assembly. Removed the frozen packing claim and
   expanded the per-package fill guard to piece counts. Re-rendered, reviewed,
   and republished the full 72 records.

Literal escaped-newline and explicit per-package fill failures now fail the
deterministic wording gate early. These checks are not presented as a complete
semantic parser; unrelated product meaning still requires full-text review.
The final independent audit asserts exact DG description/PSN equality on all
three chemical samples, in addition to the main pipeline's edit replay.

## Performance, costs and operational implications

- Same 720-draw workload: sampling **0.375 → 0.190 seconds**, about 49% lower
  elapsed time, through cached compatible donor/identity pools.
- One-time catalog initialization **4.15 → 4.56 seconds** in those measurements,
  including broader registry classification. This roughly 0.41-second startup
  increase does not recur per sample; it is reported rather than hidden.
- 2,400 draws: **0.572 seconds** after **4.419 seconds** loading; peak process RSS
  **436.7 MiB**. No baseline RSS was captured, so no memory delta is claimed.
- Final 72-sample rendering/checks: **3.088 seconds** after initialization.
- Final coordinate pass: **3.789 seconds** after initialization.
- Final targeted suite: **292 passed, 1 skipped, 14.80 seconds**. The skip is
  recompiling the original DG spreadsheets outside the pinned synthesis
  environment. The compiled, hash-pinned DG registry was loaded and exercised
  by the sampler and full pilot. Targeted Ruff and `git diff --check` passed.

All billed campaign attempts, including discarded generations, correction
attempts and repeated reviews, total **$0.07818274**:

| Stage | USD |
|---|---:|
| Main wording | 0.03203395 |
| Company contacts | 0.00162100 |
| Postal corrections | 0.00303320 |
| Full rendered reviews, including repeated reviews | 0.04006840 |
| Scoped goods correction calls | 0.00073205 |
| Bounded wording-validation repair | 0.00069414 |

That is **$0.001086 per accepted sample**. Linear arithmetic is about **$10.86
per 10,000**, but this is a development-pilot extrapolation, not a campaign quote:
it includes repeated paid work and excludes manual engineering/adjudication.
No extra API cost is incurred by deterministic sampling, casing or positions.

## Coordinate checks and coverage

Final coordinates cover **5,767 / 7,385 content lines (78.09%)**. The remaining
**1,618** retain the explicit empty ` ||` suffix; text is never shortened to
force a fit. Of 155 changed-region placement abstentions, 72 lacked local space,
38 intersected unowned source text, and 45 lacked source anchors. Region counts
are not line counts and therefore do not sum to the unknown-line count.

Across **120 pages**, 119 use accepted uniform scale/translation and one uses
the explicitly recorded integer-translation-only mode to preserve geometry.
**5,766 of 5,767 known points moved**. Independent replay verified text/target
preservation, page bounds, axis ordering/alignment and nearest-neighbour ties.
All **120 intentionally out-of-page mutations were rejected**. Expanded regions
use measured height/clearance; 13 used local reflow and five source-span
interpolation. These are plausible source-layout anchors, not measured glyph
centroids for a newly rendered PDF. Better model performance is still an
experimental question, not a result of these geometry checks.

## Completion and next use

The registry integration and the complete 24-source/72-sample pilot are done.
No training was launched. The real 600/60 dataset was not edited. The new plain
and positioned pilot datasets are separate publications with current hashes.

For scaling, keep these exact sampling/ownership/review gates, choose the
desired campaign size and family proportions, and audit additional source
contracts before adding new families. The current domain is intentionally
broader than donor-only goods but not a certification of all historical
templates or every DG regime. Do not bypass failed semantic reviews merely
because deterministic replay passes. The DG correction above removes a
specific semantic degree of freedom that an LLM reviewer proved insufficient
to police reliably.
