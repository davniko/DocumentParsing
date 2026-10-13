# Synthesis repair lineage and independent pre-training re-audit

Started: 2026-10-12; completed 2026-10-13 local time. Owner: root; no Codex
sub-agents. Training is not started.

## Scope and acceptance contract

Audit the complete repair history feeding the current 360-template registry,
the original retained 2,500 synthetic examples and the replacement UAE 2,500.
The fresh whole-sample audit focuses on all **2,500 replacement examples** in
`data/curated/mpci-bl-real600-synthetic5000-uae-coverage-r18-positions-v1`.
The immutable dataset, validation split and earlier provider responses remain
protected while investigation proceeds. If a confirmed sample defect requires
a new revision, publish the correction with exact lineage rather than modify a
committed manifest's contents silently.

For each repair distinguish:

- **Prevention:** the source/contract/renderer makes the failure impossible along
  the repaired path.
- **Fail-closed detection:** an invalid generated proposal can recur, but it is
  rejected before publication and requires a reviewed repair/retry.
- **Sample-only semantic correction:** a bounded lexical mistake was corrected;
  no source-specific Python rule or general guarantee is invented.
- **Historical/non-objective finding:** not a currently applicable defect;
  preserve the user's extraction-focused policy and do not over-repair.

Faithful targets and their correct OCR ownership are required. Exact tariff-to-
product mapping, incidental per-unit-fill arithmetic, street deliverability and
complete anonymization of excluded static carrier/Marks/footer text are not new
acceptance requirements. Email and website domains need not match. Package
targets and allocations use the declared shipment accounting unit, not an
automatic inner/outer preference. Whole product passages retain embedded packing
detail but exclude separate loading declarations, Marks and documentary fields.
PDF/layout may establish ownership, never insert missing OCR values into labels.

## Confirmed gaps — corrected and validated

- [x] **L01 — source-authority selection was not enforced upstream.** The source23
  consignee phone correction exists in the current dataset and in a corrected
  dataset/contract/auxiliary triple. The registry documents that triple, but the
  production Campaign does not consume the override declaration. An actual
  fresh Campaign using the original matched source/contract pair passes planning
  and preflight and recreates the goods-area-only phone as a consignee target.
  Add a source-owned, fail-closed target-absence requirement at compilation,
  verify the unsafe authority rejects, and prove current accepted renders stay
  exact. Do not infer party blocks with a proximity heuristic.
- [x] **L02 — packing-detail review expectations needed policy alignment.** The
  current validator deliberately permits incidental per-bag fill/cone capacity,
  consistent with the user's extraction objective. The generation/review prose
  still describes all such details as requiring explicit host support. Trace and
  test a precise clarification so reviewers distinguish product detail from a
  newly declared shipment total; retain checks for explicit totals, equipment
  claims, declared package units and actual product-state contradictions.

The paragraphs describe the diagnosed pre-fix state. The completion evidence
below records the actual source/code changes and execution-path validations.

- [x] **L03 — carton framing was model-owned in source `819af5e4`.** The
  prior campaign explicitly deferred this rebase. Its free product variable
  covers `1800 CARTON(S) X 13.61 KGS OF IQF WILD BLUEBERRIES FROZEN`, swallowing
  the only printed carton-type evidence and one occurrence of the shipment
  quantity. The footer quantity is already host-owned. Accepted descendants
  were individually reconciled, but future generation could omit that type or
  invent a second count. The tested proposal narrows the product owner, exposes
  the inline count to its existing typed owner, and retains the complete
  description through deterministic assembly. No new per-source Python branch
  is necessary. Preserve the old immutable generation authority when promoting
  the revised contract; prove source/gold identity replay and test fresh output.
- [x] **L04 — postal country continuations required ownership reconciliation.**
  Fresh review identified source `0d495914`'s explicit `CN>` continuation and
  source `23e145de`'s starred country footnote. The corresponding consignee
  address/country labels omit these countries. The first has independently
  marked country ownership; the second has now been checked against its actual
  PDF layout and matching OCR asterisks. Do not transfer goods-area contacts
  into consignee labels or copy a merely equal country from unrelated Marks.
- [x] **L05 — six generated oil descriptions needed internally consistent
  product-state wording.** A fresh review exposed one crude/refined ambiguity;
  a complete 5,600-row sweep found nine co-occurrences. Three were valid distinct
  variants, a negative qualifier or derivation from crude feedstock. Six required
  bounded wording corrections. Their descriptions and printed product passages
  were amended together, preserving HS codes, accounting, equipment, dates,
  identities and numeric coordinates. Instructions now explicitly distinguish
  separate variants from competing states assigned to one product. These are
  sample-only lexical corrections plus general generation/review guidance, not
  six hardcoded source rules or a global ban on the words “crude” and “refined.”

## Historical repair inventory

This inventory follows the fixes that feed the active registry and dataset,
not obsolete proposed rules from the earlier GROUND 15 repair experiments.
The [evidence catalog](analysis/synthesis-repair-lineage-reaudit-20261012/historical-evidence-catalog.json)
indexes **123 Markdown records and 165 structured repair/proposal records**.
It includes reports of delegated work; those reports are not independent proof
unless the resulting source declarations, code path and accepted data agree.
Repeated reports of one correction are grouped below rather than counted as
separate defects. The earlier immutable sample authorities and provider outputs
are historical evidence, not files to overwrite to make old hashes pass.

`Prevention` means the specific repaired deterministic path is closed.
`Detection` means a bad model proposal is still possible but cannot be approved
without resolving a reported violation. Neither term means a language model
can never make a new semantic mistake.

| ID / repair | Sample/source/template layers | Production prevention or detection | Disposition in this re-audit |
| --- | --- | --- | --- |
| H01 — reduced target projection | Real and synthetic labels remove carrier, Marks, forward/export references and vessel flag; source targets use the same reduced task. OCR intentionally retains excluded fields. | Task schema, projection and assembly validate the reduced target. | No new defect: excluded flavor text is not missing supervision. |
| H02 — uppercase text | Real target repair and synthetic target normalization; identifiers/emails/URLs retain their own case. | Shared final target normalization, configurable synthesis casing, separate OCR casing presentation. | Inspect all published targets; do not “repair” harmless OCR title case. |
| H03 — complete postal address rather than stripped city/country | Source target/contract rebases retain complete postal wording and separately printed country. A separate city target was removed. | Source-owned expressions and postal component/country declarations; no literal suffix stripping. | Existing path retained; L04 closes two newly confirmed continuation omissions. |
| H04 — names versus postal/tax/contact fields | The field-boundary campaign repaired 151 synthetic and two real records in 41 families; source identity/lexical declarations were corrected where the source owner was wrong. | Pydantic field descriptions, explicit identity-kind certification, lexical ownership and complete-render review. Genuine named facilities and uncaptioned proprietors remain allowed. | Detection, not a global “delete company-like words” regex. Current generated values are reviewed afresh. |
| H05 — primary A in `A O/B B`, `A VIA B` | Later user policy supersedes the old composite-name repair: retain primary A, B in audit metadata, and omit an address whose A/B ownership is unresolved. Source `ca74` required one real plus four retained-synthetic copied corrections and new-family reconciliation. | Primary-entity source decisions and exact owner mappings; no inferred forwarder/NVOCC from O/B alone. | Retain legitimate missing address targets in composite-party cases; a reviewer cannot assign B's address to A merely because it is printed nearby. |
| H06 — consignee contacts in its own block | Source `23a49094` and six retained descendants had a goods-area-only phone removed; two new descendants correctly keep its anonymized OCR occurrence targetless. | Corrected source dataset/contract/auxiliary triple existed, but original matched authorities could still pass. | **Open root gap L01**, independently reproduced; corrected current samples alone were insufficient. |
| H07 — explicit country/phone continuation ownership | Source-certified postal owners, separate contacts and marker fragments were used instead of nearest-heading heuristics. | Country bindings and whole-address assembly. | L04 repairs postal omissions; the direct-block contact policy does not prohibit a clearly marked postal continuation. |
| H08 — repeated postal country | `84b` postal owner now includes the separately repeated suffix; one new descendant loses only the duplicated trailing country. | Exact source-owner rebase; one coherent replacement across owned repetitions. | Upstream deterministic fix, not a final whole-document country deletion. |
| H09 — DD-MM numeric dates | The mistaken “omit ambiguous date” guideline was superseded. Repair covered 31 real, four validation and 105 synthetic records; all relevant date fields were searched. | Date field descriptions, date policy, source date recipes and sampled ISO-to-source-format rendering. | Preserve actual role ownership; apply DD-MM only when both orders remain possible, not based on country/language. |
| H10 — negotiability and notify reference | Current real/synthetic labels and template eligibility were back-checked. Actual consignee order instructions determine negotiability; notify `sameAs` needs printed reference wording. | Field descriptions, source capabilities and configurable negotiable-template sampling. | Conditional preprinted “if to order”/“consignee or order” headings are not actual order instructions. Missing consignee evidence can remain null. |
| H11 — unsupported delivery roles | Equal locality survived under discharge/payment while delivery occurrence was absent. Five real, one validation and 21 synthetic labels were removed; route capabilities/owners were reconciled. | Role-specific source evidence and route-presence contract, not a text-equality permission. | No PDF-only values may enter targets. |
| H12 — description excludes Marks | Real/validation main-description decisions and template contracts were repaired; synthetic descriptions were replayed. | Source-certified description blocks, described extraction fields and complete-render review. | Do not restore detached Marks product specifications just because they describe goods. |
| H13 — complete main product passage | Embedded quantities, packing capacities and qualifiers were restored where previous policy silently stripped them. `1012322163DXH DEGREE 1` was retained by the user's final decision. | Main-block/true-continuation policy and assembly, no arbitrary numeric cleanup. | Distinct from standalone loading declarations and shipment-total fields. |
| H14 — loading/STC headings and detached tracking | Source decisions removed leading `6 PALLETS STC`, `1 PALLET STC`, separate declared totals and cherry-picked detached blocks. Source 363's target is the unified product wording, without later tackling/packing fragments. | Certified starts/ends and description-block replay; standalone declarations stay separately host-owned. | Inspect actual target as well as OCR: an excluded heading can legitimately remain printed. |
| H15 — declared package accounting unit | Real/source labels and synthetic totals/allocations were corrected from automatic inner-package selection to the declared shipping unit. | Physical registry, declared-unit ownership and allocation validation. | Supporting pallets/inner bags can differ from the public declared unit. |
| H16 — sole-container STC allocations | A sole identified container's explicit count/declaration can establish its allocation, even without a table. | Source allocation evidence and relation validators. | Preserve these legitimate allocations; do not infer quantities for unidentified/multiple containers. |
| H17 — package-word totals and repeated counts | Three source families had written totals/repeated package occurrences repaired. Conditional legal-clause numbers were not all treated as mutable package counts. | Exact typed owners and rendering surfaces; deterministic package probes and source identity replay. | Prevention applies to owned declaration occurrences, not every package-like noun in the document. |
| H18 — independently typed gross/net/volume | Source `489` free product owner had swallowed a separate gross prefix. Eleven new slots and three approved retained copies were repaired. | Compiler rejects separately typed measurement owners or sum derivations enclosed in generated product spans. | Upstream guard exists and needs retained regression tests. Embedded per-product capacities remain allowed. |
| H19 — explicit net total generated in a description | One generated sorghum sample declared a net total but lacked the target. The one sample's label was corrected. | Generation scope and review check whole-shipment totals; no broad arithmetic/HS learning objective introduced. | One-off correction was justified. Other NET/NW sources were individually adjudicated with unit evidence. |
| H20 — shared-unit NET/NW decisions | Three real sources and six descendants were assessed. `635b24d9`'s separate unitless NET line stays absent because only a distinct container gross row has KGS; other shared-unit contexts were resolved accordingly. | Source unit decisions/private numeric recipe, not “every nearby KG supplies every mass unit.” | Fresh review flags on the known unitless line do not supersede this explicit adjudication. |
| H21 — incidental bag/cone fill arithmetic | Added 50-kg bag fill and 1.5-kg cone capacities were examined; exact extraction still agrees. No target asks the model to enforce universal payload or unit-fill multiplication. | Current validator permits incidental per-unit detail, while explicit shipment totals remain checked. | **Open expectation gap L02**: review/generation prose still over-restricts these non-objective details. |
| H22 — source `819` declared carton framing | Its 24 new descendants were reconciled; no authority contract was changed before the finite campaign publication. | The tested inline count/type/fill owner separation remained a proposal. | **Open root gap L03**; future variable product wording must not erase the only type evidence. |
| H23 — registry goods diversity | Sampling moved from a small source/donor subset to common HS, dangerous-goods and thermal registries with package compatibility. | Shipment sampling and physical-form capability checks; model writes wording around host facts. | Source structure constrains where facts can print, not which supported commodity identity is sampled. |
| H24 — classification context versus physical form | Earlier concatenated briefs could imply conflicting whole/dressed, crude/refined or packaging forms. | Separate classification reference and authoritative shipment-form facts in generation requests. | Exact tariff boundaries are not targets; genuine mutually contradictory physical-state claims remain actionable. |
| H25 — semantic product-wording repairs | Pilot/campaign corrections removed crude/refined contradictions, FLEXITANK versus sampled IBC claims, UNPACKAGED versus declared drums, stale Canada/equipment wording, and chicken shell-on terminology. | General shipment/physical-form instructions plus scoped full-text review; corrected lexical receipts retain before/after provenance. | Mostly **sample-only + detection**. No source-ID Python rule is warranted for a one-off lexical mistake; current source-fact wording is not blindly copied. |
| H26 — canonical equipment and aliases | Original typeDescription cases, NOR/reefer, open-top, 20HO, 40EC source OCR, 8x6/9x6 and missing family/length were adjudicated. Labels and registries were updated, including independent type-without-size where length truly absent. | Shared semantics and alias registry; source presentation style plus configurable alias coverage. | ISO `45G1`/`45R0` are 40-foot high-cube codes, not literal 45-foot lengths. Native review false positives must be rejected. |
| H27 — thermal category and carrying settings | Missing refrigerated category was repaired on source and associated template; carrying temperature/ventilation evidence was distinguished from product-processing temperature. | Thermal compatibility registry and typed rendering; setpoint units and source ownership retained. | Fresh checks include all 300 newly sampled thermal shipments. |
| H28 — optional mixed equipment | 85 compatible templates were mapped to per-ID equipment owners and tested with mixed/uniform descendants; aggregate-only mixed sources do not get invented per-ID assignments. | Configurable mixed/uniform sampling, inventory binding and alias realization, with uniform controls. | Current assigned and aggregate-only mixed counts are separately reported. |
| H29 — per-container tare mapping | Source `e7dd` had two repeated tare occurrences assigned to wrong indices. All six affected new renders were checked; old uniform descendants had no demonstrated tare defect. | Correct exact source bindings and stage-level source-readiness validation. | Fix was upstream; do not rewrite unaffected old samples for symmetry. |
| H30 — height-only equipment predicate | A `HIGH CUBE`/height-only span was incorrectly replaced with a whole size/type alias. Sixteen new renderings in four families changed, with facts/targets/numbers unchanged. | Height projection returns only the predicate; complete alias owners stay independent. | Source-aware deterministic prevention and integrated regression probes. |
| H31 — zero-padded equipment counts | The independent audit needed to recognize a valid `02X` count. A production grammar expansion changed alias ownership in unrelated sources and was rolled back after 334 replay comparisons. | Independent count decoder accepts padding; production owner admission remains unchanged. | This is a tested rollback, not an unresolved label issue or license to broaden rendering regexes. |
| H32 — realistic email/web wording | Deterministic hash/example domains were replaced with native generated business/free-mail contacts; malformed/duplicated/source-copied values were repaired. | Contact schema and syntax/source-copy/placeholder validation, preserved source scheme presentation. | **No shared-domain requirement**. Company website/email can differ; existing good contacts need no regeneration. |
| H33 — phone-prefix source ownership | Nine inherited `9ad49fcc` descendants and two real OCR originals were previously recorded with doubled signs; they were not silently rewritten in that admission pass. New-source plans received complete-phone ownership, while the old family's residual signs remain documented. | Exact prefix/number ownership where rebased; no global punctuation stripping. | Residual `++` is OCR/presentation noise, not a missing or fabricated labeled phone: current target numbers retain the grounded canonical number. Do not claim all historical phone typography was repaired. |
| H34 — secondary identity/identifier casting | EXID co-reference (source `524`), exporter/importer tax IDs and declared VAT aliases (`643668997`/`643668977`) were reconciled at owner/auxiliary recipe level; unrelated actors remain independent. | Exact alias/canonical-owner declarations and source-private recipe rendering; no fuzzy VAT spelling match. | Excluded static carrier, signature, third-party and Marks identities may remain by policy. |
| H35 — external-reference footer | Stale footer references were replaced; user explicitly allowed keeping the prefix/format. | Shared auxiliary footer owner/template correction. | Cosmetic prefix meaning was not promoted into a target taxonomy or extra required complexity. |
| H36 — source-position alignment | Repeated lines/words and table rows use occurrence-aware GLM-to-Paddle anchors; unavailable source anchors remain explicit. | Measured source alignments and hash-pinned registry assets. | No assignment to the first equal token when repeated owners have different positions. |
| H37 — synthetic geometry | Initial anchor copying/interpolation was replaced with coherent page transforms and elastic local reflow using measured font/space constraints. Generated descriptions gained informative geometry. | Bounds/order/collision/region/transform and preservation checks; explicit unknown ` ||` when infeasible. | Geometry is training signal, not a claim to exact PDF re-typesetting. |
| H38 — rejected reflows/source-transfer receipts | Four rejected reflows were traced/replayed; local geometry fixes preserve raw/target text and do not fabricate missing boxes. | Position replay and exact text reconstruction; source-coordinate receipts. | Re-audit all current positions and quantify source versus generated gaps. |
| H39 — immutable manual correction provenance | Lexical/source-context recoveries explicitly adopted reviewed strings instead of editing the saved provider response. Stale native review hashes were reissued for 85 source units, preserving 184 adjudications. | Candidate/request/source/receipt hash binding; manual repairs have before/after provenance and invalidate stale approval. | A paid response is never relabelled as if it answered a different prompt. |
| H40 — registry pins | Two manifest pins were stale after valid source-owner corrections (`489`, `ca74`); registry metadata was reconciled without data changes. | All case/shared/override files pinned and compared; dependency drift blocks resumption. | Verify again after this task's new source revisions. |
| H41 — provider interruptions and budget | Saved successful calls were resumed exactly; rejected/rate-limited/invalid JSON attempts remain in accounting and are not reviews. | Bounded concurrency, explicit retry and all-attempt cost ledger. | Operational failures cannot become silent “pass”; new audit requires exact 2,500 unique reviewed IDs. |
| H42 — byte-preserving mixture publication | R18 replaces only the new UAE cohort and explicitly approved retained corrections; 60 validation rows stay protected. | Atomic publication, required exact counts, duplicate/leakage checks and strict copied-row approvals. | If L04 changes targets, publish a new revision and update the training config instead of silently invalidating R18. |

### Evidence map for grouped historical fixes

The catalog contains exact artifact paths/hashes. Principal visible authorities:

- [Policy decisions](kie-real-baseline-label-policy-2026-10-05.md).
- [Field-boundary repair](kie-synthesis-field-boundary-repair-2026-10-10.md)
  and [probes](kie-synthesis-field-boundary-probes-2026-10-10.md): H03–H08,
  H32–H34. Later primary-entity decisions supersede earlier composite names.
- [Party/date/route correction](kie-party-date-route-repair-2026-10-10.md):
  H05, H09–H11. Earlier random-audit date-absence findings are historical,
  not today's policy.
- [Consistency audit](kie-synthesis-consistency-audit-2026-10-10.md): H18–H21,
  including explicit unit-context adjudications.
- [Goods layout pilot](kie-synthesis-goods-layout-pilot-2026-10-07.md),
  [registry integration](kie-synthesis-registry-integration-2026-10-07.md),
  [follow-up integrity](kie-synthesis-followup-integrity-2026-10-08.md):
  H12–H17, H23–H25, H34–H35.
- [Mixed-equipment integration](kie-mixed-equipment-production-integration-2026-10-12.md)
  and [mixed pilot](kie-mixed-equipment-pilot-2026-10-11.md): H26–H30,
  legacy actor-equality corrections and configurable uniform controls.
- [Position reflow investigation](kie-synthesis-position-reflow-investigation-2026-10-07.md):
  H36–H38.
- [Replacement campaign](kie-synthesis-uae2500-replacement-2026-10-12.md)
  and its `proposals/`, lexical reconciliation and final audit reports:
  H18, H22, H25, H29–H31, H39–H42.

The following sections record independent checks against the executable paths
and final dataset, not merely the conclusions of historical reports.

## Fresh sample audit

**Complete: all 2,500 replacement samples were reviewed, all findings were
adjudicated, all confirmed fixes were applied, and the successor was validated.
There are zero unresolved sample, source-rebase or geometry holds in this task.**

### Independent full-text review, not another reading of prior approval flags

The new native review used GLM-5.3-Flash, low reasoning, native structured outputs
and concurrency eight. It received the **entire plain rendered OCR**, complete
target and source-certified/sampled context for every sample. It used a separate
audit instruction contract aligned with current policy. Saved successes were
reused only for an identical request; an invalid JSON attempt remained billed
and unapproved until its recorded retry succeeded. OCR was not sent as a JSON
input wrapper. No Codex sub-agents were used.

The [complete review receipt](analysis/synthesis-repair-lineage-reaudit-20261012/fresh-semantic-review.json)
covers **2,500 unique IDs exactly once**. It raised **133 claims in 118 samples**.
Root independently reconciled all 133 claims against actual targets, source
contracts, policy and, where layout was material, the actual PDF. **15 claims
were actionable**: 14 postal-country omissions and one product-state ambiguity.
**118 claims were rejected**, not applied speculatively. The family sweep caught
two additional country omissions beyond those flagged, and the product-state
sweep caught five additional wording issues. Thus model review alone was not
treated as a complete checker.

The [individual adjudication inventory](analysis/synthesis-repair-lineage-reaudit-20261012/adjudication.json)
preserves every original claim, candidate hash, review path, root decision and
reason. Nonliteral or paraphrased model quotations are explicitly identified;
they are not automatically accepted as evidence.

Important rejected claims demonstrate why automatic “repair every reviewer
finding” would damage good data:

- ISO `45G1`/`45R0` are **40-foot high-cube**, not literal 45-foot lengths.
  `25G1` is 20HC, `42G1` is 40STD, `55G1`/`L5G1` are 45HC; documented `40RQ`
  and `40RH` are high-cube reefers. `9x6` supplies height evidence. Multiple
  claims also stated a target category the actual container-by-ID target did
  not contain. Independent equipment replay rejected those claims.
- The 13-row `f36409a0` sum is 71,362.3 kg after target rounding, not the
  reviewer's sum with an extra row. `4799921c`'s five weights/allocations also
  reconcile; its reviewer duplicated a weight.
- `635b24d9`'s unitless NET field does not inherit KGS from a different gross
  row. `9ad49fcc`'s unitless GW likewise does not authorize a kilograms target.
- An O/B company's source-only address is not the primary A party's address.
  A private instruction to generate B does not mean B replaces the A target.
- The actual `e70b7185` PDF places its product summary **inside description**,
  not Marks. The source-certified summary and detailed passage remain valid.
  The actual `3d0d3748` PDF/coordinates establish receipt/payment ownership
  despite flattened OCR header order. Those targets were not swapped.
- `SAID TO CONTAIN` remained in OCR, but was already absent from the actual
  `97efe837` description target. `PACKAGE_PACKAGE` is the canonical enum.
  `GMA CGM BARRACUDA` is the OCR spelling; a carrier name does not authorize
  silently changing it to `CMA`. Harmless casing received no repair.
- Per-bag fill multiplication and payload arithmetic are not new extraction
  criteria. Two fresh carton-pilot flags were rejected under that policy; the
  printed totals and extraction labels remain consistent with one another.

### Exact corrections to the training data

| Correction | Real train | Retained non-UAE synthetic | New UAE synthetic | Input/coordinate impact |
| --- | ---: | ---: | ---: | --- |
| `0d495914`: complete consignee address with its explicit `CN>` country | 1 | 0 | 4 | Targets only; all input bytes unchanged |
| `23e145de`: complete consignee address with matched-star country footnote | 1 | 6 | 12 | Targets only; notify phone ownership unchanged |
| Oil product-state wording | 0 | 0 | 6 | One exact product-owner edit per sample, in OCR and description together; numeric positions unchanged |
| **Total** | **2** | **6** | **22** | **30 distinct training records corrected** |

Examples:

```text
Consignee postal text ... CAIRO, CN>
...
CN>
EGYPT
TEL: ...
```

The address now includes `EGYPT` and its country target is `EGYPT`. The
goods-area phone is **not** transferred into the consignee target. In the new
descendants the owned country is `UNITED ARAB EMIRATES` instead. For the starred
source, a matching `*` after the address and `* EGYPT TEL: ...` establishes the
postal continuation; it does not change its independently owned notify phone.
Six retained descendants use their actual printed countries, including
Lithuania, United States, Macao, Svalbard and Jan Mayen, Djibouti and Israel,
not a blanket Egypt/UAE replacement.

The oil correction distinguishes two valid sampled variants explicitly:

```text
Before: REFINED SOYBEAN OIL, NON-DEGUMMED EDIBLE GRADE LIQUID —
        CRUDE SOYA-BEAN OIL AND FRACTIONS
After:  SOYBEAN OILS AND FRACTIONS: REFINED EDIBLE GRADE AND
        CRUDE NON-DEGUMMED GRADE, LIQUID
```

The other five bounded edits remove a conflicting adjective/specification,
without changing any weights, packages, equipment or classifications. All
before/after wording and target hashes are in
[the publication receipt](analysis/synthesis-repair-lineage-reaudit-20261012/dataset-revision.json).
This is not a new HS-to-product supervision objective.

### Upstream repairs and their observed behavior

**L01:** `curated_ownership.py` now enforces source-owned
`required_target_absences`. It validates the path against the current reduced
schema, rejects malformed paths/indices/reasons, checks exact subtree boundaries,
and rejects populated excluded facts. It never deletes OCR or substitutes a
default. The actual production Campaign with the original matched source and
old contract now fails **before paid generation** with:

```text
source target contradicts required absence:
documentPatch.parties.consignee.contactDetails.phoneNumbers
```

[The actual-path probe](analysis/synthesis-repair-lineage-reaudit-20261012/actual-absence-guard-validation.json)
records zero API calls/cost. Current valid source content remains supported.
The source contract, source target, auxiliary recipe and registry authority
metadata were promoted together, rather than leave a preferred-path note as
the sole safety mechanism.

**L02/L05:** concise generation/review prose and product field requirements now
distinguish incidental per-unit capacity from declared shipment accounting,
and separate product variants from incompatible states of one variant. Existing
numeric/accounting/context-caption guards remain active. Positive per-unit
examples pass; added explicit net/count totals still reject. Historical wording
caches intentionally fail under the new prompt hash: tests replay old artifacts
against their explicitly saved old provider prompt, not a production bypass.

**L03:** source `819af5e4`'s product variable now covers only
`IQF WILD BLUEBERRIES FROZEN`. Its existing count owner covers both inline
`1800` and footer `1.800`; carton type and the source's packing qualifier stay
outside free generation. Description assembly retains the complete printed
passage. Two fresh descendants print the host's carton count/type even though
the requested product fragment no longer owns those facts. No new source-ID
branch was added to production code. The 24 already-sound accepted descendants
were not needlessly regenerated or added new qualifiers.

**L04:** source country bindings and address expressions were corrected in both
source authorities and registry cases. Fresh actual preflight initially caught
two **duplicate semantic owners**: a newly public country still had its former
private auxiliary recipe. Those two obsolete recipes were removed, not silently
overridden. Both sources then passed normal generation, render and review. This
is why compilation alone was not called final validation.

The complete pre-change registry snapshot is preserved, as are R18 and every
original provider response. Corrected future authorities are centrally under
`artifacts/synthesis-templates/mpci-bl-v7-reviewed/source-authorities/reaudit-20261012/`
(`real660` and `wide`). New campaigns must select the corrected authority and
current registry contracts/auxiliary together. Completed historical cohort
configs are history, not configurations to resume under a new instruction/source
version and pretend their old approval is current.

All **360 cases / 2,160 case files / four shared files** pass the refreshed
[registry-pin check](analysis/synthesis-repair-lineage-reaudit-20261012/registry-pin-validation.json).
The four revised cases also link their fresh two-descendant certification.

### Fresh synthesis and post-repair validation

[The ordinary-path pilot](analysis/synthesis-repair-lineage-reaudit-20261012/pilot-certification.json)
generated **10 new inspection samples from five templates**, including all four
source rebases plus an oil-form control. It used ordinary native wording,
contacts, render, review, publication and positional synthesis. Root adjudicated
the two non-objective fill-weight review claims. All 10 pass publication and
full geometry checks: **18 pages, 1,068 inside-page boxes, 15 generated blocks**.
The existing bounded postal correction operation handled three rejected fresh
postal proposals; no rejected wording was silently accepted. Inspection samples
were **not added to training**.

All **30 corrected dataset records** received a separate, full-OCR, scoped
post-repair native review. One suggested including a separately owned H.S CODE
suffix in description; root rejected it under the existing description-boundary
policy. The classification remains printed and separately labeled. No unresolved
post-repair claim remains.

The [fresh complete replay](analysis/synthesis-repair-lineage-reaudit-20261012/complete-replay-geometry.json)
replayed every original candidate from its immutable source/scenario/wording
authority, independently checked its equipment and geometry, then verified the
approved successor edits:

- **2,500 samples**, all 45 cohorts; **4,900 pages**.
- **5,663 equipment rows** independently checked, with no unresolved per-ID or
  printed-alias decisions.
- **207,458 inside-page boxes**, **3,028 generated blocks**, **7,921 exact
  generated centroids** checked. Four declined reflows are verified retained
  source transfer, not applied rejected layouts.
- **7,500 deliberate corruptions rejected**: changed positioned target, extra
  invented OCR line and foreign source-line ownership, once each per sample.
- All numeric coordinate values are unchanged by this repair. Text preservation
  and exact bounded edit checks also apply to the six modified product lines.
- Read-only audit took **323.02 s**, peak **955.23 MiB**. No API calls or candidate
  writes. Original coordinate coverage remains **84.03%**; labels alone do not
  invent missing anchors.

### Tests, performance and training handoff

The integrated production/source regression run passed **1,246 tests** in
**51.33 s**, with one host Torch-dependent skip. The added bounded dataset-repair
and absence-guard suite passed **29 tests** in **9.66 s**; the final rank/config
and historical-replay checks also pass. These overlap and are not additive
unique-test counts. Targeted Ruff checks and `git diff --check` pass.

The source-absence gate adds no hot-path filesystem I/O. A seven-repeat,
10,000-call benchmark measured **0.097 µs** with no declaration and **11.889 µs**
for the one reviewed exclusion over 41 source leaves. It runs during source
compilation/readiness, not once per generated line or GPU forward pass; its
cost is negligible relative to source compilation and provider requests.

R19 is published at:

`data/curated/mpci-bl-real600-synthetic5000-uae-coverage-r19-positions-v1`

It contains **600 real + 5,000 synthetic training records, and the same 60
validation records**. Relative to immutable R18, **5,570 train rows remain exact
copies**, all 60 validation rows are byte-identical, and all non-address/country
and non-adjudicated-description targets stay unchanged. In the new 2,500 cohort,
**2,478 complete rows are byte-identical** and 22 have the recorded bounded fixes.
The inherited generation audit remains historical; `postpublicationRepair` binds
the new review, root decisions, replay and exact successor changes. It is not a
claim that the old approval alone covers new labels.

The new [R19 training config](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic5000_uae_coverage_r19_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml)
changes only dataset/run provenance, not hyperparameters: rank 48, alpha 32,
EVA/RSLoRA, 10 epochs, microbatch 1 / accumulation 32, input-only prompt, compact
targets, 88 warmup updates and eval/save every 438 updates, all checkpoints kept.
Train SHA-256:
`001fa2a63ac426109993547524d0cc5b5ca7e0408b7a8bdbc8a681486fdb5d49`.
Validation SHA-256:
`2948e9001d2429c63e9e918597d9dd10e7ebec3b26d9f4c1758a1dfe168318be`.

The rebuilt **actual Compose image** passes `validate-config`, `inspect-dataset`
and tokenizer-only `prepare-dataset`: **5,600/60 retained, zero exclusions or
truncations**, maximum train input/target lengths **8,066/3,160**. Dataset inspection
took **3.61 s** (1,568.84 records/s); uncached tokenization took **45.16 s**.
No model weights or training were launched.

Additional provider spend for this task is **$1.20955714**: $1.19087674 for the
whole-set review, $0.00993715 for post-repair review, and $0.00874325 for fresh
generation/review/postal correction. Costs include recorded retries; reused
responses are not charged twice. This is approximately **$0.000484 per newly
audited sample**, including the bounded follow-up work. The predecessor campaign
plus this additional audit is **$3.77151501 confirmed**, with the previously
documented $0.018146 interrupted-request uncertainty separate, not a new charge.

**Final disposition:** L01–L05 are corrected and validated. All 2,500 new samples
are approved for inclusion under the stated extraction policy, after the 22
bounded sample corrections. No known unresolved training-data defect or review
hold remains from this inventory. This is a completed pre-training acceptance,
not a promise that any future unseen model-generated text is automatically good.
Future synthesis still passes its ordinary source, render, semantic review,
adjudication and geometry gates.

[The final machine-readable certificate](analysis/synthesis-repair-lineage-reaudit-20261012/completion-certificate.json)
also verifies all **5,600 current training targets**, their uppercase policy,
every committed file hash, exact repair replay against R18 and train/validation
disjointness. It records **zero unresolved issues**, exact all-attempt costs,
5,570 unchanged train rows and byte-identical validation. This final check took
**6.73 s**, peak **147.15 MiB**, without generation or training.

```bash
mpci_docker_config="$(mktemp -d /tmp/mpci-docker-public.XXXXXX)"
DOCKER_CONFIG="$mpci_docker_config" docker --config "$mpci_docker_config" compose \
  --profile training run --rm --build kie-trainer train \
  --config configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic5000_uae_coverage_r19_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml \
  --project-root /workspace
```
