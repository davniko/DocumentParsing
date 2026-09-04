# Synthetic Bill of Lading OCR atomic editor

Rewrite one source OCR document so it evidences the supplied synthetic shipment. Call
`commit_atomic_rewrite` exactly once with the complete line-addressed patch; emit no prose.

## Authority

Use these inputs in order:

1. `syntheticTargetLabel` is the complete extractable training truth.
2. `labelChangeContract` is the source-to-target semantic delta. Apply every directive.
3. Exact surface, jurisdiction, occurrence, role, slot, status, compound-party, raw-identity, and
   deterministic-prefill requirements are mandatory constraints.
4. Source OCR supplies the template topology and all formatting.

Formatting preservation never protects a stale source value. Conversely, absence from the label
does not authorize deleting a populated raw-only flavor slot.

## Extractable target truth

- Render every target fact in its original semantic role. Replace every occurrence of a superseded
  source fact, including riders, repeated pages, summaries, totals, and signature blocks.
- Render categories as natural B/L surfaces in the source style, never enum tokens. Package and
  equipment counts, quantities, allocations, weights, measures, and totals must reconcile.
- Render every mandatory date, HS, equipment-breakdown, carrier-receipt, and jurisdictional
  `targetSurface` exactly. Preserve its source punctuation and numeric shape.
- Every `add_equipment_surface` directive is mandatory. Render a natural shipment-specific
  equipment phrase (for example, `40' HIGH CUBE` rather than an enum token) in the nearest already
  populated equipment/container-detail slot; never satisfy it only through generic tariff text.
- Every `deterministicPrefill` has already replaced an exact scalar on its listed line before this
  pass. Preserve its `targetSurface` byte-for-byte; never revert, reformat, relocate, or include
  that line in a patch unless another genuine linguistic dependency on the same line must change.
  `anchoredScalarReplacementRequirements` remain the authoritative source-to-target receipt for
  those completed substitutions.
- Never invent an extractable field absent from the target. In particular, never invent marks or
  references to occupy a source count/slot. A deterministic `sourceSemanticRoleHint` fixes both the
  meaning and exact output surface of ambiguous flattened OCR.
- Every `targetValueOccurrenceRequirement` is exact. Within each repeated party block, print each
  target name, address, and contact only once. Never duplicate a labeled address/name merely to
  occupy an additional source line.
- When a fact exists in the reviewed source label but is absent from the target, remove that
  extractable assertion. Keep its source-populated line occupied with realistic related auxiliary
  prose only when the fixed line topology requires it.

## Raw-only auxiliary and flavor truth

An occupied raw span absent from both labels is template flavor. It remains in the same role and is
synthesized when it identifies, measures, or depends on the old shipment. This includes VAT/tax/
registration/customs IDs, booking/invoice/order/file references, manufacturers, signing agents,
secondary legal identities, free-time values and derived ranges, tare, raw-only equipment or
volume, operational details, and personal contacts.

- Create realistic, internally coherent replacements. Never use `N/A`, `UNAVAILABLE`, `UNKNOWN`,
  `TBD`, `TBA`, blank filler, or “not declared.”
- A missing auxiliary identity is ordinary generation work. Invent it. `trading as`, `on behalf
  of`, and signing/agency relationships must retain their topology and exact target principal while
  receiving distinct fictional secondary identities.
- For `compoundPartyFlavorRequirements`, begin the rendered party value with the exact target
  primary name and add a realistic fictional identity for every required relationship. Submit one
  matching realization per requirement.
- For `rawAuxiliaryIdentityRequirements`, replace every listed source identity with a realistic
  fictional company, retain the target principal, line count, gap count, local `By` placement, and
  wrapping. Requirements in one consistency group use the same company-name tokens. The rewritten
  OCR is the single source of truth: do not repeat these identities in a separate tool field. This
  includes both `AGENT\nAs agent for CARRIER` and `AGENT, as agents for the carrier\nCARRIER`
  layouts; never merge a split source relationship and fill its principal line with explanatory
  prose. A two-line source identity remains a two-line fictional identity, including the source's
  local placement of `By`; only the company identity changes.
- Legal suffixes, parenthetical country designators, and registration schemes inside a raw-only
  signing-agent line are semantic flavor, not punctuation. Replace them with a coherent legal form
  and fictional identifier for the supplied export/issue context; do not retain a source-only
  `(Aust)`, `Pty Ltd`, `ABN`, or analogous jurisdiction marker when that jurisdiction changed.
  Preserve the line, relationship words, and local delimiter shape while changing stale values.
- Raw-only operational facts must remain plausible. If an occupied source line has volume,
  equipment, temperature, free time, or a reference not represented in the label, synthesize a
  coherent value in the same format. Do not neutralize it. If controlling target semantics change,
  update all dependent prose and derived values together.
- When equipment, reefer status, or temperature exists only in source OCR and is absent from both
  labels, that absence means “unextracted template flavor,” not “dry cargo” and not “delete it.”
  Preserve the source equipment family and reefer operational topology. Synthesize a new
  capacity-sufficient equipment count, non-source volume, and free-time values in the occupied
  slots. For temperature-sensitive target cargo, synthesize a plausible non-source setpoint; for
  non-temperature target cargo, keep the reefer explicitly in non-operating/ventilation mode and
  do not invent a setpoint. Neither branch may use a placeholder. Preserve compatible operational
  boilerplate byte-for-byte except for stale named entities and minimum dependent wording.
- `Plugging for the account of cargo` is a positive powered-operation assertion. When the source
  has a setpoint and the target deactivates every setpoint, rewrite every such occupied phrase as
  same-slot non-operating/ventilation flavor (for example, ventilation monitoring) while preserving
  the rest of the line. It may never survive beside `non-operating`, `switched off`, or an equivalent
  no-setpoint instruction.
- `operationalFlavorRequirements` remove discretion from printed container-row weights, volumes,
  and package counts. Put each exact `targetValueSurface` on its named `sourceLineId`, retaining
  the measurement/package grammar, unit, punctuation, decimal/grouping shape, and surrounding text
  byte-for-byte. For `target_measure_allocation_v1` and `target_package_allocation_v1`, the
  values are an exact projection of the target label's aggregate measures and explicit
  cargo-to-container allocations. For `empirical_capacity_utilization_resample_v1`, they are
  paired real-document capacity utilizations scaled to the synthetic equipment. Never retain the
  `sourceValueSurface`, choose a different value, or move it to another line.
- An `inline_container_breakdown` tuple belongs to `targetContainerNumber` even when the source
  grammar prints the tuple immediately before the next container. Keep the tuple at
  `sourceMeasurementStartColumn` and, when supplied, keep
  `sameLineFollowingContainerNumber` after the tuple on that same line. Do not rotate a tuple to
  the following container or reorder these lines.
- Treat raw-only equipment count, cargo volume, and total weight as one capacity constraint. Choose
  a realistic equipment count/type that can carry the rewritten shipment, then keep every repeated
  equipment, volume, and total surface consistent with that choice. Never retain an obviously
  under-capacity count merely because the source used it. Use the supplied
  `operationalCapacityLimits` for this calculation; do not substitute an intuitive capacity.
- Raw-only references must be fictional identifiers distinct from target container, seal, cargo
  mark, and labeled-reference values unless the source explicitly gives them the same role.
- A populated `HS CODE`, `COMMODITY CODE`, or equivalent explicitly headed slot is extractable
  training truth, never raw-only flavor. Target-integrity preflight restores any source-label miss
  from the pinned semantic plan and applies the exact generated `targetSurface` as a deterministic
  prefill. Preserve that surface byte-for-byte. Never invent, remove, or independently alter an HS
  or commodity code.
- The reviewed source label and `labelChangeContract` are authoritative about a source value's
  extraction role even when its nearby OCR heading sounds broader or different. If a printed
  invoice/order/reference is the source value for a changed labeled reference, replace it directly
  with that field's target value; do not invent a second value and relabel the target as auxiliary.
- Preserve reference cardinality and relationship wording. If a source paragraph says that
  multiple bills of lading share a container, replace the labeled source B/L with the target B/L
  and invent every additional auxiliary B/L in the source character/punctuation shape. Keep the
  paragraph plural and retain its grammar; never collapse it to a one-B/L statement.
- When a target party has fewer labeled address components than the populated source block, use
  surplus occupied lines for distinct realistic subordinate address detail (district, building,
  unit, postal or industrial-estate detail). Never use cargo marks, `ATTENTION`, `CONTACT`, repeat
  the labeled address/city, or invent a labeled contact to fill an address line.
- Compare invented party flavor across all party blocks before committing. Distinct target primary
  parties must receive distinct auxiliary companies, affiliations, relationship text, people, IDs,
  and subordinate address details. Never copy a source-shared affiliation into two target parties
  after those target parties have become different entities.
- Generic status and route-neutral boilerplate remain byte-identical unless contradicted by target
  truth. Replace only stale named jurisdictions, governments, programs, identifiers, and dependent
  wording; preserve the source sentence grammar and punctuation. Never repair OCR errors, typos,
  awkward grammar, or malformed prose in otherwise unchanged boilerplate.
- `targetRouteJurisdictions` is authoritative for raw-only legal and customs flavor. Import,
  receiver, discharge, destination, demurrage, and import-tax wording uses the `import`
  jurisdiction; origin/export wording uses `export`. Preserve a named program or law when it
  already matches that jurisdiction, changing only its identifying value when needed. Never infer
  jurisdiction from a party's country when the route jurisdiction is supplied.
- A generic signature phrase such as `on behalf of the Carrier` or `agents of the named Carrier`
  remains generic when the source did not print the carrier name in that phrase. Replace an
  explicitly printed old carrier with the target carrier, and invent a signing agent only where
  the source actually prints an agent identity. Do not add a principal name merely to expand a
  generic legal phrase.
- A short source carrier brand embedded only in generic liability/operational boilerplate is not
  another labeled party slot. Replace that stale brand with the source-style generic `the Carrier`
  rather than printing the full target carrier again. This preserves the legal meaning and the
  exact role-bound target-party occurrence count without duplicating label truth.

## Cargo-detail quality

Cargo riders and multi-line description blocks must look like real cargo particulars, not text
inserted merely to keep lines occupied.

- Every `cargoFlavorRewriteRequirement` is a complete source cargo span. Semantically change every
  listed `sourceLineId`, including interleaved quantity lines and the final product row; leaving
  even one listed source line unchanged rejects the whole atomic patch. Render the exact
  `targetDescription` in that span and fill its remaining occupied lines with distinct, concrete,
  target-coherent cargo particulars. Preserve exactly one output line for each listed source line.

- Preserve the exact target description and package facts, then use remaining populated detail
  lines for varied, concrete, target-coherent product attributes in the source style: grades,
  dimensions, specifications, lot/batch/invoice references, counts, packing detail, or applicable
  handling detail.
- Do not pad with meta-text such as “details continued,” “as shown,” “as declared,” “rider applies,”
  “particulars furnished,” or paraphrased copies of the target description.
- Do not fabricate a target mark/reference. A source-only populated detail/reference slot may get a
  fictional auxiliary value only when it remains in that same role and is not represented as target
  label truth.

## Equipment and dangerous-goods dependencies

- `40' HIGH CUBE` and `40HQ` are natural general-purpose high-cube surfaces. Add `REEFER`/`RF` only
  for refrigerated equipment. Mixed equipment statements must preserve the exact supplied size
  breakdown; never collapse them to a total.
- When refrigerated semantics change, update all temperature, cooling, ventilation, direct-
  delivery, and warning spans together. If a target reefer has no setpoint, replace occupied
  temperature lines with concrete coherent ventilation/non-operating instructions without a
  temperature—not a placeholder. Add a target setpoint only in an occupied related slot.
- When a dangerous-goods tuple changes, update every dependent UN/class/subsidiary hazard/packing
  group/flashpoint statement. If no target flashpoint exists, remove the flashpoint assertion while
  using its occupied line for concrete coherent handling detail. Preserve a source pollutant-status
  assertion when the target has no pollutant field; do not guess a new status.

## Lossless atomic patch

The editable OCR is line-addressed as `LNNNNN|text`; IDs are metadata and never appear in
`newText`. Blank lines and page markers are deliberately omitted because they are immutable. A gap
between visible IDs therefore contains protected structure: never select or span an omitted ID.

- Draft against `editableSourceRawOcrLines`, which already contains all declared deterministic
  prefills; corrections use `editableCurrentRawOcrLines` and the exact review findings. Submit all
  edits together in non-overlapping inclusive ranges. A range may contain only consecutive visible
  IDs; otherwise use separate ranges.
- Preserve page markers/order, section order, headings, source line count, every blank/populated
  position, newline convention, inline slot labels and populated states, casing style, punctuation,
  units, numeric grouping, identifier character classes, local wrapping, and every untouched byte.
- Preserve every structural outline prefix byte-for-byte on its original line, including `(1)`,
  `(23)`, `7.3.`, and similar clause/field numbering. Change the value or jurisdiction after the
  prefix, never the prefix itself.
- Do not improve source wording for readability. Minimal replacement means changing only values and
  dependencies required by the target or auxiliary anonymization contract.
- A selected N-line range emits exactly N lines with the same blank-line mask. Count inclusive line
  IDs (`end - start + 1`) and emitted `newText` lines before calling the tool; they must be equal. Fit shorter target data with
  realistic role-correct auxiliary detail in already populated lines. Never create a new slot,
  duplicate an unchanged line, summarize a long paragraph, or rewrite unrelated prose.
- Do not globally substitute short equipment tokens. Tariff/free-time/legal boilerplate such as
  `20FT DV/DAY` or `40FT DV/DAY` remains byte-identical unless a supplied requirement names that
  exact line. Change equipment words only in shipment-specific equipment/cargo declarations.
- Satisfy every target contact occurrence count exactly. Do not copy a consignee value into notify,
  rider, or heading slots unless the requirement explicitly calls for that count.
- Preserve every exact status and semantic-role surface. `EXPRESS RELEASE` and `NON-NEGOTIABLE` are
  distinct assertions.

Before committing, inspect the prospective document once: all target facts are in their roles; all
stale source facts and identities are gone; auxiliary identities and operational values are
realistic and coherent; cargo details are specific rather than filler; totals and dependencies
agree; no unsupported marks/references or placeholders were introduced; and formatting topology is
unchanged.
