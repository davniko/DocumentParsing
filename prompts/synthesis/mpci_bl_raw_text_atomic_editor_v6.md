# Synthetic Bill of Lading OCR atomic editor

Rewrite one source OCR document so it evidences the supplied synthetic shipment. Call
`commit_atomic_rewrite` exactly once with the complete line-addressed patch; emit no prose.

## Authority

Use these inputs in order:

1. `syntheticTargetLabel` is the complete extractable training truth.
2. `labelChangeContract` is the source-to-target semantic delta. Apply every directive.
3. Exact surface, jurisdiction, occurrence, role, slot, status, compound-party, and raw-identity
   requirements are mandatory deterministic constraints.
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
- Every `anchoredScalarReplacementRequirement` is literal find-and-replace authority: put its
  target scalar on every listed source line, in the same role and local grammar. Do not move a
  labeled reference into a nearby empty heading or invent a substitute value for its old slot.
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
  wrapping. Requirements in one consistency group use the same company-name tokens. Submit one
  matching realization per requirement. This includes both `AGENT\nAs agent for CARRIER` and
  `AGENT, as agents for the carrier\nCARRIER` layouts; never merge a split source relationship and
  fill its principal line with explanatory prose.
- Raw-only operational facts must remain plausible. If an occupied source line has volume,
  equipment, temperature, free time, or a reference not represented in the label, synthesize a
  coherent value in the same format. Do not neutralize it. If controlling target semantics change,
  update all dependent prose and derived values together.
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

## Cargo-detail quality

Cargo riders and multi-line description blocks must look like real cargo particulars, not text
inserted merely to keep lines occupied.

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

- Draft against `editableSourceRawOcrLines`; corrections use `editableCurrentRawOcrLines` and the
  exact review findings. Submit all edits together in non-overlapping inclusive ranges. A range may
  contain only consecutive visible IDs; otherwise use separate ranges.
- Preserve page markers/order, section order, headings, source line count, every blank/populated
  position, newline convention, inline slot labels and populated states, casing style, punctuation,
  units, numeric grouping, identifier character classes, local wrapping, and every untouched byte.
- Do not improve source wording for readability. Minimal replacement means changing only values and
  dependencies required by the target or auxiliary anonymization contract.
- A selected N-line range emits exactly N lines with the same blank-line mask. Count the selected
  source lines and emitted `newText` lines before calling the tool. Fit shorter target data with
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

