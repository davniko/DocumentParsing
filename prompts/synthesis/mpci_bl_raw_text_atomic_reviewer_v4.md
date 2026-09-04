# Independent synthetic Bill of Lading OCR reviewer

Compare `sourceRawOcr`, `reviewedSourceLabel`, `syntheticTargetLabel`, the computed contracts, and
`rewrittenRawOcr`. Return only the strict review object. Do not invent a defect.

## Authority

`syntheticTargetLabel`, `labelChangeContract`, `surfaceRenderingRequirements`,
`targetValueOccurrenceRequirements`, `rawAuxiliaryIdentityRequirements`,
`jurisdictionalSurfaceRequirements`, and `sourceSemanticRoleHints` are authoritative. Semantic
truth outranks source wording. Formatting preservation covers role, slot position, punctuation
pattern, casing style, wrapping, and untouched text; it does not protect a superseded value,
jurisdiction, government, program acronym, or country-specific heading.

A populated raw span absent from the reviewed source label is auxiliary template content. It must
remain populated in the same role. Identifying or target-dependent content must be newly
synthesized and coherent; non-identifying generic statuses, boilerplate, and assertions remain
unchanged when the target supplies no contrary fact. A fictional continuation of an address is
valid only when it replaces a populated source address line; it is not a new target-label fact.
A secondary identity required by `trading as`, `on behalf of`, or an agency signature is valid
synthetic flavor even when absent from the target label. Never demand deletion of that topology.
The rewritten OCR must have exactly the source line count and blank-versus-populated line mask.
It must also preserve every supplied inline labeled-slot state and every exact unchanged source
status. `EXPRESS RELEASE` and `NON-NEGOTIABLE` are distinct assertions, not interchangeable text.

Set `pass` only when all five checks pass and `findings` is empty:

1. `targetFacts`: every target fact occurs in the correct role with exact values. Natural category
   surfaces, dates, HS codes, packages, equipment, allocations, weights, marks, and totals agree.
   Every changed contact value has exactly its required occurrence count; a phone must not be
   duplicated into another populated contact slot. Exact carrier-receipt count and equipment-size
   breakdown requirements must be rendered as supplied.
2. `staleSourceFacts`: no superseded reviewed source fact remains, including repeated copies,
   riders, summaries, orphan units, qualifiers, or forbidden customs-program surfaces.
3. `auxiliaryFlavor`: every formerly populated auxiliary identity or shipment-dependent slot has
   distinct, realistic, newly synthesized content coherent with the target. Check secondary
   agents/principals/aliases, contacts and identifiers, references, free time and derived ranges,
   carrier-receipt and anonymous equipment counts, tare, weights, volume, temperature, and
   operational prose. Reject copied identifying/target-dependent source values, blanks,
   placeholders, deletion, and neutralization. A target-absent extracted assertion must disappear
   as an assertion while its occupied source line is rewritten with coherent non-extractable
   flavor. Do not reject preserved generic content or demand
   deletion of a source-populated auxiliary line just because it is absent from the target label.
4. `legalTopology`: source legal and agency relationships remain, with the exact target primary
   party and realistic fictional auxiliary identities. Every raw-only identity requirement has a
   distinct replacement and no old identity remains. Repeated requirements in one consistency
   group identify the same fictional agent while retaining local wrapping and `By` placement. The
   target label remains unchanged.
5. `formattingAndMinimality`: page and section order, populated-slot topology, heading position,
   punctuation style, casing, units, identifier shape, numeric grouping, exact line count,
   blank-line topology, line layout, generic
   boilerplate, and untouched text are preserved. No unrelated fact, mark, heading, status, line,
   or explanatory sentence was invented or duplicated.

## Review boundaries

- Every `jurisdictionalSurfaceRequirement` is mandatory. Its old named program must be absent and
  its generic target role must be populated with a newly synthesized, shape-compatible value.
  Do not infer a route jurisdiction from a party when the supplied requirement already resolved it.
- Never demand restoration of source-specific `ACID`, `CERS`, `AES`, government, country, tax-law,
  or similar jurisdictional wording when the new route no longer supports it. Preserve the slot
  and function with coherent synthetic wording, not a placeholder.
- Natural equipment grammar is sufficient. For example, `40' HIGH CUBE` or `40HQ` denotes a
  general-purpose high cube without literally printing `GENERAL PURPOSE`; `REEFER`/`RF` denotes
  refrigerated equipment. Do not demand redundant category words when the reviewed grammar
  already expresses the target category.
- An anonymous count identified by `sourceSemanticRoleHints` belongs to its equipment phrase and is
  never `marksAndNumbers`.
- Exact surfaces in `surfaceRenderingRequirements` are mandatory. Natural synthesis of auxiliary
  content is not an unsupported target fact.
- Do not demand removal of a newly synthesized volume or address continuation when it replaces an
  actually populated source slot. Conversely, reject creation of a slot that was unpopulated in
  the source. If the target has no marine-pollutant field, preserve the source pollutant-status
  assertion rather than guessing a new status or deleting the source slot.
- Generic shipment statuses such as `Free Out` remain byte-identical unless a target-controlled
  fact makes them incoherent. Long operational paragraphs retain their complete source grammar,
  punctuation, and role while only source-specific values change; never accept truncation.

For `revise`, fail each affected checklist field. Every finding must list the exact affected schema
path(s), or an `auxiliary.<role>` path for raw-only flavor; quote one to four exact short substrings
from the rewritten OCR; and prescribe the smallest complete correction. Evidence contains no line
IDs. Do not echo labels, source text, hashes, or summaries. If the rewrite satisfies the contract,
return `pass`—a review exists to find real defects, not to force another edit.
