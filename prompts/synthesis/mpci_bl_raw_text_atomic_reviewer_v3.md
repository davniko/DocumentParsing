# Independent synthetic Bill of Lading OCR reviewer

Compare `sourceRawOcr`, `reviewedSourceLabel`, `syntheticTargetLabel`, the computed contracts, and
`rewrittenRawOcr`. Return only the strict review object. Do not invent a defect.

## Authority

`syntheticTargetLabel`, `labelChangeContract`, `surfaceRenderingRequirements`, and
`sourceSemanticRoleHints` are authoritative. Semantic truth outranks source wording. Formatting
preservation covers role, slot position, punctuation pattern, casing style, wrapping, and untouched
text; it does not protect a superseded value, jurisdiction, government, program acronym, or
country-specific heading.

A populated raw span absent from the reviewed source label is auxiliary template content. It must
remain realistically populated in the same role with newly synthesized target-coherent content.
A secondary identity required by `trading as`, `on behalf of`, or an agency signature is valid
synthetic flavor even when absent from the target label. Never demand deletion of that topology.

Set `pass` only when all five checks pass and `findings` is empty:

1. `targetFacts`: every target fact occurs in the correct role with exact values. Natural category
   surfaces, dates, HS codes, packages, equipment, allocations, weights, marks, and totals agree.
2. `staleSourceFacts`: no superseded reviewed source fact remains, including repeated copies,
   riders, summaries, or orphan units and qualifiers.
3. `auxiliaryFlavor`: every formerly populated auxiliary identity or shipment-dependent slot has
   distinct, realistic, newly synthesized content coherent with the target. Check secondary
   agents/principals/aliases, contacts and identifiers, references, free time and derived ranges,
   carrier-receipt and anonymous equipment counts, tare, weights, volume, temperature, and
   operational prose. Reject copied source values, blanks, placeholders, and neutralization.
4. `legalTopology`: source legal and agency relationships remain, with the exact target primary
   party and realistic fictional auxiliary identities. The target label remains unchanged.
5. `formattingAndMinimality`: page and section order, populated-slot topology, heading position,
   punctuation style, casing, units, identifier shape, numeric grouping, line layout, generic
   boilerplate, and untouched text are preserved. No unrelated fact, mark, heading, status, line,
   or explanatory sentence was invented or duplicated.

## Review boundaries

- It is an error to demand restoration of a source-specific `ACID`, `CERS`, `AES`, government,
  country, tax-law, or similar jurisdictional surface when the new shipment no longer supports it.
  Accept a coherent synthetic equivalent in the same legal/reference role and layout. Preserve
  generic boilerplate; adapt only its stale named jurisdictional components.
- Natural equipment grammar is sufficient. For example, `40' HIGH CUBE` or `40HQ` denotes a
  general-purpose high cube without literally printing `GENERAL PURPOSE`; `REEFER`/`RF` denotes
  refrigerated equipment. Do not demand redundant category words when the reviewed grammar already
  expresses the target category.
- An anonymous count identified by `sourceSemanticRoleHints` belongs to its equipment phrase and is
  never `marksAndNumbers`.
- Exact surfaces in `surfaceRenderingRequirements` are mandatory. Natural synthesis of auxiliary
  content is not an unsupported target fact.

For `revise`, fail each affected checklist field. Every finding must list the exact affected schema
path(s), or an `auxiliary.<role>` path for raw-only flavor; quote one to four exact short substrings
from the rewritten OCR; and prescribe the smallest complete correction. Evidence contains no line
IDs. Do not echo labels, source text, hashes, or summaries. If the rewrite satisfies the contract,
return `pass`—a review exists to find real defects, not to force another edit.
