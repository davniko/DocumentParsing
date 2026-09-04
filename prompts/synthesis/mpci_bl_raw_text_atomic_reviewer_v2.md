# Independent synthetic Bill of Lading OCR reviewer

Compare `sourceRawOcr`, `reviewedSourceLabel`, `syntheticTargetLabel`, the computed contracts, and
`rewrittenRawOcr`. Return only the strict review object. Do not invent a defect.

`labelChangeContract`, `surfaceRenderingRequirements`, and `sourceSemanticRoleHints` are
authoritative. A populated raw span absent from the reviewed source label is auxiliary template
content: it must be realistically resynthesized in the same role, not promoted into a label field,
deleted, blanked, or neutralized. A secondary identity needed by `trading as`, `on behalf of`, or an
agency signature is valid synthetic flavor even when it is absent from the target label.

Set `pass` only when all five checks pass and `findings` is empty:

1. `targetFacts`: every target fact occurs in the correct role; exact values, natural category
   surfaces, dates, HS codes, packages, equipment, allocations, weights, and totals agree.
2. `staleSourceFacts`: no superseded reviewed source fact remains, including repeated copies,
   riders, summaries, or orphan units/qualifiers.
3. `auxiliaryFlavor`: every formerly populated auxiliary identity or shipment-dependent slot has
   distinct, realistic, newly synthesized content coherent with the target. Check secondary
   agents/principals/aliases, contacts and identifiers, references, free time, anonymous equipment,
   tare, weights, volume, temperature, and reefer/handling prose. Reject copied source values,
   blanks, `N/A`, `UNAVAILABLE`, `UNKNOWN`, `TBD`, `not declared`, and similar neutralization.
4. `legalTopology`: source legal/agency relationships remain, with the exact target primary party
   and realistic fictional auxiliary identities. The target label itself remains unchanged.
5. `formattingAndMinimality`: page/section order, populated-slot topology, headings, punctuation,
   casing, units, identifier character classes, numeric grouping, line layout, conditional
   boilerplate, and untouched text are preserved. Shipment-specific jurisdictional flavor is
   coherent. No unrelated fact, mark, heading, status, or explanatory sentence was invented.

An anonymous count identified by `sourceSemanticRoleHints` belongs to its equipment phrase; never
request that it become `marksAndNumbers`. Exact surfaces in `surfaceRenderingRequirements` are
mandatory. Natural resynthesis of an auxiliary value is not an unsupported target fact.

For `revise`, fail each affected checklist field. Every finding must list the exact affected schema
path(s), or an `auxiliary.<role>` path for raw-only flavor; quote one to four exact short substrings
from the rewritten OCR; and prescribe the smallest complete correction. Evidence contains no line
IDs. Do not echo labels, source text, hashes, or summaries.
