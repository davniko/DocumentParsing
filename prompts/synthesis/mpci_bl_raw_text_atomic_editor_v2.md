# Synthetic Bill of Lading OCR atomic editor

Produce one coherent synthetic OCR document by replacing the source shipment with the supplied
synthetic shipment. Call `commit_atomic_rewrite` exactly once with the complete line-addressed
patch. A successful call ends this pass; emit no prose.

## Authoritative inputs

- `syntheticTargetLabel` is the complete training label that the rewritten OCR must support.
- `labelChangeContract` is the deduplicated source-to-target delta. Each item may list several
  schema paths because one printed value can support multiple relational fields. Apply every item.
- `surfaceRenderingRequirements` gives exact source-style date and HS-code surfaces. Render each
  `targetSurface` exactly and remove its superseded `sourceSurface` from the corresponding slot.
- `sourceSemanticRoleHints` resolves flattened OCR grammar. In particular, an anonymous equipment
  count is not a mark or container identifier.
- `compoundPartyFlavorRequirements` identifies reviewed party names whose template topology uses
  `on behalf of` or `trading as`.

Do not reinterpret or second-guess these computed contracts.

## Required document semantics

1. Replace every target-controlled source fact with its target value in the same semantic slot and
   at its repeated occurrences. Remove a target-controlled assertion only when the target omits it,
   and remove the complete assertion without an orphan unit or qualifier.
2. Preserve every populated auxiliary slot that is absent from the training label. Replace its old
   identifying or shipment-dependent content with realistic fictional content coherent with the
   target. This includes secondary legal identities, signing agents, manufacturers, contacts,
   VAT/tax/registration/customs identifiers, booking/invoice/order references, free-time values,
   anonymous equipment counts, tare, row weights, and volume.
3. Never neutralize populated flavor with `N/A`, `UNAVAILABLE`, `UNKNOWN`, `TBD`, blank filler, or a
   statement that data is not declared. If a source-only reefer/equipment/volume/free-time slot is
   present, synthesize a plausible value or naturally adapt the complete populated assertion to the
   target cargo/equipment. Reconcile row measures and printed totals.
4. Do not add a new extractable fact absent from `syntheticTargetLabel`. Auxiliary flavor remains
   auxiliary: preserve its source role and topology instead of moving it into a target field.
5. Keep repeated identities and references internally consistent. Keep distinct identities
   distinct. An explicit `same as consignee` surface may remain relational wording; an explicitly
   printed party must contain the target party data.

## Compound and raw-only identities

For every `compoundPartyFlavorRequirement`, retain the relationship while leaving the training
label unchanged:

- begin `renderedName` with the exact `targetPrimaryName`;
- invent a distinct realistic auxiliary company identity for every required relationship;
- place the complete compound surface in the same party-name slot; and
- submit exactly one corresponding `compoundPartyFlavorRealizations` item.

The same rule applies to relations found only in raw signing/agency text: synthesize the secondary
agent or principal. A missing auxiliary identity is never a reason to block or delete the legal
relationship.

## Categories, auxiliary operations, and jurisdiction

Render semantic equipment/package categories as natural B/L text in the source's style. Examples:
`20' DRY`, `40HQ`, `40' HIGH CUBE REEFER`, `PIECE(S)`, or `PACKAGE(S)`. Never print enum tokens.
Preserve correct existing equipment wording when instructed. Container/cargo allocations,
quantities, gross/net values, and totals must reconcile.

An anonymous equipment summary stays anonymous; do not invent a numbered container. A source-only
volume remains a newly synthesized plausible volume. When the target changes reefer state, adapt
temperature and reefer-specific prose coherently rather than erasing or neutralizing populated
slots.

Keep route-neutral and conditional legal boilerplate unchanged except embedded identities. Adapt
an unconditional shipment-specific customs/export assertion only when its jurisdiction conflicts
with the target route. Preserve the slot and actor grammar, use a route-neutral natural heading such
as `CUSTOMS REFERENCE` or `EXPORT REFERENCE` when no target-specific nomenclature is supplied, and
synthesize its value in the source format. Do not infer jurisdiction from a party country alone.

## Fidelity and atomic patching

Preserve page markers and order, section order, headings, punctuation, capitalization, OCR
character, newline convention, local wrapping, identifier character classes, number grouping,
units, and every untouched byte. Use the exact target values while rendering canonical dates and
HS codes according to `surfaceRenderingRequirements`.

The OCR uses `LNNNNN|text` addresses. IDs are metadata and must not appear in `newText`.

- Draft against `sourceRawOcrLines`; corrections use `currentRawOcrLines`.
- Use exact inclusive ranges. Ranges cannot overlap or contain a page marker.
- Prefer minimal separate ranges. Copy unchanged surrounding text exactly.
- `newText` is the complete replacement block without a required trailing newline.
- Submit all required edits together. The output function validates and commits atomically.

Before calling the output function, inspect the prospective document once: all target facts are in
their original roles; all superseded source facts are gone; populated auxiliary slots contain new,
realistic, coherent values; compound relationships have fictional secondary identities; required
date/HS surfaces are exact; totals and relations agree; and there are no placeholders, orphaned
units, invented marks, duplicate headings, or unnecessary changes.
