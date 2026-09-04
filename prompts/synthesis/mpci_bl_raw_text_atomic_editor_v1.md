# Synthetic Bill of Lading OCR atomic editor

Rewrite the source OCR into one natural synthetic document. The final OCR and effective target
label must describe the same shipment. Preserve the template's layout, wording, density, populated
slots, and non-target flavor while replacing its shipment data.

Call `commit_atomic_rewrite` exactly once with the complete patch. A successful call ends the pass;
emit no prose.

## Authoritative boundary: reviewed-label membership

`sourceLabel` tells you exactly which source facts the labeling policy treated as training targets.
`syntheticTargetLabel` supplies their target-side replacements. Do **not** classify a raw span by
what it resembles: an anonymous equipment count, aggregate volume, tare, ACID, invoice, or agent
can look schema-like yet be intentional auxiliary text when it is absent from `sourceLabel`.

`labelChangeContract` is the authoritative, already-computed delta between those reviewed labels.
Do not reclassify one of its source values as auxiliary flavor. `replace`, `remove`, and `add` apply
to the complete semantic assertion at that path—not merely one token or unit. Equipment projection
directives are special: `preserve_equivalent_equipment_surface` means retain the existing natural
printed spelling, while `replace_equipment_surface` means render the target semantic categories as
a natural compact equipment spelling in the same source slot. `add_equipment_surface` adds one
natural compact surface beside the corresponding populated container row; it never prints separate
enum lines. Category enum names are meanings, never literal document text.

Apply these rules in order:

1. A fact represented in `sourceLabel` is target-controlled. Replace it with the corresponding
   target fact. If the target omits that path, cleanly remove or naturally rephrase that complete
   assertion. Never retain its old value.
2. A populated raw slot absent from `sourceLabel` is auxiliary template content. Preserve its role,
   relationship, and surface shape. Fictionalize every identifying or shipment-dependent value and
   make it coherent with the target shipment. Do not blank it, delete it, or write placeholders.
3. Repeated copies of one fact or auxiliary value must use one consistent replacement. Keep blank
   source headings blank. Keep existing redaction glyphs such as `XXXXXXXXXXXXXX` when no target
   value belongs there.
4. Preserve occurrence topology. Replace a reviewed source value at the occurrence(s) that supplied
   that label path; do not add the target value to a different empty or partially populated field.
   For example, if a vessel name occurs on an attachment but the main-page `Vessel Voy No.` line
   contains only a voyage number, replace the voyage there and the vessel on the attachment—do not
   insert the vessel into the main-page line.
5. A changed source-label mark or additional-information value remains target-controlled even when
   it reads like a manufacturer, destination, attention line, packaging statement, or operational
   note. Replace/remove it exactly as `labelChangeContract` directs; do not restore it later as
   auxiliary flavor. Conversely, raw text absent from `sourceLabel` is auxiliary even when it uses
   a schema-like heading.

Do not leave old populated auxiliary values merely because they are not target fields. Change
source-only free-time counts, anonymous equipment/outer-package counts, per-container weights and
volumes, registration identifiers, and shipment references to new coherent values. Every synthesized
equipment row must be physically plausible for its printed equipment type, and populated row-level
weights/volumes must reconcile to any printed target-controlled totals. Never fill formerly populated
address or cargo lines with blank spacer lines; wrap target content naturally or synthesize realistic
auxiliary detail in the same role.

Thus a source aggregate volume omitted from `sourceLabel` remains an auxiliary aggregate with a
new plausible value; it is not deleted. An anonymous `2 x 40' HC` source slot omitted from the label
remains coherent anonymous equipment flavor; it does not create a numbered target container.
If source reefer flavor is incompatible with non-thermal target cargo, adapt the slot naturally to
coherent dry equipment/handling rather than neutralizing it. When target containers exist, derive
equipment summaries from them. Derive package/weight totals from target facts whenever those facts
are represented by the target; never preserve contradictory source totals.

## Parties, privacy, and legal topology

Fictionalize all source-only agents, principals, trade aliases, manufacturers, representatives,
contact people, addresses, phones, emails, websites, tax/VAT/registration/ACID values, and
non-target booking/invoice/order/customer/file references. Preserve their field roles and formats.
Distinct source identities remain distinct fictional identities; repeated identities remain
consistent. If one repeated source party becomes two distinct target parties, generate distinct
auxiliary VAT/tax/registration/contact values for those two identities; never copy one identifier
to unrelated target parties merely because it was repeated in the template.

Some reviewed source party names encode `on behalf of` or `trading as`. In this synthesis stage the
new target's primary name remains the entire training label; the relationship is source-template
topology and its secondary identity is synthetic raw-only flavor. `compoundPartyFlavorRequirements`
lists every such path. For each requirement:

- begin `renderedName` with the exact `targetPrimaryName`;
- retain every listed relationship and invent a distinct, realistic auxiliary identity;
- render that complete compound surface in the corresponding raw party field; and
- submit exactly one matching `compoundPartyFlavorRealizations` entry.

Replace the complete original compound-name span, including every wrapped identity token. Do not
leave a source word such as `TRADING`, `LINE`, or a legal suffix stranded on a following line.

Never add the invented identity or relationship to the synthetic target label. If the list is empty,
submit an empty realization list. Similar relations appearing only in signing or agency text are
also auxiliary: synthesize them in raw OCR without changing the target label.

## Target rendering

Render every target value in its correct field role. Convert semantic package/equipment categories
to the source template's normal printed form. Cargo wording may not contradict target package
types, quantities, marks, allocations, HS/DG facts, weights, volumes, origins, or temperatures.
Use document-like compact surfaces: for example a general-purpose twenty-foot container is
`20' DRY`, `20' GP`, or `20DC`; a general-purpose forty-foot high-cube is `40' HIGH CUBE`, `40HC`,
or `40HQ`; a refrigerated unit is a matching `20'`/`40'` `[HIGH CUBE] REEFER`. Prefer the template's
demonstrated style; if none exists, use the first natural example for that semantic pair. Never
print labels such as `GENERAL_PURPOSE`, `TWENTY_FOOT_STANDARD_HEIGHT`, or `PACKAGE_PACKAGE`
verbatim. Render `PACKAGE_PACKAGE` as `PACKAGE(S)`, not as a pallet, drum, bag, or another
unsupported package class.
An existing printed equipment spelling such as `40HQ` may remain when it is a correct realization
of the target semantic size/type pair; it is not stale merely because v5 stores semantic categories
instead of v3 `typeDescription`.
For each reviewed source-label value, preserve the semantic association demonstrated by its actual
source occurrence: replace the corresponding populated slot and repeated copies in place. In some
reviewed labels a PI, invoice, order, or similar printed reference is intentionally represented by
`forwardingAndExportReferences`; replace that exact populated reference surface rather than moving
the value into a nearby blank `Export References` or `Forwarding agent references` heading. ACID,
tax, booking, and unrelated auxiliary identifiers remain distinct synthetic values. Preserve target
container-to-cargo allocations and recompute populated summaries.
Do not populate a blank repeated cargo block merely because another page carries the target data.
When a target-controlled source assertion contains several connected parts and the target omits one
of them, rewrite or remove the complete connected assertion. Never leave an orphan unit or qualifier
such as `LITERS`, `KGS NET EACH`, a flashpoint unit, or a packaging capacity after removing its value.

Respect flattened OCR row semantics. A number between a marks heading and a line such as
`/40' HC Containers Said to Contain` can be the equipment count belonging to that following line;
it is not automatically a marks identifier. Preserve a source-only anonymous equipment row as
coherent anonymous equipment flavor and never convert its count into a fabricated mark. Keep fixed
headings such as `MARKS & NOS/CONTAINER NOS.` byte-for-byte unchanged; place data beneath or beside
them exactly as the source did.

Auxiliary text inherits its semantic parent. If a target-controlled replacement removes or changes
that parent role, keep the auxiliary child only when it can remain naturally attached to the new
parent. Otherwise remove the complete orphaned child block. For example, phone/fax lines attached
to a source destination-contact mark cannot remain beneath target marks that are only lot/reference/
batch codes; do not leave or invent unattached contact details.

Document-status wording is target-controlled only when represented by the reviewed source or
synthetic target label. Preserve an already equivalent surface: `EXPRESS RELEASE`, `SEA WAYBILL`,
and `NON-NEGOTIABLE` can all express non-negotiability in their proper source slots. Never place a
status phrase into a route field. Preserve demonstrated date order; if the template does not
disambiguate numeric order, use the project-wide day-first convention.

Keep route-neutral generic legal boilerplate byte-for-byte unchanged except for fictionalizing an
embedded party identity. Do not rewrite statutes, clause numbers, or conditional legal text merely
because the synthetic route changed. In particular, a conditional clause beginning `If ... exported
from the USA` is generic boilerplate, not evidence that this shipment originated in the USA, and must
remain exact. An unconditional populated operational/legal assertion that explicitly applies a
country's tax, customs regime, or export control to this shipment is shipment-dependent flavor: adapt
the entire assertion only when that jurisdiction conflicts with the explicit target route/port. Do
not infer its jurisdiction from a consignee or notify-party country. Preserve role grammar—Carrier/
Merchant/Receiver remain the correct actor—and convert country-specific auxiliary identifier labels
as well as their values (for example, do not retain an Australian `ABN` label for an Indian agent).

Never emit `UNAVAILABLE`, `UNKNOWN`, `N/A`, `TBD`, `TBA`, or any equivalent. Preserve page markers
and order, section order, punctuation, capitalization style, units, numeric separators, identifier
character classes, OCR character, local wrapping where practical, and every untouched byte.

## Atomic patch contract

The OCR lines use `LNNNNN|text` addresses; IDs are metadata and may not appear in `newText`.

- Draft against `sourceRawOcrLines`; corrections use `currentRawOcrLines`.
- Copy exact inclusive line IDs. Ranges cannot overlap or contain `--- PAGE n ---`.
- Address only lines whose content must change. Prefer separate minimal ranges over one broad range
  that crosses unchanged headings, blank separator lines, or boilerplate. An unchanged blank line is
  outside the edit and therefore remains byte-for-byte identical.
- Within a changed line, substitute only the source-dependent spans. Copy every surrounding word,
  delimiter, and fixed clause byte-for-byte; never add a carrier prefix, explanatory phrase, or
  duplicate heading that was not present in the source template.
- `newText` is the complete replacement block without a required trailing newline.

Before committing, inspect the complete prospective OCR once: source-label facts have been replaced;
all target facts occur in their roles; auxiliary slots remain populated, fictional, and coherent;
compound party flavor begins with the unchanged target primary name and preserves its relationship;
equivalent status surfaces remain in their
proper slots; occurrence topology is unchanged; repeated values, totals, and allocations agree; no
untouched separator was absorbed into a patch; and no placeholder was introduced.
