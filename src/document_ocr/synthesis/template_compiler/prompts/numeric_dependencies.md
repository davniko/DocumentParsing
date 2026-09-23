Interpret every listed source-only numeric binding using the SOURCE bill of lading and source target. This is a one-time dependency-compilation step, not synthesis. Return exactly the requested keys.

For each, give its canonical numeric source_value as a decimal string; the interpretation must reproduce every printed source surface exactly. Distinguish decimal separators and thousands grouping using the document. Identify the actual role. divisor must be 1 except for target_average below.

A shipment quantity explicitly in MT/tonnes is cargo_mass, not a package count.
Use independently repeated numeric forms to disambiguate separators: 105.930
repeated as 105930.000 Kgs. proves thousands grouping for that amount. Do not
choose the interpretation that merely agrees with an erroneous source label.
If the label conflicts with the proved notation, request review rather than
inventing an arithmetic relationship. An unqualified MEASUREMENT column does
not establish CBM or any other unit. A numeric mode alone does not prove its
physical unit, cargo owner, or compatibility with equipment.

A complete printed equipment count/type such as 3X20'DC can describe a private
physical inventory when individual container identities are genuinely absent.
Never invent container IDs or size/type extraction labels. The host can use a
single complete inventory with one cargo owner and an explicit GROSS WEIGHT or
VOLUME caption for its private physical checks. Ambiguous inventories, multiple
cargo owners, and unexplained measurement scope still require review; a private
unit selection does not establish those relationships.

If a mass/volume unit is printed in a table heading or separate OCR line rather
than adjacent to the number, provide printed_unit_quote as an EXACT contextual
source quote proving that unit. It must contain the unit and surrounding source
context, not an isolated unit token. Inspect its semantic column/row ownership:
do not borrow units from unrelated cargo, another field, or the source label.
The host checks the quote and rejects conflicting units. Leave the field null
when units are locally attached or no unit is printed. A printed_unit_quote and
synthetic_unit are mutually exclusive; do not convert an OCR error into a quote.

The same printed_unit_quote proof is supported for source_fixed and
sampled_equipment_tare equipment tare.
Do not treat a second weight column as tare without a printed TARE declaration.
Tare is empty equipment mass, not cargo load; it must not constrain cargo capacity
as though it were goods. A row appearing before its container number in OCR needs
an explicitly reviewed container dependency in the template; a numeric contract
alone cannot establish row ownership. Do not invent missing row weights. For an
explicit TARE caption with genuinely absent units, synthetic_unit may be
"kilogram": this is an audited private synthetic choice, never an original-unit
claim, and it must not add units to text or extraction labels.

Use sampled_equipment_tare with role tare when the template explicitly owns the
empty-equipment mass by one container row or the complete inventory. Keep
target_paths and dependency_bindings empty, multiplier "1", divisor 1. The host
selects physically supported tares jointly with equipment and renders exact sums
for aggregate owners. It will reject absent ownership, contradictory source
row/aggregate arithmetic, or unsupported equipment; never invent ownership to
enable this mode. Use source_fixed only for a deliberately retained, evidenced
physical constraint, not because a new tare was unavailable.

For a source-only scalable cargo mass or volume with genuinely UNPRINTED units,
synthetic_unit may explicitly choose "kilogram" or "cubic_metre", respectively.
This is private synthetic scenario metadata, NOT an interpretation of the
original unit. It never adds a unit to the OCR or extraction target. Otherwise
leave synthetic_unit null. Inspect the complete document including table
headings first: never use this to override a printed unit, an unrecognized unit
token, area/length, or an ambiguous measurement role. It is valid only for
source_scaled with empty target_paths/dependency_bindings, multiplier "1" and
divisor 1. Physical container ownership must still be independently established;
choosing a private unit cannot supply that ownership or license equal partition
of an aggregate total. Printed captions such as REAL CBM remain printed evidence.

Prefer target_sum when the number equals one target numeric leaf or a sum of known leaves. Choose exact target_paths from numericTargetLeaves (dot/bracket syntax, such as documentPatch.cargoPackages[0].quantity, never JSON pointers). multiplier converts FROM target units INTO the printed numeric units: a target in tonnes and a printed kg number require multiplier "1000", not "0.001". The host verifies exact source equality. Refer to numeric leaves, never entire objects or arrays. Do not infer a relationship from a coincidental equal number: it must be the same shipment fact.

Use source_scaled for shipment-level/per-container/per-cargo-lot total mass, total volume or cargo quantity absent from the structured label. Those totals scale together with the newly sampled shipment; the host performs arithmetic and formatting. Do NOT classify tare, capacity, density, unit package weight, per-unit volume, temperatures, charges or contractual counts as scalable shipment totals.

An absent structured mass/volume leaf is NOT itself grounds for review: that is exactly why source_scaled exists. However, use review_required if a total depends on other numeric bindings and no supplied mode expresses the relationship; never scale and round coupled integer totals independently.

Use target_share for a printed constituent of a target total when ALL the constituents occur among the listed numeric bindings. For example, 810 cartons plus 750 cartons partition a target quantity of 1560. Give both constituents the SAME target_paths referring to that total, their individual source_value, and multiplier "1". The host proves that all constituents sum exactly to the source total and allocates the new target total proportionally with exact integer/precision conservation. Never freeze or independently round such constituents. Do not use target_share when constituents overlap or a constituent is missing.

Use binding_sum for a source-only package-count total whose constituents are other supplied numeric bindings rather than target leaves. Set dependency_bindings to their exact logicalKey strings, target_paths [], multiplier "1", divisor 1 and role cargo_quantity. The host proves source equality, rejects cycles, and adds the already rounded generated constituents exactly. Do not independently scale both a total and its constituents. Other modes must have dependency_bindings []. Existing verified contracts may be dependencies but must not be redefined. If a required constituent is missing or its meaning is uncertain, request review rather than inventing a relationship.

binding_sum also supports cargo_mass and cargo_volume when every constituent
and aggregate has the same proved physical dimension and unit. Supply explicit
printed_unit_quote evidence, or the permitted audited private synthetic_unit
where units are genuinely absent. Include every constituent exactly once;
partial visible row coverage does not prove an aggregate equation. Never borrow
units or add an unseen row merely to make the source total add up.

Use target_average when ONE binding repeats an equal measurement for multiple distinct constituents and their total is in the target. For example, a single binding printing 27510 kg for each of three containers with a target net total of 82530 kg requires divisor 3 and the target netWeight.value path. This is distinct from multiple aliases for the SAME shipment total, which require target_sum and divisor 1. The host proves source equality and enforces printed precision and divisibility on generated totals. Do not use target_share for a binding representing all equal constituents at once.

A single explicit statement of weight PER CONTAINER can also use target_average with divisor equal to the documented fixed container count; it need not be printed once for every container.

Equal package-count rows can also use target_average with role cargo_quantity,
one cargoPackages[N].quantity target, multiplier "1", and the proved number of
equal constituents as divisor. For example two 330-BAG container rows and a
660-BAG package total require divisor 2. The host draws an integer total divisible
by 2 before generating either row. Do not confuse repeated aliases of the same
total with separate constituents, or use this for unrelated equal counts.

Use target_converted for a second gross/net mass representation in pounds versus kilograms. Give exactly one grossWeight.value or netWeight.value path, divisor 1, and multiplier "kg_to_lb" when the target unit is kilogram and this surface is pounds; use "lb_to_kg" for the reverse. The host uses the exact international conversion and proves that SOURCE rounding intervals overlap using the printed precision on both sides. It then calculates every new converted surface deterministically. Do not use an approximate decimal multiplier or independently scale a converted alias.

Use source_fixed for genuinely fixed equipment tare/capacity, per-unit measures, density, commercial rates/charges, free-time days or operational counts. These are explicit retained scenario constraints, not fallback values. Provide a specific reason tied to source evidence. Container and document-copy counts are fixed because topology is fixed. A shipment mass or package total must not be frozen as operational text. Carrying temperatures and product flashpoints must follow their actual equipment/goods owners; source_fixed alone does not provide that dependency. If the owner or physical contract is missing, return review_required instead of freezing a condition while its goods change.

Use surface_fixed for an explicitly retained non-scalar surface such as a product dimension tuple, commercial clock time, operational range, or literal formatting component. Set source_value to the EXACT first sourceSurfaces string, target_paths [], multiplier "1", divisor 1. Permitted roles are dimensions, commercial, operational, temperature, metadata. Include a source-grounded reason; this mode must NEVER hide a shipment quantity/total or identity. Use metadata for file size, page/date formatting fragments and document metadata, not unknown. These retained dimensional/operational constraints will be supplied to cargo generation.

When a printed per-unit measure determines a target total, use unit_product, NOT source_fixed. target_paths must contain exactly [quantity path, measurement value path] in that order. The equation is target_measurement = target_quantity * source_value * multiplier. Example: 11525 cartons X 20 KGS = 230.500 MT NET, use source_value "20", multiplier "0.001", and paths to package quantity and netWeight.value. The host generates the total from the new quantity before freezing labels and checks the equation again at rendering. source_fixed is only valid for per-unit values with no target total to constrain. Never classify a shipment total as a per-unit value.

If source arithmetic is contradictory, dependencies cannot be distinguished, or no supplied mode faithfully expresses the relationship, return review_required with the exact issue. Never invent a dependency to make validation pass. For modes other than target_sum, target_share, target_average, target_converted and unit_product, target_paths must be empty and multiplier must be "1".
