# Synthetic Bill of Lading cargo language

Generate the target-facing printed cargo text for every supplied cargo group in one synthetic
Bill of Lading scenario. The caller supplies authoritative synthetic goods identities, dangerous-
goods semantics, structured package/equipment/measurement facts, route context, source text used
only as a style reference, and the exact output-field topology.

Rules:

1. Return one `cargoGroups` row per input row, in the same order, with the exact `groupId`.
2. Generate only these task-facing textual leaves: `description`, `additionalInformation`,
   `marksAndNumbers`, and `handlingInstructions`. Do not generate package, container, or HS-code
   printed-surface fields; those categorical/identifier surfaces are handled during final raw-text
   patching.
3. Preserve the field contract exactly. Populate `description` only when `descriptionPresent` is
   true. Each array must contain exactly one position per supplied slot, in slot order. Use an
   empty array when no slots are supplied. Every requested position must contain substantive text;
   never emit a textual or JSON null placeholder for an occupied source slot.
4. Treat all source values as formatting/style evidence only. Do not copy source goods, brands,
   order numbers, marks, parties, identifiers, or distinctive phrases into newly generated values.
   Match useful traits such as capitalization, terse-versus-descriptive wording, and approximate
   complexity without making a light edit of the source. When a source slot is entirely uppercase
   or entirely lowercase, keep that casing style exactly for its replacement; mixed-case slots may
   use natural casing appropriate to the synthetic value.
5. The description must name the supplied synthetic goods identity accurately. HS descriptions are
   hierarchical: interpret each leaf together with its supplied heading and chapter. Never emit a
   fragment such as `Other` or `Of polyurethanes` without the parent noun that makes it a complete
   goods identity. When several goods identities belong to one group, cover all of them naturally
   in one description. Keep package quantities, weights, marks, route text, and unrelated
   commercial prose out of the description.
6. For thermal goods, say `FROZEN` or `CHILLED` consistently with the supplied profile. Handling
   instructions, when requested, must be compatible with the supplied thermal profile, equipment,
   and temperature setpoint; do not invent a conflicting setpoint.
7. For dangerous goods, use the supplied proper shipping name and do not change the substance,
   hazard class, packing group, subsidiary hazard, or flashpoint semantics. Do not invent a second
   dangerous substance. The structured dangerous-goods fields remain outside this text-only output.
8. `additionalInformation` is auxiliary printed cargo text. Source text identifies the slot's
   style; the supplied `semanticRole` is the authoritative slot function, and
   `sourceFactKinds` lists any structured fact classes explicitly printed there. Never change one
   role into another merely because another target fact is available. When the slot states
   a task fact such as package count, weight, temperature, or origin, ground the complete statement
   in the supplied synthetic facts. When it is source-only flavor such as wrapping, formulation,
   grade, or operational detail, generate a new realistic fictional statement of the same role that
   is compatible with the supplied goods/packages/equipment and does not assert a contradictory
   task fact. Treat each slot independently: keep its broad fact class and approximate information
   density, and do not append unrelated weights, volumes, temperatures, equipment facts,
   references, or other structured facts merely because they are available elsewhere in the seed.
   Never copy a source brand, identifier, or distinctive private value, and never use a null,
   unknown, unavailable, or generic filler placeholder.
   - `package_hierarchy_or_quantity`: print every supplied target package level and quantity in
     one coherent hierarchy/equivalence statement. If categories repeat, distinguish their
     outer/inner positions in the prose; do not collapse levels or invent a total. Category names
     such as `PACKAGE_PIECE` are schema vocabulary, not printable words: write `2 PIECES`, never
     `2 PACKAGE PIECES` or a duplicated form such as `2 PIECES PIECES`.
   - `packing_method_or_per_unit_measure`: retain a packing-method statement and name at least one
     supplied target package fact. Do not turn it into an unrelated gross-weight statement.
   - `measurement_statement`: retain exactly the source fact classes and use their supplied target
     values; do not substitute a different measure.
   - `consolidation_status`: emit a fresh consolidation statement, not weight, packages, or route.
   - `transit_or_bonded_movement`: retain transit/bonded-movement meaning and use a supplied target
     route locality.
   - `dangerous_goods_status`: match the supplied target DG state; for DG cargo include each
     supplied UN number, and for non-DG cargo state non-hazardous status.
   - `purpose_or_end_use`: retain purpose/end-use syntax and ground it in the supplied goods.
   - `commercial_or_product_identifier`: generate a new identifier with the same broad printed
     grammar and no source value.
   - `product_attribute_or_condition`: generate compatible lexical product flavor without adding
     an unrelated package, measure, route, or identifier fact.
9. For a marks slot whose action is `preserve_literal`, copy `sourceStyleReference` exactly; this is
   a generic literal such as `N/M` or `NO MARKS`, not private data. For `generate`, create a new,
   realistic fictional shipping mark or reference with a similar broad shape but no source value,
   party identity, or real-world claim.
10. Handling instructions are printed cargo instructions, not advice or explanatory prose. Generate
    them only for supplied slots and keep them operational, concise, and cargo-coherent.
11. Do not add reasoning, commentary, evidence, confidence, formatting directions, source values,
    headings, or fields outside the provider-enforced output schema.

The JSON Schema supplied by the caller is authoritative. If a general instruction conflicts with
the schema or a field contract, follow the schema and field contract.
