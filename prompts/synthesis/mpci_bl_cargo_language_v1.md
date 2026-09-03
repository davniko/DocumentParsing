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
   empty array when no slots are supplied. An `additionalInformation` position may be `null` under
   rule 8; other requested positions must contain text.
4. Treat all source values as formatting/style evidence only. Do not copy source goods, brands,
   order numbers, marks, parties, identifiers, or distinctive phrases into newly generated values.
   Match useful traits such as capitalization, terse-versus-descriptive wording, and approximate
   complexity without making a light edit of the source.
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
8. `additionalInformation` is auxiliary printed cargo text only. Source text identifies the broad
   slot style; it is not authority for retaining the source fact. Return replacement text only when
   the supplied synthetic goods identity, dangerous-goods facts, or `structuredFacts` ground the
   complete statement. Return `null` for a slot with no target-grounded semantic replacement; this
   explicitly tells final patching to remove the source slot. Never fill an ungrounded formulation,
   grade, brand, package, origin, or measurement slot with generic prose.
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
