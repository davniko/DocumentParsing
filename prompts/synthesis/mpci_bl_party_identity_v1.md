# Synthetic maritime party identity

Generate one fully fictional party identity for a synthetic maritime Bill of Lading sample.
The caller supplies a party role, a target locality, cargo semantics, the exact target-field
presence to preserve, and—when available—a real source name used only as a style reference.

Rules:

1. Return exactly one party and use the supplied `partyRole`, `city`, and `country` verbatim.
2. Generate a genuinely new fictional entity. The source name is only evidence about approximate
   capitalization, length, legal-name complexity, and document style. Never copy it, lightly edit
   it, preserve its distinctive core words, or claim that the generated entity is real.
3. Make the entity plausible for the supplied role and locality. Cargo may inform the entity's
   business character, but do not mechanically paste a product or HS description into the name.
   A carrier, forwarder, or delivery agent should remain a transport/logistics entity rather than
   become the owner or manufacturer of every listed cargo item.
4. `address` is one joined, single-line postal address. It excludes the party name, city, country,
   contact details, tax or registration identifiers, and headings. Do not repeat the supplied city
   or country in `address` because those are separate target fields.
5. Preserve `fieldPresence` exactly. A requested scalar must be populated; an unrequested scalar
   must be null. Each contact array must contain exactly its requested number of distinct values.
6. Phone numbers must be plausible for the target country and retain a realistic printable style.
   Email and website values must be syntactically valid and internally coherent with the newly
   generated entity. Do not use example.com, placeholder values, masked digits, or prose.
7. A contact person's name, when requested, is a fictional human name plausible for the target
   locality. It is not the company name or a role heading.
8. Do not add reasoning, commentary, confidence, evidence, alternative candidates, auxiliary
   identifiers, or facts outside the provider-enforced output schema.

The JSON Schema supplied by the caller is authoritative. If a general instruction appears to
conflict with the schema or `fieldPresence`, follow the schema and field-presence contract.
