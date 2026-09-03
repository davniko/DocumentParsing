# Synthetic maritime party identity

Generate one fully fictional party identity or contact-only override for a synthetic maritime
Bill of Lading sample. The caller supplies the party role, any resolved target locality, cargo
semantics, the exact target-field presence to preserve, and—when present—a real source name used
only as a broad document-style reference.

Rules:

1. Return exactly one party. Copy `partyRole`, `targetLocality.city`, and
   `targetLocality.country` exactly, including nulls.
2. Generate only fields marked present in `fieldPresence`. Every requested scalar must be
   populated; every unrequested scalar must be null. Each contact array must have exactly the
   requested number of distinct values.
3. Generate a genuinely new fictional entity. A source name is evidence only for approximate
   capitalization, length, legal-name complexity, and document style. Never copy it, lightly edit
   it, preserve its distinctive core words, or claim that the generated entity is real.
4. Make the entity plausible for the supplied role and available locality. Cargo may inform its
   business character, but do not mechanically paste a product or HS description into the name.
   Carriers, forwarders, delivery agents, and consolidators remain transport/logistics entities.
5. `address` is one fictional, joined, single-line postal address. It excludes the party name,
   contact data, tax/registration identifiers, and headings. When city or country is separately
   supplied, do not add it as a standalone comma-, slash-, pipe-, or semicolon-delimited address
   component. A genuine street or district name that happens to contain a locality word is valid.
6. When `sameAsReference` is non-null, this request is a contact-only override for a deterministic
   `sameAs` relation. Keep name, address, city, and country null and generate only the requested
   contacts. Do not expand or duplicate the referenced party.
7. Phone numbers must be plausible for the supplied country when one is available and must retain
   a realistic printable style. Emails and websites must be syntactically valid and internally
   coherent with the newly generated entity. Do not use `example.com`, placeholders, masked
   digits, or prose.
8. A requested contact person's name is a fictional human name plausible for the available
   locality. It is not a company name or role heading.
9. Do not add reasoning, commentary, confidence, evidence, alternative candidates, auxiliary
   identifiers, or facts outside the provider-enforced output schema.

The JSON Schema supplied by the caller is authoritative. If a general instruction conflicts with
the schema or `fieldPresence`, follow the schema and exact field-presence contract.
