# Bill of Lading goods/package compatibility

Select one or more physically plausible, task-facing package-category signatures for the supplied
synthetic dangerous-goods context.

Rules:

1. Use only category tokens supplied in `allowedPackageCategories`. The provider-enforced JSON
   Schema restricts every value to this vocabulary.
2. Every candidate must contain exactly `context.packageCount` ordered categories. A candidate is
   the complete package signature for one cargo group, not a loose set of suggestions.
3. Base compatibility on the proper shipping name, HS identity, physical form implied by that
   identity, hazard class, subsidiary hazards, and packing group. Prefer categories normally
   capable of safely containing and transporting that physical form by sea.
4. Choose packages at the requested declared shipment accounting level. Contained packing and
   supporting handling units are not additional additive shipment rows. Neither innermost nor
   outermost packaging is a universal default; pallets/skids require a declared handling-unit role.
5. A liquid must not be assigned a bag, bale, roll, sheet, or other solid-only package. A gas must
   not be assigned an ordinary carton or sack as its direct containment. Apply the corresponding
   physical-form discipline to solids and articles.
6. Return several candidates only when they are independently plausible. Do not pad the list with
   weak alternatives. Keep signatures distinct and rank the strongest first.
7. `PACKAGE_PACKAGE` is a generic last resort, not a substitute for a supported specific category.
   `PACKAGE_UNPACKED_OR_UNPACKAGED` is valid only for goods genuinely transported without a package.
8. The rationale is short evidence for audit. Do not add legal claims, regulatory citations,
   fabricated product facts, or fields outside the schema.

The supplied data is synthetic. The goal is a realistic training example for reading a printed
Bill of Lading, not validation of a real shipment or a recommendation for actual transport.
