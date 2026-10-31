# Reviewed internal package definitions

Reviewed 2026-10-25. This is a code-to-properties reference, **not** the public
extraction schema, an alias registry, a filing-code selector or a sampling rollout.

- All 405 codes from the saved MPCI application registry are covered.
- SET and generic BALE are separate semantic extensions, with no fabricated
  application code.
- All 48 active task categories and the additional BARREL, JERRICAN, BUCKET, TUB
  and KIT families are validated. CYLINDER/GAS BOTTLE definitions are included as
  part of the full code list; loading them does not enable runtime categories.
- Four reviewed shards retain primary material, explicit attributes, component
  roles, measurements, standards/designations and definition notes. The manifest
  pins those shards, the application authority and active task constraints.

## Unspecified is a valid broader description

```text
DR: DRUM
  1A: DRUM + STEEL
    QA: DRUM + STEEL + NON_REMOVABLE_HEAD
    QB: DRUM + STEEL + REMOVABLE_HEAD

5M: BAG + PAPER
  XJ: BAG + PAPER + MULTI_WALL
    XK: BAG + PAPER + MULTI_WALL + WATER_RESISTANT
```

A steel-drum declaration need not select a head. A paper-bag declaration need not
select multi-wall or water resistance. Unknown fields are neither negative facts
nor errors. Generic BALE similarly does not select compressed/non-compressed.
These are examples of increasing specificity; not every axis makes one tree.

## Internal interface

```python
from pathlib import Path
from document_ocr.package_ontology import load_package_ontology, definition_includes

root = Path("/mnt/d/projects/documentparsing")
reference = load_package_ontology(
    root / "configs/ontology/package-properties-v1/manifest.json",
    project_root=root,
)
steel = reference.by_code["1A"]
generic = reference.by_code["DR"]
assert definition_includes(steel.properties, generic.properties)
assert not definition_includes(generic.properties, steel.properties)
```

Returned definitions and nested mappings are immutable. Code/category lookups
are exact; `DRUM`, commercial abbreviations, differently cased codes and DG
designators such as `1A1` are not guessed into another namespace.

`definition_includes` compares reviewed explicit properties. It does not link OCR
regions, infer undocumented material-superclass relationships, prove arbitrary
numeric inequalities, convert units or transfer inner/support material to a body.
Distinct codes may encode the same reviewed properties: `44` and `EC` both
establish plastic bags. The original identities remain distinct; no unique inverse
filing code is invented.

## Definition versus printed evidence

The ordinary phrase GAS BOTTLE does not automatically assert every property in
the supplemented formal GB definition. Likewise the word OCTABIN alone must not
silently populate every property of the corresponding code definition. Future
evidence interpretation must distinguish an actually printed, namespace-resolved
code from a natural family name or a contextually shortened repetition.

Only information supported for the same declared package is eligible for its
extraction target. Component materials stay on their components; code-definition
pressure/temperature conditions are not shipment equipment settings. Quantity
does not select NF, and a named cargo form's substance is not enclosing material.

The saved application's exact labels/categories are joined at load time. The
OASIS supplementary source is an older, explicitly pinned UNECE revision, not a
claim to have downloaded the newest list. Edition differences and uncertainty
(such as WJ's absent pressure comparator or OX dimension units) remain explicit.
This reference is reviewed internal modeling, not NAIC certification of a new
submission interpretation.

## Maintenance and validation

1. Review changes against the exact application source and cited definitions.
2. Edit the owning shard; do not synthesize unprinted qualifiers or conflate roles.
3. Regenerate manifest pins/vocabularies after that review:

   ```bash
   .venv/bin/python scripts/prepare_package_ontology.py
   ```

4. Run the maintained tests:

   ```bash
   TMPDIR=/tmp .venv/bin/pytest -q tests/test_package_ontology.py
   ```

Regenerating a hash is not semantic approval. Tests establish coverage, coherent
ownership, valid broad/narrow distinctions and corruption guards; reviewers still
own whether an authored property is supported by the cited definition.
