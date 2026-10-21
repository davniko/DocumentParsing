# Dataset edge-case inventory

Current working dataset: **R28**, `data/curated/mpci-bl-r28-positions-v1`.
Earlier entries explicitly naming R27 describe that frozen historical dataset;
their artifacts are not rewritten by R28 repairs. Subsequent work stays in R28
unless the user explicitly authorizes another version.
This inventory separates accepted unusual source presentations, optional cleanup,
and unresolved annotation questions. An unusual value is not automatically a
defect. Historical audit snapshots remain preserved.

## Country presentation: accepted document-specific interpretation

**EC-COUNTRY-001 — `c7060e22`; accepted unchanged, 2026-10-16.**

- Occurrence: original template source, notify party postal block:
  `ARAB ORGANIZATION FOR INDUSTRIALIZATION. (A.O.I.)`,
  `KM 17 CAIRO, SUEZ DESER ROAD, ALMAZA, CAIRO, ARE`.
- Existing target: `country: EGYPT`.
- User adjudication: retain the Egyptian country target as correct for this
  occurrence. The previous review hold is closed; no label or source edit is needed.
- This is a document-specific decision, not a global `ARE → EGYPT` alias.
  Undotted ISO3 `ARE` remains UAE in the general registry.
- The original is not in real training or validation. Its six current synthetic
  descendants independently print their sampled countries and already pass.
- Those R27 `synthetic-train.jsonl` descendants are
  `syn_full_v7_a073d1c175e6e797323aab88`,
  `syn_full_v7_0eef88598c28453f1eb3518b`,
  `syn_full_v7_9fb56645426dd24f0764f24d`,
  `syn_full_v7_108b625c0f3518c696fa9583`,
  `syn_full_v7_062a0ccc96ed919811ed9f8a`, and
  `syn_full_v7_3087bbef86c60e90bf215085`.
- [OCR](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_c7060e220634333eba11aecef9d3a01cbad4a0d51a2a9ec6c7fc27db2b2bf170/source.txt#L15),
  [PDF](../data/corpora/paddle-bl-swb-filtered-max5/files/e4cde1eb8292da5bfebc06fb057ce897d984fd5525a52c9658cc21160e33a8ff.pdf).

## Repeated package declarations: source-faithful, optional cleanup only

**EC-REPETITION-001 — `ebb856c7`; accepted unchanged, 2026-10-16.**

The original OCR itself prints:

```text
22 Bundle(s) of Bundle(s) of 22 Bundle(s) of SPECIAL HIGH GRADE ZINC 99.995 PCT
```

The five identified R27 descendants preserve this repetition with their sampled
packages and quantities. The source rendering is the basis; this is not a new
country-migration or synthetic-only defect. User decision: retain it; an optional
future cleanup must not be represented as necessary target correction.

| R27 synthetic ID | Per-container counts | Labeled total | Location |
| --- | --- | ---: | --- |
| `syn_full_v7_a6c8047e6a4b22124c938de1` | 585 + 585 packages | 1,170 | `synthetic-train.jsonl`, page 2 goods rows |
| `syn_full_v7_2b27d4110aa2bb6279bb6fc5` | 1,060 + 1,059 bags | 2,119 | same |
| `syn_full_v7_891086e8bd466b039a4ae56f` | 372 + 371 pieces | 743 | same |
| `syn_full_v7_b66624c801b6655ce9eca197` | 2,007 + 2,007 packages | 4,014 | same |
| `syn_full_v7_c4c4cba5d4320c491c11661f` | 14 + 13 bins | 27 | same |

Each has two repeated-prefix occurrences. Product description targets omit the
leading package declarations, while quantities/allocations agree. No new
target inconsistency is established by the repetition.

[Original OCR](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_ebb856c7220ab2a66071c536b6f662b26cf593566ca2b4ee0973c705d2abb5ee/ocr.txt#L145),
[all five rendered quotations and descriptions](analysis/country-production-repair-20261016/unrelated-packaging-wording-review.json).

## Telephone lists: unresolved completeness review

**EC-CONTACT-001 — validation `a44bb232`; unchanged review item.**

The delivery-agent block prints four telephone numbers separated by slashes;
the current target retains only the first. Six comparable real-training sources
retain all four. The earlier audit deliberately did not change frozen or current
validation. This is a completeness/segmentation decision, unlike the two accepted
cases above. Carrier-footer contacts elsewhere are separately owned and must not
be borrowed into that party.

[Source-grounded evidence and six comparisons](kie-failing-run-deep-audit-2026-10-15.md#52-validation-phone-list-completeness-hold),
[current R27 readback](analysis/r27-validation-real-overlap-20261016/current-a44bb232.json).

## Split independence: related partial-container shipment records

**EC-SPLIT-001 — validation `2809b1ea` / real training `b7eadb8c`; audit finding,
no split change authorized or made.**

R27 validation row 4 and real-training row 400 share container `TEMU7634743`,
seal `YMAR415201`, shipper, vessel/voyage and date. Both explicitly say the cargo
is part of one container. They contain different cars, VINs and consignees, so
they are not duplicate copies. They do belong to the same physical shipment
group, which matters if validation is intended to be shipment-disjoint rather
than merely document-disjoint.

[Full comparison and OCR/PDF inspection links](kie-r27-failure-followup-and-validation-overlap-2026-10-16.md#shared-container-and-seal-a-real-split-independence-concern).

## Opaque numeric Marks: scoped anonymization closures

**EC-MARKS-001 — `978a3990`; resolved in R28, 2026-10-20.**

The company name in `HANSA 2973 MADE ...` was already correctly regenerated.
The user explicitly authorized anonymizing the opaque four-digit code without
assigning it a private-reference/product subtype. All fourteen current R28
synthetic records / twenty-four active-and-retained copies now have new
format-preserving codes in the exact reviewed Marks occurrence. The source
template has its own deterministic private identifier binding. Company/origin
wording, targets and coordinates are unchanged; the already-anonymized real
`NORVIK 7492` row is unchanged. Historical probe descendants remain historical,
not silently rewritten or certified by this closure.

**EC-MARKS-002 — `88bab156`; resolved in R28, 2026-10-20.**

Source CUSTOMS/Marks lines print seven consecutive codes5464747–5464753, separate
from product/HS wording, quantities, seals and container IDs. PDF review confirms
opaque castable references; no more specific subtype or one-to-one container
assignment is inferred. The current anonymized real record's6837291–6837297
sequence remains unchanged. All twenty-one current R28 synthetic records /
twenty-nine physical rows now use independently sampled same-width consecutive
sequences excluding every original member. All seven template owners have
private deterministic recipes sharing one new sequence base, including the
continuation page. Targets and coordinate suffixes remain unchanged. Two fresh
ordinary-production controls pass complete text/target/privacy/geometry review.

[Original review history](analysis/r28-legacy-closure-20261019/REPORT.md#additional-notes-discovered-during-full-review)
and [applied closure, exact scope and fresh controls](analysis/r28-held-branch-repair-20261020/marks/REPORT.md).
The combined publisher independently verifies thirty-five Marks records plus
two phone-target records; it does not imply universal anonymization beyond the
reviewed inventory.

## Inherited non-target geographic context

**EC-CONTEXT-001 — `dfaddb83`; advisory, 2026-10-19.**

Two phone-production controls retain source state/postcode suffixes`TX` and
`CA92008-880` after sampled locality names. Name-only targets do not infer a
country from them. Root treats these as non-target context under the user's
flavor-text policy, not an extraction-label defect. Native findings remain
preserved; the two controls have explicit inspection-only geometry rather than
bypassing ordinary publication. No current R28input rewrite is made for this
cosmetic context.
