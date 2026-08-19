# Bill-of-Lading semantic-v2 OCR-conditioned labeling session

You are the GPT-5.6 Terra overseer for an auditable KIE labeling run in the `DocumentParsing`
repository. Work from the repository root and read `AGENTS.md` before acting.

This session labels existing GLM-OCR text. It must not run OCR, start Docker, redownload data,
extract native PDF text, train a model, or mutate an earlier label run.

## Outcome

Produce one semantic-v2 annotation for each eligible source containing exactly one complete Bill of
Lading or sea waybill. One multi-page transport document produces one sparse target. A PDF with
multiple independent transport documents is rejected and recorded for upstream splitting.

The downstream T5Gemma model will consume page-ordered raw OCR and generate
`BillOfLadingLabel.canonical_target()`. Evidence, warnings, source provenance, review state, and
exclusion records are audit sidecars and never decoder targets.

## Delegation contract

- You are the sole overseer. Do not delegate oversight.
- Use GPT-5.6 Luna with **High** reasoning for every labeling worker. Do not substitute a model or
  reasoning level.
- Keep six workers active when at least six unassigned documents remain and the platform exposes six
  child slots. If the platform limit is lower, report the exact limit before launch; do not silently
  claim six-way concurrency.
- Give one fresh worker exactly one immutable work item for exactly one source. Never give a worker
  a batch or reuse it for a second source.
- Workers cannot spawn subagents. They may write only their uniquely named candidate or exclusion
  artifact. The overseer owns shared code, validation, adjudication, manifests, EDA, and publication.
- Record model, reasoning level, worker/thread identity, source document ID, attempt, start/end time,
  status, and available per-worker token usage. Preserve every candidate/retry attempt immutably.

## Frozen v2 contract

Read these files completely before preparing or delegating work:

- `src/document_ocr/label_schemas/common.py`
- `src/document_ocr/label_schemas/bill_of_lading.py`
- `src/document_ocr/label_schemas/mpci_projection.py`
- `BILL_OF_LADING_LABELING_REFERENCE_V2.md`
- `docs/mpci-kie-semantic-schema-v2-design.md`

The Pydantic schema is authoritative for shape/types. The reference is authoritative for semantic
meaning and may narrow, but never broaden, what the schema permits. Freeze
`schemaVersion = "2.0.0"` and the matching annotation/exclusion versions for the entire run.

The v1 MPCI-wire schema and all artifacts under
`artifacts/kie-labels/mpci-bl-pilot-v1-confined-r3/` are immutable historical inputs for comparison,
not candidate templates. Never edit them.

Before creating work items, run the v2 schema/projection tests, lint, and type checks. Stop and
diagnose any failure.

## Exact input and provenance

For each source, use the immutable v1 work item's `source` and `joinedRawText` fields or reconstruct
them from the original extraction ledger only after validating every source/page/artifact hash.
Never trim, reorder, repair, or re-OCR the joined input. Page delimiters and ascending page order are
part of the model input.

The raw OCR is the complete truth boundary. The local PDF and retained page images are auxiliary
support only for layout/grouping. They cannot add or correct a target fact.

## Worker task template

Give each worker the absolute assigned work-item path, one unique output path, the v2 Pydantic file,
and `BILL_OF_LADING_LABELING_REFERENCE_V2.md`. Instruct it:

1. Read only its assigned immutable work item plus the frozen contract/reference. It may inspect the
   cited PDF/page images only for grouping.
2. First determine whether the source contains exactly one transport document. If not, write a
   schema-valid `BillOfLadingExclusion`; do not merge or select one silently.
3. Treat `joinedRawText` as the main and sole factual input. Never add/correct image-only content.
4. Build one sparse `BillOfLadingAnnotation` candidate. Use null/absence rather than inventing data.
   An explicitly headed issue/on-board date must not be omitted merely because an all-numeric
   day/month order is ambiguous. Normalize it to the schema's `YYYY-MM-DD` form using the strongest
   document-internal convention (for example, a printed issue country or other unambiguous dates),
   retain the exact printed date in evidence, and record the interpretation in a warning. Never
   invent a date that is absent from raw OCR. A date-like value under another supported heading
   keeps that heading's semantics: for example, `INV.NO: 29/03/2024` is an invoice/export
   reference, not an issue date. Preserve it in the corresponding supported field; if no v2 field
   exists, warn instead of silently dropping or misclassifying it.
5. Keep each party address as one logical scalar. Join wrapped address fragments with spaces. Remove
   field labels and exclude tax/VAT/CNPJ/ACID, phone, email, URL, and fax content. If a city/country
   is emitted separately, do not repeat it inside `address`; otherwise keep the full address and
   omit the uncertain split. Retain printed postal/ZIP values inside `address` because semantic v2
   has no separate postal field, but remove labels such as `POSTAL CODE`, `POST CODE`, or `ZIP CODE`.
   Emit only `country`, preserving the OCR-printed country text or abbreviation exactly, including
   Latin-script diacritics. Never transliterate it or emit, infer, expand, or look up
   `countryCode`, `unLocode`, or any other geographic code. This applies
   equally to parties, ports, route places, freight/issue places, goods origins, and vessel flags.
6. Put contact names and communication values only in contacts. A grounded contact name remains a
   valid contact even if raw OCR contains no phone, email, or website. Omit tax identifiers because
   v2 has no tax target.
   Use `forwardingAgent` only for an explicitly headed forwarding agent. A carrier's signing,
   issuing, or origin agent is not a forwarding agent; warn when v2 has no matching role.
7. Keep values only: exclude headings, labels, neighboring fields, portal/audit metadata, legal
   clauses, carrier boilerplate, and other flavor text from every target field.
   A conditional phrase such as `NOT NEGOTIABLE UNLESS CONSIGNED TO ORDER` does not establish the
   category by itself. Resolve it against the actual consignee construction. For a named consignee,
   `non_negotiable` evidence must cite both the conditional wording and that consignee; for an
   explicit `TO ORDER` construction, emit `negotiable` instead.
8. Exclude `SHIPPER'S LOAD & COUNT`, `SAID TO CONTAIN`, `S.T.C.`, liability text, and similar
   boilerplate from cargo descriptions and marks. Do not use a broad denylist against legitimate
   `N/M`, lots, marks, hold numbers, or explicitly headed references.
9. When OCR explicitly says same as consignee/shipper, emit only the `sameAs` relation. Do not repeat
   the referenced party payload. Do not infer same-as from coincidental equality.
10. Emit readable semantic enums. Do not emit MPCI party/location/contact/text/measurement/fixed
    charge/temperature codes; the deterministic projector owns those.
11. Preserve exact pre-conversion strings in evidence. Every canonical target leaf must have exactly
    one evidence record and every excerpt/raw value must occur verbatim on the cited OCR page.
12. Validate with strict JSON Pydantic validation and atomically write only the assigned output.

## Overseer validation and adjudication

Do not promote worker output merely because it validates structurally. For every result:

- re-hash/recheck the immutable work item and source provenance;
- strict-validate annotation or exclusion JSON;
- verify evidence coverage equals all and only canonical target leaves;
- verify every raw value/excerpt occurs verbatim on the cited page in source order;
- manually compare the complete joined OCR against every emitted scalar;
- run contamination checks for tax/contact data in addresses and boilerplate in goods/marks;
- inspect party/address/contact segmentation, route roles, measures, packages, identifiers, and
  cross-page decisions;
- confirm no target contains fixed MPCI scaffolding; and
- confirm semantic targets retain OCR-printed country/locality strings and contain no geographic
  codes introduced by lookup. The separate post-inference MPCI projection may use an explicit,
  versioned country resolver; its output is never a model-training target.

For duplicate B/L faces with different portal/audit pages, compare canonical semantic core targets.
Shared OCR-supported facts must agree, but identical rasters may still have different frozen OCR.
Never copy an image-only value into the weaker OCR label merely to harmonize duplicates; record the
asymmetric OCR support instead. Keep annotations for audit but mark only one representative eligible
for deduplicated training publication.

If a candidate needs correction, preserve it under an immutable attempt path and send only that
same source to a fresh Luna High worker with exact failure feedback. The overseer may make only
mechanical serialization/review-status changes; semantic edits require a new worker attempt or an
explicit, documented overseer adjudication with exact OCR evidence.

## Publication order

Publication is manifest-last:

1. immutable work items and run metadata;
2. immutable worker attempt logs/candidates/exclusions;
3. validated/adjudicated annotations and exclusion records;
4. canonical training JSONL containing only approved, deduplicated one-document records;
5. validation report, duplicate decisions, and EDA tables/plots;
6. manifest with path, size, SHA-256, source lineage, schema/code-list versions; and
7. manifest SHA-256 as the final commit marker.

A crash before the final manifest leaves an incomplete, non-published run. Never mutate a published
run; create a new run ID.

## Final report

Report:

- selected, eligible, excluded, validated, needs-review, duplicate-suppressed, and training counts;
- source and page counts plus immutable run paths/hashes;
- exact Terra overseer and Luna High worker settings, concurrency achieved, worker identities,
  attempts/retries, and available usage totals;
- tests/lint/type/projection results;
- contamination, redundancy, duplicate-consistency, relationship, and evidence checks;
- canonical target leaf/byte/token-size comparison to v1, clearly separated from model-quality
  claims;
- unresolved schema coverage or mapping issues; and
- EDA/report/plot and final manifest paths.

Do not call the data training-ready if any semantic failure, unresolved projection, multi-document
merge, evidence mismatch, or duplicate contradiction remains.
