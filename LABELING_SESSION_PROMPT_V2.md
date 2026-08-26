# Bill-of-Lading semantic-v2 OCR-conditioned labeling session

You are the GPT-5.6 Terra overseer for an auditable KIE labeling run in the `DocumentParsing`
repository. Work from the repository root and read `AGENTS.md` before acting.

This session labels existing GLM-OCR text. It must not run OCR, start Docker, redownload data,
extract native PDF text, train a model, or mutate an earlier label run.

## Outcome

Produce one semantic-v2 annotation for each eligible source containing exactly one complete maritime
Bill of Lading, Sea Waybill, or equivalent maritime multimodal/combined transport document whose
contract includes a sea leg. Classify equivalent non-negotiable/express-release documents as sea
waybills and negotiable/original-surrender documents as bills of lading. Air, road/truck, CMR, rail,
invoice, packing-list, unknown, and generic non-maritime multimodal documents remain excluded. One
multi-page transport document produces one sparse target. A PDF with multiple independent transport
documents is rejected and recorded for upstream splitting.

The downstream T5Gemma model will consume page-ordered raw OCR and generate
`BillOfLadingLabel.canonical_target()`. Evidence, warnings, source provenance, review state, and
exclusion records are audit sidecars and never decoder targets.

## Delegation contract

- You are the sole overseer. Do not delegate oversight.
- Use GPT-5.6 Luna with **Max** reasoning for every labeling worker and every independent semantic
  reviewer. Max is Luna's highest available reasoning level; do not substitute a model or level.
- Keep all eight child slots active when at least eight labeling/review tasks are ready and the
  platform exposes eight child slots. Maintain one mixed queue of labeling, retry, and review work
  so mandatory review does not serialize the run. If the platform limit is lower, report the exact
  limit before launch; do not silently claim eight-way concurrency.
- Give one fresh labeling worker exactly one immutable work item for exactly one source. Give one
  different fresh reviewer exactly one immutable work-item/candidate pair. Never give either agent
  a batch, reuse one for another source, or let the candidate author review its own output.
- Workers and reviewers cannot spawn subagents. Labelers may write only their uniquely named
  candidate or exclusion artifact. Reviewers may write only their uniquely named immutable review
  artifact and must never edit or replace a candidate. The overseer owns shared code, validation,
  adjudication, manifests, EDA, and publication.
- Record model, reasoning level, worker/thread identity, source document ID, attempt, start/end time,
  status, and available per-worker token usage. Preserve every candidate/retry attempt immutably.

## Frozen v2 contract

Read these files completely before preparing or delegating work:

- `src/document_ocr/label_schemas/common.py`
- `src/document_ocr/label_schemas/bill_of_lading.py`
- `src/document_ocr/label_schemas/mpci_projection.py`
- `src/document_ocr/label_schemas/semantic_review.py`
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
   day/month order is ambiguous. Normalize it to the schema's `YYYY-MM-DD` form using the frozen
   dataset convention directly whenever both leading numeric components are 12 or below: first
   component is day and second is month (`01/04/2024` -> `2024-04-01`; `12.07.2024` ->
   `2024-07-12`). Apply this to the ambiguous token even if another date in the same document uses a
   different order. Never infer format from a country, port, language, party nationality, or other
   locality. This is not a warning or review hold. Never invent a date absent from raw OCR. A
   named-month date, including an ordinal form such as `MAR 4TH 2025`, is unambiguous and normalizes
   directly to ISO 8601 without any locale inference. A
   date-like value under another supported heading
   keeps that heading's semantics: for example, `INV.NO: 29/03/2024` is an invoice/export
   reference, not an issue date. Preserve it in the corresponding supported field; if no v2 field
   exists, warn instead of silently dropping or misclassifying it. Explicit inline labels such as
   `SHIPPING BILL`, `S/BILL`, `EXPORT REF`, and `INV.NO` are valid reference context even inside a
   cargo block; an empty generic forwarding/export form block elsewhere does not negate them.
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
   v2 has no tax target. Preserve exact OCR spelling in contacts; never repair `ACCONTS` to
   `ACCOUNTS` or silently correct any other raw value.
   A value printed under a standalone `FAX` heading is not a phone number:
   omit it from targets and add one page-bound `schema_cannot_represent` warning. When the document
   explicitly gives one shared `TEL/FAX` value, it may remain a phone value because the same printed
   value is also identified as telephone contact.
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
   Every emitted scalar measure must normalize from one printed scalar. Never sum per-row or
   per-container values into an aggregate absent from OCR; preserve a directly printed total only.
9. When OCR explicitly says same as consignee/shipper, emit only the `sameAs` relation. Do not repeat
   the referenced party payload. Do not infer same-as from coincidental equality.
10. Emit readable semantic enums. Do not emit MPCI party/location/contact/text/measurement/fixed
    charge/temperature codes; the deterministic projector owns those.
11. Preserve exact pre-conversion strings in evidence. Every canonical target leaf must have exactly
    one evidence record and every excerpt/raw value must occur verbatim on the cited OCR page. An
    `ocrExcerpt` is one exact contiguous substring of that page, never a reconstruction with skipped
    headers/columns or image-visible text. Preserve the immutable `source` object byte-for-value.
12. Validate with strict JSON Pydantic validation and atomically write only the assigned output. If
   a goods item has a package quantity and container allocations, the allocation quantities must
   cover exactly one emitted package level or the total of multiple emitted package levels. Every
   allocation must then carry a quantity; never emit a partial or guessed allocation merely to
   satisfy the downstream projection. When OCR has flattened a table and row/column grouping
   affects a target, inspect the retained raster to recover grouping only. Never copy a value that
   is visible only in the raster into the target; omit it and warn instead.
13. Before publication, make a second source-order completeness sweep over every explicit heading,
    party block, route/date/freight field, and cargo/container row. Account for each supported fact
    or meaningful unsupported transport fact as an emitted target, an explicit relation, or a
    warning. Routine tax/regulatory, portal/audit, administrative/security, upload, filename, and
    blockchain metadata is omitted silently unless it creates genuine ambiguity for a supported
    target. This sweep must catch omissions such as delivery agents, printed marks, type
    descriptions, seals, allocations, and localities that a field-by-field pass can overlook.

## Independent semantic-review task template

Every candidate that passes automated validation receives a second, fresh Luna Max pass before the
overseer may accept it. Give the reviewer the exact absolute work-item path, candidate-attempt path,
assigned review-output path, candidate/work-item SHA-256 values, the v2 schema/reference files, and
`src/document_ocr/label_schemas/semantic_review.py`. Instruct it:

1. Read the complete page-ordered `joinedRawText` and the complete candidate independently. Do not
   assume that structurally valid evidence means the extraction is semantically complete.
2. Reconstruct the document's supported facts from raw OCR, then compare them field-by-field against
   the candidate. Inspect the PDF/raster only when layout, row association, or headings are genuinely
   ambiguous; never use it to add or correct image-only facts.
3. Thoroughly check the truth boundary, exact evidence, missing explicit fields, wrong role/column
   assignments, party/locality/contact segmentation, route and date semantics, container/seal/goods
   relationships and exact allocations, contamination, redundancy, and whether the source contains
   one document or multiple independent transport documents. When OCR has flattened a table,
   inspect the retained raster for grouping and column association while continuing to enforce the
   OCR-only factual boundary.
4. Do not create a replacement label or offer broad stylistic suggestions. For every blocking issue,
   provide an exact actionable finding with target path(s) and contiguous raw-OCR evidence. A passing
   review must pass every check and contain no blocking findings. A correct candidate is allowed and
   expected to pass: never invent a defect merely because this is a review task. Conversely, finding
   one real defect does not end the review; finish the complete source-order and field-by-field sweep
   and report every independently grounded blocker in the same review. Within each finding, order
   `rawOcrEvidence` by the first occurrence of each excerpt in `joinedRawText`, including within the
   same page, and do not use overlapping excerpts or list a later occurrence before earlier support.
   This ordering rule applies within each finding's `rawOcrEvidence` sequence. The candidate's outer
   `evidence[]` array follows canonical target traversal and has no global OCR-order requirement.
   Do not demand warnings for routine tax/regulatory, portal/audit, administrative/security,
   upload, filename, or blockchain metadata merely because it is present in OCR or the raster.
   The lifecycle validator enforces this stronger source-order contract even when the Pydantic review
   schema alone accepts the artifact.
5. Strict-validate `BillOfLadingIndependentReview` and atomically write only the assigned review
   output. Bind it to the assigned document ID, attempt, review number, work-item SHA-256, candidate
   SHA-256, model, and Max reasoning setting.

If a review fails, preserve both artifacts and send the same source plus the exact findings to a new
Luna Max labeling worker. A corrected attempt must receive a new review from another fresh Luna Max
reviewer. Reviewer findings are quality-control sidecars, never decoder targets.

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
same source to a fresh Luna Max worker with exact failure feedback. The overseer may make only
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
- exact Terra overseer and Luna Max labeler/reviewer settings, concurrency achieved, worker and
  reviewer identities, attempts/retries/reviews, and available usage totals;
- tests/lint/type/projection results;
- contamination, redundancy, duplicate-consistency, relationship, and evidence checks;
- canonical target leaf/byte/token-size comparison to v1, clearly separated from model-quality
  claims;
- unresolved schema coverage or mapping issues; and
- EDA/report/plot and final manifest paths.

Do not call the data training-ready if any semantic failure, unresolved projection, multi-document
merge, evidence mismatch, or duplicate contradiction remains.
