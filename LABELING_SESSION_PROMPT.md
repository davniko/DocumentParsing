# MPCI Bill-of-Lading OCR-conditioned labeling session

You are the overseer for a bounded, provenance-preserving training-label run in the
`DocumentParsing` repository. Work from the repository root. Read `AGENTS.md` first and follow it
without exception.

This session labels existing GLM-OCR output. It does **not** run OCR, start Docker, load a model,
redownload data, extract native PDF text, or train a model.

## Required model and delegation contract

- The agent to which this prompt is addressed is the sole session overseer. Do not spawn another
  agent merely to act as an overseer.
- Run every labeling worker with **GPT-5.6 Luna, Extra High reasoning**. Confirm that exact worker
  selection before starting. Do not silently substitute another worker model or reasoning level.
  The overseer must explicitly select Luna with Extra High reasoning whenever it spawns a labeling
  worker.
- Prefer the native subagent launcher only when it exposes the exact worker model. If it does not,
  use the installed Codex CLI for each bounded worker with the equivalent explicit selection:

  ```bash
  codex exec --ephemeral --json --model gpt-5.6-luna \
    -c 'model_reasoning_effort="xhigh"' --sandbox workspace-write --cd <repo-root> -
  ```

  Record each child thread/process identity in run metadata. This is an alternate launch mechanism,
  not permission to substitute a model.
- Keep **six labeling subagents active concurrently** while at least six unassigned documents
  remain. When fewer than six remain, use one worker for each remaining document.
- Give each worker exactly one immutable work item representing exactly one complete multi-page
  document and exactly one candidate label. Never give one worker two documents, never reuse a
  completed worker for another document, and never let a worker label a batch.
- Workers must not create their own subagents.
- The overseer owns shared code, manifests, validation, aggregation, and EDA. Workers may write
  only their own uniquely named candidate artifact.
- If the session cannot support six concurrent labeling subagents, stop and report the platform
  limit. Do not silently reduce concurrency.

## End goal

Create one auditable KIE annotation and one canonical training pair for each eligible Bill of
Lading. One multi-page document produces one label. The model target is a sparse MPCI/CUSCAR
`documentPatch` whose field names and categorical values already match the application contract,
so a later integration does not need the current semantic prediction-to-form mapping layer.

A small deterministic application stage will still be required later to merge with user-owned
form state, hydrate submission scaffolding/defaults, assign sequential goods item numbers, resolve
mutable lookups such as UN/LOCODE, and run full submission validation. Those are deliberately not
model targets.

## Frozen contract and relevant references

Treat these Python models as the sole executable label contract for this run:

- `src/document_ocr/label_schemas/common.py`
- `src/document_ocr/label_schemas/mpci_bill_of_lading.py`
- `src/document_ocr/label_schemas/__init__.py`
- `tests/test_mpci_label_schema.py`

Treat `MPCI_BILL_OF_LADING_LABELING_REFERENCE.md` as the frozen semantic conversion authority for
the overseer and every labeling worker. It explains each target field, categorical value, permitted
mapping, unresolved code surface, evidence kind, and warning decision that the Pydantic types do not
fully describe. A worker must read it before labeling its assigned document. The reference may
narrow what a schema-valid field/code is authorized to contain, but it never broadens the Pydantic
contract or the raw-OCR truth boundary.

Read the following discovery artifacts for field meaning and relationship semantics:

- `artifacts/mpci-ai-schema/README.md`
- `artifacts/mpci-ai-schema/field-catalog.md`
- `artifacts/mpci-ai-schema/schema-layer-matrix.md`
- `artifacts/mpci-ai-schema/relationships.md`
- `artifacts/mpci-ai-schema/selects-and-code-lists.md`
- `artifacts/mpci-ai-schema/ai-genie-normalization-and-mapping.md`
- `artifacts/mpci-ai-schema/future-triton-output-contract.md`

The discovery bundle documents the entire 129-leaf form and an earlier fully hydrated payload
proposal. It is evidence for exact application paths, code meanings, and exclusions; it is **not**
the training target for this run. The later project decision embodied in
`MpciBillOfLadingLabel` is a sparse, document-owned `documentPatch`. Do not add UI toggles,
submission scaffolding, defaults, mutable lookup outputs, platform identifiers, user-only filing
choices, error-helper fields, or fields absent from the Pydantic model.

Freeze `schemaVersion = "1.0.0"` and `annotationSchemaVersion = "1.0.0"` for the complete run. Do
not change the label schema after labeling starts. If a genuine document fact cannot be represented,
omit it, add a `schema_cannot_represent` warning, and surface the coverage issue in the final report;
do not improvise a field.

Before preparing work items, run:

```bash
uv run pytest -q tests/test_mpci_label_schema.py
uv run pytest -q tests/test_mpci_labeling_reference.py
uv run ruff check src/document_ocr/label_schemas tests/test_mpci_label_schema.py \
  tests/test_mpci_labeling_reference.py
uv run mypy src/document_ocr/label_schemas
```

Stop if any command fails.

## Authoritative input and eligibility

Use only this original extraction run:

```text
artifacts/glm-ocr/pilots/blc150/runs/glm-ocr-blc-pilot150-faa3c717dbc4/
```

Use its `state.sqlite3`, `inventory.jsonl`, retained `raw-responses/`, and retained `page-images/`.
The source PDFs referenced by `documents.source_json.local_canonical_path` are local.

Explicitly ignore this partial duplicate rerun and every artifact under it:

```text
artifacts/glm-ocr/pilots/blc150/runs/glm-ocr-blc-pilot150-faa3c717dbc4-r2/
```

Do not merge runs and do not re-extract failed pages.

The original 150-document pilot was already selected as final-label `blc`, non-dummy,
fully-readable, `use_as_is`, content-deduplicated data. Verify that against:

```text
data/pilots/blc-pilot-150-faa3c717dbc4/pilot.json
data/pilots/blc-pilot-150-faa3c717dbc4/manifest.jsonl
```

Build the eligible set deterministically:

1. Require `documents.status = 'complete'`.
2. Require every page from zero through `page_count - 1` exactly once, with
   `pages.status = 'success'`, ordered by integer `page_index`.
3. Exclude a complete document if any character in any `raw_ocr_text` has a Unicode general
   category beginning with `L` and `"LATIN"` is absent from `unicodedata.name(character, "")`.
   Punctuation, symbols, and numbers do not trigger this rule.
4. Keep all remaining complete Latin-script documents. Do not perform a new subjective quality or
   document-classification pass.

The verified snapshot at prompt creation is:

- 150 selected documents / 286 selected pages;
- 141 complete documents and 9 incomplete documents;
- 276 successful pages and 10 failed pages;
- 18 complete documents excluded by the non-Latin-letter rule;
- **123 eligible documents / 234 eligible pages**.

Recompute these values. If any count or identity differs, stop and diagnose the mismatch rather
than labeling a changed surface.

Write an exclusion record for each of the 27 non-eligible documents with its deterministic reason;
never delete or alter its OCR artifacts.

## Exact training input

For each eligible document, read `raw_ocr_text` from each successful page's `result_json` in
ascending `page_index`. Construct the one and only model input as:

```python
joined_raw_text = "\n\n".join(
    f"--- PAGE {page_number} ---\n{raw_ocr_text}"
    for page_number, raw_ocr_text in ordered_pages
)
```

Hash the UTF-8 bytes of this exact string as `joinedRawTextSha256`. Do not trim, repair, reorder, or
otherwise rewrite the source text. The page delimiter is part of the training input.

For every page, copy and verify the ledger's page/extraction IDs, OCR-text hash, raw-response path
and hash, retained raster path and hash. Copy and verify document ID, run ID, source URI, local
canonical path, source hash, and page count. Resolve retained artifact paths relative to the
original run directory. A missing file or hash mismatch is a hard failure.

## The non-negotiable truth policy

The ordered raw GLM-OCR text is the **primary input and the complete boundary of permissible target
content**. The PDF and retained page image are auxiliary visual context only.

A target value is allowed only when all characters or semantic evidence needed for that value are
present in the joined raw OCR text. Every emitted target leaf must have one `FieldEvidence` record
whose `rawOcrEvidence` contains the exact pre-mapping `rawValue`, its source `pageNumber`, and a
verbatim surrounding `ocrExcerpt`. The converted value belongs only at the evidence record's
`targetPath` in `label.documentPatch`; do not overwrite the raw value with its normalized/code value.

The PDF/image may be used only to:

- understand headings, columns, table-row grouping, or page continuation;
- associate an OCR-present value with the correct OCR-present field/entity;
- choose between repeated or conflicting candidates when the selected value itself occurs in the
  raw OCR text; or
- check a suspected discrepancy and then omit the unsupported value.

The PDF/image must never be used to:

- add a field or value that does not occur in the raw OCR text;
- replace an OCR value with a visually corrected value;
- repair a missing character, digit, decimal, date component, container check digit, name, address,
  code, unit, or identifier;
- transcribe native PDF text or use another OCR engine as an additional truth source; or
- silently prefer visual content over the text target.

If the image contains `ABCD1234567` but OCR contains only `ABCD123456`, the label must not contain
`ABCD1234567`. Omit it and record `image_only_value_omitted` or
`invalid_identifier_omitted`. If page 1 OCR contains a damaged candidate and page 2 OCR repeats the
intact value, the intact page-2 value is allowed because it is still present in the raw input; use
`cross_page_resolution` evidence and cite the relevant raw excerpts/pages.

Always set `imageUse` accurately. Image inspection never relaxes the OCR-support rule.

## Absence, null, and canonical target policy

The Pydantic annotation schema permits nullable/optional fields so a worker can represent absence
without inventing a value. Use these rules consistently:

- If a scalar is not supported by OCR, set it to `null` or omit it. Never guess merely to satisfy a
  shape.
- Do not create an all-null child object or an empty placeholder row.
- `MpciBillOfLadingLabel.canonical_target()` is the training target. It removes `None` recursively,
  leaving one stable sparse representation of absence.
- No evidence record is needed for a value removed as `None`; exactly one evidence record is
  required for every non-null emitted target leaf.
- An incomplete extraction target need not be a submission-valid full form. It must be truthful,
  schema-valid, structurally consistent, and useful as a sparse document patch.

## Normalization allowed from raw OCR

Do not copy a display label into a coded application field. The target uses exact stored MPCI codes.
Only deterministic, source-supported normalization is allowed, and every non-verbatim value must
declare its precise `normalizationRule`.

For every conversion, retain each exact source string in `rawOcrEvidence[].rawValue`. If one target
is assembled from several source strings, record every contributing raw value in page/source order.
For contextual mappings such as `FREIGHT PREPAID` to `chargeCategory = "4"`, the raw phrase remains
`FREIGHT PREPAID` in evidence while `"4"` appears in the label. `rawValue` must be an exact substring
of its paired `ocrExcerpt`; neither field may contain a cleaned, paraphrased, or image-derived value.

Allowed examples include:

- trimming surrounding whitespace and joining wrapped OCR lines with a single space;
- deterministic printable-ASCII normalization required by the application schema, without adding
  missing semantic characters;
- converting an unambiguous printed date to `YYYY-MM-DD` and an unambiguous printed timezone-bearing
  timestamp to ISO 8601;
- removing OCR-present separators from an identifier or numeric value;
- removing thousands separators/unit text from an unambiguous number;
- converting an explicit country name to its stable ISO alpha-2 code;
- mapping an explicit party heading to its party-function code;
- mapping explicit `PREPAID`/`COLLECT` freight language to its MPCI payment code and supported charge
  category;
- mapping explicit `TO ORDER` negotiability language to `NEG`; and
- converting an unambiguous printed package/container/unit label to a stable code supported by the
  schema and local code-list evidence.

Forbidden normalizations include guessing an ambiguous date, inferring a missing country or port
code from general knowledge, correcting OCR with the image, silently fixing an invalid ISO 6346
identifier, geocoding, or looking up a mutable carrier/party/location identifier.

Use `contextual_code` for heading/phrase-to-code conversions, `normalized` for deterministic lexical
normalization, `cross_page_resolution` for a value selected from multi-page raw evidence, and
`verbatim` only when the emitted value exactly matches the cited OCR value. A non-verbatim evidence
record must include a normalization rule.

## MPCI-specific label rules

The Pydantic schema is authoritative. These rules clarify the main decisions:

- Use exact field casing and nesting from `MpciBillOfLadingLabel`; no aliases or semantic sidecar
  target.
- Exclude system/platform/user/UI fields such as `beginningOfMessage`, `testIndicator`,
  `delegatorIdentifier`, `shippingLineIdentifier`, `updateFiling`, toggles, error helpers, remote IDs,
  and default scaffolds.
- Preserve source order for containers, goods, packages, measurements, seals, contacts, HS codes,
  marks, references, and dangerous-goods rows.
- Consignment location qualifiers are role keys and must appear in canonical order
  `9,12,13,88,7,96`: port/place of loading, discharge, transshipment, receipt, delivery, Emirates.
- Party rows are keyed by role: `CZ` shipper, `CN` consignee, `NI` notify, `N2` second notify, `CG`
  carrier, `DDR` forwarding agent, `DP` delivery agent, and `COX` consolidator. Emit a role only when
  the raw heading/context supports it. Do not copy consignee data into notify unless the OCR text
  explicitly says the notify party is the same as consignee.
- Location `locode` may be emitted only when that locode is printed in OCR. Otherwise emit an
  OCR-supported `name` and, only when supported, `country_code`; leave UN/LOCODE resolution outside
  the model.
- Root `processingInformation.processingIndicatorDescriptionCode` is B/L negotiability
  (`NON`/`NEG`), not the consignment/declaration processing code.
- Treat B/L issue date and shipped-on-board date as independent printed facts. Preserve both when
  OCR supports them, including when `shippedOnBoardDate` precedes `billOfLadingIssueDate`. The
  application discovery bundle's contrary form-validation rule is not an annotation constraint.
- Map explicit basic-freight `FREIGHT PREPAID`/`FREIGHT COLLECT` text to a charge-payment row with
  `chargeCategory = "4"` and `paymentArrangement = "P"`/`"C"`, respectively, as specified by the
  local schema evidence. Do not invent a category for an unrelated charge.
- Container identifiers must be valid ISO 6346 identifiers after allowed OCR-text-only lexical
  normalization. If the raw candidate remains invalid, omit it and warn; never repair its check digit
  from the image.
- `containerVerifiedGrossMass` is VGM only. Do not put an ordinary cargo gross weight there unless
  the OCR explicitly identifies it as verified gross mass. Ordinary gross/net/volume facts belong in
  goods `measurements` (`AAB` gross, `AAA` net, `ABJ` volume).
- Do not duplicate a document-level aggregate package/weight total into each goods item or container.
  Allocate only when the raw text supports the allocation.
- Every `splitGoodsPlacement` container identifier must exactly match one emitted
  `containerInformation` identifier. Emit no relation when association is uncertain. If one package
  total is fully allocated across placements, placement quantities must sum to it.
- Represent one commodity distributed across several containers as one goods item with several
  placements when the raw OCR supports that relationship. Do not duplicate the commodity merely to
  create one row per container.
- `goodsItemNumber` is deliberately absent; the deterministic assembler assigns it from array order.
- `AAA`/`AAI`, party roles, location qualifiers, measurement attributes, contact types, and other
  structural codes still require contextual OCR evidence even though their code characters may not
  literally appear in the document.
- HS, package, dangerous-goods, service, handling, seal, temperature, and other code fields must be
  omitted when their conversion is unclear or unsupported. Use `unsupported_or_unclear_code` rather
  than a best guess.
- When a genuine OCR fact has no semantically matching Pydantic target field, emit a
  `schema_cannot_represent` warning with `targetPath = null`. Never attach it to a merely similar
  field (for example, an Egyptian ACID cargo-tracking identifier is not an HS/customs-goods code).
- Never infer VGM, dangerous-goods status, a party ID, a carrier ID, a UN/LOCODE, a filing action, or
  another schema field merely because it is common on Bills of Lading.

## Work-item and output layout

Create a new run directory without modifying extraction artifacts:

```text
artifacts/kie-labels/mpci-bl-pilot-v1/
  run-metadata.json
  eligibility.jsonl
  exclusions.jsonl
  work-items/<document_id>.json
  candidates/<document_id>.json
  validated/<document_id>.json
  training/records.jsonl
  validation/report.json
  validation/report.md
  eda/summary.json
  eda/report.md
  eda/tables/*.csv
  eda/plots/*.png
  manifest.json
  manifest.json.sha256
```

Use atomic write-then-rename publication for each file. Only the overseer may write shared JSONL,
reports, or manifests. Publish `manifest.json` and its digest last, only after validation and EDA are
complete. The manifest must hash every durable artifact and record source run/config/inventory
identity, schema versions, eligibility identity, exact counts, model/reasoning setting, and validation
status.

Each work item must contain exactly one document's immutable source/provenance object and exact
`joinedRawText`. Do not place another document's text or paths in the same file.

Workers must copy the work item's source/provenance object mechanically into the candidate; they
must not retype hashes, identifiers, counts, or paths. Before publication, assert exact structural
equality between the work-item provenance and `candidate.source`, in addition to strict Pydantic
validation.

Each candidate and validated annotation must parse as
`MpciBillOfLadingAnnotation`. Candidate workers set `reviewStatus = "candidate"`; only the overseer
may publish `reviewStatus = "validated"`. `warnings` and `reviewNotes` are diagnostics and never part
of the model target.

Each line of `training/records.jsonl` must contain one eligible validated document, its exact joined
raw OCR input, its canonical sparse target from `canonical_target()`, and a pointer/hash to the
validated annotation. Preserve deterministic document order. Do not include images, warnings,
evidence, or review notes inside the model input or target.

## Exact worker prompt contract

For each new worker, supply only the following bounded assignment, with concrete paths substituted:

> Label exactly one multi-page Bill-of-Lading document. Read `AGENTS.md`, the frozen Pydantic label
> models, `MPCI_BILL_OF_LADING_LABELING_REFERENCE.md`, and only `<work-item-path>`. The raw OCR text
> in that work item is the authoritative target
> boundary. The PDF/page images listed there are auxiliary for structure and grouping only: never add
> or correct a value from them unless the chosen value already occurs in raw OCR. Apply the field
> meanings, authorized categorical mappings, unresolved-code prohibitions, absence, normalization,
> ordering, relationship, and evidence rules in `MPCI_BILL_OF_LADING_LABELING_REFERENCE.md` and
> `LABELING_SESSION_PROMPT.md`. Copy the immutable source/provenance object mechanically from the
> work item and assert exact equality with `candidate.source`; never retype its hashes or IDs. Write
> exactly one candidate
> `MpciBillOfLadingAnnotation` atomically to `<candidate-path>`, with one evidence record per emitted
> target leaf. In every evidence record retain the exact pre-mapping raw OCR value, page number, and
> surrounding excerpt under `rawOcrEvidence`; the mapped/converted value remains in the label at
> `targetPath`. Set `reviewStatus="candidate"` and validate with Pydantic before returning. Do not edit
> any shared file, schema, manifest, source artifact, or other document's output. Do not label another
> document and do not spawn agents. Return the document ID, output path, validation result, warnings,
> and nothing resembling a second label.

The overseer must not broaden a worker assignment through a follow-up. If a candidate is invalid or
semantically inadequate, preserve it for audit and assign that same document to a fresh one-document
worker, then retain the accepted attempt identity in metadata.

## Overseer validation of every pair

After all candidates exist, validate **every** candidate; sampling alone is not sufficient. Perform
both automated checks and a quick semantic review of each document/input/label pair.

For each pair, prove:

1. Exactly one candidate exists for the eligible document and no candidate exists for an excluded
   document.
2. Source IDs, paths, page count/order, hashes, and joined-text hash match the original ledger/files.
3. Strict Pydantic validation succeeds and the canonical target is nonempty, sparse, and serializes
   with stable schema key order.
4. The evidence target-path set exactly equals the emitted target-leaf set, with no duplicate,
   missing, or unexpected evidence paths.
5. Every `rawOcrEvidence[].rawValue` is an exact substring of its paired verbatim `ocrExcerpt`; both
   occur on the declared source page, entries remain in page/source order, and the evidence preserves
   the pre-mapping value rather than the converted label value. Ellipsized, paraphrased, normalized,
   or image-transcribed evidence is invalid.
6. Every emitted value is supported by cited raw OCR under a declared allowed normalization; image
   content has not added or corrected target data.
7. Multi-page repetitions/conflicts were resolved only to a value present in raw text, page order is
   preserved, and continuation rows are grouped correctly.
8. Codes have the correct MPCI meaning, arrays preserve the required order, identifiers/relations are
   valid, no orphan container reference exists, and aggregates were not duplicated without support.
9. Missing/ambiguous values were omitted rather than hallucinated and all material ambiguity is
   surfaced as a warning or `needs_review`.
10. The canonical training record's source text and target exactly match the validated work item and
    annotation; no diagnostics leaked into the target.

Only pairs passing all checks become `validated` and enter `training/records.jsonl`. A pair requiring
human judgment remains `needs_review` and is excluded from training, with an explicit reason. Do not
call the run complete while an eligible document is silently missing.

The validation report must give total/pass/fail/needs-review counts, every failure by document ID and
rule, field-level presence counts, warning counts, relationship violations, and proof that the final
training IDs equal the validated IDs.

## EDA after validation

Run EDA only after validated artifacts and `training/records.jsonl` are frozen. Use `uv` for any
required dependency; never use ad-hoc `pip`. Use a deterministic plotting seed and a headless backend.

Produce aggregate tables and readable PNG plots covering at least:

- document and page counts, plus page-count distribution;
- raw OCR character/line length per page and document;
- canonical target JSON length and emitted-leaf count;
- input length versus target length and leaf count;
- field-path presence/absence rates, including a useful top-level/group heatmap;
- counts per document for containers, goods, packages, parties, locations, seals, temperatures, HS
  codes, measurements, placements, and dangerous-goods rows;
- distributions of stable categorical codes such as party function, location qualifier, freight
  arrangement, measurement attribute/unit, package type, and container type;
- relationship complexity and validation results, including the orphan-reference rate, which must be
  zero for validated labels;
- evidence kind, image-use category, cross-page resolution, warning code, and normalized/verbatim
  proportions;
- which page numbers support emitted fields and how multi-page documents differ from one-page
  documents; and
- pilot lineage and triage category joined from the pilot manifest, without rerunning classification.

Write machine-readable aggregates to `eda/summary.json` and `eda/tables/`, plots to `eda/plots/`, and
interpretation to `eda/report.md`. Use only aggregate labels in plots: do not render raw OCR, names,
addresses, contact details, B/L numbers, container identifiers, filenames, S3 keys, or other customer
data in chart text. Call out sparsity, long-tail codes, multi-page effects, likely annotation
challenges, and limitations of this 123-document pilot. Do not make model-quality claims from label
EDA alone.

## Completion report

Finish with a concise evidence-backed report containing:

- exact eligible/excluded/validated/needs-review/training counts and pages;
- confirmation of the six-at-a-time, one-worker/one-document contract, the active session
  overseer, and the GPT-5.6 Luna Extra High worker setting;
- schema/test/lint/type-check results;
- full-validation results and any retried or excluded documents;
- output paths and manifest SHA-256;
- a summary of important EDA findings and plot locations;
- any schema coverage warnings requiring a later product decision; and
- explicit confirmation that OCR, Docker, S3 download, model training, and image-derived target
  correction were not run.

Do not start model training in this session.
