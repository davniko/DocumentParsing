# S3 input audit and local extraction-corpus contracts

Audit date: 2026-08-10

This document records the source audits and completed local GLM-OCR input snapshots. Each initial
S3 audit was read-only. Approved PDFs were then downloaded conditionally, content-hashed, retained
locally, and inspected with PDFium. No PDF was sent to a model, and no model, vLLM server, Docker
service, or GPU was started.

## Source and bucket properties

- Bucket: `runpod-mpci-docparsing-workspace`
- Raw prefix: `data-processing/NEW-dataset/01_raw/`
- Region: `us-east-1`
- Default encryption: SSE-S3 (`AES256`)
- Storage class: all audited raw objects are `STANDARD`
- Request payment: bucket owner
- Lifecycle configuration: none
- Versioning: not configured; audited objects have the mutable `null` version ID

The null version IDs remain a blocker for direct S3 extraction. The approved local path does not
weaken that contract: it conditionally downloads the audited ETag, computes a full SHA-256, and
commits a manifest-last local snapshot. GLM-OCR consumes that immutable local snapshot instead of
the mutable S3 keys.

## Raw inventory

| Measure | Value |
| --- | ---: |
| Objects | 3,028 |
| PDF objects | 3,017 |
| Non-PDF metadata objects | 11 |
| PDF bytes | 1,862,818,332 |
| Largest PDF | 17,294,619 bytes |
| Duplicate PDF filenames | 0 |

The PDF population consists of 1,755 objects under the 2026-02-16 ACI BLC batch and 1,262
objects under the existing classified-document prefix.

## Authoritative classification artifacts

The initially supplied manifest at
`data-processing/dataset/02_cleaned/manifest.jsonl` belongs to `OLD-dataset`. Its 816 document
identities have zero filename or UUID matches against the proposed NEW raw prefix and it must not
be used to select these PDFs.

The matching post-filter manifests are:

1. `data-processing/NEW-dataset/dataset/03_filtered/manifest.jsonl`
   - SHA-256: `82585b1b18d76cd8fb9d48b3bf2778ea9a82cbe6ab0647cfe96dbcbaed5b2842`
2. `data-processing/NEW-dataset/dataset_aci_blc_new_batch_2026-02-16/03_filtered/manifest.jsonl`
   - SHA-256: `95dec3d694dc8911a5f2d73d555068e40bf561016c885d7a10b58f82955b66f3`

Both manifest objects are also unversioned. Their hashes therefore form part of the selection
evidence and must be revalidated immediately before staging.

The `03_filtered` artifacts are required instead of the corresponding `02_cleaned` artifacts:
the filter has removed dummy and other explicitly excluded documents and records the resolved
classification in `final_classification_label`.

### Bucket-wide classification-manifest search

A read-only search of the complete `data-processing/` prefix on 2026-08-10 examined 33,989 S3
objects totaling 35,940,576,211 bytes and identified 210 manifest-, classification-, filter-, or
metadata-like artifacts. The relevant classification lineage consists of three disjoint runs:

| Stage and source lineage | Documents | Page rows | Dummy documents | Resolved final label |
| --- | ---: | ---: | ---: | --- |
| OLD `02_cleaned` | 816 | 1,612 | 218 | No |
| Existing NEW `02_cleaned` | 1,260 | 3,549 | 154 | No |
| New ACI batch `02_cleaned` | 1,944 | 4,326 | 60 | No |
| **Three-way initial union** | **4,020** | **9,487** | **432** | **No** |
| OLD `03_filtered` | 590 | 1,280 | 0 | Yes |
| Existing NEW `03_filtered` | 1,106 | 3,315 | 0 | Yes |
| New ACI batch `03_filtered` | 1,882 | 4,192 | 0 | Yes |
| **Three-way final union** | **3,578** | **8,787** | **0** | **Yes** |

The additional initial manifests are:

- `data-processing/NEW-dataset/dataset/02_cleaned/manifest.jsonl`, SHA-256
  `8c91e29fe65838b72e77dceb3505c33213db2649c4f9ba699aea1d90deb03111`;
- `data-processing/NEW-dataset/dataset_aci_blc_new_batch_2026-02-16/02_cleaned/manifest.jsonl`,
  SHA-256 `6926bf1afffa28934e5a8df1e5d8ea7cae79344149eb5d7da5f0daced0c63370`.

The three-way rows are a verified union, not a single object currently stored on S3. Across the
three lineages, their `doc_id` values, original PDF paths, and full filenames are disjoint. The
final-label population is 169 AWBC, 2,398 BLC, 91 COO, 106 invoice, 99 packing list, 696 SWB,
2 unclassified, and 17 unknown documents.

The newer candidate at
`data-processing/dataset/03_filtered_complete/manifest.jsonl` is not a complete three-way union.
It was created on 2026-03-16, contains 2,472 documents and 5,472 page rows, and has SHA-256
`6e1867c73d8eefc7d0db411d169b36187e43883f3a78b77d27f0f88866c5043c`. Its merge report names
only the OLD final manifest and the new ACI-batch final manifest. A byte-level check confirms that
the object is their exact concatenation, with no new rows or deduplication. It omits all 1,106
documents from the existing NEW final manifest: 118 AWBC, 650 BLC, 4 invoice, 6 packing-list, and
328 SWB documents.

Other similarly named artifacts do not close that gap:

- `NEW-dataset/dataset/02_cleaned_BACKUP/manifest.jsonl` is a strict 253-document subset of the
  current 1,260-document existing-NEW initial manifest;
- the OLD `manifest copy - backup.jsonl` is a stale 592-document final variant with two
  undocumented extra rows; the later aggregate deliberately uses the current 590-document file;
- per-folder raw `manifest.json` files contain upload/source identifiers but no classifier,
  dummy, triage, or resolved-label decision;
- image, annotation, extraction, and training manifests are downstream subsets or enrichments,
  not broader classification inventories;
- no newer classification manifest exists after `03_filtered_complete`; later JSONL artifacts are
  training-run outputs.

Coverage against source files is:

| Population | Meaningful unique PDFs | Three `02_cleaned` manifests | Three `03_filtered` manifests | Stored `03_filtered_complete` |
| --- | ---: | ---: | ---: | ---: |
| Prepared local source snapshots | 3,471 | 3,471 | 3,304 | 2,330 |
| NEW raw plus real OLD processing PDFs on S3 | 3,841 | 3,826 | 3,446 | 2,344 |

For the prepared local population, the 167 documents absent from the three-way final union are all
explicitly dummy in an initial manifest; therefore every retained local source has classification
and dummy evidence. The unprocessed OLD prefix is a mirrored S3 path rather than a fourth
classification run, so its 294 files join to OLD evidence by exact full filename instead of exact
S3 key.

Across the broader meaningful raw population, 15 PDFs have no `02_cleaned` row: 6 NEW files
(3 `standard_blc`, 2 `standard_mbl`, and 1 `standard_swb`) and 9 OLD files (1 BLC, 1 COO, 6 MPCI,
and 1 packing list). They represent 12 distinct byte payloads because three pairs are exact
duplicates. None shares a UUID with any classified document, so a label cannot be safely inherited.
Conversely, the three initial manifests contain 194 references whose declared raw key is absent
from the audited raw prefixes: 1 OLD, 4 existing-NEW, and 189 ACI-batch references.

The most complete trustworthy classification catalog is therefore constructed as a lineage-aware
union: retain all three `02_cleaned` records for initial classifier, dummy, and triage evidence,
then left-join the three matching `03_filtered` records for resolved labels. A missing final row is
resolved against the initial dummy/triage fields and filter reports rather than being assigned a
label. The stored object named `03_filtered_complete` is not used alone.

## Completed classification catalog

The catalog is configured by `configs/classification_catalog.yaml` and stored under:

`data/catalogs/classification-catalog-d52a5273b036/`

It contains one row per logical document, not one row per page. Exact source line numbers point
back to every page row in the copied initial and final manifests. Each row retains the full initial
classifier, dummy, and triage objects; the final label when one exists; exact dropped, relabeled,
and review CSV evidence; every matching primary/mirror S3 alias; and every matching local snapshot
record with its full PDF SHA-256 and snapshot commit provenance.

| Catalog status | Documents | Meaning |
| --- | ---: | --- |
| `retained_final` | 3,578 | Present in the matching final manifest |
| `excluded_reported` | 434 | Absent from final with an exact dropped-report row |
| `missing_final_without_drop_report` | 8 | Absent from final without a dropped-report reason |
| `never_classified` | 15 | Meaningful primary raw PDF with no initial-manifest row |
| **Total** | **4,035** | Complete logical union |

The 434 reported exclusions comprise 431 `Dummy document` rows and 3 `LLM triage: exclude` rows.
The initial manifests themselves mark 432 documents as dummy; one dummy is among the triage
exclusions. The eight unreported final omissions are all non-dummy OLD documents. Six have a
`Very high LLM confidence; spot-check` review row; two have neither a drop nor review row. They are
intentionally unresolved in the catalog.

Final retained labels are:

| Final label | Documents | Page rows |
| --- | ---: | ---: |
| `blc` | 2,398 | 5,567 |
| `swb` | 696 | 2,209 |
| `awbc` | 169 | 296 |
| `inv` | 106 | 229 |
| `pl` | 99 | 172 |
| `coo` | 91 | 142 |
| `unknown` | 17 | 165 |
| `unclassified` | 2 | 7 |
| **Total** | **3,578** | **8,787** |

The copied initial manifests contain 4,020 documents and 9,487 page rows. They resolve against
3,841 meaningful primary raw S3 PDFs totaling 2,283,092,699 bytes. Exactly 3,826 classified
documents have their declared primary source; 194 manifest documents have no current source
object, while 15 primary raw PDFs have no classifier row. The 294-object OLD unprocessed mirror
adds 67,813,377 bytes of alternate S3 spelling, for 4,135 total S3 aliases.

The three local snapshots contribute 3,593 aliases attached to 3,471 catalog documents. Those
aliases represent 3,325 unique full-PDF SHA-256 values: 122 aliases are duplicate paths for the
same logical document, and a further 146 different filenames share exact PDF content. The catalog
does not destructively collapse either identity class.

The immutable output contract is:

```text
<catalog>/
  sources/<lineage>/{initial-manifest,final-manifest,...}  # 13 exact S3 artifacts
  raw-inventory.jsonl                                      # 4,135 S3 aliases
  catalog.jsonl                                            # 4,035 document rows
  statistics.json                                          # recomputed breakdowns
  *.sha256                                                 # digest sidecars
  catalog.json                                             # published last
```

The content digests are:

- catalog SHA-256: `d52a5273b036a96d004cbe4d653485d78c9c2523499b40dfa4ff8719742b9a37`;
- raw-inventory SHA-256:
  `7f8176f2f4272ee83b1433004d539ae3dff6f04bc9d4fb41af8021123b6130cd`;
- statistics SHA-256:
  `b2699f3d67f42c23ccdc53f9ac68a84aebb8a2a8875846c1bb6c3c802dd09027`.

Materialization contacts S3 to conditionally fetch the 13 content-pinned evidence objects and
freeze the current raw inventory. Verification is offline and re-hashes every local snapshot PDF,
replays all joins from the pinned source copies, recomputes the statistics, and rejects any extra,
missing, or changed catalog file:

```bash
uv run document-ocr-classification-catalog materialize \
  --config configs/classification_catalog.yaml
uv run document-ocr-classification-catalog verify \
  --config configs/classification_catalog.yaml
```

## Selection rule

A raw PDF is selected only when all of the following hold:

1. Its unique filename exactly matches the basename of `original_pdf_path` in one of the two
   matching `03_filtered` manifests.
2. The manifest row is the single first-page row for its document.
3. `final_classification_label` is `blc` or `swb`.
4. The raw key's size and ETag still match the values captured when the selection is staged.

The canonical candidate-selection digest produced from source bucket, key, size, ETag,
last-modified time, final document type, classification document ID, classification-manifest key,
and page count is:

`818dff0fc583dc91ca3062c9a2eb5c398517f462f974d9bbaba7b046a74bc8d2`

## Selected workload

| Final type | Selected PDFs | Classifier pages | Actual PDFium pages | Ready PDFs | Ready pages | Bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `blc` | 2,061 | 4,865 | 4,858 | 2,060 | 4,857 | 1,606,498,057 |
| `swb` | 668 | 2,158 | 2,158 | 668 | 2,158 | 180,371,674 |
| **Total** | **2,729** | **7,023** | **7,016** | **2,728** | **7,015** | **1,786,869,731** |

The classifier counts and actual PDFium counts agree for 2,728 PDFs. One malformed BLC is retained
as an original but quarantined from extraction, as detailed below.

## Explicit exclusions

| Reason | PDFs | Bytes |
| --- | ---: | ---: |
| Dropped by the classifier's document filter | 154 | 50,254,658 |
| Final classification is not `blc` or `swb` | 128 | 20,974,122 |
| No matching cleaned-manifest evidence | 6 | 4,719,821 |
| **Total** | **288** | **75,948,601** |

## Completed local snapshot

The approved snapshot is configured by `configs/s3_snapshot.blc_swb.yaml` and stored under:

`data/snapshots/new-dataset-blc-swb-818dff0fc583/`

Its immutable commit has:

- selection SHA-256:
  `818dff0fc583dc91ca3062c9a2eb5c398517f462f974d9bbaba7b046a74bc8d2`;
- snapshot manifest SHA-256:
  `f51bf86e1afdcd96fce1595bec0b8f12b3a6d32050b1aa263c669ef82052adba`;
- 2,729 retained originals totaling 1,786,869,731 bytes;
- 2,728 extraction-ready hard links totaling 7,015 pages;
- one content-pinned quarantine record.

The structure is:

```text
<snapshot>/
  files/<blc|swb>/<raw-prefix-relative-key>       # every exact downloaded original
  extraction/<blc|swb>/<raw-prefix-relative-key>  # ready originals, hard-linked
  classification/*.jsonl                          # exact pinned classifier manifests
  selection.jsonl                                 # canonical audited S3 selection
  manifest.jsonl                                  # S3, classifier, local hash, path, page status
  snapshot.json                                   # published last; complete-snapshot marker
  download-state/                                 # crash-safe per-object resume evidence
```

Files are not streamed and deleted. Every selected original remains in `files/` for direct audit.
The ready `extraction/` entries are hard links to the same inode and consume no second PDF copy.
Every manifest row records the original S3 bucket/key/ETag/last-modified time, classification
document and manifest hash, local relative paths, complete PDF SHA-256, classifier page count,
actual PDFium page count, and extraction/quarantine status.

The live staging probe also recorded 3,017 raw PDFs, 160 raw PDFs without post-filter-manifest
evidence, and 131 post-filter manifest documents whose basename is absent from this raw prefix.

### Quarantined malformed PDF

The following selected BLC is retained but intentionally absent from `extraction/blc`:

`doc-classification-data (processing)/aci_blc/pdf/2026-02-01_382ae21e-1954-4751-8de7-87dcb67d2976.pdf`

- SHA-256: `54059da59c2198c9b77386897bf91f90311f021bfb0294118ef165bb8b7513b2`
- classifier manifest: 8 pages
- PDFium: 1 page
- pypdf recovery: 4 pages with broken object-offset warnings
- file structure: two EOF markers

Processing it with the production PDFium path would silently lose pages. The quarantine declaration
pins the source hash and both page counts; any future repaired/replaced file fails verification and
must be reviewed rather than remaining silently excluded.

Re-verify every local byte, page count, and hard-link identity without contacting S3:

```bash
uv run document-ocr-snapshot verify --config configs/s3_snapshot.blc_swb.yaml
```

The complete offline verification passed on 2026-08-10. Separate extraction inventories also
passed and joined exactly to the ready snapshot records. The original, pre-merge inventories were:

- BLC: 2,060 documents, inventory SHA-256
  `ee0eb36a858a1cde28ca788c1b2513a625de707d6db310eb2b29c1fed13fbf79`;
- SWB: 668 documents, inventory SHA-256
  `741f5f20a7843e661519c9ac1fe13d6f5ece85bc9a1d0b9f37e91b1729eec8a6`.

## OLD-dataset audit and transfer

The requested OLD source locations are:

- processing prefix:
  `data-processing/OLD-dataset/01_raw/doc-classification-data (processing)/`;
- unprocessed BLC prefix:
  `data-processing/OLD-dataset/01_raw/blc - unprocessed/`.

The containing `01_raw/` prefix has 2,528 objects totaling 881,046,908 bytes. The requested
processing subdirectory accounts for 2,222 objects and 806,001,837 bytes. It contains 952 tiny
`__MACOSX` resource-fork objects and 21 `.DS_Store` files; neither is document input. After removing
that metadata, it contains 824 PDFs totaling 420,274,367 bytes and 425 non-PDF source files totaling
385,334,121 bytes.

The 425 non-PDF files are 354 JPEGs, 7 PNGs, 3 TIFFs, 2 GIFs, and 59 XLSX workbooks. They have unique
identities rather than being alternate copies of the PDFs. They remain on S3 and are deliberately
outside this transfer: the established renderer and page-provenance contract is PDF-only. Images,
multi-frame TIFF/GIF files, and workbooks need an explicit input/page semantics contract before
they can be added without silently changing dataset meaning. The 12-PDF `fiata/` sibling is also
outside the specifically requested processing subdirectory.

### OLD classifier lineage

`data-processing/dataset/02_cleaned/manifest.jsonl` is the complete OLD PDF inventory: 816
documents and 1,612 pages. The authoritative curated artifact is
`data-processing/dataset/03_filtered/manifest.jsonl`, SHA-256
`dd172d58c0e93295a0654b7e4e23b2da50cc10d7fd8884bcd57d5fee8f61f1de`. It contains 590 documents
and 1,280 pages after dummy removal and 141 recorded relabels. Selection uses its
`final_classification_label`, not the original folder name.

A byte-pinned local copy of the originally requested `02_cleaned` artifacts is retained at:

`data/manifests/old-dataset-02-cleaned-c6ae25242e91/`

- `manifest.jsonl`: 4,592,560 bytes, SHA-256
  `c6ae25242e9144216f2ef8fc7f28d1cfdade5a259a456fdca0e704a2b155d4af`;
- `classification_index.csv`: 96,644 bytes, SHA-256
  `58acfb024a093f497d0125bb3f87afb276e3cf121477e0584b8074ad3d45a7bc`;
- `bundle.json`: exact S3 key, region, ETag, last-modified time, checksum metadata, byte count,
  local filename, and SHA-256 for both downloads.

The label fields do not all mean the same thing:

| Artifact field | Meaning |
| --- | --- |
| `classification_label` | Initial source/folder label; not a resolved document-type decision |
| `classification_llm.predicted` | Classifier prediction before dummy filtering and manual/curated resolution |
| `dummy.is_dummy` | Document-removal evidence used by the later filter |
| `final_classification_label` | Resolved label; absent from `02_cleaned` and present in `03_filtered` |

At document level, the initial manifest has 151 source-labeled BLCs but 421 classifier-predicted
BLCs. It also marks 218 of all 816 documents as dummy. The final manifest retains 590 documents,
excludes those 218 dummies plus the eight MPCI documents, and records 141 initial-to-final label
changes. Its final population includes 242 BLC references; 241 have a matching OLD raw PDF and the
remaining reference is the missing OLD source described below. The companion
`classification_index.csv` contains only `image_path` and the initial `classification_label`; it
does not contain the classifier prediction, dummy decision, or final label.

This OLD manifest has zero exact-filename and zero UUID overlap with the 2,729-document NEW
snapshot, so it must not be used to reclassify the NEW BLC/SWB population.

Five concrete document types are extraction-ready:

| Final type | PDFs present | Pages | Bytes |
| --- | ---: | ---: | ---: |
| `awbc` | 46 | 79 | 12,646,613 |
| `blc` | 241 | 510 | 152,812,529 |
| `coo` | 90 | 140 | 96,340,252 |
| `inv` | 102 | 215 | 38,668,028 |
| `pl` | 91 | 161 | 41,309,160 |
| **Total** | **570** | **1,105** | **341,776,582** |

All 570 downloaded PDFs match the classifier page counts under PDFium; none is quarantined. The
curated manifest contains one additional three-page BLC reference whose original is absent from
the OLD raw prefix:

`2024-11-19_5bf4a77b-d566-4d9e-b7bd-cd2d2153aca4.pdf`

The only bucket object with that basename is under a NEW-dataset calibration path, so it was not
silently imported across dataset boundaries. Final `unknown` and `unclassified` labels are not
configured as document types.

The immutable curated snapshot is configured by `configs/s3_snapshot.old_classified.yaml` and
stored at:

`data/snapshots/old-dataset-classified-fa3419bdc44d/`

Its selection SHA-256 is
`fa3419bdc44d61ff9dffc795047e10de9b0a4f9e10536113a482567c511e8b94`; its completed manifest
SHA-256 is `f7c106350ad10c191c79a62761af755e7b84aaa7c5ab42dfcbf03d43b7d3433d`.

### `blc - unprocessed` identity result

The separate prefix contains 294 PDFs, 487 pages, and 67,813,377 bytes. Every filename follows the
date-plus-UUID convention. Against the previously downloaded 2,729 NEW originals it has:

- zero exact-filename matches;
- zero UUID matches.

All 294 names, sizes, and ETags exactly mirror
`doc-classification-data (processing)/unclassified/pdf/`. The 112 documents that are also selected
as BLC by the curated OLD classifier were downloaded through both S3 provenance paths; all 112
match on full local SHA-256, byte count, and PDFium page count.

The newly pinned label comparison shows that the `unclassified: blc` transfer mapping is broader
than the curated classification. The authoritative final outcomes for these 294 files are:

| Final outcome | Documents | PDFium pages |
| --- | ---: | ---: |
| `blc` | 112 | 243 |
| `awbc` | 7 | 32 |
| `inv` | 1 | 1 |
| `pl` | 2 | 5 |
| `unclassified` | 2 | 7 |
| `unknown` | 3 | 7 |
| Absent from final manifest (all marked dummy) | 167 | 192 |
| **Total** | **294** | **487** |

The pre-filter classifier predicted 262 of the 294 as BLC, but 149 of those predictions were later
removed as dummy. Only the 112 rows with final label `blc` are curated BLCs, and all 112 are already
present in the curated OLD BLC snapshot. Therefore, this prefix contributes alternate S3
provenance for final BLCs but no additional unique final-BLC document.

The complete pre-filter manifest provides page membership. The snapshot configuration records the
explicit `unclassified: blc` mapping while retaining the original classification document IDs and
manifest hash; it does not rewrite the evidence. The snapshot is configured by
`configs/s3_snapshot.old_blc_unprocessed.yaml`, stored at
`data/snapshots/old-dataset-blc-unprocessed-f6d32e628074/`, and has:

- selection SHA-256:
  `f6d32e628074cbb4422bcb4b93c76b6c88b50a267a3665a3f9076283a040f85c`;
- manifest SHA-256:
  `2d6cff9cb504876e9c81724b109196e42bc2bf08d381dd239000883d8be8ead4`;
- 294 extraction-ready documents and zero quarantines.

Here, extraction-ready means structurally valid PDF input; it does not override the later semantic
classification. The snapshot remains an immutable source/audit capture rather than being deleted
or relabeled in place.

Re-verify both OLD snapshots without S3 access:

```bash
uv run document-ocr-snapshot verify --config configs/s3_snapshot.old_classified.yaml
uv run document-ocr-snapshot verify --config configs/s3_snapshot.old_blc_unprocessed.yaml
```

## Combined BLC corpus

The active BLC config now consumes a manifest-last union of three verified sources:

1. 2,060 ready NEW BLCs;
2. 241 curated OLD BLCs;
3. 294 PDFs from `blc - unprocessed`.

These sources provide 2,595 aliases. Exact filename is the deduplication key, as requested; 112
identical OLD aliases collapse to one corpus document. Distinct filenames remain distinct even if
their PDF bytes or UUID component repeat. A same-filename SHA, size, or page-count conflict fails
the build.

Because all 294 unprocessed files were included before the final-label comparison above, this
combined view is now explicitly **provisional**. It contains 182 documents that are not curated
BLCs: 167 filtered dummies, 7 AWBCs, 1 invoice, 2 packing lists, 2 unresolved unclassified files,
and 3 unknown files. No OCR run should use this BLC config until its membership policy is resolved.
Filtering the unprocessed source to final-label BLC only would remove 182 documents, 244 pages, and
27,088,368 logical bytes. The resulting projection is 2,301 unique BLC filenames, 5,367 pages,
1,757,184,746 logical bytes, and 2,413 provenance aliases.

The completed corpus is configured by `configs/corpus.blc.local.yaml` and stored at:

`data/corpora/blc-combined-6c9854ca458c/`

| Measure | Value |
| --- | ---: |
| Unique PDF filenames | 2,483 |
| Source aliases | 2,595 |
| Deduplicated aliases | 112 |
| PDFium pages | 5,611 |
| Logical PDF bytes | 1,784,273,114 |
| Manifest SHA-256 | `6c9854ca458cd66d22c2ee608e892db3819f0c27c63d19f5d91ecc7b619e5e36` |

Corpus PDFs are grouped by filename year under `files/blc/`. Every entry is a hard link to a
verified snapshot original, so the combined view adds directory entries and manifests rather than
a second 1.784 GB payload copy. Each corpus manifest row retains the complete nested snapshot
record for every alias: S3 key and metadata, local SHA-256, classifier document and manifest,
source snapshot commit, and page counts.

Rebuild/resume or verify the corpus without contacting S3:

```bash
uv run document-ocr-corpus materialize --config configs/corpus.blc.local.yaml
uv run document-ocr-corpus verify --config configs/corpus.blc.local.yaml
```

### Duplicate-name, UUID, and content audit

Across the six prepared active inventories, each type independently has zero repeated full
date-plus-UUID filenames. The combined BLC view already collapses 112 same-name, byte-identical
source aliases. Across type boundaries there are 10 repeated full filenames: 7 BLC/AWBC, 2 BLC/PL,
and 1 BLC/INV. Every pair has identical SHA-256 and size; there are zero same-name content
conflicts. All 10 arise from the provisional unprocessed-to-BLC mapping, and the final manifest
assigns them to the non-BLC type. A final-label-only BLC projection removes those cross-type
duplicates without deleting the correctly labeled AWBC, invoice, or packing-list source.

The UUID component alone is not unique in the source data. There are 32 UUIDs reused under 64
different date-plus-UUID filenames: 28 groups within BLC, 3 within SWB, and 1 spanning BLC/SWB.
Every UUID-reuse pair is byte-identical. More generally, 86 SHA-256 values are shared by 232
distinct full filenames, which would represent 146 redundant document-level OCR calls if content
hash were chosen as the inference identity.

Those different full filenames have not been destructively collapsed: they are distinct source
records and may represent separate submissions. A future content-deduplicated extraction view can
infer once per SHA-256 while retaining every filename as a provenance alias, but that is a broader
identity policy than same-name deduplication. The complete machine-readable group listing and label
projection are in `data/manifests/old-dataset-02-cleaned-c6ae25242e91/dedupe-label-audit.json`,
SHA-256 `0b9d152cd0da7ab63f523b04ad53bed68f893d2c7cdd5b3284e99a45c5d362e9`.

## Quality-filtered BLC pilot

The first OCR test uses a separate catalog-derived view configured by
`configs/pilot.blc150.yaml` and committed under
`data/pilots/blc-pilot-150-faa3c717dbc4/`. Selection requires all of the following exact evidence:

- retained final label `blc`;
- at least one extraction-ready local snapshot alias;
- `dummy.is_dummy=false`;
- `triage.requires_augmentation=false`;
- `readability=fully_readable`;
- `augmentation_need=use_as_is`;
- no more than four pages;
- unique PDF content SHA-256 within the pilot.

Of 2,301 locally present final BLCs, 1,178 satisfy the quality evidence, 1,107 also satisfy the page
cap, and those bounded candidates contain 1,077 unique content hashes. The deterministic stratified
selection has:

| Measure | Value |
| --- | ---: |
| Documents / unique content hashes | 150 / 150 |
| Pages | 286 |
| Logical PDF bytes | 123,491,086 |
| One / two / three / four-page documents | 48 / 76 / 18 / 8 |
| New ACI / NEW existing / OLD documents | 89 / 34 / 27 |
| Photo / rendered / scanned documents | 11 / 34 / 105 |
| Manifest SHA-256 | `faa3c717dbc426e22c070b3c69723bc0cfa16bdde1b330a8db53cdac904501cf` |

The pilot files are hard links to verified snapshot originals, not another 123 MB payload copy.
Each manifest row embeds the full catalog row and chosen local alias. The corresponding extraction
contract is `configs/glm_ocr.blc150.local.yaml`; no model or GPU service was started while preparing
or verifying it.

## Ready extraction inventories

The BLC union and four other useful OLD document types have independent source roots, run IDs, and
output roots. Their inventories hash-join exactly to the corpus/snapshot manifests with zero
identity mismatches:

| Type | Config | Documents | Pages | Inventory SHA-256 |
| --- | --- | ---: | ---: | --- |
| BLC (provisional) | `configs/glm_ocr.blc.local.yaml` | 2,483 | 5,611 | `189bdb446aad732511763ee240eb290c35880c13100d5bc3153829dd963a6b4b` |
| AWBC | `configs/glm_ocr.awbc.local.yaml` | 46 | 79 | `f6a4f78cdac207287337afc942ef1621103fd64dbf4eef2851f301bfc5785278` |
| COO | `configs/glm_ocr.coo.local.yaml` | 90 | 140 | `09fa3045314be66486f0f68980cf9f6939122466b381e7b659ec18806542a2a6` |
| Invoice | `configs/glm_ocr.inv.local.yaml` | 102 | 215 | `9b49763761434345b821facedaa26ed71c00f0be606a46abff01725c40bb2471` |
| Packing list | `configs/glm_ocr.pl.local.yaml` | 91 | 161 | `a339e87741e9d340d0df1210cde172827c96a63b5744c04e863b459dd5569467` |

SWB remains independently configured by `configs/glm_ocr.swb.local.yaml` with its existing 668
documents and 2,158 pages. Inventory creation and all verification above were CPU-only; extraction
has not begun. The BLC entry remains blocked from extraction by the semantic selection decision
documented above; the other inventories are unaffected.

## Local runtime readiness

The authenticated AWS CLI uses the `davniko-admin` login profile. Boto3 initially failed to load
that login credential provider because the project did not install its CRT extra. The project now
depends on `boto3[crt]`; an authenticated Boto3 list/versioning probe succeeds without embedding
credentials in configuration.
