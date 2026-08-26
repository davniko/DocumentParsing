# MPCI B/L legacy/current quality alignment and combined dataset

Date: 2026-08-26  
Status: published and independently validated

## Outcome

The previous 487-record semantic-v2 corpus was not safe to concatenate directly with the current
626-record relation-explicit corpus. Its OCR and core labels were sound, but its model-facing cargo
target still contained mapped package categories and its transport-type representation differed
from the current printed-text-only policy.

The immutable alignment retained 483 legacy records, migrated them to the exact current target
contract, and joined them with 626 current records. The resulting dataset contains **1,109 unique
documents**, with no duplicate document IDs or raw-OCR hashes:

- [combined manifest](/mnt/d/Projects/DocumentParsing/artifacts/kie-training/datasets/mpci-bl-combined1109-current-policy-v1/manifest.json)
- [combined records](/mnt/d/Projects/DocumentParsing/artifacts/kie-training/datasets/mpci-bl-combined1109-current-policy-v1/records.jsonl)
- [combined lineage](/mnt/d/Projects/DocumentParsing/artifacts/kie-training/datasets/mpci-bl-combined1109-current-policy-v1/lineage.jsonl)
- [quality comparison](/mnt/d/Projects/DocumentParsing/artifacts/kie-training/datasets/mpci-bl-combined1109-current-policy-v1/quality-comparison.json)

No source dataset or historical annotation was mutated.

## Dataset lineage

| Stage | Records | Role |
|---|---:|---|
| Older pilot | 106 | Validated semantic-v2 labels |
| Older follow-up | 381 | Validated semantic-v2 labels |
| Older combined source | 487 | Previous training corpus |
| Relation-explicit legacy transform | 483 | Four declared relation blockers omitted |
| Current consolidated corpus | 626 | Current validated training records |
| Final combined corpus | **1,109** | Current relation-explicit training contract |

The four omitted legacy documents were already declared fail-closed in the pinned relation
transform:

| Document | Reason | Current decision |
|---|---|---|
| `doc_28dd90dc...` | duplicate package quantities at two package levels | omit: relation level was not adjudicated in the source corpus |
| `doc_80e915b...` | duplicate package quantities at two package levels | omit: generic packages and inner bags are both plausible allocation levels |
| `doc_eaba3f2...` | duplicate package quantities at two package levels | omit: relation level was not adjudicated in the source corpus |
| `doc_9656e533...` | one container has no OCR-grounded allocation quantity while six do | omit: the current ontology cannot encode mixed known/unknown quantities without discarding facts or deriving the missing number |

The first and third exclusions may be recoverable with a new, explicitly reviewed relation
annotation. They remain excluded here because this alignment does not silently upgrade a historical
label from visual interpretation or an inferred relation. This is 0.82% of the 487-row source.

## Required legacy alignment

### Printed package and container types

The old relation target contained 630 mapped `typeCategory` values, while the current target retains
the type as printed and performs registry mapping downstream. The alignment therefore:

- removed 630 package category tokens;
- restored the corresponding 630 printed package types from the pinned semantic-v2 labels;
- converted 91 legacy container `typeCode` fields to `typeDescription`; and
- emitted zero package or container `typeCategory` values.

All 893 aligned legacy container type descriptions and all 651 aligned legacy package type
descriptions occur in their raw OCR after case/punctuation normalization. The conversion does not
invent a registry meaning.

### Product-only goods descriptions

The current cleanup policy keeps product wording in `description` and separately modeled package
wording in package facts. A broad candidate scan was manually reviewed to avoid deleting legitimate
product nouns such as *steel coils*, *aerosol cans*, *curing bag*, or *packaging units*.

Twenty exact, evidence-pinned corrections were required across six legacy documents: 19 cargo
descriptions and one generic package type. Examples include removing `900 KG IBC` from `FROTHER DSF
802A 900 KG IBC`, changing `CORIANDER SEEDS IN BAGS` to `CORIANDER SEEDS`, and retaining `BAG` as the
separate printed package fact. The complete before/after/evidence ledger is available in:

- [correction audit](/mnt/d/Projects/DocumentParsing/artifacts/kie-training/datasets/mpci-bl-combined487-current-policy-aligned-v1/correction-audit.jsonl)
- [pinned correction specification](/mnt/d/Projects/DocumentParsing/configs/transforms/mpci_bl_combined487_current_policy_corrections.json)

External product references were used only to disambiguate whether a suspicious suffix was part of
a product identity; they never supplied a label value absent from OCR. The inference from the
official catalog entries is that `BAG`, `ROLL.`, and `BARREL` are shipping-package wording here:
Borealis identifies the product as `BorECO BA212E` polypropylene, and Covestro lists `Desmodur N 75
MPA/X`, `Desmoseal S XP 2636`, and `Desmophen 5168 T` without those suffixes:

- [Borealis BorECO BA212E](https://www.borealisgroup.com/products/product-catalogue/boreco-ba212e-1)
- [Covestro Desmodur N 75 MPA/X](https://solutions.covestro.com/de/products/desmodur/desmodur-n-75-mpax_000000000000832200)
- [Covestro Desmoseal S XP 2636](https://solutions.covestro.com/pt/products/desmoseal/desmoseal-s-xp-2636_000000000006366759)
- [Covestro Desmophen 5168 T](https://solutions.covestro.com/en/products/desmophen/desmophen-5168-t_000000000000418595)

### Already aligned policies

No further rewrite was needed for these areas:

- **Dates:** the exhaustive day-first audit found 28 legacy documents, 43 target fields, and 44
  ambiguous numeric-date evidence occurrences; all 43 targets were already day-first.
- **Countries/localities:** values remain printed text, not ISO-2 or UN/LOCODE predictions.
- **Extended Latin:** the existing immutable correction lineage retains 13 OCR-grounded values in
  five pilot documents rather than ASCII-normalizing them.
- **Addresses:** every one of the 1,396 legacy address leaves is a scalar string; addresses are not
  split into one list item per line.
- **Flavor metadata:** no tax ID, VAT, ACID, exporter-registration, telephone, fax, or email flavor
  string appears in a legacy target outside its proper contact/semantic field. The three apparent
  `ACID` hits are legitimate chemical product descriptions.

Two country targets are not contiguous literal substrings after a simple whitespace-only scan, but
both pass the annotation evidence contract: legacy `TURKIYE` is assembled from adjacent OCR line
fragments `TUR` and `KIYE`, and current `TAIWAN, ROC` is normalized from printed `TAIWAN,R.O.C`.
They are not image-only or inferred country mappings.

## Quality comparison

| Check | Aligned legacy | Current | Combined result |
|---|---:|---:|---:|
| Strict target-schema validation | 483/483 | 626/626 | **1,109/1,109** |
| Normal/relation projection consistency | 483/483 | 626/626 | **1,109/1,109** |
| Unique document IDs | 483/483 | 626/626 | **1,109/1,109** |
| Unique raw-OCR hashes | 483/483 | 626/626 | **1,109/1,109** |
| Mapped package/container categories | 0 | 0 | **0** |
| Manifest-listed file hashes | all | all | **3/3 combined artifacts** |

The source corpora differ in document complexity, not target quality. The legacy set averages 1.28
cargo groups, 1.37 package facts, 0.85 allocation groups, and 1.96 containers per document. The
current set averages 1.11, 1.39, 0.92, and 1.75 respectively. These are coverage/composition
differences and should be preserved when producing a seeded train/evaluation split.

## Evidence-sidecar caveat

All 487 historical semantic-v2 annotations are hash-pinned, schema-valid, and have verbatim,
page-ordered OCR evidence. Nine annotations contain 12 older, redundant evidence anchors that fail
the newer *non-overlapping anchors within one field* rule—for example, a short address fragment also
appears inside a longer adjacent address excerpt. This is an evidence-representation difference,
not a target-label contradiction. It is recorded per document in the aligned lineage and quality
audit rather than hidden or rewritten:

- [legacy quality audit](/mnt/d/Projects/DocumentParsing/artifacts/kie-training/datasets/mpci-bl-combined487-current-policy-aligned-v1/quality-audit.json)

The current 626 records additionally retain their complete dual-annotation evidence. The aligned
legacy training rows use the same model-facing schema and consistency checks, while their historical
semantic-v2 sidecars plus immutable transform/correction lineage remain the source of audit truth.

## Reproducibility and validation

The alignment command is:

```bash
TMPDIR=/tmp UV_CACHE_DIR=/tmp/documentparsing-uv-cache \
uv run --frozen document-kie-align-datasets \
  --config configs/transforms/mpci_bl_combined1109_current_policy_alignment.yaml
```

Measured on this workspace:

- first full publication: 13.06 seconds, 153,132 KiB peak RSS;
- immutable deterministic rerun: 12.48 seconds, 164,628 KiB peak RSS, zero hash drift;
- independent combined-artifact validation: 1.40 seconds, 86,040 KiB peak RSS;
- targeted schema/alignment suite: 54 passed in 1.52 seconds;
- dedicated alignment regression tests: 4 passed in 0.90 seconds.

The combined artifact is unsplit. A later training configuration must create or pin a seeded,
document-level train/evaluation split; it must not treat source-corpus membership as the split.
