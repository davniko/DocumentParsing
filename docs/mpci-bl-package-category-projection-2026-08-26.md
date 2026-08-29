# MPCI B/L task-facing package-category projection

Date: 2026-08-26

## Purpose and boundary

The package-hierarchy stage produces a model-facing target in which direct goods packages remain
and outer/generic hierarchy metadata is retained outside the decoder target. This next stage is a
separate immutable projection: it converts an exact printed package `typeDescription` to a
readable Recommendation 21 category token only when that exact source string supports one proven
registry entry.

It does not fuzzy-match at runtime, does not manufacture container categories, and does not mutate
the source dataset. A description that cannot select one registry meaning remains verbatim. The
complete source normal/relation targets and the package-by-package decision trace are retained in
`category-metadata.jsonl`.

## Frozen inputs

- source dataset manifest:
  `29a2e5c8fdaec4806b91dddcbebffdc170d0b20944284e15e680b301a9ce008e`;
- package registry:
  `2476deb46c3cabf2efe9a00bbb9ad1bf6b527e4adae0e4872d8a63807bbd374f`;
- intentionally empty container semantic registry:
  `fa1b327b8e9eb2b38a5c86c2d53fc1abe8d88f26cb08b4ed1c26d5670f41f3f0`;
- prior reviewed assignments:
  `e36652e068839aa329a76a186bfe9ffe5e0c2a61fc45e61cde7c5367e59ce4a3`.

The configuration carries exact expected occurrence and variant counts. It explicitly covers every
observed source string not present in the prior review, either with one registry category or with a
reasoned null decision. An unknown string, an extra review key, a count change, a changed registry,
or a changed prior artifact blocks publication.

## Result

The published unsplit dataset is
`artifacts/kie-training/datasets/mpci-bl-combined1157-task-facing-package-categories-v2/`.

| Measure | Result |
|---|---:|
| Records validated and published | 1,157 |
| Package facts | 1,423 |
| Typed package occurrences / exact variants | 1,405 / 183 |
| Category occurrences / exact variants | 1,348 / 156 |
| Raw fallback occurrences / exact variants | 57 / 27 |
| Prior-review occurrences / exact variants reused | 1,231 / 94 |
| Extension occurrences / exact variants reviewed | 174 / 89 |
| Untyped quantity-only facts retained | 18 |
| Observed package category tokens | 47 |
| Container type occurrences / exact variants retained as printed | 1,970 / 174 |
| Container categories inferred | 0 |

The 27 raw fallback strings include ambiguous bulk forms, underspecified bale/box/bottle forms,
opaque abbreviations, and likely OCR corruption. They remain usable targets without asserting an
incorrect registry identity. The authoritative decision inventory and affected document IDs are in
`package-category-inventory.jsonl`.

All 1,157 projected records passed both the relation schema and normal/relation consistency check.
The independent real-data probe also confirmed that all normal targets and all container objects
are byte-equivalent as JSON values to their source counterparts, all complete source relation
targets are retained in metadata, no package carries both category and fallback, and the 47 tokens
in `task-constraints.json` exactly equal the tokens observed in `records.jsonl`.

## Published hashes

| Artifact | SHA-256 |
|---|---|
| `manifest.json` | `bcc7c5411792beb80889cdc800ef5029ebba82085b5f90bcdb3838d82fbb3622` |
| `records.jsonl` | `2a3e2ea3231cfff7674e85b54a52f66d0fee98b59b207bc7b04fe0f9916dfc42` |
| `task-constraints.json` | `f55ca917c95f75ddad494067aaaa2ac40d6c58a79688056c27de9dc17241911e` |
| `package-category-inventory.jsonl` | `79e18bb17b33cdb3505c8a4e9785e3a1602f5ce22ba721f022d978070d0a8b2a` |
| `container-category-inventory.jsonl` | `383038cce14a698c449c47132ae01643d46a6b9b0b80e972ba6940c2a927cca9` |

## Reproduce and measured cost

```bash
TMPDIR=/tmp UV_CACHE_DIR=/tmp/documentparsing-uv-cache \
uv run --frozen --group dev python -m document_ocr.training.category_projection \
  --config configs/transforms/mpci_bl_combined1157_task_facing_package_categories_v2.yaml
```

An idempotent full-corpus run over the published artifact took 3.40 seconds wall time, 1.50 seconds
user CPU, and 131,572 KiB maximum resident memory. It performed no network or model calls. Re-running
against the same output proved byte identity; publication fails rather than overwriting if any
artifact differs.
