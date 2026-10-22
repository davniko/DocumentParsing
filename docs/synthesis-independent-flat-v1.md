# Donor-independent flat-target synthesis

This is an explicitly selected synthesis mode, not a change to extraction
targets. It retains the current 48 flat package categories. Factored packaging
targets and DG/HS registry enrichment are separate experiments.

## Ownership and sequence

1. Compile the selected source's presentation and verify its reviewed numeric,
   wording and geography authorities. Missing certificates, incomplete repeated
   owners and unsupported equipment presentations are explicit holds.
2. Draw goods from the pinned full HS registry, or from the supported joined-DG
   registry subject to the source's printed DG-field combination. The source
   does not select a commodity donor or a shipment bundle.
3. Ask the pinned native structured-output provider for plausible flat package
   alternatives and, where required, a source-supported private packing
   relationship. `UNKNOWN` remains a hold. The chooser does not own numeric loads
   or final descriptions. Its explanatory text is audit evidence only.
4. Select among accepted alternatives, sample independent counts, physical
   quantities, equipment and private packing, then check the complete ledger.
   Every final container row must fit its selected equipment, including after
   equal-load or nested-packing composition.
5. Generate party and goods wording. Sampled shipment facts remain authoritative.
   The normal bounded wording-validation repair also runs independent locality,
   distinct-actor and numeric checks before accepted wording/contact receipts
   are written. A second invalid proposal remains held.
6. Render, replay, review complete text/targets and adjudicate findings. Only
   then publish the inspection dataset and synthesize/audit source-anchored
   coordinates. An inspection publication is not training admission.

The source owns visible roles, repeated occurrences, package/measurement slots,
container count and allocation visibility. Privately sampled allocations used
for capacity checking are not emitted where the source has no allocation.
Hardware tare support is distinct from commodity donors and remains explicit.

Mixed support/loose-package equations use a separate native typed component
choice (`independent-mixed-accounting`), with generic PACKAGE as the declared
accounting unit. The host owns every positive component count and both equation
totals. A correlated range consumes those same counts. Additive batch count,
weight and volume cells share one fresh count partition, not source ratios.
Container-scoped nesting uses each current row load; shipment inner/support
totals reconcile to the private child ledgers. Missing printed allocations stay
missing from public targets. See the [first completion batch](analysis/r28-donor-free-batch01-20261021/REPORT.md)
for exact source ownership, tests and production validation.

Ordinary related inner units are selected from the 40 receptacle categories that
do not require additional counting-context proof. The public vocabulary is still
48 categories: bulk/unpacked/discrete-article categories remain usable as counted
units with their own context, not as ordinary numbered inner receptacles. Open
platform families PALLET/SKID/RACK do not permit the reversed CONTAINS relation in
the native output schema; choose a support relation or NONE. Direct liquid fill
does not invent numbered BULK LIQUID inner packages.

Inline equal locality/country presentations require complete frame ownership for
the existing one-occurrence/name-only projection. Separate scalar owners cannot
silently print duplicate identical locality/country tokens. Independently
captioned country declarations and distinct explicit codes/aliases remain
separate evidence, not candidates for blanket text deduplication.

Reviewed handling clauses can depend on a route or a generated party name.
Their `dependent_text` declarations explicitly select `phase: scenario` (route
name/country) or `phase: wording` (party name). The latter runs after lexical
assembly; resolving it during planning would retain the source private party.
Each declaration pins its exact original clause and an owned handling target.
Unknown references, wrong phases, missing parties and conflicting targets fail.
This is template-owned interpolation, not instructions for the LLM to rewrite
shipment declarations.

A source-only organization can be related to an extracted party, or act for a
source-certified fixed public carrier. `related_actors` uses exactly one of
`context_party` and `context_carrier`; the latter must match an actual
`carrier_static` organization binding, its complete source occurrences and its
source name. It is not a fabricated extracted party or an inferred location.
The issuing agent's name is generated normally; the carrier remains public.
An explicitly reviewed public carrier office can instead remain fixed. Neither
decision follows merely from a company being in a signature block.

## Configuration

The maintained example is
[mpci_bl_curated_v7_independent_validation.yaml](../configs/synthesis/mpci_bl_curated_v7_independent_validation.yaml).
It selects `sampling_mode: independent` and pins these authorities:

```yaml
independent:
  registry: # seven explicit path/SHA-256 pins
    hsManifest: {path: ..., sha256: ...}
    hsMetadata: {path: ..., sha256: ...}
    hsReport: {path: ..., sha256: ...}
    phrases: {path: ..., sha256: ...}
    dgHmt: {path: ..., sha256: ...}
    dgEcics: {path: ..., sha256: ...}
    flash: {path: ..., sha256: ...}
  numeric_certificates: {path: ..., sha256: ...}
  wording_certificates: {path: ..., sha256: ...}
  geography_certificates: {path: ..., sha256: ...}
  locality_owner_reviews: {path: ..., sha256: ...}
  distinct_actor_reviews: {path: ..., sha256: ...}
independent_preparation:
  package_targets: {PACKAGE_BALE: 4, PACKAGE_SET: 4, PACKAGE_DRUM: 4}
  accepted_counts: {} # optional actual previously accepted category counts
  mixed_source_ids: [doc_...]
  mixed_probability: 0.3333333333333333
```

`package_targets` are **soft deficit targets**, not permission to force an
implausible type or a promise of exact final counts. Only accepted native
alternatives can be selected. Pending reservations are reported separately from
accepted-final counts; the latter must describe actual reviewed acceptances.
Mixed sampling requires source-certified row ownership and compatible equipment.

## Execution

Use `inventory-independent` with the same config for a source-only inventory.
It performs no provider calls or cargo draws, may initialize the explicitly
configured output directory, and saves an immutable diagnostic report. Missing
certificates and explicit reviewed holds have distinct dispositions. It does
not claim physical or semantic acceptance.

Run preparation next, using the actual synthesis environment:

```bash
docker compose --profile synthesis run --rm --build --entrypoint python synthesis-tools \
  -m document_ocr.synthesis.curated_campaign prepare-independent \
  --config configs/synthesis/mpci_bl_curated_v7_independent_validation.yaml \
  --project-root /workspace
```

Preparation prints an immutable summary and a pinned `preparedConfig` path.
Use that exact generated JSON configuration for the normal `generate`, `render`,
`review`, `validate`, `publish` and `positions` stages. Do not use the unprepared
YAML for those stages. Preparation preserves every requested source/variant;
only complete ready source groups receive a runnable config. Successful siblings
of a held variant are inventoried as `blocked_by_source_group`, not silently
renumbered or substituted. A partial preparation exits nonzero even when it
provides a runnable ready subset.

Packets and native receipts bind the exact request, schema, provider and received
answer. Nested contexts serialize canonically, and verification uses Runner's
typed provider identity, not the unparsed YAML representation. Altered evidence
is a hard failure. Provider retries and semantic regenerations retain their
original receipts and remain subject to the single configured output ledger's
budget; no silent provider fallback is allowed.

Explicit `--retry-invalid-output` and `--retry-rate-limited` permissions also
apply to `prepare-independent`. For another failed request, repeat
`--retry-failed-receipt-sha256 <exact SHA-256>` for each specifically reviewed
failure. A retry creates a new Runner attempt receipt; it never overwrites the
failed receipt or silently retries all requests. Valid cached successes are
reused. Semantic UNKNOWN or contradictory package alternatives are holds, not
transport failures that these flags can override.

## Certification boundary

An ordinary admitted template is not automatically certified for independent
variation. Numeric ownership, source readiness, successful physical planning,
successful wording, complete semantic review and coordinate review are separate
gates. Report their denominators separately. Never label missing provider output
as a semantic failure, or a passed arithmetic check as a complete-text audit.

The completed first 20-source production batch and current accounting migration
are in the [batch report](analysis/r28-donor-free-batch01-20261021/REPORT.md).
Earlier integration evidence and unresolved source presentations remain in the
[historical integration report](analysis/r28-production-integration-20261021/REPORT.md).
Historical experiments and R27 are not runtime dependencies and are not
rewritten by this mode. Confirmed legacy corrections are separately inventoried
and applied only to the current R28 dataset and the appropriate upstream owners.
