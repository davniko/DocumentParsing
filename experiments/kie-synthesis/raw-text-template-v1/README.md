# Carrier-bound raw-text template experiment

This package is an isolated proof path for compiling reviewed Bill of Lading OCR into reusable,
carrier-bound semantic templates. It does not alter or publish through the production synthesis
pipeline.

The current measured status, failure analysis, costs, and continuation gates are recorded in
[`audits/status-2026-09-12.md`](audits/status-2026-09-12.md).
The recurring-render correction and paired compiler-cost experiments are recorded in
[`audits/cost-optimization-2026-09-13.md`](audits/cost-optimization-2026-09-13.md).

The compiler combines pinned accepted-label evidence with a one-time preprocessing proposal, a
bounded repair loop when the host returns exact defects, and an independent literal-remainder
review. Exact byte spans, source round trips, carrier identity, topology, risk coverage, and
sentinel isolation are host-owned gates. A document produces either a fully certified template or
an explicit rejection; no partial template is published.

`audit-carriers` classifies all pinned sources before selection. A source-label carrier is accepted
only when it is reusable as a carrier identity: vessel-master labels, country-role leakage, and
explicit placeholders are rejected alongside missing labels. Every accepted label still requires
exact compiler-plus-critic OCR evidence. Generic carrier language and agent/customer names do not
qualify and are reported as requiring external enrichment.

`preflight` resolves and hashes every selected source, compiles all accepted anchors, classifies
carrier-resolution work, validates capability metadata, enumerates deterministic risk candidates,
and sizes each compiler request before any provider is constructed.

Generated runs live under `artifacts/kie-synthesis/`. The planned sequence is a 30-document
development gate, an additional 200-document transfer gate, aggregation into a 230-template
catalog, and separate 30 then 100 descendant-render experiments.

Every extraction phase requires every selected document to certify. All checked-in configurations
have `provider_launch_authorized: false`; a deliberate bounded launch requires changing only the
relevant pin after its prerequisites and provider balance are verified. Rejected runs remain
useful audit artifacts but never authorize the next phase or publish templates downstream.

Useful offline commands:

```bash
uv run raw-text-template-experiment audit-carriers --config configs/development30.yaml
uv run raw-text-template-experiment selection --config configs/development30.yaml
uv run raw-text-template-experiment preflight --config configs/development30.yaml
uv run raw-text-template-experiment offline-audit \
  --development-config configs/development30.yaml \
  --transfer-config configs/transfer200.yaml \
  --run-name YOUR_IMMUTABLE_RUN_NAME
uv run raw-text-template-experiment audit-efficiency --config configs/efficiency-audit.yaml
```

The current cost-driver, contract, and paired model/provider measurements are committed in
`mpci-bl-carrier-bound-template-extraction-efficiency-audit-v4`. Manual comparison of the live
probe outputs is recorded in
[`audits/model-efficiency-probe-manual-review-2026-09-12.yaml`](audits/model-efficiency-probe-manual-review-2026-09-12.yaml).
