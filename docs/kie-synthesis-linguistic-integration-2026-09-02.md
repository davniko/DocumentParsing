# Integrated linguistic synthesis stage

Date: 2026-09-02

## Boundary

This stage consumes one committed relation-v5 semantic-completion plan set. It generates only the
linguistic values already proved by the isolated party and cargo probes:

- fictional party names, addresses, contact names, phone numbers, emails, and websites;
- printed cargo descriptions, grounded auxiliary cargo text, substantive shipping marks, and
  handling instructions.

It does not generate package/container/HS printed surfaces and does not patch raw OCR. Successful
outputs therefore remain `trainingEligible: false`; final source-safe OCR realization is still a
separate downstream stage.

## Execution contract

- Provider-native strict JSON Schema is used on every request.
- The first attempt uses the stable schema so repeated calls can benefit from provider-side prefix
  and schema caching. A semantic failure gets one fresh, history-free attempt with a topology-
  constrained schema that fixes field presence, array cardinality, party role/locality, cargo
  group IDs, and group count.
- One shared request limiter and one bounded document-worker pool enforce the two configured
  concurrency limits. Work does not create one long-lived coroutine per corpus document.
- Plain `sameAs` notify relations require no generation. Contact-only `sameAs` overrides generate
  only their contacts. An explicitly repeated notify identity reuses the generated shipper or
  consignee unit. Locality-only party objects require no model call.
- Every attempt retains its provider-visible transcript, provider response ID, parsed output,
  semantic checks, error, duration, token buckets, and estimated cost. Unit artifacts are written
  as they finish and are reusable after interruption.
- Every successful projected target must pass the current task schema and exact relational inverse.
  Source-party identities and contact values are checked against the complete pinned source corpus.

## Provider settings

`provider.generation_settings` is optional. Omitting it sends no temperature, top-p, or text-
verbosity override, leaving provider defaults intact. Present values are forwarded directly after
strict range/type validation. `service_tier` is independently optional. Operational controls—model,
reasoning effort, output-token ceilings, timeout, SDK transport retries, and response retention—
remain explicit.

The production example intentionally omits generation settings:

```yaml
workflow:
  max_concurrent_requests: 16
  max_concurrent_documents: 16

# Optional; omit the entire mapping for provider defaults.
provider:
  generation_settings:
    temperature: 1.1
    text_verbosity: medium
```

The provider also accepts `top_p`; temperature and top-p may be configured together, although the
OpenAI API documentation recommends tuning one or the other in most cases.

## Commands

Configuration-only validation does not make an API call:

```bash
uv run --frozen document-kie-synthesis validate-linguistic-completion-config \
  --config configs/synthesis/mpci_bl_linguistic_completion_audit100_luna_high_v1.yaml \
  --project-root .
```

The paid, resumable run is explicit:

```bash
uv run --frozen document-kie-synthesis run-linguistic-completion \
  --config configs/synthesis/mpci_bl_linguistic_completion_audit100_luna_high_v1.yaml \
  --project-root .
```
