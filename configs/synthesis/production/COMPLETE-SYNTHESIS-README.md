# Current complete-synthesis launch configurations

Use only these current configurations for the replacement datasets:

- `mpci_bl_complete_synthesis10000_exact_validation_id_v10.yaml`
- `mpci_bl_complete_synthesis10000_layout_proxy_holdout_v10.yaml`

Each requests 10,000 complete, independently synthesized targets followed by
compiled-template rendering: 7,500 standard, 1,000 dangerous-goods, and 1,500
temperature-controlled samples. Carriers and supported topology remain bound
to the selected template. Repeated use of a template is intentional.

The exact-ID plan uses 1,252 sources and excludes the 100 real validation document
IDs. The stronger layout-proxy holdout uses 848 sources and excludes validation
proxy families too. The original shared validation dataset is unchanged.

Both configurations have:

- 16 concurrent physical requests and batches of up to 16 compatible scenarios;
- a separate persistent **$6 estimated-spend cap** including retries and residuals;
- pinned source catalog, numeric contracts, registries, prompts, and sample plans;
- no numeric/template compilation during synthesis and no source-copy fallback;
- publication only after exact plan coverage, complete-target variation, and
  frozen target/render validation;
- `provider_launch_authorized: false`, deliberately awaiting the full-run go-ahead.

The full 10k datasets have **not** been launched or published. Training inputs
are not ready until those datasets are generated, audited, and assembled.

## Launch procedure after explicit authorization

Set `provider_launch_authorized` to `true` in the selected configuration. Keep its
existing budget ledger path and $6 limit; a restart must never reset accounting.
Use the rebuilt synthesis container, which contains the implementation (a workspace
mount alone does not update installed imports):

```bash
docker compose --profile synthesis build synthesis-tools
docker compose --profile synthesis run --rm --no-deps --entrypoint python synthesis-tools \
  -m document_ocr.synthesis.template_compiler.complete_pipeline \
  --config configs/synthesis/production/mpci_bl_complete_synthesis10000_exact_validation_id_v10.yaml
```

For the second dataset, use the layout-proxy configuration in the same command.
No training command is run by this synthesis procedure. If a budget or validation
failure occurs, inspect the explicit incomplete result; do not bypass its checks.

The readiness evidence and cost accounting are recorded in
[the readiness report](../../../artifacts/kie-synthesis-production/repair-20260919/READINESS-REPORT.md).
