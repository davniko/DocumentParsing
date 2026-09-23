# Complete-synthesis status and configurations

## Diversified route/cargo run: finalized at 9,930 synthetic samples

**Final selection updated by user:** retain 9,930 existing synthetic samples and
exclude the 70 draws from source `910f8b9a...` without replacements. The accepted
selection and complete target subset are committed in
`synthesis-plans/mpci-bl-diversified-accepted9930-v1`; all excluded IDs are recorded
there. `mpci_bl_diversified_accepted9930_publish_v1.yaml` performs model-free replay
of the existing paid generation. Do not launch the unused v18 replacement plan
or another synthesis round. Publication, full replay, EDA and the 10,987-record
real-plus-synthetic training merge are complete. Validation remains 100 real
documents, unchanged. The same ledger remains $6.76712478; no new allowance was
opened and no additional provider calls were made during finalization.

**Quality disclosure:** final EDA identified two retained address/country
contradictions. They are documented, not silently fixed or excluded. Mechanical
passes are not an unqualified semantic sign-off. Training is prepared but not
launched.

- [Final report and training command](../../../artifacts/kie-synthesis-production/analysis/diversified-exact10k-20260920/FINAL_REPORT.md)
- [Visual overview and 60 main/comparison plots](../../../artifacts/kie-synthesis-production/analysis/diversified-exact10k-20260920/index.html)
- [New training config](../../training/production/t5gemma2_270m_lora.mpci_bl_real1057_diversified_synth9930_exact_id_v5_e5_v1.yaml)

`mpci_bl_diversified_route_cargo_synthesis10000_v5.yaml` is the separate,
completed route/locality, commodity, DG and equipment-diversified generation run.
Its immutable selection/routes are v17, with exactly 7,500 standard, 1,000 DG and
1,500 temperature-controlled samples and exact validation-source exclusion.
Unresolved source contracts are explicitly reviewed/excluded, not silently fixed
by copying source values. The 1,510-template catalog is unchanged.

**One-run budget exception:** the user authorized a $7 total cap for this v5 run
only, preserving its existing spending ledger and valid paid work. All subsequent
10k runs retain the $6 cap; do not propagate this exception when creating a new
run. The exact authorization and unchanged-charge receipt is recorded in
`artifacts/kie-synthesis-production/analysis/diversified-exact10k-20260920/current-run-only-seven-dollar-authorization.json`.
The finalized publication is `mpci-bl-diversified-accepted9930-v1-published`,
not the original 10,000-case diagnostic generation. Training is user-launched;
all current synthesis configurations are provider-locked again.

## Completed exact-ID dataset

`mpci_bl_complete_synthesis10000_exact_validation_id_v10.yaml` has completed.
Its final selection plan is v15: 10,000 complete synthetic targets rendered
through 1,240 eligible sources, excluding all 100 real validation document IDs.
Shared carrier layouts remain intentionally allowed. The source catalog is
unchanged; unresolved source contracts are excluded by the selection plan.

The published population is 7,500 standard, 1,000 dangerous-goods and 1,500
temperature-controlled documents. Carrier identity and supported topology remain
bound to each template; repeated template use is intentional. Final publication,
independent audit and preparation of 11,057 training / 100 validation records
are complete. Training itself has **not** been launched.

- [Final handoff, costs and training command](../../../artifacts/kie-synthesis-production/analysis/complete-exact10k-20260920/HANDOFF.md)
- [EDA dashboard: 42 plots](../../../artifacts/kie-synthesis-production/analysis/complete-exact10k-20260920/eda/index.html)
- [Validation evidence](../../../artifacts/kie-synthesis-production/analysis/complete-exact10k-20260920/VALIDATION.md)
- [New training config](../../training/production/t5gemma2_270m_lora.mpci_bl_real1057_complete_synth10000_exact_id_v5_e5_v1.yaml)

The user authorized one reset to the v2 $6 allowance. That ledger settled at
$4.55922723, with $5.10592553 conservatively occupied including uncertain
requests. Earlier spending remains separately recorded; the reset did not erase
it. **Do not reset this ledger again or describe reused paid work as free.**
The completed config is provider-locked and is not a new-run launch config.

## Separate layout-proxy holdout arm: not run

`mpci_bl_complete_synthesis10000_layout_proxy_holdout_v10.yaml` remains prepared
and provider-locked. It selects 848 sources and excludes validation layout-proxy
families as well as exact IDs. Its intended population is also 7,500 standard,
1,000 dangerous-goods and 1,500 temperature-controlled samples. The completed
exact-ID run does not certify this unexecuted dataset.

Both configurations use up to 16 concurrent physical requests and compatible
scenario batches, pinned inputs, persistent estimated-spend accounting, and
explicit validation before publication. Synthesis does not compile templates or
restore source targets as a fallback.

## Future execution requires explicit authorization

For an explicitly authorized run, verify its inputs, output location and budget
ledger, then enable `provider_launch_authorized` only in that selected config.
Ordinary restarts retain accounting. Rebuild the synthesis container: a workspace
mount alone does not update installed imports.

```bash
docker compose --profile synthesis build synthesis-tools
docker compose --profile synthesis run --rm --no-deps --entrypoint python synthesis-tools \
  -m document_ocr.synthesis.template_compiler.complete_pipeline \
  --config configs/synthesis/production/mpci_bl_complete_synthesis10000_layout_proxy_holdout_v10.yaml
```

This example is for the **unlaunched, separately authorized** layout-holdout arm,
not an instruction to run it now. Synthesis does not launch training. Never bypass
budget or publication checks to turn an incomplete result into a training set.
