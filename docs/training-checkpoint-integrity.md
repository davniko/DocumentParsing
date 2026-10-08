# Rank-48 comparison and best-checkpoint export integrity

Validated: 2026-10-08. No production training or GPU inference was launched.

## Experiment configuration

[Rank-48 local configuration](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic1500_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml).

Compared with the preceding rank-32 configuration, the only parsed changes are:

- LoRA rank 32 → 48.
- New run and adapter identities, and the rank tracking tag.
- `checkpoint.total_limit: null`: retain every scheduled checkpoint.
- Follow-up correction: ScheduleFree internal warmup **5 → 33 updates**, exactly **5% of 660 updates**. The comparison now changes rank and warmup, not only rank.

Alpha remains 32. Data hashes, input-only prompt, positions, compact targets, EVA settings, ScheduleFree AdamW, learning rate, seeds, microbatch/accumulation and ten epochs are unchanged. This is a fresh run, not a resume or overwrite of the rank-32 artifacts.

With 2,100 training rows and accumulation 32, the configuration retains checkpoints at steps **165, 330, 495 and 660**, approximately epochs **2.5, 5, 7.5 and 10**. Checkpoints include optimizer/trainer state, not only adapter weights. The existing intermediate prediction-publication behavior was not expanded: retained weights allow subsequent granular checkpoint evaluation.

ScheduleFree takes an integer `warmup_steps` setting: `ceil(2100 / 32) * 10 = 660`, then `ceil(0.05 * 660) = 33`. The top-level `warmup_ratio: 0.0` is intentional: an external scheduler must not apply a second warmup. The optimizer applies the 33-step warmup internally, as recommended by the [ScheduleFree documentation](https://github.com/facebookresearch/schedule_free#caveats). The regression test derives the expected count from the dataset size, batch size, accumulation and epochs, so changing the run length without updating the warmup fails the test. Existing rank32/historical configurations were not modified.

Follow-up validation: **25 tests passed in 10.65 seconds** in the training image, including the checkpoint tests and a new real Trainer/optimizer warmup probe. Trainer computes 660 total updates, every optimizer group receives `warmup_steps=33`, and all 34 tested updates match `1e-4 * min(step/33, 1)`. Update 1 uses approximately `3.0303e-6`; update 33 reaches `1e-4`; update 34 stays there. The constant external scheduler does not apply a second ramp. This is a configuration change, not additional per-step processing. No production training was launched.

Because rsLoRA remains enabled, changing rank alone also changes its mathematical scale from `32/sqrt(32) = 5.657` to `32/sqrt(48) = 4.619`. Alpha was deliberately not adjusted. Adapter parameter count grows by 50%; actual GPU throughput and memory at rank48 have not been benchmarked. Retaining four full checkpoints also increases disk use. The previous rank32 checkpoint occupied approximately 207 MiB; rank48 checkpoints will be larger.

## Confirmed defect

The previous run selected checkpoint495 but its final adapter exactly matched checkpoint660. PEFT saves a non-default adapter under `<checkpoint>/<adapter_name>/`, whereas the installed Transformers5.15 best-model loader first checks for weights directly under `<checkpoint>/`. It never reached loading for our named adapter and continued with the last weights. This directory convention is documented in the [official PEFT checkpoint format](https://huggingface.co/docs/peft/developer_guides/checkpoint).

The regression test reproduces this with an actual tiny T5Gemma2 model, PEFT and Trainer: four CPU training steps, checkpoint1 deliberately best, but the unpatched loader exports checkpoint4. No pretrained weights or external models are downloaded.

## Implementation

- [checkpoints.py](../src/document_ocr/training/checkpoints.py): resolve the native named/default adapter directory explicitly; load the single active adapter locally; reject missing files, incompatible weights and unsupported multi-adapter/distributed configurations. Unwrap compiled models through Accelerate. Non-PEFT models retain their native Transformers loader.
- [runtime.py](../src/document_ocr/training/runtime.py): use this loader for both standard and ScheduleFree trainers. ScheduleFree's existing evaluation-mode transition occurs **before** loading the selected checkpoint. Subsequent evaluation/save calls therefore cannot extrapolate the loaded best weights using the last optimizer iterate.
- After final export, compare every tensor's name, dtype and exact value against the selected checkpoint. Publish `best-adapter-export.json` and include it in the run manifest only if verification succeeds. A mismatch raises an error instead of publishing a successful run.
- Record the checkpoint module and ScheduleFree lifecycle module in loaded-source provenance.

The fix does not alter the attention path, optimizer updates, metric definitions, datasets or old run artifacts. Checkpoint resume continues to use its saved optimizer state; a final adapter export is not a resumable optimizer checkpoint.

## Validation

The actual training Docker image was rebuilt and passed its pinned-dependency verifier. Tests imported the immutable image source from `/opt/document-ocr/src`, not an overriding workspace source tree.

- **86 tests passed** in 16.55 seconds in the rebuilt image: checkpoint loading/export, ScheduleFree save/resume, config translation, callbacks, resume contracts, model adapter insertion, EVA, data preparation and relation output ordering. One CUDA-BF16 argument test was deselected for this CPU-only execution.
- **Six dataset-alignment tests passed** in the host environment. That suite depends on PDF packages intentionally absent from the training image.
- Both default/named adapters and standard/ScheduleFree optimizers load an earlier best checkpoint, preserve parameter identities/trainability, and export its exact tensors rather than the last checkpoint's tensors.
- All four intermediate checkpoint directories and optimizer states remain on disk with unlimited retention.
- Actual Trainer compilation/wrapping was exercised using the CPU `eager` compile backend. This validates loading/export integration, not GPU compilation performance.
- Missing files/tensors, extra tensors, wrong shapes, wrong exported tensor values/dtypes/keys, absent selection and unsupported adapter configurations fail explicitly.
- The rebuilt image's real `inspect-dataset` CLI validated **2,100 training + 60 validation records**, their hashes, the prompt and current schema contract in 1.37 seconds.
- Ruff and whitespace checks pass for the changes.

An initial synthetic compiled-wrapper test incorrectly replaced `Trainer.model` directly; it was corrected to exercise Trainer's actual `torch_compile` path. The resulting integration test passes. No production workaround for that artificial test setup was introduced.

### Broader-suite failures that predate this change

A broader run also encountered 15 historical metrics-fixture failures and one CPU hardware restriction. These were independently reproduced against immutable pre-change commit `acdb89e928468bb6d466beeb3059bf0b9290638f` with the same installed dependencies:

- The 15 v7 metrics fixtures omit now-required `negotiability`, and some omit notify `sameAs`.
- The warmup argument test requests CUDA BF16 while the verification container intentionally has CUDA disabled.

Those two test files produce the same **16 failed / 43 passed** on the committed baseline. They are not regressions from this fix and were not silently edited. The real current dataset passes schema validation. Thus the targeted change is validated, but this report does not claim the entire repository suite is green.

## Measured overhead and real-artifact probe

CPU measurements in the rebuilt image:

| Check | Measurement |
|---|---:|
| Native valid default-adapter loader, median of 20 warmed calls | 0.860 ms |
| New explicit loader, same model/checkpoint | 1.003 ms |
| Added one-time loader work | 0.143 ms |
| Export guard on the real rank32 adapter, first call | 0.408 s |
| Export guard median of five calls | 0.368 s |
| Tensors / total tensor bytes compared | 504 / 60,751,872 |
| Observed process peak-RSS increase during real comparisons | 117.1 MiB |

The guard compares one tensor pair at a time with memory-mapped CPU files. These costs occur once at completion, not on training steps or each evaluation. No training hot path was changed, and no GPU throughput improvement is claimed.

The real prior export passes against checkpoint660 and is **rejected against its actually selected checkpoint495**, confirming that the new guard detects the observed production defect. All old artifact files were read-only.

## Launch

```bash
docker compose --profile training run --rm --build kie-trainer train \
  --config configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic1500_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml \
  --project-root /workspace
```

The `--build` matters because training imports source copied into the image. The old rank32 run and its retained checkpoints remain untouched.

### Recurring Docker credential-helper failure

The observed `ghcr.io/astral-sh/uv ... error getting credentials ... exit status 1, out: ""` is a **client-side credential-helper failure**, before training starts. The local Docker configuration selects `credsStore: desktop.exe`, resolving to the Windows executable `/Docker/host/bin/docker-credential-desktop.exe`. Repeating the ordinary command continues to invoke that same helper. Docker's [credential-store and helper documentation](https://docs.docker.com/reference/cli/docker/login/) explains this selection and lookup protocol.

The exact empty-output failure did not reproduce during diagnosis: the helper returned the recognized missing-credentials response, and anonymous metadata lookup succeeded. We have confirmed the failing layer from the supplied error, not the internal cause of the intermittent Windows helper failure. Replacing pinned images or altering training dependencies would not address that layer.

For these **public** base images, use an isolated Docker client configuration without changing existing credentials:

```bash
mpci_docker_config="$(mktemp -d /tmp/mpci-docker-public.XXXXXX)"

docker --config "$mpci_docker_config" compose --profile training run --rm --build kie-trainer train \
  --config configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic1500_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml \
  --project-root /workspace
```

This isolated configuration was tested: default Unix-socket context, Compose discovery, exact pinned uv metadata, and the complete training-image build with dependency verification all succeeded. The training launch above is for the user; it was **not** executed during diagnosis. This avoids the failing helper for public pulls but does not supply private-registry credentials. A permanent repair to the normal client requires diagnosing Docker Desktop's Windows/WSL helper integration; no global Docker configuration or saved login was changed, and Docker Desktop was not restarted.
