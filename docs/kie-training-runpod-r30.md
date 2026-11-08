# R30 RunPod deployment — 7 November 2026

Historical rank-64 deployment. For the replacement pod and current rank-96
launcher, use [the November 8 deployment](kie-training-runpod-r30-r96.md).
The shared `run_r30_runpod.sh` launcher now selects rank 96; the direct command
below records the original rank-64 configuration.

Prepared on `root@205.196.144.42:11447`.
GPU: RTX PRO 6000 Blackwell Server Edition, 97,887 MiB; driver 595.91.07.
The authoritative repository and outputs remain in `/workspace/DocumentParsing`.
The executing repository, newly recreated venv, dataset, tokenizer/model and
tokenized cache live in `/opt/DocumentParsing`, on the Pod-local overlay.
`/workspace` is network-backed MooseFS; Python code must not execute there.
The local runtime is disposable. Keep the network volume and back up outputs
before terminating/replacing the Pod.

## Start training

In the Pod terminal:

```bash
cd /workspace/DocumentParsing
bash scripts/training/run_r30_runpod.sh
```

The launcher verifies MLflow health, sets offline/cache environment variables,
requires the validated local filesystem, and executes the local training CLI.
No Hugging Face token or model download is needed.
This is a fresh run, not a resume command.

Equivalent direct command, without the launcher:

```bash
cd /opt/DocumentParsing
HF_HOME=/opt/DocumentParsing/.cache/huggingface \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/document-kie-train train \
  --config configs/training/deployment/r30_runpod_offline.yaml \
  --project-root /opt/DocumentParsing
```

After replacing the Pod, first rebuild its local runtime from the persistent copy:

```bash
bash /workspace/DocumentParsing/scripts/training/prepare_r30_runpod_local.sh
```

Provisioning never starts training. It refuses to replace runtime files while
training is running. The venv is recreated from the frozen lock, not moved with
stale shebangs/editable-install paths. The existing network venv is left intact.

MLflow is already started in the separate `r30-mlflow` tmux session, listening
only on `127.0.0.1:5003`. Forward port 5003 through VSCode/SSH to access its UI.
If the service needs restarting after a Pod restart:

```bash
tmux new-session -d -s r30-mlflow -c /workspace/DocumentParsing \
  'bash scripts/training/start_r30_mlflow.sh'
```

## What was deployed

- Repository history through `89496f4fe547ee01099e8903075ec5edabf9b6a2`, plus the
  current working files and deployment helpers. Git credentials/hooks/config
  were not copied. The network volume does not preserve workstation ownership;
  file-content checks pass, and Git ignores its unreliable executable-mode bits.
- Exact R30 train/validation files: 9,400 synthetic + 600 anonymized real train;
  60 validation. Manifest/schema accompany the data. Intermediate migration
  backups, historical datasets and old training outputs were not transferred.
- Base model and tokenizer: `google/t5gemma-2-270m-270m`, revision
  `7c38f16641f455ef0685b18431faf1b17722d5a1`. Six cached files total
  1,605,574,366 bytes; all SHA-256 hashes match the workstation.
- Frozen task constraints and prompt, with the existing pins intact.
- `.venv`: Python 3.12.3, frozen `uv.lock` training environment, including
  Torch 2.13.0/CUDA 13.0, Transformers 5.15.0, PEFT 0.19.1, Accelerate 1.14.0,
  ScheduleFree 1.4.1. The Pod's older system Torch is not used.
- Separate `/workspace/.venvs/mlflow`, MLflow 3.15.1; installing its server
  dependencies cannot change the locked training environment.

No `.env`, SSH keys or Hugging Face token/cache credentials were copied.

## Configuration and validation

`configs/training/deployment/r30_runpod_offline.yaml` is exactly the prepared
production RunPod config, except for `model.token_env: null`,
`model.local_files_only: true`, and an absolute persistent `run.output_dir`.
A regression test checks that these are the only effective differences.
It retains rank 64, microbatch 4, accumulation 12,
effective batch 48, 10 epochs / 2,090 optimizer updates, 105 warmup steps,
522-step evaluation/save cadence, evaluation microbatch 2, and generation cap
3,329. The local profile and the production RunPod profile were not changed.

The environment contract and dependency consistency checks pass. The actual
Pod trainer prepared all 10,060 records, verified the dataset/schema pins, and
cached tokenization, excluding/truncating zero records. Train maxima are 11,229
input tokens / 3,200 target tokens; validation maxima are 8,653 / 1,356.
The first cold preparation took 224.17 seconds, including cold imports from the
network volume. Subsequent starts reuse the tokenization cache.

Local configuration regression tests: 9 passed. Repository checksum comparison
found no differing file contents; directory timestamps are not training inputs.

The offline GPU smoke passed using the actual rank-64 LoRA construction and
gradient checkpointing. Both microbatch-4 controls produced finite loss and
finite gradients in all 504 adapter tensors:

| Control | Padded input / target lengths | Forward + backward | Peak reserved VRAM |
| --- | --- | ---: | ---: |
| Four longest inputs | 11,232 / 1,152 | 10.30 s | 19.89 GiB |
| Four longest targets | 8,672 / 3,200 | 3.88 s | 47.78 GiB |

The evaluation-microbatch-2, eight-token generation smoke also passed. These
are disposable compatibility/capacity checks, not a full training throughput
benchmark: no optimizer updates, EVA calibration or checkpoints were performed.
The training process will perform its normal EVA calibration when started.

GPU report: [gpu-preflight.json](analysis/r30-runpod-20261107/gpu-preflight.json).

Preparation receipt: [prepare-dataset.json](analysis/r30-runpod-20261107/prepare-dataset.json).

## Output locations

Training output:

```text
/workspace/DocumentParsing/artifacts/kie-training/t5gemma2-270m-mpci-bl-real600-synthetic9400-r30-paddle-centroids-inputonly-v7-e10-compact-eva-a32-r64-runpod-sfadamw-v1/
```

MLflow database/artifacts: `/workspace/DocumentParsing/artifacts/mlflow/`.
Active model/tokenizer cache: `/opt/DocumentParsing/.cache/huggingface/`.
Persistent staging copy: `/workspace/.cache/huggingface/`.

## Interrupted attempt and local-disk correction

The user-started attempt was interrupted cleanly at 16:42:23 UTC on November 7,
before EVA completed and before any checkpoint or optimizer update. At the
user's request, its eight run files and empty run directory were deleted.
MLflow run `6f82e482dae04ee0986c79e3f6750018` was soft-deleted through its API;
the shared database, other runs and prepared dataset caches were retained.
The soft-deleted MLflow metadata remains recoverable; the run files were removed.
**Training was not restarted.**

Diagnosis traced EVA's per-layer seeded low-rank SVD through
`torch.manual_seed` to deferred XPU/MTIA calls that capture Python stacks.
Stack formatting repeatedly checks source-file metadata. A live 10-second
sample observed the training process waiting on FUSE in 39 of 100 observations.
Repeated metadata checks averaged 1,706 microseconds on the network filesystem
versus 2.59 microseconds locally. This is a per-check comparison, not a claim
of a 658-fold end-to-end speedup. The local deployment changes storage only;
it does not patch PyTorch/PEFT, reduce calibration, or change numerical settings.

The corrected environment contains the same 116 locked packages; dependency
and training-image contract checks pass. Imports of the project, Torch, PEFT
and Transformers are verified to resolve under `/opt/DocumentParsing`.
Local configuration regression tests: 9 passed in 24.65 seconds.
General training configuration tests: 27 passed in 20.90 seconds.

The repeated local-runtime GPU smoke passed all 10,060 input/target checks,
retained the same tokenization-cache identity, and again produced finite
gradients in all 504 adapter tensors for both tail microbatches. Input-tail
forward/backward took 5.37 seconds (19.89 GiB peak reserved); target-tail took
3.90 seconds (47.78 GiB). The eight-token generation control passed. These
smokes perform no optimizer updates and create no training-run artifacts.
See [local GPU receipt](analysis/r30-runpod-20261107/localdisk-gpu-preflight.json).
