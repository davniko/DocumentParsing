# R30 rank-96 RunPod deployment — 8 November 2026

Replacement pod: `root@198.13.252.15:18519`, RTX PRO 6000 Blackwell Server
Edition, 97,887 MiB. This is a fresh rank-96 run, not a rank-64 checkpoint resume.

## Start training

Run in the pod's VSCode terminal:

```bash
cd /opt/DocumentParsing
bash scripts/training/run_r30_runpod.sh
```

The launcher selects `configs/training/deployment/r30_r96_runpod_offline.yaml`,
checks MLflow health, and uses only the exact local model/tokenizer cache.
It does not need a Hugging Face credential.

Rank 96, alpha 32, rsLoRA, EVA, learning rate and all other settings match the
prepared rank-96 RunPod profile. Microbatch 4 × accumulation 12 gives effective
batch 48. Ten epochs yield 2,090 optimizer updates; warmup is 105 updates,
evaluation/save interval 522, logging interval 3, and generation limit 3,329.

## Storage and environments

- `/opt/DocumentParsing`: executing source, frozen-lock training `.venv`, exact
  base model/tokenizer, current dataset and tokenized dataset cache. This is
  local overlay storage, avoiding the previous network-filesystem EVA overhead.
- `/opt/.venvs/mlflow`: separate MLflow 3.15.1 server environment. Installing the
  server does not alter the locked training environment.
- `/workspace/DocumentParsing`: persistent repository and dataset copy;
  checkpoints/results under `artifacts/kie-training`, MLflow database/artifacts
  under `artifacts/mlflow`.
- `/workspace/.cache/huggingface/hub`: persistent copy of the pinned base model.

The persistent filesystem is network-backed; runtime imports must resolve under
`/opt`. Do not move a venv: recreate it with the frozen lock. The repository
includes the current working changes, not just the last Git commit.

MLflow listens on `127.0.0.1:5003` in tmux session `r30-mlflow`. If stopped, start
it in a separate terminal:

```bash
bash /opt/DocumentParsing/scripts/training/start_r30_mlflow.sh
```

Do not start a second server if the health endpoint already responds. VSCode's
Ports view / Open in Browser can open port 5003 as usual.

## Dataset and model

The corrected R30 dataset contains 9,400 synthetic + 600 anonymized real training
records and 60 validation records. The deployment config pins all three current
file hashes, including the reviewed November 8 party-label amendments. Old
training outputs, migration intermediates and stale tokenization caches are not
needed or transferred. Tokenization is rebuilt locally against current hashes.

Base model/tokenizer: `google/t5gemma-2-270m-270m`, revision
`7c38f16641f455ef0685b18431faf1b17722d5a1`. The six snapshot files are verified
against the workstation copies. No `.env`, SSH private key or Hugging Face
credential is installed on the pod.

## Preflight scope

The deployment preflight validates pinned data/contracts, prepares all examples,
loads the rank-96 model offline, and runs forward/backward passes on both the
longest-input and longest-target microbatches. It checks finite loss/gradients
and performs an eight-token generation smoke test. It performs **zero optimizer
updates**, does not run EVA calibration, and creates no training checkpoints.

Its machine-readable result is
`docs/analysis/r30-r96-runpod-20261108/preflight.json`.

## Verified results

- All 10,060 records validated; no skipped targets or truncated inputs. Dataset
  preparation, including fresh tokenization, took 47.66 seconds.
- Maximum training input/target: 11,229 / 3,200 tokens; validation:
  8,653 / 1,356. The existing generation limit remains sufficient for gold targets.
- Rank 96 activates 45,563,904 trainable parameters across 252 modules.
  All 504 adapter tensors had finite gradients in both tail tests.
- Longest-input microbatch: 4 × 11,232 padded input / 4 × 1,152 target,
  5.43 seconds forward/backward, 19.72 GiB peak allocated GPU memory.
- Longest-target microbatch: 4 × 8,672 input / 4 × 3,200 target,
  3.80 seconds, 47.49 GiB peak allocated (47.89 GiB reserved).
- Offline generation passed for a batch of two. These are disposable preflight
  measurements, not full EVA/training throughput or an optimizer-memory benchmark.
- Local configuration tests: 3 passed; the Torch-dependent shuffle test was
  unavailable on the workstation and its exact assertions passed on the pod.
  Shell syntax, Ruff and whitespace checks passed. The frozen 116-package
  training environment passes the version/import contract and dependency check.
- All selected working-source files match the workstation by checksum; all six
  model files and dataset splits match SHA-256. Runtime imports for project,
  Torch, Transformers and PEFT resolve locally under `/opt/DocumentParsing`.
- A 200-call metadata probe measured median `stat` latency of 0.00162 ms locally
  versus 0.10599 ms on `/workspace`. This verifies the storage distinction;
  it is not an end-to-end training speedup claim. Local storage has about 36 GiB free.
- MLflow health and UI return HTTP 200; database/artifacts are persistent.
  Training has not been launched and no optimizer update/checkpoint was produced.

## Rebuild after replacing the pod

If this persistent volume is retained, rebuild the disposable environments with:

```bash
bash /workspace/DocumentParsing/scripts/training/prepare_r30_runpod_local.sh
```

This restores both local environments and verifies training imports without
starting training. Start MLflow separately, then launch training yourself. The
helper includes non-ignored uncommitted source/configuration files as well as
tracked files, so the reviewed working-tree repairs are preserved.
