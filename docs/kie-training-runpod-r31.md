# R31 rank64 RunPod deployment — 9 November 2026

Pod: `root@136.255.118.0:23096`, RTX PRO 6000 Blackwell Server Edition,
97,887 MiB. This is a fresh R31 rank64 run, not a checkpoint resume.

## Start training

In the pod's VSCode terminal:

```bash
cd /opt/DocumentParsing
bash scripts/training/run_r31_runpod.sh
```

Equivalent direct command:

```bash
cd /opt/DocumentParsing
HF_HOME=/opt/DocumentParsing/.cache/huggingface \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/document-kie-train train \
  --config configs/training/deployment/r31_runpod_offline.yaml \
  --project-root /opt/DocumentParsing
```

The launcher verifies local storage and MLflow health. MLflow is already running
in tmux session `r31-mlflow`, on `127.0.0.1:5003`. If stopped, start it in another
terminal with `bash /opt/DocumentParsing/scripts/training/start_r31_mlflow.sh`.
Do not start a second server while that port is healthy.

## Configuration and storage

- Data:10,150 synthetic +600 anonymized real training records,60 validation.
  Includes the final two-description readiness amendment and current file hashes.
- Rank64,alpha32,rsLoRA,EVA and the existing training hyperparameters are unchanged.
  Microbatch4 × accumulation12 gives effective batch48. Ten epochs yield2,240
  optimizer updates; warmup112,eval/save560,logging3,generation cap3,329.
- Deployment differs from the prepared production RunPod config only in offline
  model access/no credential requirement and the absolute persistent output path.
- `/opt/DocumentParsing`: executing source,new frozen-lock `.venv`,dataset,pinned
  model/tokenizer and tokenized cache. Imports and caches stay on local overlay.
- `/opt/.venvs/mlflow`: separate MLflow3.15.1 server environment; it cannot change
  the training environment's locked dependencies.
- `/workspace/DocumentParsing`: persistent repository and dataset copy. Training
  outputs are under `artifacts/kie-training`; tracking DB/artifacts under
  `artifacts/mlflow`. Keep/back up this volume before terminating the pod.

Unlike the initial R30 pod, this pod mounts `/workspace` as XFS on `/dev/md1`, not
MooseFS. The local/persistent split is nevertheless retained. Measured median
metadata latency was4.84 microseconds at `/opt` and4.32 at `/workspace`; neither
showed the previous millisecond-scale metadata delay.

The base model/tokenizer is `google/t5gemma-2-270m-270m`, exact revision
`7c38f16641f455ef0685b18431faf1b17722d5a1`. All six files match workstation SHA-256.
Weights were fetched directly to the pod using temporary file-specific download
authorization:1,572,182,264 bytes in25.63s,about61.34MB/s. No Hugging Face token,
workstation `.env`, private SSH key or Git credential configuration was copied.

Repository history through `bc9fd1638d16c53ac24ea6fb3d535891ebb4eeb4`, plus the
current working source, was transferred. Historical datasets, old training runs
and migration intermediates are unnecessary for training and were not copied.
The transfer manifest records2,416 source files,five dataset files and six model
files; both persistent and active copies passed exact checksums.

## Rebuild the disposable runtime

If the persistent copy survives replacement of this pod:

```bash
bash /workspace/DocumentParsing/scripts/training/prepare_r31_runpod_local.sh
```

This recreates environments from the frozen training lock and exact model copy,
without starting training or deleting prior outputs. It deliberately does not
copy an old venv or tokenization cache. Start MLflow separately after rebuilding.

Evidence is under `docs/analysis/r31-runpod-20261109/` locally and on the pod.

## Verified preflight

The frozen116-package training environment passes dependency and exact-version
checks (Torch2.13.0/CUDA13.0,Transformers5.15.0,PEFT0.19.1). Local regression tests
confirm that deployment changes only storage/model access:seven tests passed.

All10,810 records validated and tokenized, with zero source truncations or target
exclusions. Cold preparation took66.74s; subsequent startup reuses the local
tokenization cache. Maximum training input/target lengths are11,229/3,182;
validation8,653/1,356. No cap or sample-count adjustment is needed.

The actual configured rank64 model passes disposable GPU controls with finite
loss and gradients in all504 adapter tensors:

| Microbatch4 control | Padded input/target | Forward and backward | Peak reserved VRAM |
|---|---|---:|---:|
| Four longest inputs | 11,232 /1,120 | 6.70s | 19.42GiB |
| Four longest targets | 8,672 /3,184 | 3.85s | 47.55GiB |

An eight-token, evaluation-batch2 generation smoke also passes. These are
compatibility/capacity probes, not a full optimizer-memory or training-throughput
benchmark. No optimizer update or training checkpoint was produced.

The four-forward disposable EVA probe completed in22.45s, with2.92GiB peak
reserved memory. Its10,568 `stat` calls consumed0.0775s in total; the profile is
dominated by the expected low-rank SVD work, not network metadata. This is close
to the earlier repaired rank64 local-storage probe (22.36s), not a prediction of
full calibration convergence time. The production calibration ceiling remains512;
the four-pass limit exists only in the disposable probe. Training has not started.
