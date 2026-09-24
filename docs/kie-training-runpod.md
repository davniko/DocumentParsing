# KIE training on RunPod (no nested Docker)

Prepared on 2026-09-23 in `/workspace/DocumentParsing`, cloned from
`davniko/DocumentParsing` at commit `6b98d18717a84a3602a30c8d42bd96bde27ed20b`.
The RunPod configuration is an additional, uncommitted file; the workstation
configuration and its artifacts are not replaced.

## Prepared environment and data

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB reported memory.
- NVIDIA driver: 595.91.07; compute capability 12.0.
- Training environment: `/workspace/DocumentParsing/.venv`, Python 3.12.3,
  Torch 2.13.0+cu130, Transformers 5.15.0, PEFT 0.19.1, Accelerate 1.14.0.
- Full MLflow server 3.15.1: `/workspace/.venvs/mlflow` (separate from training).
- Model/tokenizer cache: `/workspace/.cache/huggingface`, pinned revision
  `7c38f16641f455ef0685b18431faf1b17722d5a1` of `google/t5gemma-2-270m-270m`.
- Dataset: `artifacts/kie-training/datasets/mpci-bl-real1057-synthetic29910-recovered-v5-v1/`.
  This includes train/validation JSONLs, the frozen vocabulary and accompanying receipts.
- The repository, dataset, configuration and prompt retain their expected relative paths.
  Synthesis/template catalogs are not needed for training.

The model-access credential was passed in memory for preparation, not persisted
on the Pod. The GitHub credential was not placed in the remote URL or Git config.
Provide `HF_TOKEN` in the training shell as shown below. Do not paste secrets into
the committed config or runbook. The temporary GitHub token can be revoked after
setup if no further private-repository operations are required.

**Storage warning:** at preparation time `/workspace` was on the Pod's 30 GB
container overlay, not an independently persistent volume. Back up outputs
before stopping, restarting, replacing or terminating this Pod. Merely naming
a directory `/workspace` does not make it persistent. Configure a persistent
RunPod volume for future Pods before provisioning them.

## Training configuration

`configs/training/production/t5gemma2_270m_lora.mpci_bl_real1057_synthetic29910_recovered_v5_e5_eva_a32_runpod_v1.yaml`

| Setting | Value |
| --- | ---: |
| Raw training records | 30,967 |
| Target cutoff, including EOS | 5,500 tokens |
| Excluded training records | 68 |
| Retained training records | 30,899 |
| Unchanged real validation records | 100 |
| Microbatch | 4 |
| Gradient accumulation | 12 |
| Effective optimizer batch, one GPU | 48 |
| Gradient checkpointing | Off |
| Epochs | 5 |
| Training batches per epoch | 7,725 |
| Optimizer updates per epoch | 644 |
| Total optimizer updates | 3,220 |
| Evaluation and checkpoint interval | 322 updates |
| Scheduled evaluations | 10 |

Calculation: `ceil(ceil(30899 / 4) / 12) = 644`. The final accumulation
group of each epoch is partial because `drop_last` remains false. Evaluation
steps are 322, 644, 966, 1288, 1610, 1932, 2254, 2576, 2898 and 3220.
There is no separate start/final evaluation.

EVA settings, rank 32, alpha 32, rsLoRA, BF16, SDPA, learning rate 0.0001,
source limit 19,200, evaluation microbatch 2 and generation limit 2,048 remain
unchanged. The effective batch is 48 instead of 32, so five epochs have fewer
optimizer updates than the workstation configuration. No learning-rate scaling
was applied automatically.

The GPU and environment were checked, but **microbatch 4 with checkpointing off
has not been memory/throughput benchmarked with the full model**. This is the
requested test configuration, not a guarantee that every padded long batch fits.
No EVA calibration, training, evaluation or persistent MLflow server was started
during preparation.

## Connect and forward MLflow

From the workstation, using local port 5001 to avoid its existing MLflow server:

```bash
ssh -p 11834 -i ~/.ssh/id_ed25519 \
  -L 5001:127.0.0.1:5000 root@103.196.86.190
```

## Start MLflow on the Pod

In that SSH shell, create a persistent terminal session:

```bash
tmux new -s kie-mlflow
```

Then run:

```bash
cd /workspace/DocumentParsing
mkdir -p artifacts/mlflow/artifacts
/workspace/.venvs/mlflow/bin/mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --workers 1 \
  --backend-store-uri sqlite:////workspace/DocumentParsing/artifacts/mlflow/mlflow.db \
  --artifacts-destination /workspace/DocumentParsing/artifacts/mlflow/artifacts
```

Wait for server startup, then detach with **Ctrl+B, D**. Do not press Ctrl+C.
Open `http://localhost:5001` on the workstation while the SSH tunnel stays open.
The server binds only to loopback; no unauthenticated public MLflow port is needed.

## Prepare and start training on the Pod

Create a second persistent terminal session:

```bash
tmux new -s kie-train
```

Inside it:

```bash
cd /workspace/DocumentParsing
export HF_HOME=/workspace/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
read -rsp 'Hugging Face read token: ' HF_TOKEN
export HF_TOKEN
KIE_RUNPOD_CONFIG=configs/training/production/t5gemma2_270m_lora.mpci_bl_real1057_synthetic29910_recovered_v5_e5_eva_a32_runpod_v1.yaml

.venv/bin/document-kie-train validate-config \
  --config "$KIE_RUNPOD_CONFIG" --project-root /workspace/DocumentParsing

.venv/bin/document-kie-train prepare-dataset \
  --config "$KIE_RUNPOD_CONFIG" --project-root /workspace/DocumentParsing

curl --fail http://127.0.0.1:5000/health

.venv/bin/document-kie-train train \
  --config "$KIE_RUNPOD_CONFIG" --project-root /workspace/DocumentParsing
```

Only run the final command if preparation and the MLflow health check succeed.
The dataset-preparation command was already run during provisioning and can
reuse the tokenization cache. It reports filtered rows without truncating labels.
`HF_TOKEN` is required by the configured access contract even when weights are cached.

Detach with **Ctrl+B, D** to leave training running. Reattach with
`tmux attach -t kie-train`. Logs/checkpoints are written to:

`/workspace/DocumentParsing/artifacts/kie-training/t5gemma2-270m-mpci-bl-30k-eva-a32-runpod-b4-ga12-nogc-v1/`

A subsequent fresh attempt must use a new run ID or explicitly archive the old
failed output directory; existing artifacts are never overwritten silently.

## Recreate the environments if needed

The following setup has already been performed. It does not start training:

```bash
cd /workspace/DocumentParsing
uvx --from uv==0.11.8 uv sync --frozen --group train --no-dev --python 3.12
.venv/bin/python docker/training/verify_environment.py
uvx --from uv==0.11.8 uv pip check --python .venv/bin/python
```

For a new Pod where the MLflow environment does not yet exist:

```bash
uvx --from uv==0.11.8 uv venv --python 3.12 /workspace/.venvs/mlflow
uvx --from uv==0.11.8 uv pip install \
  --python /workspace/.venvs/mlflow/bin/python mlflow==3.15.1
```

MLflow is isolated so installing the full server cannot alter the locked
training environment. Do not copy the workstation's entire `.env`, virtual
environment, caches, historical training outputs or synthesis artifacts.

References: [RunPod storage](https://docs.runpod.io/pods/storage/types),
[MLflow tracking server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/).
