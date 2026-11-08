#!/usr/bin/env bash
# Source and Python must live on local disk; only run outputs use the network volume.
# Current campaign: repaired R30, rank 96. This does not resume the rank-64 run.
set -euo pipefail
cd /opt/DocumentParsing
if [[ $(findmnt -n -o FSTYPE -T "$PWD") != overlay ]]; then
  echo 'Expected the validated Pod-local overlay at /opt/DocumentParsing.' >&2
  exit 1
fi
export HF_HOME=/opt/DocumentParsing/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
curl --fail --silent --show-error http://127.0.0.1:5003/health >/dev/null
exec .venv/bin/document-kie-train train \
  --config configs/training/deployment/r30_r96_runpod_offline.yaml \
  --project-root "$PWD"
