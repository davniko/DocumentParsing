#!/usr/bin/env bash
# Fresh R31 rank64 run; runtime/cache local, checkpoints and tracking persistent.
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
  --config configs/training/deployment/r31_runpod_offline.yaml \
  --project-root "$PWD"
