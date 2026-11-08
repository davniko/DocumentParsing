#!/usr/bin/env bash
# Loopback-only tracking; run in its own tmux session on the prepared Pod.
set -euo pipefail
cd /workspace/DocumentParsing
mkdir -p artifacts/mlflow/artifacts
exec /opt/.venvs/mlflow/bin/mlflow server \
  --host 127.0.0.1 --port 5003 --workers 1 \
  --backend-store-uri "sqlite:///$PWD/artifacts/mlflow/mlflow.db" \
  --artifacts-destination "$PWD/artifacts/mlflow/artifacts"
