#!/usr/bin/env bash
# Rebuild R31's disposable runtime; keep outputs persistent. Never starts training.
set -euo pipefail
cd /workspace/DocumentParsing
if pgrep -f '[d]ocument-kie-train train' >/dev/null; then
  echo 'Stop training before replacing its runtime files.' >&2
  exit 1
fi
if [[ $(findmnt -n -o FSTYPE -T /opt) != overlay ]]; then
  echo 'Expected the validated Pod-local overlay at /opt.' >&2
  exit 1
fi
mkdir -p /opt/DocumentParsing
git ls-files -z --cached --others --exclude-standard |
  rsync -rt --from0 --files-from=- ./ /opt/DocumentParsing/

dataset_dir=data/curated/mpci-bl-r31-paddle-centroids-v1
mkdir -p "/opt/DocumentParsing/$dataset_dir"
rsync -rt \
  "$dataset_dir/synthetic-train.jsonl" \
  "$dataset_dir/real-train.jsonl" \
  "$dataset_dir/validation.jsonl" \
  "$dataset_dir/manifest.json" \
  "$dataset_dir/schema.json" \
  "/opt/DocumentParsing/$dataset_dir/"

model_dir=models--google--t5gemma-2-270m-270m
mkdir -p /opt/DocumentParsing/.cache/huggingface/hub
rsync -rt "/workspace/.cache/huggingface/hub/$model_dir/" \
  "/opt/DocumentParsing/.cache/huggingface/hub/$model_dir/"

cd /opt/DocumentParsing
export UV_CACHE_DIR=/opt/.cache/uv
# Recreate here: a copied venv retains old shebangs and editable import paths.
UV_LINK_MODE=copy uvx --from uv==0.11.8 uv sync \
  --frozen --group train --no-dev --python /usr/bin/python3
uvx --from uv==0.11.8 uv pip check --python .venv/bin/python
.venv/bin/python docker/training/verify_environment.py
.venv/bin/python - <<'PY'
from pathlib import Path
import document_ocr
import peft
import torch
import transformers
root = Path('/opt/DocumentParsing')
for module in (document_ocr, peft, torch, transformers):
    path = Path(module.__file__).resolve()
    if not path.is_relative_to(root):
        raise RuntimeError(f'Non-local runtime import: {path}')
    print(f'{module.__name__}: {path}')
PY
# Isolate the server dependencies from the locked trainer environment.
if [[ ! -x /opt/.venvs/mlflow/bin/python ]]; then
  uvx --from uv==0.11.8 uv venv /opt/.venvs/mlflow --python /usr/bin/python3
fi
UV_LINK_MODE=copy uvx --from uv==0.11.8 uv pip install \
  --python /opt/.venvs/mlflow/bin/python mlflow==3.15.1
uvx --from uv==0.11.8 uv pip check --python /opt/.venvs/mlflow/bin/python
printf '%s\n' 'R31 local runtime prepared. Training was NOT started.'
