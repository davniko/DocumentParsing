#!/usr/bin/env bash
set -Eeuo pipefail

readonly poll_seconds=60
readonly training_run_id="t5gemma2-270m-lora-mpci-bl-pilot106-v5"
readonly ocr_config="configs/glm_ocr.blc500-followup.local.yaml"
readonly ocr_output="artifacts/glm-ocr/pilots/blc500-followup"

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "${project_root}"

export TMPDIR="/tmp"
ocr_entrypoint="${project_root}/.venv/bin/document-ocr"
if [[ ! -x "${ocr_entrypoint}" ]]; then
  printf 'Missing executable OCR entry point: %s\n' "${ocr_entrypoint}" >&2
  exit 1
fi

training_manifest="artifacts/kie-training/${training_run_id}/manifest.json"
while [[ ! -f "${training_manifest}" ]]; do
  trainer_container="$({
    docker ps --quiet \
      --filter "label=com.docker.compose.project=document-ocr" \
      --filter "label=com.docker.compose.service=kie-trainer"
  } | head -n 1)"
  if [[ -z "${trainer_container}" ]]; then
    printf 'Training stopped without publishing %s; refusing to start OCR.\n' \
      "${training_manifest}" >&2
    exit 1
  fi
  printf '[%s] Training is still running; checking again in %s seconds.\n' \
    "$(date --iso-8601=seconds)" "${poll_seconds}"
  sleep "${poll_seconds}"
done

python3 - "${training_manifest}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
manifest = json.loads(path.read_text(encoding="utf-8"))
if manifest.get("status") != "complete":
    raise SystemExit(f"training manifest is not complete: {path}")
PY
printf '[%s] Training completed successfully.\n' "$(date --iso-8601=seconds)"

if [[ ! -r .env ]]; then
  printf 'Missing readable project .env file.\n' >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
source ./.env
set +a
: "${VLLM_API_KEY:?VLLM_API_KEY must be set in .env}"

vllm_was_running=false
if [[ -n "$(docker compose ps --status running --quiet glm-ocr-vllm)" ]]; then
  vllm_was_running=true
else
  while true; do
    gpu_memory="$({
      nvidia-smi \
        --query-gpu=memory.total,memory.free \
        --format=csv,noheader,nounits
    } | head -n 1)"
    IFS=',' read -r gpu_total_mib gpu_free_mib <<<"${gpu_memory}"
    gpu_total_mib="${gpu_total_mib//[[:space:]]/}"
    gpu_free_mib="${gpu_free_mib//[[:space:]]/}"
    required_free_mib=$((gpu_total_mib * 90 / 100 + 512))
    if ((gpu_free_mib >= required_free_mib)); then
      break
    fi
    printf '[%s] GPU has %s MiB free; GLM-OCR requires at least %s MiB. Checking again in %s seconds.\n' \
      "$(date --iso-8601=seconds)" "${gpu_free_mib}" "${required_free_mib}" "${poll_seconds}"
    sleep "${poll_seconds}"
  done
fi

log_root="${ocr_output}/operator-logs"
mkdir -p -- "${log_root}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
progress_log="${log_root}/followup500-${timestamp}.progress.jsonl"
result_log="${log_root}/followup500-${timestamp}.result.json"
vllm_log="${log_root}/followup500-${timestamp}.vllm.log"
vllm_log_since="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

vllm_log_pid=""
stop_started_vllm() {
  if [[ "${vllm_was_running}" == false ]]; then
    docker compose stop glm-ocr-vllm >/dev/null
  fi
  if [[ -n "${vllm_log_pid}" ]]; then
    kill "${vllm_log_pid}" 2>/dev/null || true
    wait "${vllm_log_pid}" 2>/dev/null || true
  fi
}
trap stop_started_vllm EXIT

printf '[%s] GPU is free; starting GLM-OCR vLLM.\n' "$(date --iso-8601=seconds)"
docker compose up --detach --wait --wait-timeout 3600 glm-ocr-vllm

docker compose logs \
  --follow \
  --no-color \
  --timestamps \
  --since "${vllm_log_since}" \
  glm-ocr-vllm \
  | VLLM_LOG_SECRET="${VLLM_API_KEY}" python3 -u -c '
import os
import sys

secret = os.environ["VLLM_LOG_SECRET"]
for line in sys.stdin:
    sys.stdout.write(line.replace(secret, "[REDACTED]"))
    sys.stdout.flush()
' >"${vllm_log}" &
vllm_log_pid=$!

# Reacquire the working-directory handle after the long-running service startup.
# This directly guards the observed stale-handle failure, and uv project discovery
# is unnecessary once the pinned environment has been prepared.
cd /
cd -- "${project_root}"

printf '[%s] Starting the 500-document raw OCR extraction.\n' "$(date --iso-8601=seconds)"
"${ocr_entrypoint}" run \
  --config "${ocr_config}" \
  --project-root "${project_root}" \
  > >(tee "${result_log}") \
  2> >(tee "${progress_log}" >&2)

printf '[%s] OCR extraction completed; result: %s\n' \
  "$(date --iso-8601=seconds)" "${result_log}"
