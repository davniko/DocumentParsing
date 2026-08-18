#!/usr/bin/env bash
# Launch immutable one-document Luna assignments with six active workers whenever possible.
set -euo pipefail

root=/mnt/d/Projects/DocumentParsing
run=${MPCI_BL_LABEL_DEST:?MPCI_BL_LABEL_DEST must name the prepared retry directory}
attempt=${MPCI_BL_LABEL_ATTEMPT:-1}
retry_guidance=${MPCI_BL_LABEL_RETRY_GUIDANCE:-}
if [[ ! $attempt =~ ^[1-9][0-9]*$ ]]; then
  echo 'MPCI_BL_LABEL_ATTEMPT must be a positive integer' >&2
  exit 1
fi
case "$run" in
  "$root"/artifacts/kie-labels/*) ;;
  *) echo "retry directory must be under $root/artifacts/kie-labels" >&2; exit 1 ;;
esac

mapfile -t all_ids < <(TMPDIR=/tmp UV_CACHE_DIR=/tmp/documentparsing-uv-cache uv run python - <<'PY'
import json
import os
from pathlib import Path

run = Path(os.environ["MPCI_BL_LABEL_DEST"])
print(*json.loads((run / "run-metadata.json").read_text())["sampling"]["selectedDocumentIdsOrdered"], sep="\n")
PY
)
declare -A known_ids=()
declare -A requested_ids=()
for id in "${all_ids[@]}"; do
  known_ids["$id"]=1
done
if [[ -n ${MPCI_BL_LABEL_DOCUMENT_IDS:-} ]]; then
  IFS=, read -r -a ids <<<"$MPCI_BL_LABEL_DOCUMENT_IDS"
else
  ids=("${all_ids[@]}")
fi
if (( attempt == 1 )) && [[ ${#ids[@]} -ne 15 ]]; then
  echo 'first pass must launch exactly 15 work items' >&2
  exit 1
fi
if (( attempt > 1 )) && [[ ${#ids[@]} -eq 0 || ${#ids[@]} -gt 5 ]]; then
  echo 'a retry pass must launch between one and five work items' >&2
  exit 1
fi
if (( attempt == 1 )) && [[ -n $retry_guidance ]]; then
  echo 'retry guidance is allowed only for a retry attempt' >&2
  exit 1
fi
for id in "${ids[@]}"; do
  if [[ -z ${known_ids[$id]:-} || -n ${requested_ids[$id]:-} ]]; then
    echo "retry includes an unknown document ID: $id" >&2
    exit 1
  fi
  requested_ids["$id"]=1
done

if (( attempt == 1 )); then
  event_log="$run/worker-logs/launcher-events.jsonl"
else
  event_log="$run/worker-logs/launcher-events.attempt-$attempt.jsonl"
fi
if [[ -e $event_log ]]; then
  echo "refusing existing launcher event log: $event_log" >&2
  exit 1
fi
: >"$event_log"

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

declare -A pid_to_id=()
active=0
next=0
failures=0

launch() {
  local id=$1 work candidate log stage pid retry_context
  work="$run/work-items/$id.json"
  candidate="$run/candidates/$id.json"
  log="$run/worker-logs/$id.attempt-$attempt.jsonl"
  if [[ ! -f $work || -e $candidate || -e $log ]]; then
    echo "refusing invalid worker paths for $id" >&2
    exit 1
  fi
  stage=$(mktemp -d "$run/.worker-stage-$id.XXXXXX")
  retry_context=
  if [[ -n $retry_guidance ]]; then
    retry_context=$(cat <<EOF
This is a fresh replacement after the overseer rejected an earlier candidate. Independently apply
the frozen reference, with particular care for this rejection criterion:
$retry_guidance
Do not inspect, reconstruct, or rely on the earlier candidate.

EOF
)
  fi
  (
    set +e
    bwrap --die-with-parent --new-session \
      --ro-bind / / \
      --proc /proc \
      --dev /dev \
      --bind /home/davidn/.codex /home/davidn/.codex \
      --tmpfs "$root" \
      --dir "$root/artifacts" \
      --dir "$root/artifacts/kie-labels" \
      --dir "$run" \
      --dir "$run/work-items" \
      --dir "$run/candidates" \
      --ro-bind "$root/.git" "$root/.git" \
      --ro-bind "$root/AGENTS.md" "$root/AGENTS.md" \
      --ro-bind "$root/LABELING_SESSION_PROMPT.md" "$root/LABELING_SESSION_PROMPT.md" \
      --ro-bind "$root/MPCI_BILL_OF_LADING_LABELING_REFERENCE.md" "$root/MPCI_BILL_OF_LADING_LABELING_REFERENCE.md" \
      --ro-bind "$root/pyproject.toml" "$root/pyproject.toml" \
      --ro-bind "$root/uv.lock" "$root/uv.lock" \
      --ro-bind "$root/.python-version" "$root/.python-version" \
      --ro-bind "$root/.venv" "$root/.venv" \
      --ro-bind "$root/src" "$root/src" \
      --ro-bind "$work" "$work" \
      --bind "$stage" "$run/candidates" \
      --tmpfs /tmp \
      --setenv TMPDIR /tmp \
      --setenv TEMP /tmp \
      --setenv TMP /tmp \
      --setenv UV_CACHE_DIR /tmp/documentparsing-uv-cache \
      --chdir "$root" \
      codex exec --ephemeral --json --model gpt-5.6-luna -c 'model_reasoning_effort="xhigh"' --sandbox workspace-write --cd /mnt/d/Projects/DocumentParsing - >"$log" 2>&1 <<EOF
$retry_context
Label exactly one multi-page Bill-of-Lading document. Read AGENTS.md, the frozen Pydantic label
models, MPCI_BILL_OF_LADING_LABELING_REFERENCE.md, and only $work. The raw OCR text
in that work item is the authoritative target
boundary. The PDF/page images listed there are auxiliary for structure and grouping only: never add
or correct a value from them unless the chosen value already occurs in raw OCR. Apply the field
meanings, authorized categorical mappings, unresolved-code prohibitions, absence, normalization,
ordering, relationship, and evidence rules in MPCI_BILL_OF_LADING_LABELING_REFERENCE.md and
LABELING_SESSION_PROMPT.md. Copy the immutable source/provenance object mechanically from the
work item and assert exact equality with candidate.source; never retype its hashes or IDs. Write
exactly one candidate
MpciBillOfLadingAnnotation atomically to $candidate, with one evidence record per emitted
target leaf. In every evidence record retain the exact pre-mapping raw OCR value, page number, and
surrounding excerpt under rawOcrEvidence; the mapped/converted value remains in the label at
targetPath. Set reviewStatus="candidate" and validate with Pydantic before returning. Do not edit
any shared file, schema, manifest, source artifact, or other document's output. Do not label another
document and do not spawn agents. Return the document ID, output path, validation result, warnings,
and nothing resembling a second label.
EOF
    worker_status=$?
    if (( worker_status == 0 )); then
      mapfile -t staged_files < <(find "$stage" -maxdepth 1 -type f -printf '%f\n')
      if [[ ${#staged_files[@]} -ne 1 || ${staged_files[0]} != "$id.json" || -e $candidate ]]; then
        printf '%s\n' 'worker candidate staging invariant failed' >>"$log"
        worker_status=1
      elif ! mv "$stage/$id.json" "$candidate"; then
        printf '%s\n' 'worker candidate transfer failed' >>"$log"
        worker_status=1
      fi
    fi
    if [[ -d $stage ]] && [[ -z $(find "$stage" -mindepth 1 -print -quit) ]]; then
      rmdir "$stage"
    fi
    exit "$worker_status"
  ) &
  pid=$!
  pid_to_id["$pid"]=$id
  printf '{"event":"launched","documentId":"%s","attempt":%s,"launcherProcessId":%s,"retryGuidanceProvided":%s,"timestamp":"%s"}\n' "$id" "$attempt" "$pid" "$([[ -n $retry_guidance ]] && echo true || echo false)" "$(timestamp)" >>"$event_log"
}

while (( next < ${#ids[@]} || active > 0 )); do
  while (( failures == 0 && active < 6 && next < ${#ids[@]} )); do
    launch "${ids[next]}"
    next=$((next + 1))
    active=$((active + 1))
  done
  if (( active == 0 )); then
    break
  fi
  completed_pid=
  if wait -n -p completed_pid "${!pid_to_id[@]}"; then
    exit_status=0
  else
    exit_status=$?
  fi
  id=${pid_to_id[$completed_pid]:-}
  if [[ -z $id ]]; then
    echo "launcher could not associate completed worker process" >&2
    exit 1
  fi
  unset 'pid_to_id[$completed_pid]'
  active=$((active - 1))
  candidate="$run/candidates/$id.json"
  printf '{"event":"completed","documentId":"%s","attempt":%s,"launcherProcessId":%s,"exitStatus":%s,"candidatePresent":%s,"timestamp":"%s"}\n' \
    "$id" "$attempt" "$completed_pid" "$exit_status" "$([[ -f $candidate ]] && echo true || echo false)" "$(timestamp)" >>"$event_log"
  if (( exit_status != 0 )) || [[ ! -f $candidate ]]; then
    failures=1
  fi
done

if (( failures != 0 )); then
  echo 'one or more Luna workers failed; remaining work was not launched' >&2
  exit 1
fi
