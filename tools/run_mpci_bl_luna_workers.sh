#!/usr/bin/env bash
# Launch immutable one-document Luna assignments with six active workers whenever possible.
set -euo pipefail

root=/mnt/d/Projects/DocumentParsing
run=${MPCI_BL_LABEL_DEST:?MPCI_BL_LABEL_DEST must name the prepared retry directory}
case "$run" in
  "$root"/artifacts/kie-labels/*) ;;
  *) echo "retry directory must be under $root/artifacts/kie-labels" >&2; exit 1 ;;
esac

mapfile -t ids < <(TMPDIR=/tmp UV_CACHE_DIR=/tmp/documentparsing-uv-cache uv run python - <<'PY'
import json
import os
from pathlib import Path

run = Path(os.environ["MPCI_BL_LABEL_DEST"])
print(*json.loads((run / "run-metadata.json").read_text())["sampling"]["selectedDocumentIdsOrdered"], sep="\n")
PY
)
if [[ ${#ids[@]} -ne 15 ]]; then
  echo 'expected exactly 15 work items' >&2
  exit 1
fi

event_log="$run/worker-logs/launcher-events.jsonl"
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
  local id=$1 attempt=1 work candidate log pid
  work="$run/work-items/$id.json"
  candidate="$run/candidates/$id.json"
  log="$run/worker-logs/$id.attempt-$attempt.jsonl"
  if [[ ! -f $work || -e $candidate || -e $log ]]; then
    echo "refusing invalid worker paths for $id" >&2
    exit 1
  fi
  bwrap --die-with-parent --new-session \
    --ro-bind / / \
    --proc /proc \
    --dev /dev \
    --bind /home/davidn/.codex /home/davidn/.codex \
    --bind "$root" "$root" \
    --tmpfs /tmp \
    --setenv TMPDIR /tmp \
    --setenv TEMP /tmp \
    --setenv TMP /tmp \
    --setenv UV_CACHE_DIR /tmp/documentparsing-uv-cache \
    --chdir "$root" \
    codex exec --ephemeral --json --model gpt-5.6-luna -c 'model_reasoning_effort="xhigh"' --sandbox workspace-write --cd /mnt/d/Projects/DocumentParsing - >"$log" 2>&1 <<EOF &
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
  pid=$!
  pid_to_id["$pid"]=$id
  printf '{"event":"launched","documentId":"%s","attempt":1,"launcherProcessId":%s,"timestamp":"%s"}\n' "$id" "$pid" "$(timestamp)" >>"$event_log"
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
  printf '{"event":"completed","documentId":"%s","attempt":1,"launcherProcessId":%s,"exitStatus":%s,"candidatePresent":%s,"timestamp":"%s"}\n' \
    "$id" "$completed_pid" "$exit_status" "$([[ -f $candidate ]] && echo true || echo false)" "$(timestamp)" >>"$event_log"
  if (( exit_status != 0 )) || [[ ! -f $candidate ]]; then
    failures=1
  fi
done

if (( failures != 0 )); then
  echo 'one or more Luna workers failed; remaining work was not launched' >&2
  exit 1
fi
