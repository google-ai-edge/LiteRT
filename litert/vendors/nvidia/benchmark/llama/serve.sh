#!/usr/bin/env bash
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
# Starts or stops a llama-server for one model, the way the text and MTP measurements and the
# evaluation use it: one slot, the context asked for, the MTP drafter when asked, under GNU time,
# with the GPU memory sampled while it lives.
#
#   serve.sh start HANDLE_DIR PROFILE --ctx N [--mtp] [--port P] [--no-gpu-sampling] [--arg ARG]...
#       starts the server, waits until /health answers (up to 900 s), prints the handle directory.
#       HANDLE_DIR receives server.log, time.txt, gpu.tsv, serve.json (command, start, ready
#       time) and the pids. Exit 1 when the server dies or does not become ready.
#   serve.sh stop HANDLE_DIR
#       interrupts the server (its own shutdown, so GNU time reports it), stops the sampler.
set -euo pipefail
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/env.sh"

cmd=${1:-}; handle=${2:-}
[[ -n $cmd && -n $handle ]] || { sed -n '2,13p' "$0"; exit 2; }

if [[ $cmd == stop ]]; then
  if [[ -f $handle/sampler.pid ]]; then kill "$(cat "$handle/sampler.pid")" 2>/dev/null || true; fi
  if [[ -f $handle/server.pid ]]; then
    pid=$(cat "$handle/server.pid")
    # The server is GNU time's child; interrupt the server itself so time writes its report.
    child=$(pgrep -P "$pid" 2>/dev/null | head -1 || true)
    kill -INT "${child:-$pid}" 2>/dev/null || true
    for _ in $(seq 1 100); do kill -0 "$pid" 2>/dev/null || break; sleep 0.2; done
    kill -KILL "${child:-$pid}" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  rm -f "$handle/server.pid" "$handle/sampler.pid"
  exit 0
fi
[[ $cmd == start ]] || lb_die "serve.sh: start or stop"

profile=${3:-}; shift 3 || lb_die "serve.sh start HANDLE_DIR PROFILE ..."
ctx= mtp=false port=$LLAMA_PORT sample=true extra=()
while (($#)); do
  case "$1" in
    --ctx) ctx=$2; shift 2 ;;
    --mtp) mtp=true; shift ;;
    --port) port=$2; shift 2 ;;
    --no-gpu-sampling) sample=false; shift ;;
    --arg) extra+=("$2"); shift 2 ;;
    *) lb_die "serve.sh: unknown option $1" ;;
  esac
done
[[ -n $ctx ]] || lb_die "serve.sh: --ctx is required"
model=$(lb_model_file "$profile")
mapfile -t model_args < <(lb_model_args "$profile")
args=("$LLAMA_BIN/llama-server" -m "$model" "${model_args[@]}" -lm "$LLAMA_LOAD_MODE" -c "$ctx" -np 1 --port "$port" --no-warmup)
if [[ $mtp == true ]]; then
  args+=(--spec-type draft-mtp -md "$(lb_draft_file "$profile")" --spec-draft-n-max 3 -ngld 99)
fi
args+=("${extra[@]}")
mkdir -p "$handle"
! ss -ltn 2>/dev/null | rg -q ":$port " || lb_die "Port $port is in use"
lb_runtime_env
smi=$(lb_smi)
if [[ $sample == true ]]; then
  ( while :; do printf '%s\t%s\n' "$(date +%s.%N)" "$("$smi" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"; sleep 0.1; done ) > "$handle/gpu.tsv" 2>/dev/null &
  echo $! > "$handle/sampler.pid"
fi
export LD_LIBRARY_PATH="$LLAMA_BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
start=$(date +%s.%N)
start_monotonic=$(python3 -c 'import time; print(time.monotonic())')
/usr/bin/time -v -o "$handle/time.txt" "${args[@]}" > "$handle/server.log" 2>&1 &
pid=$!
echo "$pid" > "$handle/server.pid"
ready= ready_monotonic=
for _ in $(seq 1 4500); do
  if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
    ready_monotonic=$(python3 -c 'import time; print(time.monotonic())')
    ready=$(date +%s.%N)
    break
  fi
  kill -0 "$pid" 2>/dev/null || break
  sleep 0.2
done
python3 - "$handle/serve.json" "$start" "${ready:-}" "$start_monotonic" "$ready_monotonic" "$pid" "${args[@]}" <<'PY'
import json, sys
path, start, ready, start_monotonic, ready_monotonic, pid, *command = sys.argv[1:]
json.dump({'command': command, 'pid': int(pid), 'start_epoch': float(start),
           'ready_epoch': float(ready) if ready else None,
           'elapsed_clock': 'monotonic',
           'ready_s': (float(ready_monotonic) - float(start_monotonic)) if ready else None},
          open(path, 'w'), indent=1)
PY
if [[ -z $ready ]]; then
  echo "serve.sh: the server did not become ready; see $handle/server.log" >&2
  "$0" stop "$handle"
  exit 1
fi
echo "$handle"
