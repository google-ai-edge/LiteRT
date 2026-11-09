#!/usr/bin/env bash
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
# One llama.cpp measurement, the counterpart of litert/vendors/nvidia/benchmark/bench.sh: the
# same models, workloads and metrics, with llama.cpp's own tools as the data. llama-bench's
# JSON gives the synthetic prefill and decode numbers, llama-server's completion timings the
# real-text and speculative ones; nvidia-smi samples and GNU time run around them. report.py
# turns run directories into tables.
#
# usage: bench.sh --model e2b|e4b|12b|26b --out DIR [options]
#   --workload short|8k|16k|30k|128k  prompt tokens / decode tokens / context (default short)
#   --metric latency|memory|cold          latency (default): llama-bench prefill of the whole
#                                         prompt and decode at that depth; with --prompt-file,
#                                         llama-server completions of the file's text
#                                         memory: one prefill+decode pass (llama-bench -pg)
#                                         under GPU sampling and GNU time
#                                         cold: server start to ready after evicting the model
#                                         files from the page cache, then again warm;
#                                         with --prompt-file also prefill and decode once
#   --mtp                                 speculative decoding with the model's MTP drafter GGUF
#                                         (completions only)
#   --prompt-file FILE                    the file's text through llama-server instead of
#                                         synthetic tokens (stop tokens honored, at most the
#                                         workload's decode tokens)
#   --iterations N                        llama-bench repetitions or completions (default by workload)
#   --label NAME                          run directory name under --out (default derived)
#   --sampling greedy|model               the completions' sampler (default: LiteRT-LM's for the model)
#   --arg ARG                             an extra llama-bench / llama-server flag (repeatable)
# Output: DIR/LABEL/{meta.json, pp.json+tg.json | mem.json | client.N.{json,txt}+server/ |
#   cold/+warm/, time.*.txt, gpu.tsv}.
set -euo pipefail
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/env.sh"

model_arg= out= workload=short metric=latency mtp=false prompt_file= iterations= label= sampling= extra=()
while (($#)); do
  case "$1" in
    --model) model_arg=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --workload) workload=$2; shift 2 ;;
    --metric) metric=$2; shift 2 ;;
    --mtp) mtp=true; shift ;;
    --prompt-file) prompt_file=$(realpath "$2"); shift 2 ;;
    --iterations) iterations=$2; shift 2 ;;
    --label) label=$2; shift 2 ;;
    --sampling) sampling=$2; shift 2 ;;
    --arg) extra+=("$2"); shift 2 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) lb_die "Unknown option $1" ;;
  esac
done
[[ -n $model_arg && -n $out ]] || lb_die "--model and --out are required"
[[ $workload =~ ^(short|8k|16k|30k|128k)$ ]] || lb_die "Invalid workload"
[[ $metric =~ ^(latency|memory|cold)$ ]] || lb_die "Invalid metric"
[[ $mtp == false || -n $prompt_file || $metric == cold ]] || lb_die "--mtp needs --prompt-file (llama-bench has no speculative mode)"
model=$(lb_model_file "$model_arg")
profile=$model_arg; [[ $profile =~ ^(e2b|e4b|12b|26b)$ ]] || profile=$(basename "${model%.gguf}")
draft=; [[ $mtp == false ]] || draft=$(lb_draft_file "$profile")
sampling=${sampling:-$(lb_sampling "$profile")}
[[ -x $LLAMA_BIN/llama-bench && -x $LLAMA_BIN/llama-server ]] || lb_die "llama.cpp binaries missing in $LLAMA_BIN"
[[ -x /usr/bin/time ]] || lb_die "GNU time is required at /usr/bin/time"
export LD_LIBRARY_PATH="$LLAMA_BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
lb_runtime_env
smi=$(lb_smi)

# The workload presets of the LiteRT benchmark: prompt/decode tokens and server
# context. llama-bench instead sizes each phase from prompt + generation + depth.
case "$workload" in
  short) input=1024 output=256 context=2048 default_iterations=5 warmups=1 ;;
  8k) input=8192 output=256 context=9216 default_iterations=3 warmups=1 ;;
  16k) input=16384 output=256 context=17408 default_iterations=3 warmups=1 ;;
  30k) input=30720 output=256 context=31744 default_iterations=3 warmups=1 ;;
  128k) input=128001 output=128 context=130048 default_iterations=2 warmups=1 ;;
esac
iterations=${iterations:-$default_iterations}
[[ $metric == latency ]] || { iterations=1; warmups=0; }
mode=plain; [[ $mtp == false ]] || mode=mtp
label=${label:-${profile}_${workload}_${mode}_${metric}}
run=$out/$label
mkdir -p "$run"
mapfile -t model_args < <(lb_model_args "$profile")
busy=$("$smi" --query-compute-apps=pid --format=csv,noheader | wc -l)
[[ $busy == 0 ]] || lb_die "Another GPU compute process is running"
baseline=$("$smi" --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
# Synthetic runs execute llama-bench. Bind its executable and adjacent libraries
# before running; a rebuild during capture invalidates that measurement.
bench_binary_hashes() {
  python3 - "$LLAMA_BIN" <<'PY'
import hashlib, json, pathlib, sys
hashes, resolved = {}, {}
for p in sorted(pathlib.Path(sys.argv[1]).iterdir()):
    if p.is_file() and (p.name == 'llama-bench' or p.name.startswith(('libllama', 'libggml'))):
        target = p.resolve()
        if target not in resolved:
            with target.open('rb') as f:
                resolved[target] = hashlib.file_digest(f, 'sha256').hexdigest()
        hashes[str(p)] = resolved[target]
print(json.dumps(hashes, sort_keys=True))
PY
}
binary_before='{}'
bench_model_identity() {
  python3 - "$model" <<'PY'
import hashlib, json, os, sys
with open(sys.argv[1], 'rb') as f:
    identity = {'bytes': os.fstat(f.fileno()).st_size,
                'sha256': hashlib.file_digest(f, 'sha256').hexdigest()}
print(json.dumps(identity, sort_keys=True))
PY
}
model_before='{}'
if [[ -z $prompt_file && $metric != cold ]]; then
  binary_before=$(bench_binary_hashes)
  model_before=$(bench_model_identity)
fi
start=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
commands=()
rc=0

# llama_bench NAME SAMPLE_GPU ARGS...: one llama-bench run; its JSON in $run/NAME.json, its log
# in $run/NAME.log, GNU time in $run/time.NAME.txt, nvidia-smi samples in $run/gpu.tsv.
llama_bench() {
  local name=$1 sample=$2; shift 2
  local args=("$LLAMA_BIN/llama-bench" -m "$model" "${model_args[@]}" -lm "$LLAMA_LOAD_MODE" -lzm off -o json "$@" "${extra[@]}")
  commands+=("${args[*]}")
  echo "$label ($name): ${args[*]}" | cut -c1-240
  local sampler=
  if [[ $sample == 1 ]]; then
    ( while :; do printf '%s\t%s\n' "$(date +%s.%N)" "$("$smi" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"; sleep 0.1; done ) > "$run/gpu.tsv" 2>/dev/null &
    sampler=$!
  fi
  local status=0
  /usr/bin/time -v -o "$run/time.$name.txt" "${args[@]}" > "$run/$name.json" 2> "$run/$name.log" || status=$?
  [[ -z $sampler ]] || { kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null || true; }
  return $status
}

# completions HANDLE N [PREFIX]: N completions, optionally kept in a separate state directory.
completions() {
  local handle=$1 n=$2 prefix=${3:-$run/client} i status=0
  for ((i = 1; i <= n; i++)); do
    python3 "$LLAMA_DIR/client.tool" --prompt-file "$prompt_file" --out "$prefix.$i" \
      --n-predict "$output" --port "$LLAMA_PORT" --sampling "$sampling" --seed "$i" || status=$?
  done
  return $status
}

serve_args=(--ctx "$context" --port "$LLAMA_PORT")
[[ $mtp == false ]] || serve_args+=(--mtp)
for a in "${extra[@]}"; do serve_args+=(--arg "$a"); done
handle=
cleanup() { [[ -z $handle ]] || "$LLAMA_DIR/serve.sh" stop "$handle" || true; }
trap cleanup EXIT

case "$metric" in
  latency)
    if [[ -z $prompt_file ]]; then
      llama_bench pp 0 -p "$input" -n 0 -d 0 -r "$iterations" || rc=$?
      llama_bench tg 0 -p 0 -n "$output" -d "$input" -r "$iterations" || rc=$?
    else
      handle=$run/server
      if "$LLAMA_DIR/serve.sh" start "$handle" "$profile" "${serve_args[@]}" > /dev/null; then
        commands+=("$(jq -r '.command | join(" ")' "$handle/serve.json")")
        completions "$handle" "$iterations" || rc=$?
      else
        rc=1
      fi
      "$LLAMA_DIR/serve.sh" stop "$handle"; handle=
    fi ;;
  memory)
    if [[ -z $prompt_file ]]; then
      llama_bench mem 1 -pg "$input,$output" -p 0 -n 0 -d 0 -r 1 || rc=$?
    else
      handle=$run/server
      if "$LLAMA_DIR/serve.sh" start "$handle" "$profile" "${serve_args[@]}" > /dev/null; then
        commands+=("$(jq -r '.command | join(" ")' "$handle/serve.json")")
        completions "$handle" 1 || rc=$?
      else
        rc=1
      fi
      "$LLAMA_DIR/serve.sh" stop "$handle"; handle=
    fi ;;
  cold)
    # The model files out of the page cache, then the server's start to its first /health
    # answer; the warm start right after it for comparison.
    lb_evict "$model" ${draft:+"$draft"}
    for state in cold warm; do
      handle=$run/$state
      if "$LLAMA_DIR/serve.sh" start "$handle" "$profile" "${serve_args[@]}" --no-gpu-sampling > /dev/null; then
        commands+=("$(jq -r '.command | join(" ")' "$handle/serve.json")")
        if [[ -n $prompt_file ]]; then
          completions "$handle" 1 "$handle/client" || rc=$?
        fi
      else
        rc=1
      fi
      "$LLAMA_DIR/serve.sh" stop "$handle"; handle=
    done ;;
esac
trap - EXIT
end=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
binary_changed=false
if [[ $binary_before != '{}' && $binary_before != "$(bench_binary_hashes)" ]]; then
  binary_changed=true
  rc=1
  echo 'llama-bench binaries changed during capture; measurement invalid' >&2
fi
model_changed=false
if [[ $model_before != '{}' ]]; then
  model_after=$(bench_model_identity) || model_after='{}'
  if [[ $model_before != "$model_after" ]]; then
    model_changed=true
    rc=1
    echo 'llama-bench model changed during capture; measurement invalid' >&2
  fi
fi
printf '%s\n' "${commands[@]}" > "$run/commands.txt"
# The host's clocks (clock_check.py in LiteRT's benchmark, through BENCH or LITERT_RT_DIR: the kernel
# tick and frequency, the monotonic clock's rate against the real-time clock): recorded with the run;
# a formal or idle-verified run refuses a clock that is not fit, since llama-bench's timer is the
# monotonic clock. Without the check (no LiteRT tree named), host_clock is null.
LB_HOST_CLOCK=
clock_check=${BENCH:-${LITERT_RT_DIR:-/nonexistent}/litert/vendors/nvidia/benchmark}/clock_check.py
if [[ -f $clock_check ]]; then
  LB_HOST_CLOCK=$(python3 "$clock_check" 3) || case ${CI_HOST_MODE:-unspecified} in
    formal|idle-verified) echo "The host's clock is not fit for a $CI_HOST_MODE run: $LB_HOST_CLOCK" >&2; exit 1 ;;
    *) echo "Warning: the host's clock is not fit: $LB_HOST_CLOCK" >&2 ;;
  esac
fi
LB_LABEL=$label LB_PROFILE=$profile LB_MODEL=$model LB_DRAFT=$draft LB_WORKLOAD=$workload LB_METRIC=$metric \
LB_MTP=$mtp LB_PROMPT_FILE=$prompt_file LB_SAMPLING=$sampling LB_INPUT=$input LB_OUTPUT=$output LB_CONTEXT=$context \
LB_ITERATIONS=$iterations LB_WARMUPS=$warmups LB_BUILD_ID=$(lb_build_id) LB_RC=$rc LB_START=$start LB_END=$end \
LB_BASELINE=$baseline LB_RUN=$run LB_BINARIES=$binary_before LB_BINARY_CHANGED=$binary_changed \
LB_HOST_CLOCK=$LB_HOST_CLOCK LB_MODEL_IDENTITY=$model_before LB_MODEL_CHANGED=$model_changed python3 - <<'PY'
import json, os
e = os.environ
model_identity = json.loads(e['LB_MODEL_IDENTITY'])
meta = {
  'label': e['LB_LABEL'], 'runtime': 'llama.cpp', 'profile': e['LB_PROFILE'], 'model': e['LB_MODEL'],
  'draft': e['LB_DRAFT'] or None, 'workload': e['LB_WORKLOAD'], 'metric': e['LB_METRIC'], 'mtp': e['LB_MTP'] == 'true',
  'prompt_file': e['LB_PROMPT_FILE'] or None, 'sampling': e['LB_SAMPLING'], 'input_tokens': int(e['LB_INPUT']),
  'output_tokens': int(e['LB_OUTPUT']), 'context': int(e['LB_CONTEXT']), 'iterations': int(e['LB_ITERATIONS']),
  'warmups': int(e['LB_WARMUPS']), 'build_id': e['LB_BUILD_ID'], 'exit_code': int(e['LB_RC']), 'start': e['LB_START'],
  'end': e['LB_END'], 'gpu_baseline_mib': int(e['LB_BASELINE']),
  'model_bytes': model_identity['bytes'] if model_identity else os.path.getsize(e['LB_MODEL']),
  'model_sha256': model_identity.get('sha256'),
  'model_changed_during_run': e['LB_MODEL_CHANGED'] == 'true',
  'host_mode': e.get('CI_HOST_MODE', 'unspecified'),
  'host_clock': json.loads(e['LB_HOST_CLOCK']) if e.get('LB_HOST_CLOCK') else None,
  'load_mode': e['LLAMA_LOAD_MODE'],
  'physical_batch_tokens': int(e['LLAMA_UBATCH']),
  'cuda_cache_path': e.get('CUDA_CACHE_PATH'),
  'file_cache': ('model hashed before timing; OS file cache retained' if model_identity else
                 'eviction requested (best effort), then warm' if e['LB_METRIC'] == 'cold' else
                 'pre-read by CI when used'),
  'draft_bytes': os.path.getsize(e['LB_DRAFT']) if e['LB_DRAFT'] else None,
  'commands': [l.rstrip('\n') for l in open(os.path.join(e['LB_RUN'], 'commands.txt')) if l.strip()],
  'binary_sha256': json.loads(e['LB_BINARIES']),
  'binary_changed_during_run': e['LB_BINARY_CHANGED'] == 'true',
  'measurement_cwd': os.getcwd(),
  'measurement_environment': {k: e.get(k) for k in (
      'LLAMA_CPP_DIR', 'LLAMA_BIN', 'LLAMA_LOAD_MODE', 'LLAMA_UBATCH',
      'LLAMA_THREADS', 'CI_HOST_MODE')},
}
meta['server_context_preset'] = meta['context']
if not meta['prompt_file'] and meta['metric'] in ('latency', 'memory'):
  padded = lambda n: ((n + 255) // 256) * 256
  populated = padded(meta['input_tokens'] + meta['output_tokens'])
  meta['context_scope'] = 'llama-bench'
  meta['context'] = populated
  meta['context_by_phase'] = ({'pp': padded(meta['input_tokens']), 'tg': populated}
                             if meta['metric'] == 'latency' else {'mem': populated})
else:
  meta['context_scope'] = 'llama-server'
json.dump(meta, open(os.path.join(e['LB_RUN'], 'meta.json'), 'w'), indent=1)
PY
echo "Run: $run (exit $rc)"
exit $rc
