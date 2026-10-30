#!/usr/bin/env bash
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
# One measurement = one run of the unchanged LiteRT-LM advanced CLI in benchmark mode on the
# NVIDIA (TensorRT-RTX) backend. The CLI's own benchmark report is the data; this script only
# chooses the flags of a workload, keeps the run's files together and samples the GPU memory
# with nvidia-smi for the memory metric. report.py turns run directories into tables.
#
# usage: bench.sh --model e2b|e4b|12b|26b|FILE --out DIR [options]
#   --workload short|8k|16k|30k|32k|128k  prefill tokens / decode tokens / context (default short;
#                                         1,024-token prefill batches throughout)
#   --metric latency|memory|verify|cold   latency: the benchmark (default); memory: one benchmark
#                                         iteration with GPU sampling and the CLI's peak RSS;
#                                         verify: the Paris answer check on the NPU and the CPU;
#                                         cold: one iteration on an empty cache (compiles AOT),
#                                         sampled like memory (the profiler logs the compile phases)
#   --mtp                                 speculative decoding with the model's MTP drafter
#   --prompt-file FILE                    prefill the file's text instead of synthetic tokens
#                                         and decode the model's answer, at most the workload's
#                                         decode tokens (acceptance needs real text)
#   --iterations N                        benchmark iterations (default by workload)
#   --label NAME                          run directory name under --out (default derived)
#   --pool DIR                            model cache pool (default: by model and build id)
#   --cli-arg ARG                         an extra CLI flag, after the workload's (repeatable)
#   --env NAME=VALUE                      an environment variable for the CLI (repeatable;
#                                         after the pool's standard settings)
# Environment: see env.sh. Output: DIR/LABEL/{cli.log,time.txt,gpu.tsv,meta.json}.
set -euo pipefail
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/env.sh"
nb_resolve_paths

model_arg= out= workload=short metric=latency mtp=false prompt_file= iterations= label= pool= extra=() env_extra=()
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
    --pool) pool=$2; shift 2 ;;
    --cli-arg) extra+=("$2"); shift 2 ;;
    --env) env_extra+=("$2"); shift 2 ;;
    -h|--help) sed -n '3,24p' "$0"; exit 0 ;;
    *) nb_die "Unknown option $1" ;;
  esac
done
[[ -n $model_arg && -n $out ]] || nb_die "--model and --out are required"
[[ $workload =~ ^(short|8k|16k|30k|32k|128k)$ ]] || nb_die "Invalid workload"
[[ $metric =~ ^(latency|memory|verify|cold)$ ]] || nb_die "Invalid metric"
model=$(nb_model_file "$model_arg")
profile=$model_arg; [[ $profile =~ ^(e2b|e4b|12b|26b)$ ]] || profile=$(basename "${model%.litertlm}")
engine=$NB_RUNTIME_LIBS/litert_lm_advanced_main
[[ -x $engine && -f $NB_RUNTIME_LIBS/libLiteRtDispatch_Nvidia.so ]] || nb_die "Run build.sh first ($NB_RUNTIME_LIBS)"
[[ -x /usr/bin/time ]] || nb_die "GNU time is required at /usr/bin/time"
smi=$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)

# Workload presets: prefill tokens, decode tokens, context, iterations, warm-ups, prefill batch.
case "$workload" in
  short) input=1024 output=256 context=2048 default_iterations=8 warmups=2 prefill=1024 ;;
  8k) input=8192 output=256 context=9216 default_iterations=4 warmups=1 prefill=1024 ;;
  16k) input=16384 output=256 context=17408 default_iterations=4 warmups=1 prefill=1024 ;;
  30k) input=30720 output=256 context=31744 default_iterations=4 warmups=1 prefill=1024 ;;
  32k) input=32768 output=256 context=33792 default_iterations=4 warmups=1 prefill=1024 ;;
  128k) input=128001 output=128 context=130048 default_iterations=4 warmups=1 prefill=1024 ;;
esac
iterations=${iterations:-$default_iterations}
[[ $metric != memory && $metric != cold ]] || { iterations=1; warmups=0; }
mode=plain; [[ $mtp == false ]] || mode=mtp
label=${label:-${profile}_${workload}_${mode}_${metric}}
run=$out/$label
mkdir -p "$run"
build_id=$(nb_build_id)
if [[ $metric == cold ]]; then
  pool=${pool:-$NB_ROOT/cache/scratch/$label-$(date -u +%Y%m%dT%H%M%S)}
  [[ ! -d $pool ]] || nb_die "A cold run needs an empty pool: $pool"
else
  pool=${pool:-$NB_ROOT/cache/models/$profile/$build_id}
fi
nb_runtime_env "$pool" "$model"
for assignment in "${env_extra[@]}"; do export "$assignment"; done

# Plain runs of the larger models compile only the signatures they run. An MTP run compiles
# every signature: a selection would also apply to the drafter model, which has none of them.
selected=()
if [[ $mtp == false && ( $profile == 12b || $profile == 26b || $prefill == 128 ) ]]; then
  selected=("--selected_signatures=prefill_$prefill,decode")
fi
base_args=("--model_path=$model" "--prefill_batch_sizes=$prefill" "--litert_dispatch_lib_dir=$NB_RUNTIME_LIBS"
  --min_log_severity=0)
benchmark_args() {  # ITERATIONS
  local args=(--benchmark=true "--num_iterations=$1" "--max_output_tokens=$output")
  if [[ -n $prompt_file ]]; then
    # Real text: the model's own answer, up to $output tokens, stop tokens honored (a decode
    # token count would keep a speculative round going past a stop token).
    args+=(--benchmark_prefill_tokens=0 --benchmark_decode_tokens=0 "--input_prompt_file=$prompt_file")
  else
    args+=("--benchmark_prefill_tokens=$input" "--benchmark_decode_tokens=$output" '--input_prompt=Write one sentence explaining why CUDA is useful for neural network inference:')
  fi
  printf '%s\n' "${args[@]}"
}

# invoke NAME BACKEND SAMPLE_GPU ARGS...: one CLI run into $run/NAME.* (NAME empty: $run/cli.*).
invoke() {
  local name=$1 backend=$2 sample=$3; shift 3
  local prefix=$run/${name:+$name.}
  local args=("$engine" "${base_args[@]}" "--backend=$backend" "--max_num_tokens=$context"
    "--cache_dir=$pool/$([[ $backend == cpu ]] && echo cpu || echo compiler)")
  [[ $backend == cpu ]] || args+=("${selected[@]}" "--enable_speculative_decoding=$mtp")
  args+=("$@" "${extra[@]}")  # --cli-arg flags last, so they override the workload's
  export LITERT_NVIDIA_MEMORY_PROFILE=$([[ $sample == 1 ]] && echo 1 || echo 0)
  local baseline; baseline=$("$smi" --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  local busy; busy=$("$smi" --query-compute-apps=pid --format=csv,noheader | wc -l)
  [[ $busy == 0 ]] || nb_die "Another GPU compute process is running"
  local sampler=
  if [[ $sample == 1 ]]; then
    ( while :; do printf '%s\t%s\n' "$(date +%s.%N)" "$("$smi" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"; sleep 0.1; done ) > "${prefix}gpu.tsv" 2>/dev/null &
    sampler=$!
  fi
  local start end rc=0
  start=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
  echo "$label${name:+ ($name)}: ${args[*]}" | cut -c1-240
  (cd "$NB_LM" && /usr/bin/time -v -o "${prefix}time.txt" "${args[@]}" > "${prefix}cli.log" 2>&1) || rc=$?
  end=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
  [[ -z $sampler ]] || { kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null || true; }
  python3 - "${prefix}meta.json" "$rc" "$start" "$end" "$baseline" "${args[@]}" <<'PY'
import json, os, sys, hashlib
meta_path, rc, start, end, baseline, *command = sys.argv[1:]
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1 << 23), b''): h.update(block)
    return h.hexdigest()
libs = os.environ['NB_RUNTIME_LIBS']
meta = {
  'label': os.environ['NB_LABEL'], 'name': os.path.basename(meta_path).replace('meta.json', '').rstrip('.') or None,
  'profile': os.environ['NB_PROFILE'], 'model': os.environ['NB_MODEL'], 'workload': os.environ['NB_WORKLOAD'],
  'metric': os.environ['NB_METRIC'], 'mtp': os.environ['NB_MTP'] == 'true', 'prompt_file': os.environ.get('NB_PROMPT_FILE') or None,
  'input_tokens': int(os.environ['NB_INPUT']), 'output_tokens': int(os.environ['NB_OUTPUT']), 'context': int(os.environ['NB_CONTEXT']),
  'iterations': int(os.environ['NB_ITERATIONS']), 'warmups': int(os.environ['NB_WARMUPS']),
  'build_id': os.environ['NB_BUILD_ID'], 'pool': os.environ['NB_POOL'], 'command': command, 'exit_code': int(rc),
  'host_mode': os.environ.get('CI_HOST_MODE', 'unspecified'),
  'start': start, 'end': end, 'gpu_baseline_mib': int(baseline),
  'environment': {k: v for k, v in os.environ.items() if k.startswith('LITERT_NVIDIA_')},
  'binaries': {n: sha(os.path.realpath(os.path.join(libs, n))) for n in ('libLiteRtCompilerPlugin_Nvidia.so', 'libLiteRtDispatch_Nvidia.so', 'litert_lm_advanced_main')},
  'sources': {k: os.environ[k] for k in ('NB_RT', 'NB_LM')},
}
json.dump(meta, open(meta_path, 'w'), indent=1)
PY
  return $rc
}
export NB_LABEL=$label NB_PROFILE=$profile NB_MODEL=$model NB_WORKLOAD=$workload NB_METRIC=$metric NB_MTP=$mtp
export NB_PROMPT_FILE=$prompt_file NB_INPUT=$input NB_OUTPUT=$output NB_CONTEXT=$context NB_ITERATIONS=$iterations
export NB_WARMUPS=$warmups NB_BUILD_ID=$build_id NB_POOL=$pool

# A pool without AOT artifacts compiles in its first run; that run is kept apart as "prepare"
# so the measured runs read warm caches (the cold metric is that first run itself).
if [[ $metric != cold ]] && ! compgen -G "$pool/aot/tensorrt_aot*.bin" >/dev/null; then
  mapfile -t prep < <(benchmark_args 1)
  invoke prepare npu 0 "${prep[@]}" || nb_die "Preparation failed; see $run/prepare.cli.log"
fi
rc=0
case "$metric" in
  latency)
    mapfile -t args < <(benchmark_args "$iterations")
    invoke "" npu 0 "${args[@]}" || rc=$? ;;
  cold)
    # Sampled like the memory metric: the memory profiler also logs the compiler's phases.
    mapfile -t args < <(benchmark_args 1)
    invoke "" npu 1 "${args[@]}" || rc=$? ;;
  memory)
    mapfile -t args < <(benchmark_args 1)
    invoke "" npu 1 "${args[@]}" --report_peak_memory_footprint=true || rc=$? ;;
  verify)
    answer=(--benchmark=false --max_output_tokens=16 '--input_prompt=Answer with only the capital city: What is the capital of France?' --expected_output=Paris)
    invoke npu npu 0 "${answer[@]}" || rc=$?
    invoke cpu cpu 0 "${answer[@]}" --enable_speculative_decoding=false || rc=$? ;;
esac
echo "Run: $run (exit $rc)"
exit $rc
