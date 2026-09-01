#!/usr/bin/env bash
# Build and run the existing LiteRT + LiteRT-LM HEADs with TensorRT RTX.
#
# TensorRT RTX SDK prerequisite:
#   1. Download the Linux x86_64, CUDA 12.9 TensorRT for RTX SDK from:
#        https://developer.nvidia.com/tensorrt-rtx
#      The baseline integration used:
#        TensorRT-RTX-1.5.0.114-Linux-x86_64-cuda-12.9-Release-external.tar.zst
#   2. Extract the archive so its bin/, include/, and lib/ directories are
#      directly under $TENSORRT_RTX_ROOT.
#
# Required environment variables are already defined on cuda-wsl:
#   TENSORRT_RTX_ROOT  CUDA_HOME  G4MODEL  LITERT_G3_HEAD  LITERT_LM_G3_HEAD
#
# Usage (execute this script; do not source it):
#   ./run_head.sh all             # build, numerics, benchmark
#   ./run_head.sh build
#   ./run_head.sh numeric
#   ./run_head.sh benchmark
#   ./run_head.sh memory-profile     # cold compile + one short inference run
#   ./run_head.sh download-model  # optional; not run by default
#
# Optional overrides:
#   NUM_PROMPT='...' BENCH_PROMPT='...' ./run_head.sh all
#   PREDEQUANT_MODE=cuda_gemv ./run_head.sh all

# Keep shell options and variables contained if the script is sourced by
# accident. In particular, leaking `set -euo pipefail` into an interactive
# shell can make the shell exit after this script has completed successfully.
(

set -euo pipefail

download_model_with_hf() {
  : "${G4MODEL:?G4MODEL must point to the desired .litertlm file}"
  local model_dir
  model_dir=$(dirname -- "$G4MODEL")
  mkdir -p "$model_dir"
  hf download \
    litert-community/gemma-4-E2B-it-litert-lm \
    gemma-4-E2B-it.litertlm \
    --local-dir "$model_dir"
}

require_environment() {
  : "${TENSORRT_RTX_ROOT:?TENSORRT_RTX_ROOT is not set}"
  : "${CUDA_HOME:?CUDA_HOME is not set}"
  : "${G4MODEL:?G4MODEL is not set}"
  : "${LITERT_G3_HEAD:?LITERT_G3_HEAD is not set}"
  : "${LITERT_LM_G3_HEAD:?LITERT_LM_G3_HEAD is not set}"

  test -d "$TENSORRT_RTX_ROOT/lib"
  test -d "$CUDA_HOME"
  test -f "$G4MODEL"
  test -d "$LITERT_G3_HEAD/.git"
  test -d "$LITERT_LM_G3_HEAD/.git"

  local repo
  for repo in "$LITERT_G3_HEAD" "$LITERT_LM_G3_HEAD"; do
    if [[ -n "$(git -C "$repo" status --porcelain=v1)" ]]; then
      echo "Warning: building a dirty checkout: $repo" >&2
      git -C "$repo" status --short >&2
    fi
  done
}

record_source_state() {
  {
    echo "LiteRT remote: $(git -C "$LITERT_G3_HEAD" remote get-url origin)"
    echo "LiteRT HEAD:   $(git -C "$LITERT_G3_HEAD" rev-parse HEAD)"
    echo "LiteRT worktree:"
    git -C "$LITERT_G3_HEAD" status --short
    echo "LiteRT-LM remote: $(git -C "$LITERT_LM_G3_HEAD" remote get-url origin)"
    echo "LiteRT-LM HEAD:   $(git -C "$LITERT_LM_G3_HEAD" rev-parse HEAD)"
    echo "LiteRT-LM worktree:"
    git -C "$LITERT_LM_G3_HEAD" status --short
    echo "Predequant mode: $PREDEQUANT_MODE"
    /usr/lib/wsl/lib/nvidia-smi \
      --query-gpu=name,driver_version,memory.total \
      --format=csv,noheader
  } | tee "$LOG_DIR/source_and_gpu.txt"
}

build_head() {
  (
    cd "$LITERT_G3_HEAD"
    TENSORRT_RTX_ROOT="$TENSORRT_RTX_ROOT" \
    CUDA_HOME="$CUDA_HOME" \
    env -u ANDROID_HOME -u ANDROID_SDK_HOME -u ANDROID_NDK_HOME \
      bazel build -c opt \
        --repo_env=TENSORRT_RTX_ROOT="$TENSORRT_RTX_ROOT" \
        --repo_env=CUDA_HOME="$CUDA_HOME" \
        --noincompatible_enable_android_toolchain_resolution \
        --action_env=CC=/usr/bin/clang \
        --action_env=CXX=/usr/bin/clang++ \
        //litert/vendors/nvidia/compiler:compiler_plugin_so \
        //litert/vendors/nvidia/dispatch:dispatch_api_so \
        --verbose_failures
  ) 2>&1 | tee "$LOG_DIR/build_litert.log"

  (
    cd "$LITERT_LM_G3_HEAD"
    env -u ANDROID_HOME -u ANDROID_SDK_HOME -u ANDROID_NDK_HOME \
      bazel build -c opt \
        --override_repository=litert="$LITERT_G3_HEAD" \
        --noincompatible_enable_android_toolchain_resolution \
        --action_env=CC=/usr/bin/clang \
        --action_env=CXX=/usr/bin/clang++ \
        //runtime/engine:litert_lm_main \
        //runtime/engine:litert_lm_advanced_main \
        --verbose_failures
  ) 2>&1 | tee "$LOG_DIR/build_litert_lm.log"
}

prepare_runtime() {
  local compiler_so="$LITERT_G3_HEAD/bazel-bin/litert/vendors/nvidia/compiler/libLiteRtCompilerPlugin_Nvidia.so"
  local dispatch_so="$LITERT_G3_HEAD/bazel-bin/litert/vendors/nvidia/dispatch/libLiteRtDispatch_Nvidia.so"

  test -f "$compiler_so"
  test -f "$dispatch_so"
  test -x "$ENGINE"

  ln -sfn "$compiler_so" "$RUNTIME/libLiteRtCompilerPlugin_Nvidia.so"
  ln -sfn "$dispatch_so" "$RUNTIME/libLiteRtDispatch_Nvidia.so"

  LD_LIBRARY_PATH="$RUNTIME_LD_PATH" \
    ldd "$RUNTIME/libLiteRtDispatch_Nvidia.so" | tee "$LOG_DIR/dispatch_ldd.txt"
  if rg -q 'not found' "$LOG_DIR/dispatch_ldd.txt"; then
    echo "The NVIDIA dispatch library has unresolved dependencies." >&2
    return 1
  fi
}

run_numeric() {
  echo "Running NPU numeric prompt: $NUM_PROMPT"
  env \
    -u LITERT_NVIDIA_TENSORRT_SKIP_SUBGRAPHS \
    -u LITERT_LM_EXCLUDE_PREFILL_SIGNATURES \
    LD_LIBRARY_PATH="$RUNTIME_LD_PATH" \
    LITERT_NVIDIA_TENSORRT_PARTITION_POLICY=gemma4 \
    LITERT_NVIDIA_TENSORRT_FP16_ACTIVATIONS=bf16 \
    LITERT_NVIDIA_TENSORRT_PREDEQUANTIZE_FC_WEIGHTS="$PREDEQUANT_MODE" \
    LITERT_NVIDIA_DISPATCH_RUNTIME_CACHE_DIR="$NUM_CACHE/runtime_cache" \
    "$ENGINE" \
      --model_path="$G4MODEL" \
      --backend=npu \
      --prefill_batch_sizes=1024 \
      --max_num_tokens=2048 \
      --max_output_tokens="$NUM_OUTPUT_TOKENS" \
      --input_prompt="$NUM_PROMPT" \
      --cache_dir="$NUM_CACHE/compiler_cache" \
      --litert_dispatch_lib_dir="$RUNTIME" \
      --min_log_severity=0 \
      2>&1 | tee "$LOG_DIR/numeric_npu.log"

  echo "Running CPU reference prompt: $NUM_PROMPT"
  env \
    LD_LIBRARY_PATH="$LITERT_LM_G3_HEAD/prebuilt/linux_x86_64:${LD_LIBRARY_PATH:-}" \
    "$ENGINE" \
      --model_path="$G4MODEL" \
      --backend=cpu \
      --prefill_batch_sizes=1024 \
      --max_num_tokens=2048 \
      --max_output_tokens="$NUM_OUTPUT_TOKENS" \
      --input_prompt="$NUM_PROMPT" \
      --cache_dir="$CPU_CACHE" \
      --min_log_severity=0 \
      2>&1 | tee "$LOG_DIR/numeric_cpu.log"
}

run_benchmark() {
  echo "Running eight-iteration HEAD benchmark with a fresh cache."
  env \
    -u LITERT_NVIDIA_TENSORRT_SKIP_SUBGRAPHS \
    -u LITERT_LM_EXCLUDE_PREFILL_SIGNATURES \
    LD_LIBRARY_PATH="$RUNTIME_LD_PATH" \
    LITERT_NVIDIA_TENSORRT_PARTITION_POLICY=gemma4 \
    LITERT_NVIDIA_TENSORRT_FP16_ACTIVATIONS=bf16 \
    LITERT_NVIDIA_TENSORRT_PREDEQUANTIZE_FC_WEIGHTS="$PREDEQUANT_MODE" \
    LITERT_NVIDIA_DISPATCH_RUNTIME_CACHE_DIR="$BENCH_CACHE/runtime_cache" \
    "$ENGINE" \
      --model_path="$G4MODEL" \
      --backend=npu \
      --benchmark=true \
      --benchmark_prefill_tokens=1024 \
      --benchmark_decode_tokens=256 \
      --num_iterations=8 \
      --prefill_batch_sizes=1024 \
      --max_num_tokens=2048 \
      --max_output_tokens=256 \
      --input_prompt="$BENCH_PROMPT" \
      --cache_dir="$BENCH_CACHE/compiler_cache" \
      --litert_dispatch_lib_dir="$RUNTIME" \
      --min_log_severity=0 \
      2>&1 | tee "$BENCH_LOG"

  python3 - "$BENCH_LOG" <<'PY'
import pathlib
import re
import statistics
import sys

text = pathlib.Path(sys.argv[1]).read_text()
prefill = [float(x) for x in re.findall(r"Prefill Speed: ([0-9.]+)", text)]
decode = [float(x) for x in re.findall(r"Decode Speed: ([0-9.]+)", text)]
assert len(prefill) == len(decode) == 8, (len(prefill), len(decode))

print("prefill_all:", prefill)
print("decode_all:", decode)
print("steady_prefill_mean:", statistics.fmean(prefill[2:]))
print("steady_prefill_median:", statistics.median(prefill[2:]))
print("steady_decode_mean:", statistics.fmean(decode[2:]))
print("steady_decode_median:", statistics.median(decode[2:]))
PY
}

run_memory_profile() {
  echo "Running cold compilation and first-invocation memory profile."
  env \
    -u LITERT_NVIDIA_TENSORRT_SKIP_SUBGRAPHS \
    -u LITERT_LM_EXCLUDE_PREFILL_SIGNATURES \
    LD_LIBRARY_PATH="$RUNTIME_LD_PATH" \
    LITERT_NVIDIA_MEMORY_PROFILE=1 \
    LITERT_NVIDIA_TENSORRT_PARTITION_POLICY=gemma4 \
    LITERT_NVIDIA_TENSORRT_FP16_ACTIVATIONS=bf16 \
    LITERT_NVIDIA_TENSORRT_PREDEQUANTIZE_FC_WEIGHTS="$PREDEQUANT_MODE" \
    LITERT_NVIDIA_DISPATCH_RUNTIME_CACHE_DIR="$MEMORY_CACHE/runtime_cache" \
    "$ENGINE" \
      --model_path="$G4MODEL" \
      --backend=npu \
      --benchmark=true \
      --benchmark_prefill_tokens=1024 \
      --benchmark_decode_tokens="$MEMORY_PROFILE_DECODE_TOKENS" \
      --num_iterations=1 \
      --prefill_batch_sizes=1024 \
      --max_num_tokens=2048 \
      --max_output_tokens="$MEMORY_PROFILE_DECODE_TOKENS" \
      --input_prompt="$BENCH_PROMPT" \
      --cache_dir="$MEMORY_CACHE/compiler_cache" \
      --litert_dispatch_lib_dir="$RUNTIME" \
      --min_log_severity=0 \
      2>&1 | tee "$MEMORY_LOG"

  python3 - "$MEMORY_LOG" "$MEMORY_CSV" <<'PY'
import csv
import pathlib
import re
import sys

log_path = pathlib.Path(sys.argv[1])
csv_path = pathlib.Path(sys.argv[2])
pattern = re.compile(
    r"NVIDIA memory profile "
    r"component=(\S+) phase=(\S+) context=(\S+) "
    r"monotonic_ns=(\d+) "
    r"cpu_available=(\d+) cpu_rss_bytes=(\d+) cpu_peak_rss_bytes=(\d+) "
    r"cuda_available=(\d+) cuda_device_used_bytes=(\d+) "
    r"cuda_device_free_bytes=(\d+) cuda_device_total_bytes=(\d+)"
)

rows = []
for match in pattern.finditer(log_path.read_text()):
  component, phase, context = match.group(1, 2, 3)
  values = [int(value) for value in match.groups()[3:]]
  rows.append({
      "component": component,
      "phase": phase,
      "context": context,
      "monotonic_ns": values[0],
      "cpu_available": values[1],
      "cpu_rss_bytes": values[2],
      "cpu_peak_rss_bytes": values[3],
      "cuda_available": values[4],
      "cuda_device_used_bytes": values[5],
      "cuda_device_free_bytes": values[6],
      "cuda_device_total_bytes": values[7],
  })

if not rows:
  raise RuntimeError(f"No NVIDIA memory profile records found in {log_path}")

rows.sort(key=lambda row: row["monotonic_ns"])
start_ns = rows[0]["monotonic_ns"]
base_cpu = rows[0]["cpu_rss_bytes"]
base_cuda = rows[0]["cuda_device_used_bytes"]
for sequence, row in enumerate(rows):
  row["sequence"] = sequence
  row["elapsed_ms"] = (row["monotonic_ns"] - start_ns) / 1_000_000
  row["cpu_rss_delta_bytes"] = row["cpu_rss_bytes"] - base_cpu
  row["cuda_device_used_delta_bytes"] = (
      row["cuda_device_used_bytes"] - base_cuda
  )

fieldnames = [
    "sequence", "elapsed_ms", "component", "phase", "context",
    "cpu_available", "cpu_rss_bytes", "cpu_peak_rss_bytes",
    "cpu_rss_delta_bytes", "cuda_available", "cuda_device_used_bytes",
    "cuda_device_used_delta_bytes", "cuda_device_free_bytes",
    "cuda_device_total_bytes",
]
with csv_path.open("w", newline="") as output:
  writer = csv.DictWriter(output, fieldnames=fieldnames)
  writer.writeheader()
  writer.writerows({name: row[name] for name in fieldnames} for row in rows)

mib = 1024 * 1024
peak_cpu = max(rows, key=lambda row: row["cpu_rss_bytes"])
peak_cuda = max(rows, key=lambda row: row["cuda_device_used_bytes"])
print(f"memory_checkpoints: {len(rows)}")
print(f"memory_csv: {csv_path}")
print(
    "sampled_cpu_rss_peak_mib: "
    f"{peak_cpu['cpu_rss_bytes'] / mib:.1f} "
    f"at {peak_cpu['component']}/{peak_cpu['phase']} "
    f"context={peak_cpu['context']}"
)
print(f"process_cpu_peak_rss_mib: {max(row['cpu_peak_rss_bytes'] for row in rows) / mib:.1f}")
print(
    "sampled_cuda_device_used_peak_mib: "
    f"{peak_cuda['cuda_device_used_bytes'] / mib:.1f} "
    f"at {peak_cuda['component']}/{peak_cuda['phase']} "
    f"context={peak_cuda['context']}"
)
print("Note: CUDA used is device-wide total-minus-free; use deltas on an otherwise idle GPU.")
PY
}

ACTION=${1:-all}

if [[ "$ACTION" == "download-model" ]]; then
  download_model_with_hf
  exit 0
fi

require_environment

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_ROOT=${RUN_ROOT:-"$SCRIPT_DIR/results/$RUN_ID"}
RUNTIME="$RUN_ROOT/nvidia_runtime"
NUM_CACHE="$RUN_ROOT/numeric_cache"
BENCH_CACHE="$RUN_ROOT/benchmark_cache"
MEMORY_CACHE="$RUN_ROOT/memory_profile_cache"
CPU_CACHE="$RUN_ROOT/cpu_cache"
LOG_DIR="$RUN_ROOT/logs"
ENGINE="$LITERT_LM_G3_HEAD/bazel-bin/runtime/engine/litert_lm_advanced_main"
BENCH_LOG="$LOG_DIR/head_benchmark_8iter.log"
MEMORY_LOG="$LOG_DIR/memory_profile.log"
MEMORY_CSV="$LOG_DIR/memory_profile.csv"
PREDEQUANT_MODE=${PREDEQUANT_MODE:-fp8}
NUM_PROMPT=${NUM_PROMPT:-"Answer with only the capital city: What is the capital of France?"}
NUM_OUTPUT_TOKENS=${NUM_OUTPUT_TOKENS:-16}
BENCH_PROMPT=${BENCH_PROMPT:-"Write one sentence explaining why CUDA is useful for neural network inference:"}
MEMORY_PROFILE_DECODE_TOKENS=${MEMORY_PROFILE_DECODE_TOKENS:-16}

mkdir -p \
  "$RUNTIME" \
  "$NUM_CACHE/compiler_cache" \
  "$NUM_CACHE/runtime_cache" \
  "$BENCH_CACHE/compiler_cache" \
  "$BENCH_CACHE/runtime_cache" \
  "$MEMORY_CACHE/compiler_cache" \
  "$MEMORY_CACHE/runtime_cache" \
  "$CPU_CACHE" \
  "$LOG_DIR"

RUNTIME_LD_PATH="$RUNTIME:$LITERT_LM_G3_HEAD/prebuilt/linux_x86_64:$TENSORRT_RTX_ROOT/lib:$CUDA_HOME/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

case "$ACTION" in
  build)
    record_source_state
    build_head
    ;;
  numeric)
    record_source_state
    prepare_runtime
    run_numeric
    ;;
  benchmark)
    record_source_state
    prepare_runtime
    run_benchmark | tee "$LOG_DIR/benchmark_summary.txt"
    ;;
  memory-profile)
    record_source_state
    prepare_runtime
    run_memory_profile | tee "$LOG_DIR/memory_profile_summary.txt"
    ;;
  all)
    record_source_state
    build_head
    prepare_runtime
    run_numeric
    run_benchmark | tee "$LOG_DIR/benchmark_summary.txt"
    ;;
  *)
    echo "Usage: $0 {all|build|numeric|benchmark|memory-profile|download-model}" >&2
    exit 2
    ;;
esac

echo "Results: $RUN_ROOT"
)
