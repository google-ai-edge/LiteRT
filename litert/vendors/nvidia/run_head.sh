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
# Usage:
#   ./run_head.sh all             # build, numerics, benchmark
#   ./run_head.sh build
#   ./run_head.sh numeric
#   ./run_head.sh benchmark
#   ./run_head.sh download-model  # optional; not run by default
#
# Optional prompt overrides:
#   NUM_PROMPT='...' BENCH_PROMPT='...' ./run_head.sh all

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
      echo "Refusing to build a dirty checkout: $repo" >&2
      git -C "$repo" status --short >&2
      return 1
    fi
  done
}

record_source_state() {
  {
    echo "LiteRT remote: $(git -C "$LITERT_G3_HEAD" remote get-url origin)"
    echo "LiteRT HEAD:   $(git -C "$LITERT_G3_HEAD" rev-parse HEAD)"
    echo "LiteRT-LM remote: $(git -C "$LITERT_LM_G3_HEAD" remote get-url origin)"
    echo "LiteRT-LM HEAD:   $(git -C "$LITERT_LM_G3_HEAD" rev-parse HEAD)"
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
    LITERT_NVIDIA_TENSORRT_PREDEQUANTIZE_FC_WEIGHTS=fp8 \
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
    LITERT_NVIDIA_TENSORRT_PREDEQUANTIZE_FC_WEIGHTS=fp8 \
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
CPU_CACHE="$RUN_ROOT/cpu_cache"
LOG_DIR="$RUN_ROOT/logs"
ENGINE="$LITERT_LM_G3_HEAD/bazel-bin/runtime/engine/litert_lm_advanced_main"
BENCH_LOG="$LOG_DIR/head_benchmark_8iter.log"
NUM_PROMPT=${NUM_PROMPT:-"Answer with only the capital city: What is the capital of France?"}
NUM_OUTPUT_TOKENS=${NUM_OUTPUT_TOKENS:-16}
BENCH_PROMPT=${BENCH_PROMPT:-"Write one sentence explaining why CUDA is useful for neural network inference:"}

mkdir -p \
  "$RUNTIME" \
  "$NUM_CACHE/compiler_cache" \
  "$NUM_CACHE/runtime_cache" \
  "$BENCH_CACHE/compiler_cache" \
  "$BENCH_CACHE/runtime_cache" \
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
  all)
    record_source_state
    build_head
    prepare_runtime
    run_numeric
    run_benchmark | tee "$LOG_DIR/benchmark_summary.txt"
    ;;
  *)
    echo "Usage: $0 {all|build|numeric|benchmark|download-model}" >&2
    exit 2
    ;;
esac

echo "Results: $RUN_ROOT"

