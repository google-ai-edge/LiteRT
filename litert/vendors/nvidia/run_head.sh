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
#   ./run_head.sh memory-profile     # cold in-memory JIT + short inference
#   ./run_head.sh memory-profile-aot # compare cold/hot AOT and runtime caches
#   ./run_head.sh verify-aot         # cold AOT creation + warm AOT reuse
#   ./run_head.sh download-model  # optional; not run by default
#
# Optional overrides:
#   NUM_PROMPT='...' BENCH_PROMPT='...' ./run_head.sh all
#   PREDEQUANT_MODE=cuda_gemv ./run_head.sh all
#   LITERT_NVIDIA_TENSORRT_JIT_HANDLE=0 ./run_head.sh memory-profile
#   AOT_MEMORY_SHARED_WEIGHTS=0 ./run_head.sh memory-profile-aot
#   AOT_VERIFY_SHARED_WEIGHTS=0 ./run_head.sh verify-aot

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
  git -C "$LITERT_G3_HEAD" rev-parse --is-inside-work-tree >/dev/null
  git -C "$LITERT_LM_G3_HEAD" rev-parse --is-inside-work-tree >/dev/null

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

run_aot_verification_pass() {
  local compiler_cache=$1
  local runtime_cache=$2
  local log_path=$3

  env \
    -u LITERT_NVIDIA_DISPATCH_LAYER_PROFILE \
    -u LITERT_NVIDIA_DISPATCH_PROFILE \
    -u LITERT_NVIDIA_MEMORY_PROFILE \
    -u LITERT_NVIDIA_TENSORRT_AOT_FORCE_CONTENT_VALIDATION \
    -u LITERT_NVIDIA_TENSORRT_JIT_HANDLE \
    -u LITERT_NVIDIA_TENSORRT_SKIP_SUBGRAPHS \
    -u LITERT_LM_EXCLUDE_PREFILL_SIGNATURES \
    LD_LIBRARY_PATH="$RUNTIME_LD_PATH" \
    LITERT_NVIDIA_TENSORRT_AOT_CACHE_DIR="$AOT_ARTIFACT_CACHE" \
    LITERT_NVIDIA_TENSORRT_AOT_MODEL_PATH="$G4MODEL" \
    LITERT_NVIDIA_TENSORRT_PARTITION_POLICY=gemma4 \
    LITERT_NVIDIA_TENSORRT_FP16_ACTIVATIONS=bf16 \
    LITERT_NVIDIA_TENSORRT_PREDEQUANTIZE_FC_WEIGHTS="$PREDEQUANT_MODE" \
    LITERT_NVIDIA_TENSORRT_SHARED_WEIGHTS="$AOT_VERIFY_SHARED_WEIGHTS" \
    LITERT_NVIDIA_DISPATCH_RUNTIME_CACHE_DIR="$runtime_cache" \
    "$ENGINE" \
      --model_path="$G4MODEL" \
      --backend=npu \
      --prefill_batch_sizes=1024 \
      --max_num_tokens=2048 \
      --max_output_tokens="$AOT_VERIFY_OUTPUT_TOKENS" \
      --input_prompt="$AOT_VERIFY_PROMPT" \
      --cache_dir="$compiler_cache" \
      --litert_dispatch_lib_dir="$RUNTIME" \
      --min_log_severity=0 \
      2>&1 | tee "$log_path"
}

verify_aot_path() {
  local cache_dir
  for cache_dir in \
    "$AOT_ARTIFACT_CACHE" \
    "$AOT_COLD_COMPILER_CACHE" \
    "$AOT_COLD_RUNTIME_CACHE" \
    "$AOT_WARM_COMPILER_CACHE" \
    "$AOT_WARM_RUNTIME_CACHE"; do
    if [[ -n "$(find "$cache_dir" -mindepth 1 -print -quit)" ]]; then
      echo "AOT verification requires an empty cache directory: $cache_dir" >&2
      return 1
    fi
  done

  echo "Running cold AOT pass; this must compile and persist the artifacts."
  run_aot_verification_pass \
    "$AOT_COLD_COMPILER_CACHE" \
    "$AOT_COLD_RUNTIME_CACHE" \
    "$AOT_COLD_LOG"

  find "$AOT_ARTIFACT_CACHE" -maxdepth 1 -type f \
    -printf '%f\t%D\t%i\t%s\t%T@\t%C@\t%m\n' | sort >"$AOT_IDENTITY_BEFORE"

  echo "Running warm AOT pass with a fresh LiteRT compiler cache; this must reuse the persisted artifacts."
  run_aot_verification_pass \
    "$AOT_WARM_COMPILER_CACHE" \
    "$AOT_WARM_RUNTIME_CACHE" \
    "$AOT_WARM_LOG"

  find "$AOT_ARTIFACT_CACHE" -maxdepth 1 -type f \
    -printf '%f\t%D\t%i\t%s\t%T@\t%C@\t%m\n' | sort >"$AOT_IDENTITY_AFTER"
  if ! cmp -s "$AOT_IDENTITY_BEFORE" "$AOT_IDENTITY_AFTER"; then
    echo "AOT artifacts changed during the warm reuse pass:" >&2
    diff -u "$AOT_IDENTITY_BEFORE" "$AOT_IDENTITY_AFTER" || true
    return 1
  fi

  python3 - \
    "$AOT_COLD_LOG" \
    "$AOT_WARM_LOG" \
    "$AOT_ARTIFACT_CACHE" \
    "$AOT_VERIFY_EXPECTED_OUTPUT" \
    "$AOT_VERIFY_SHARED_WEIGHTS" <<'PY'
import pathlib
import re
import stat
import sys

cold_log = pathlib.Path(sys.argv[1])
warm_log = pathlib.Path(sys.argv[2])
artifact_dir = pathlib.Path(sys.argv[3]).resolve()
expected_output = sys.argv[4]
shared_weights_enabled = sys.argv[5] not in ("", "0")
cold = cold_log.read_text()
warm = warm_log.read_text()


def require(condition, message):
  if not condition:
    raise RuntimeError(message)


require(cold.count("NVIDIA TensorRT-RTX AOT cache miss:") == 1,
        "cold pass did not report exactly one AOT cache miss")
require("NVIDIA TensorRT-RTX AOT cache hit:" not in cold,
        "cold pass unexpectedly reported an AOT cache hit")

hits = re.findall(
    r"NVIDIA TensorRT-RTX AOT cache hit: partitions=(\d+) modules=(\d+)",
    warm,
)
require(len(hits) == 1,
        "warm pass did not report exactly one AOT cache hit")
partitions, modules = map(int, hits[0])
require(partitions > 0 and modules == partitions,
        f"expected one AOT shard per partition, got partitions={partitions} "
        f"modules={modules}")
require("NVIDIA TensorRT-RTX AOT cache miss:" not in warm,
        "warm pass unexpectedly reported an AOT cache miss")
require(warm.count("NVIDIA TensorRT-RTX compiled ") == 0,
        "warm pass rebuilt TensorRT engines instead of reusing AOT artifacts")
require(cold.count("NVIDIA TensorRT-RTX compiled ") == partitions,
        "cold pass did not compile exactly one TensorRT engine per partition")
if shared_weights_enabled:
  stripped_partitions = cold.count(
      "NVIDIA TensorRT-RTX compiled stripped partition"
  )
  cold_refits = cold.count("NVIDIA dispatch refitted TensorRT plan")
  warm_refits = warm.count("NVIDIA dispatch refitted TensorRT plan")
  require("NVIDIA TensorRT-RTX AOT shards ready:" in cold,
          "cold pass did not create shared-weight AOT shards")
  require(stripped_partitions > 0,
          "shared weights were enabled but no stripped plan was built")
  require(cold_refits == warm_refits and cold_refits > 0,
          "cold and warm passes did not refit the same positive number of "
          "stripped plans")

mapping_pattern = re.compile(
    r"NVIDIA dispatch mapped TensorRT AOT artifact for (\S+) "
    r"\(bytes=(\d+) validation=(\S+) path=(.+)\)"
)
cold_mappings = mapping_pattern.findall(cold)
warm_mappings = mapping_pattern.findall(warm)
require(len(cold_mappings) == partitions,
        f"cold pass mapped {len(cold_mappings)} artifacts, expected {partitions}")
require(len(warm_mappings) == partitions,
        f"warm pass mapped {len(warm_mappings)} artifacts, expected {partitions}")
require(all(row[2] == "trusted_file_identity" for row in cold_mappings),
        "cold pass did not use trusted file-identity validation for every shard")
require(all(row[2] == "trusted_file_identity" for row in warm_mappings),
        "warm pass did not use trusted file-identity validation for every shard")
require([row[3] for row in cold_mappings] ==
        [row[3] for row in warm_mappings],
        "cold and warm passes did not map the same AOT artifacts")

indexes = list(artifact_dir.glob("tensorrt_aot_index_v*.bin"))
artifacts = list(artifact_dir.glob("tensorrt_aot_v*.bin"))
require(len(indexes) == 1,
        f"expected one AOT index, found {len(indexes)}")
require(len(artifacts) == modules,
        f"expected {modules} AOT artifact shards, found {len(artifacts)}")
for artifact in artifacts:
  require(artifact.resolve().parent == artifact_dir,
          f"AOT artifact escaped its cache directory: {artifact}")
  require(stat.S_IMODE(artifact.stat().st_mode) & 0o222 == 0,
          f"AOT artifact is not sealed read-only: {artifact}")

if expected_output:
  for label, text in (("cold", cold), ("warm", warm)):
    require(any(line.strip() == expected_output for line in text.splitlines()),
            f"{label} pass did not produce expected output "
            f"{expected_output!r}")

artifact_bytes = sum(path.stat().st_size for path in artifacts)
print("AOT verification passed")
print(f"partitions: {partitions}")
print(f"artifact_shards: {modules}")
print(f"artifact_bytes: {artifact_bytes}")
print("cold_cache_result: miss and compiled")
print("warm_cache_result: hit without compilation")
print("validation: trusted_file_identity for every cold and warm mapping")
print("artifact_identity: unchanged across warm reuse")
if shared_weights_enabled:
  print(f"shared_weights: enabled; stripped_partitions={stripped_partitions}; "
        f"refitted_partitions={cold_refits}")
if expected_output:
  print(f"cold_and_warm_output: {expected_output}")
PY
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

run_memory_profile_pass() {
  local compiler_cache=$1
  local runtime_cache=$2
  local log_path=$3
  shift 3

  env \
    -u LITERT_NVIDIA_TENSORRT_SKIP_SUBGRAPHS \
    -u LITERT_LM_EXCLUDE_PREFILL_SIGNATURES \
    LD_LIBRARY_PATH="$RUNTIME_LD_PATH" \
    LITERT_NVIDIA_MEMORY_PROFILE=1 \
    LITERT_NVIDIA_TENSORRT_PARTITION_POLICY=gemma4 \
    LITERT_NVIDIA_TENSORRT_FP16_ACTIVATIONS=bf16 \
    LITERT_NVIDIA_TENSORRT_PREDEQUANTIZE_FC_WEIGHTS="$PREDEQUANT_MODE" \
    LITERT_NVIDIA_DISPATCH_RUNTIME_CACHE_DIR="$runtime_cache" \
    "$@" \
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
      --cache_dir="$compiler_cache" \
      --litert_dispatch_lib_dir="$RUNTIME" \
      --min_log_severity=0 \
      2>&1 | tee "$log_path"
}

summarize_memory_profile() {
  local log_path=$1
  local csv_path=$2

  python3 - "$log_path" "$csv_path" <<'PY'
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

run_memory_profile() {
  local jit_handle=${LITERT_NVIDIA_TENSORRT_JIT_HANDLE:-1}
  echo "Running cold compilation and first-invocation memory profile " \
       "(jit_handle=$jit_handle)."
  run_memory_profile_pass \
    "$MEMORY_CACHE/compiler_cache" \
    "$MEMORY_CACHE/runtime_cache" \
    "$MEMORY_LOG" \
    LITERT_NVIDIA_TENSORRT_AOT_CACHE_DIR= \
    LITERT_NVIDIA_TENSORRT_AOT_MODEL_PATH= \
    LITERT_NVIDIA_TENSORRT_JIT_HANDLE="$jit_handle"
  summarize_memory_profile "$MEMORY_LOG" "$MEMORY_CSV"
}

run_aot_memory_profile() {
  local cache_dir
  for cache_dir in \
    "$AOT_MEMORY_ARTIFACT_CACHE" \
    "$AOT_MEMORY_COLD_COMPILER_CACHE" \
    "$AOT_MEMORY_COLD_RUNTIME_CACHE" \
    "$AOT_MEMORY_WARM_COMPILER_CACHE" \
    "$AOT_MEMORY_WARM_RUNTIME_CACHE" \
    "$AOT_MEMORY_HOT_COMPILER_CACHE" \
    "$AOT_MEMORY_HOT_RUNTIME_CACHE"; do
    if [[ -n "$(find "$cache_dir" -mindepth 1 -print -quit)" ]]; then
      echo "AOT memory profiling requires an empty cache directory: $cache_dir" >&2
      return 1
    fi
  done

  local -a aot_environment=(
    LITERT_NVIDIA_TENSORRT_AOT_CACHE_DIR="$AOT_MEMORY_ARTIFACT_CACHE"
    LITERT_NVIDIA_TENSORRT_AOT_MODEL_PATH="$G4MODEL"
    LITERT_NVIDIA_TENSORRT_SHARED_WEIGHTS="$AOT_MEMORY_SHARED_WEIGHTS"
    LITERT_NVIDIA_TENSORRT_AOT_FORCE_CONTENT_VALIDATION=0
    LITERT_NVIDIA_TENSORRT_JIT_HANDLE=0
    LITERT_NVIDIA_DISPATCH_LAYER_PROFILE=0
    LITERT_NVIDIA_DISPATCH_PROFILE=0
  )

  echo "State 1/3: cold AOT cache and cold TensorRT runtime cache."
  run_memory_profile_pass \
    "$AOT_MEMORY_COLD_COMPILER_CACHE" \
    "$AOT_MEMORY_COLD_RUNTIME_CACHE" \
    "$AOT_MEMORY_COLD_LOG" \
    "${aot_environment[@]}"
  summarize_memory_profile \
    "$AOT_MEMORY_COLD_LOG" \
    "$AOT_MEMORY_COLD_CSV" | tee "$AOT_MEMORY_COLD_SUMMARY"

  find "$AOT_MEMORY_ARTIFACT_CACHE" -maxdepth 1 -type f \
    -printf '%f\t%D\t%i\t%s\t%T@\t%C@\t%m\n' | sort \
    >"$AOT_MEMORY_IDENTITY_BEFORE"

  echo "State 2/3: hot AOT cache and cold TensorRT runtime cache."
  run_memory_profile_pass \
    "$AOT_MEMORY_WARM_COMPILER_CACHE" \
    "$AOT_MEMORY_WARM_RUNTIME_CACHE" \
    "$AOT_MEMORY_WARM_LOG" \
    "${aot_environment[@]}"
  summarize_memory_profile \
    "$AOT_MEMORY_WARM_LOG" \
    "$AOT_MEMORY_WARM_CSV" | tee "$AOT_MEMORY_WARM_SUMMARY"

  if [[ -z "$(find "$AOT_MEMORY_WARM_RUNTIME_CACHE" -type f -print -quit)" ]]; then
    echo "The cold TensorRT runtime-cache pass produced no cache files." >&2
    return 1
  fi
  cp -a "$AOT_MEMORY_WARM_RUNTIME_CACHE"/. "$AOT_MEMORY_HOT_RUNTIME_CACHE"/

  echo "State 3/3: hot AOT cache and hot TensorRT runtime cache."
  run_memory_profile_pass \
    "$AOT_MEMORY_HOT_COMPILER_CACHE" \
    "$AOT_MEMORY_HOT_RUNTIME_CACHE" \
    "$AOT_MEMORY_HOT_LOG" \
    "${aot_environment[@]}"
  summarize_memory_profile \
    "$AOT_MEMORY_HOT_LOG" \
    "$AOT_MEMORY_HOT_CSV" | tee "$AOT_MEMORY_HOT_SUMMARY"

  find "$AOT_MEMORY_ARTIFACT_CACHE" -maxdepth 1 -type f \
    -printf '%f\t%D\t%i\t%s\t%T@\t%C@\t%m\n' | sort \
    >"$AOT_MEMORY_IDENTITY_AFTER"
  if ! cmp -s "$AOT_MEMORY_IDENTITY_BEFORE" "$AOT_MEMORY_IDENTITY_AFTER"; then
    echo "AOT artifacts changed during memory profiling:" >&2
    diff -u "$AOT_MEMORY_IDENTITY_BEFORE" "$AOT_MEMORY_IDENTITY_AFTER" || true
    return 1
  fi

  python3 - \
    "$AOT_MEMORY_COLD_CSV" \
    "$AOT_MEMORY_WARM_CSV" \
    "$AOT_MEMORY_HOT_CSV" \
    "$AOT_MEMORY_COLD_LOG" \
    "$AOT_MEMORY_WARM_LOG" \
    "$AOT_MEMORY_HOT_LOG" \
    "$AOT_MEMORY_ARTIFACT_CACHE" \
    "$AOT_MEMORY_HOT_RUNTIME_CACHE" \
    "$AOT_MEMORY_COMPARISON_CSV" \
    "$AOT_MEMORY_SHARED_WEIGHTS" <<'PY' | tee "$AOT_MEMORY_SUMMARY"
import csv
import pathlib
import re
import stat
import sys

cold_csv, warm_csv, hot_csv = map(pathlib.Path, sys.argv[1:4])
cold_log, warm_log, hot_log = map(pathlib.Path, sys.argv[4:7])
artifact_dir = pathlib.Path(sys.argv[7]).resolve()
runtime_cache_dir = pathlib.Path(sys.argv[8]).resolve()
comparison_csv = pathlib.Path(sys.argv[9])
shared_weights_enabled = sys.argv[10] not in ("", "0")


def require(condition, message):
  if not condition:
    raise RuntimeError(message)


def read_rows(path):
  rows = list(csv.DictReader(path.open()))
  require(rows, f"No memory checkpoints found in {path}")
  for row in rows:
    for field in (
        "sequence", "cpu_rss_bytes", "cpu_peak_rss_bytes",
        "cuda_device_used_bytes", "cuda_device_used_delta_bytes"
    ):
      row[field] = int(row[field])
  return rows


def profile(label, csv_path, log_path):
  rows = read_rows(csv_path)
  text = log_path.read_text()
  compiler_rows = [row for row in rows if row["component"] == "compiler"]
  unmaps = [row for row in rows if row["phase"] == "aot_artifact_unmapped"]
  invokes = [row for row in rows if row["phase"] == "invoke_begin"]
  init_times = [float(value) for value in re.findall(
      r"Init Executor: ([0-9.]+) ms", text
  )]
  engine_build_times = [float(value) for value in re.findall(
      r"Engine generation completed in ([0-9.]+) seconds", text
  )]
  require(unmaps, f"{label}: no AOT artifact-unmap checkpoint")
  require(len(invokes) >= 2, f"{label}: expected prefill and decode invocations")
  require(init_times, f"{label}: no Init Executor timing")
  return {
      "state": label,
      "cpu_hwm_mib": max(row["cpu_peak_rss_bytes"] for row in rows) / 2**20,
      "compiler_hwm_mib": (
          max(row["cpu_peak_rss_bytes"] for row in compiler_rows) / 2**20
          if compiler_rows else 0.0
      ),
      "post_unmap_mib": unmaps[-1]["cpu_rss_bytes"] / 2**20,
      "prefill_rss_mib": invokes[0]["cpu_rss_bytes"] / 2**20,
      "decode_rss_mib": invokes[1]["cpu_rss_bytes"] / 2**20,
      "cuda_peak_mib": max(
          row["cuda_device_used_bytes"] for row in rows
      ) / 2**20,
      "cuda_peak_delta_mib": max(
          row["cuda_device_used_delta_bytes"] for row in rows
      ) / 2**20,
      "init_executor_s": init_times[-1] / 1000,
      "engine_build_s": sum(engine_build_times),
      "checkpoints": len(rows),
      "log": str(log_path),
      "csv": str(csv_path),
  }


cold_text = cold_log.read_text()
warm_text = warm_log.read_text()
hot_text = hot_log.read_text()
require(cold_text.count("NVIDIA TensorRT-RTX AOT cache miss:") == 1,
        "cold state did not report exactly one AOT cache miss")
require("NVIDIA TensorRT-RTX AOT cache hit:" not in cold_text,
        "cold state unexpectedly hit the AOT cache")

hit_pattern = re.compile(
    r"NVIDIA TensorRT-RTX AOT cache hit: partitions=(\d+) modules=(\d+)"
)
warm_hits = hit_pattern.findall(warm_text)
hot_hits = hit_pattern.findall(hot_text)
require(len(warm_hits) == len(hot_hits) == 1,
        "both AOT-hot states must report exactly one AOT cache hit")
partitions, modules = map(int, warm_hits[0])
require(tuple(map(int, hot_hits[0])) == (partitions, modules),
        "AOT-hot states reported different partition/module counts")
require(partitions > 0 and modules == partitions,
        "expected one AOT artifact per partition")
require(cold_text.count("NVIDIA TensorRT-RTX compiled ") == partitions,
        "cold state did not compile exactly one engine per partition")
require(warm_text.count("NVIDIA TensorRT-RTX compiled ") == 0 and
        hot_text.count("NVIDIA TensorRT-RTX compiled ") == 0,
        "an AOT-hot state recompiled a TensorRT engine")
require(cold_text.count("loaded runtime cache") == 0 and
        warm_text.count("loaded runtime cache") == 0,
        "a runtime-cold state unexpectedly loaded a TensorRT runtime cache")
require(hot_text.count("loaded runtime cache") == partitions,
        "runtime-hot state did not load one cache per partition")
if shared_weights_enabled:
  require("NVIDIA TensorRT-RTX AOT shards ready:" in cold_text,
          "shared-weight cold state did not create AOT shards")
  require(cold_text.count("compiled stripped partition") > 0,
          "shared weights produced no stripped plans")

mapping_pattern = re.compile(
    r"NVIDIA dispatch mapped TensorRT AOT artifact for \S+ "
    r"\(bytes=\d+ validation=(\S+) path=(.+)\)"
)
for label, text in (
    ("cold AOT / cold runtime", cold_text),
    ("hot AOT / cold runtime", warm_text),
    ("hot AOT / hot runtime", hot_text),
):
  mappings = mapping_pattern.findall(text)
  require(len(mappings) == partitions,
          f"{label}: expected {partitions} AOT mappings, found {len(mappings)}")
  require(all(validation == "trusted_file_identity"
              for validation, _ in mappings),
          f"{label}: not every artifact used trusted file identity")

artifacts = sorted(artifact_dir.glob("tensorrt_aot_v*.bin"))
indexes = sorted(artifact_dir.glob("tensorrt_aot_index_v*.bin"))
runtime_caches = sorted(runtime_cache_dir.glob("*.trt_rtx_runtime_cache"))
require(len(artifacts) == modules and len(indexes) == 1,
        "unexpected AOT artifact/index count")
require(len(runtime_caches) == partitions,
        "unexpected TensorRT runtime-cache count")
require(all(stat.S_IMODE(path.stat().st_mode) & 0o222 == 0
            for path in artifacts),
        "an AOT artifact is not sealed read-only")

profiles = [
    profile("Cold AOT + cold runtime", cold_csv, cold_log),
    profile("Hot AOT + cold runtime", warm_csv, warm_log),
    profile("Hot AOT + hot runtime", hot_csv, hot_log),
]

fields = [
    "state", "cpu_hwm_mib", "compiler_hwm_mib", "post_unmap_mib",
    "prefill_rss_mib", "decode_rss_mib", "cuda_peak_mib",
    "cuda_peak_delta_mib", "init_executor_s", "engine_build_s",
    "checkpoints", "log", "csv",
]
with comparison_csv.open("w", newline="") as output:
  writer = csv.DictWriter(output, fieldnames=fields)
  writer.writeheader()
  writer.writerows(profiles)


def print_table(headers, rows, widths):
  def line(values):
    return "  ".join(
        f"{value:<{width}}" if index == 0 else f"{value:>{width}}"
        for index, (value, width) in enumerate(zip(values, widths))
    )
  print(line(headers))
  print(line(["-" * width for width in widths]))
  for row in rows:
    print(line(row))


print("AOT memory profile comparison")
print("CPU memory (MiB)")
print_table(
    ["State", "Process HWM", "Compiler HWM", "Post-unmap", "Prefill", "Decode"],
    [[
        row["state"], f'{row["cpu_hwm_mib"]:.1f}',
        f'{row["compiler_hwm_mib"]:.1f}', f'{row["post_unmap_mib"]:.1f}',
        f'{row["prefill_rss_mib"]:.1f}', f'{row["decode_rss_mib"]:.1f}',
    ] for row in profiles],
    [28, 11, 12, 10, 8, 8],
)
print()
print("Timing and CUDA device-wide memory")
print_table(
    ["State", "Init s", "TRT build s", "CUDA peak MiB", "CUDA delta MiB"],
    [[
        row["state"], f'{row["init_executor_s"]:.3f}',
        f'{row["engine_build_s"]:.3f}', f'{row["cuda_peak_mib"]:.1f}',
        f'{row["cuda_peak_delta_mib"]:.1f}',
    ] for row in profiles],
    [28, 10, 11, 13, 14],
)
print()
print("CPU HWM reductions")
reductions = [
    ("Cold -> hot AOT", profiles[0], profiles[1]),
    ("Cold -> fully hot", profiles[0], profiles[2]),
    ("Runtime cache cold -> hot", profiles[1], profiles[2]),
]
print_table(
    ["Transition", "Saved MiB", "Saved %", "Init saved s"],
    [[
        label,
        f'{before["cpu_hwm_mib"] - after["cpu_hwm_mib"]:.1f}',
        f'{(1 - after["cpu_hwm_mib"] / before["cpu_hwm_mib"]) * 100:.2f}',
        f'{before["init_executor_s"] - after["init_executor_s"]:.3f}',
    ] for label, before, after in reductions],
    [28, 10, 8, 12],
)
print()
print(f"AOT artifacts: {len(artifacts)} shards, "
      f"{sum(path.stat().st_size for path in artifacts)} bytes")
print(f"TensorRT runtime caches: {len(runtime_caches)} files, "
      f"{sum(path.stat().st_size for path in runtime_caches)} bytes")
print(f"Comparison CSV: {comparison_csv}")
print("CUDA values are device-wide total-minus-free; use the delta only after "
      "establishing an idle GPU baseline.")
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
AOT_MEMORY_ROOT="$RUN_ROOT/aot_memory_profile"
AOT_MEMORY_ARTIFACT_CACHE="$AOT_MEMORY_ROOT/aot_artifacts"
AOT_MEMORY_COLD_COMPILER_CACHE="$AOT_MEMORY_ROOT/cold/compiler_cache"
AOT_MEMORY_COLD_RUNTIME_CACHE="$AOT_MEMORY_ROOT/cold/runtime_cache"
AOT_MEMORY_WARM_COMPILER_CACHE="$AOT_MEMORY_ROOT/aot_hot_runtime_cold/compiler_cache"
AOT_MEMORY_WARM_RUNTIME_CACHE="$AOT_MEMORY_ROOT/aot_hot_runtime_cold/runtime_cache"
AOT_MEMORY_HOT_COMPILER_CACHE="$AOT_MEMORY_ROOT/aot_hot_runtime_hot/compiler_cache"
AOT_MEMORY_HOT_RUNTIME_CACHE="$AOT_MEMORY_ROOT/aot_hot_runtime_hot/runtime_cache"
AOT_ARTIFACT_CACHE="$RUN_ROOT/aot_artifacts"
AOT_COLD_COMPILER_CACHE="$RUN_ROOT/aot_verify/cold/compiler_cache"
AOT_COLD_RUNTIME_CACHE="$RUN_ROOT/aot_verify/cold/runtime_cache"
AOT_WARM_COMPILER_CACHE="$RUN_ROOT/aot_verify/warm/compiler_cache"
AOT_WARM_RUNTIME_CACHE="$RUN_ROOT/aot_verify/warm/runtime_cache"
CPU_CACHE="$RUN_ROOT/cpu_cache"
LOG_DIR="$RUN_ROOT/logs"
ENGINE="$LITERT_LM_G3_HEAD/bazel-bin/runtime/engine/litert_lm_advanced_main"
BENCH_LOG="$LOG_DIR/head_benchmark_8iter.log"
MEMORY_LOG="$LOG_DIR/memory_profile.log"
MEMORY_CSV="$LOG_DIR/memory_profile.csv"
AOT_MEMORY_COLD_LOG="$LOG_DIR/aot_memory_cold.log"
AOT_MEMORY_COLD_CSV="$LOG_DIR/aot_memory_cold.csv"
AOT_MEMORY_COLD_SUMMARY="$LOG_DIR/aot_memory_cold_summary.txt"
AOT_MEMORY_WARM_LOG="$LOG_DIR/aot_memory_aot_hot_runtime_cold.log"
AOT_MEMORY_WARM_CSV="$LOG_DIR/aot_memory_aot_hot_runtime_cold.csv"
AOT_MEMORY_WARM_SUMMARY="$LOG_DIR/aot_memory_aot_hot_runtime_cold_summary.txt"
AOT_MEMORY_HOT_LOG="$LOG_DIR/aot_memory_aot_hot_runtime_hot.log"
AOT_MEMORY_HOT_CSV="$LOG_DIR/aot_memory_aot_hot_runtime_hot.csv"
AOT_MEMORY_HOT_SUMMARY="$LOG_DIR/aot_memory_aot_hot_runtime_hot_summary.txt"
AOT_MEMORY_SUMMARY="$LOG_DIR/aot_memory_comparison.txt"
AOT_MEMORY_COMPARISON_CSV="$LOG_DIR/aot_memory_comparison.csv"
AOT_MEMORY_IDENTITY_BEFORE="$LOG_DIR/aot_memory_identity_before.tsv"
AOT_MEMORY_IDENTITY_AFTER="$LOG_DIR/aot_memory_identity_after.tsv"
AOT_COLD_LOG="$LOG_DIR/aot_cold.log"
AOT_WARM_LOG="$LOG_DIR/aot_warm.log"
AOT_IDENTITY_BEFORE="$LOG_DIR/aot_identity_before.tsv"
AOT_IDENTITY_AFTER="$LOG_DIR/aot_identity_after.tsv"
PREDEQUANT_MODE=${PREDEQUANT_MODE:-fp8}
NUM_PROMPT=${NUM_PROMPT:-"Answer with only the capital city: What is the capital of France?"}
NUM_OUTPUT_TOKENS=${NUM_OUTPUT_TOKENS:-16}
BENCH_PROMPT=${BENCH_PROMPT:-"Write one sentence explaining why CUDA is useful for neural network inference:"}
MEMORY_PROFILE_DECODE_TOKENS=${MEMORY_PROFILE_DECODE_TOKENS:-16}
AOT_MEMORY_SHARED_WEIGHTS=${AOT_MEMORY_SHARED_WEIGHTS:-1}
AOT_VERIFY_PROMPT=${AOT_VERIFY_PROMPT:-"Answer with only the capital city: What is the capital of France?"}
AOT_VERIFY_OUTPUT_TOKENS=${AOT_VERIFY_OUTPUT_TOKENS:-16}
AOT_VERIFY_EXPECTED_OUTPUT=${AOT_VERIFY_EXPECTED_OUTPUT:-PARIS}
AOT_VERIFY_SHARED_WEIGHTS=${AOT_VERIFY_SHARED_WEIGHTS:-1}

mkdir -p \
  "$RUNTIME" \
  "$NUM_CACHE/compiler_cache" \
  "$NUM_CACHE/runtime_cache" \
  "$BENCH_CACHE/compiler_cache" \
  "$BENCH_CACHE/runtime_cache" \
  "$MEMORY_CACHE/compiler_cache" \
  "$MEMORY_CACHE/runtime_cache" \
  "$AOT_MEMORY_ARTIFACT_CACHE" \
  "$AOT_MEMORY_COLD_COMPILER_CACHE" \
  "$AOT_MEMORY_COLD_RUNTIME_CACHE" \
  "$AOT_MEMORY_WARM_COMPILER_CACHE" \
  "$AOT_MEMORY_WARM_RUNTIME_CACHE" \
  "$AOT_MEMORY_HOT_COMPILER_CACHE" \
  "$AOT_MEMORY_HOT_RUNTIME_CACHE" \
  "$AOT_ARTIFACT_CACHE" \
  "$AOT_COLD_COMPILER_CACHE" \
  "$AOT_COLD_RUNTIME_CACHE" \
  "$AOT_WARM_COMPILER_CACHE" \
  "$AOT_WARM_RUNTIME_CACHE" \
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
  memory-profile-aot)
    record_source_state
    prepare_runtime
    run_aot_memory_profile
    ;;
  verify-aot)
    record_source_state
    prepare_runtime
    verify_aot_path | tee "$LOG_DIR/aot_verification_summary.txt"
    ;;
  all)
    record_source_state
    build_head
    prepare_runtime
    run_numeric
    run_benchmark | tee "$LOG_DIR/benchmark_summary.txt"
    ;;
  *)
    echo "Usage: $0 {all|build|numeric|benchmark|memory-profile|memory-profile-aot|verify-aot|download-model}" >&2
    exit 2
    ;;
esac

echo "Results: $RUN_ROOT"
)

