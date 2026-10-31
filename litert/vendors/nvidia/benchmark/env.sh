# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
#
# Paths and the TensorRT-RTX runtime environment of the NVIDIA benchmarks.
# Sourced by build.sh, bench.sh and matrix.sh; nothing here runs a command. An env.local.sh
# next to this file, if present, is sourced first: a machine's paths without editing this one.
#
# Inputs (environment, all optional):
#   LITERT_RT_DIR      this LiteRT tree (default: the tree this file is in)
#   LITERT_LM_DIR      LiteRT-LM checkout (default: ../../llm/LiteRT-LM next to this tree)
#   TENSORRT_RTX_ROOT  TensorRT-RTX SDK (default: ~/opt/tensorrt-rtx-sdk)
#   CUDA_HOME          CUDA toolkit (default: /usr/local/cuda)
#   LITERT_BENCH_ROOT  where builds, caches and runs live
#                      (default: ~/.local/state/litert-nvidia-benchmark-wl)
#   LITERT_BENCH_MODELS directory of the litertlm files (default: $LITERT_LM_DIR/../models)
#   LITERT_BENCH_DISTDIR a Bazel --distdir for offline builds

if [[ -f "$(dirname "${BASH_SOURCE[0]}")/env.local.sh" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/env.local.sh"
fi

nb_die() { echo "ERROR: $*" >&2; exit 2; }

nb_resolve_paths() {
  local here
  here=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
  export NB_RT NB_LM NB_ROOT NB_SDK NB_CUDA NB_MODELS NB_BUILD_RT NB_BUILD_LM NB_RUNTIME_LIBS
  NB_RT=$(realpath -m "${LITERT_RT_DIR:-$here/../../../..}")
  NB_LM=$(realpath -m "${LITERT_LM_DIR:-$NB_RT/../../llm/LiteRT-LM}")
  NB_ROOT=$(realpath -m "${LITERT_BENCH_ROOT:-$HOME/.local/state/litert-nvidia-benchmark-wl}")
  export LITERT_BENCH_ROOT=$NB_ROOT
  NB_SDK=$(realpath -m "${TENSORRT_RTX_ROOT:-$HOME/opt/tensorrt-rtx-sdk}")
  NB_CUDA=$(realpath -m "${CUDA_HOME:-/usr/local/cuda}")
  NB_MODELS=$(realpath -m "${LITERT_BENCH_MODELS:-$NB_LM/../models}")
  NB_BUILD_RT=$NB_ROOT/cache/build/litert
  NB_BUILD_LM=$NB_ROOT/cache/build/litert-lm
  NB_RUNTIME_LIBS=$NB_BUILD_RT/runtime-libs
}

# nb_model_file NAME -> the litertlm path of a profile name (e2b, e4b, 12b, 26b), or NAME itself
# when it is a file.
nb_model_file() {
  case "$1" in
    e2b) echo "$NB_MODELS/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm" ;;
    e4b) echo "$NB_MODELS/gemma-4-E4B-it-litert-lm/gemma-4-E4B-it.litertlm" ;;
    12b) echo "$NB_MODELS/gemma-4-12B-it-litert-lm/gemma-4-12B-it.litertlm" ;;
    26b) echo "$NB_MODELS/gemma-4-26B-moe-mtp-external/gemma-4-26B-moe-mtp.external.litertlm" ;;
    *) [[ -f $1 ]] || nb_die "Model is neither a profile name (e2b, e4b, 12b, 26b) nor a file: $1"; realpath "$1" ;;
  esac
}

# nb_build_id -> "<litert sha>[+]-<litert-lm sha>[+]" (a + marks local changes); the model
# caches of one build live under it, because the compiler, dispatch and CLI binaries decide
# what the AOT and runtime caches hold. The LiteRT side is the last commit that changed
# anything but this benchmark directory: its scripts and tools are not part of the binaries,
# so editing them keeps the pools warm.
nb_build_id() {
  local rt lm
  local -a sources=(. ':(exclude)litert/vendors/nvidia/benchmark')
  rt=$(git -C "$NB_RT" log -1 --format=%h --abbrev=9 HEAD -- "${sources[@]}")
  [[ -z $(git -C "$NB_RT" status --porcelain --untracked-files=no -- "${sources[@]}") ]] || rt+=+
  lm=$(git -C "$NB_LM" rev-parse --short=8 HEAD)
  [[ -z $(git -C "$NB_LM" status --porcelain --untracked-files=no) ]] || lm+=+
  echo "$rt-$lm"
}

# nb_runtime_env POOL MODEL: exports the TensorRT-RTX environment of one model cache pool
# (AOT artifacts, runtime caches and the CUDA JIT cache under POOL). The settings are the
# same for every Gemma 4 model; the compiler and dispatch choose the rest by model.
nb_runtime_env() {
  local pool=$1 model=$2 variable
  for variable in ${!LITERT_NVIDIA_@}; do unset "$variable"; done
  mkdir -p "$pool/aot" "$pool/runtime" "$pool/compiler/cuda" "$pool/cpu"
  export LD_LIBRARY_PATH=$NB_RUNTIME_LIBS:$NB_LM/prebuilt/linux_x86_64:$NB_SDK/lib:$NB_CUDA/lib64:/usr/lib/wsl/lib
  export LITERT_NVIDIA_TENSORRT_PARTITION_POLICY=gemma4
  export LITERT_NVIDIA_TENSORRT_FP16_ACTIVATIONS=bf16
  export LITERT_NVIDIA_TENSORRT_PREDEQUANTIZE_FC_WEIGHTS=cuda_gemv
  export LITERT_NVIDIA_TENSORRT_SHARED_WEIGHTS=1
  export LITERT_NVIDIA_TENSORRT_WEIGHT_STORE=1
  export LITERT_NVIDIA_TENSORRT_JIT_HANDLE=0
  export LITERT_NVIDIA_TENSORRT_AOT_MODEL_PATH=$model
  export LITERT_NVIDIA_TENSORRT_AOT_CACHE_DIR=$pool/aot
  export LITERT_NVIDIA_DISPATCH_RUNTIME_CACHE_DIR=$pool/runtime
  export LITERT_NVIDIA_DISPATCH_SHARED_WEIGHT_SEGMENTS=1
  export LITERT_NVIDIA_DISPATCH_LAZY_AOT_ENGINES=0
  export CUDA_CACHE_DISABLE=0 CUDA_CACHE_PATH=$pool/compiler/cuda
  export CUDA_HOME=$NB_CUDA TENSORRT_RTX_ROOT=$NB_SDK
}
