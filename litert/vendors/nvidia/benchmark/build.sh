#!/usr/bin/env bash
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
# Builds what the NVIDIA benchmarks run: the compiler plugin and dispatch libraries of this
# LiteRT tree and the unchanged LiteRT-LM advanced CLI against it, with Bazel output bases
# under the benchmark root so the benchmark caches and the binaries stay together.
# usage: build.sh [--rt-only | --lm-only]    (environment: see env.sh)
set -euo pipefail
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/env.sh"
nb_resolve_paths
what=${1:-all}
for command in bazel git realpath flock; do command -v "$command" >/dev/null || nb_die "Missing prerequisite: $command"; done
# The system clang, not the caller's CC/CXX: toolchain managers (swiftly) export proxies named
# clang that print their own help when Bazel calls them. LITERT_BENCH_CC/CXX override.
cc=${LITERT_BENCH_CC:-/usr/bin/clang} cxx=${LITERT_BENCH_CXX:-/usr/bin/clang++}
[[ -x $cc && -x $cxx ]] || nb_die "clang and clang++ are needed (LITERT_BENCH_CC, LITERT_BENCH_CXX)"
[[ -f $NB_RT/MODULE.bazel || -f $NB_RT/WORKSPACE ]] || nb_die "Not a LiteRT tree: $NB_RT"
[[ -f $NB_LM/runtime/engine/litert_lm_advanced_main.cc ]] || nb_die "LiteRT-LM checkout missing: $NB_LM (set LITERT_LM_DIR)"
[[ -f $NB_SDK/lib/libtensorrt_rtx.so && -x $NB_CUDA/bin/nvcc ]] || nb_die "SDK files missing; set TENSORRT_RTX_ROOT and CUDA_HOME"
mkdir -p "$NB_ROOT/cache/build" "$NB_RUNTIME_LIBS"
exec 9>"$NB_ROOT/.lock"
flock 9

bazel_build() {
  local repo=$1 base=$2; shift 2
  # A clean environment: the build does not depend on the caller's shell.
  (cd "$repo" && exec env -i HOME="$HOME" USER="${USER:-$(id -un)}" LOGNAME="${USER:-$(id -un)}" \
    SHELL=/bin/bash LANG=en_US.UTF-8 PATH="$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    CUDA_HOME="$NB_CUDA" TENSORRT_RTX_ROOT="$NB_SDK" \
    bazel --batch "--output_base=$base" build -c opt \
      --noincompatible_enable_android_toolchain_resolution \
      "--repo_env=CC=$cc" "--repo_env=CXX=$cxx" "--action_env=CC=$cc" "--action_env=CXX=$cxx" \
      "--jobs=${LITERT_BENCH_JOBS:-8}" "--local_ram_resources=${LITERT_BENCH_BUILD_RAM_MB:-8192}" \
      --disk_cache= --remote_cache= --remote_executor= --noremote_accept_cached \
      ${LITERT_BENCH_DISTDIR:+--distdir=$LITERT_BENCH_DISTDIR} "$@")
}

if [[ $what != --lm-only ]]; then
  echo "Building the LiteRT NVIDIA compiler plugin and dispatch libraries ($NB_RT)"
  bazel_build "$NB_RT" "$NB_BUILD_RT" "--repo_env=CUDA_HOME=$NB_CUDA" "--repo_env=TENSORRT_RTX_ROOT=$NB_SDK" \
    //litert/vendors/nvidia/compiler:compiler_plugin_so //litert/vendors/nvidia/dispatch:dispatch_api_so
  ln -sfn "$(realpath "$NB_RT/bazel-bin/litert/vendors/nvidia/compiler/libLiteRtCompilerPlugin_Nvidia.so")" "$NB_RUNTIME_LIBS/libLiteRtCompilerPlugin_Nvidia.so"
  ln -sfn "$(realpath "$NB_RT/bazel-bin/litert/vendors/nvidia/dispatch/libLiteRtDispatch_Nvidia.so")" "$NB_RUNTIME_LIBS/libLiteRtDispatch_Nvidia.so"
fi
if [[ $what != --rt-only ]]; then
  echo "Building the LiteRT-LM advanced CLI ($NB_LM) against this LiteRT tree"
  bazel_build "$NB_LM" "$NB_BUILD_LM" "--override_repository=litert=$NB_RT" //runtime/engine:litert_lm_advanced_main
  ln -sfn "$(realpath "$NB_LM/bazel-bin/runtime/engine/litert_lm_advanced_main")" "$NB_RUNTIME_LIBS/litert_lm_advanced_main"
fi
echo "Build id: $(nb_build_id)"
echo "Runtime libraries and CLI: $NB_RUNTIME_LIBS"
