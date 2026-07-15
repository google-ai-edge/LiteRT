#!/bin/sh
############################################################
# Copyright (C) 2025 Broadcom.
############################################################
# Cross-compile LiteRT dispatch + example into lib/ (host flatc + ARM64 CMake).
#
# Usage: ./build_binaries.sh [--clean]
# Env:   BRCM_TOOLCHAIN_TOP (default /opt/toolchains/stbgcc-12.4-2.0)
#
# lib/libbstorm_core.so must exist before the ARM64 configure/link step (CMake IMPORTED
# target). Source it from the BSTORM Core SDK build output, for example after:
#   cd <bstorm_sdk>/core && make bstorm_core bstorm_simple_server build_linux_driver \
#        BSTM_CHIP=74110 BSTM_CHIP_REV=b0 BSTORM_COMPILE_FOR_TARGET=y -j16
# Then either copy by hand into lib/, or set BSTORM_SDK_CORE_OUT to the out directory
# (the folder that contains libbstorm_core.so, bstorm_simple_server, bstm.ko), e.g.:
#   export BSTORM_SDK_CORE_OUT=<bstorm>/core/out/core.74110b0.2.64.target
#   ./build_binaries.sh
# Optional: BSTORM_SDK_ROOT=<bstorm> if bstm_simple_run.sh is not found relative to
# BSTORM_SDK_CORE_OUT (otherwise .../core/scripts/target/bstm_simple_run.sh is derived).

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LIB_DIR="${SCRIPT_DIR}/lib"
BUILD_DIR="${SCRIPT_DIR}/build_arm64"
HOST_TOOLS_DIR="${SCRIPT_DIR}/host_tools_build"
BRCM_TOOLCHAIN_TOP="${BRCM_TOOLCHAIN_TOP:-/opt/toolchains/stbgcc-12.4-2.0}"

stage_bstorm_sdk_into_lib() {
    if [ -z "${BSTORM_SDK_CORE_OUT:-}" ] || [ ! -d "${BSTORM_SDK_CORE_OUT}" ]; then
        return 0
    fi
    echo "[Broadcom] copying BSTORM SDK artifacts from ${BSTORM_SDK_CORE_OUT} -> ${LIB_DIR}/"
    mkdir -p "${LIB_DIR}"
    for f in libbstorm_core.so bstorm_simple_server bstm.ko; do
        if [ -f "${BSTORM_SDK_CORE_OUT}/${f}" ]; then
            cp -f "${BSTORM_SDK_CORE_OUT}/${f}" "${LIB_DIR}/"
        fi
    done
    run_sh=""
    if [ -n "${BSTORM_SDK_ROOT:-}" ] && [ -f "${BSTORM_SDK_ROOT}/core/scripts/target/bstm_simple_run.sh" ]; then
        run_sh="${BSTORM_SDK_ROOT}/core/scripts/target/bstm_simple_run.sh"
    elif [ -f "${BSTORM_SDK_CORE_OUT}/../../scripts/target/bstm_simple_run.sh" ]; then
        run_sh="${BSTORM_SDK_CORE_OUT}/../../scripts/target/bstm_simple_run.sh"
    fi
    if [ -n "${run_sh}" ]; then
        cp -f "${run_sh}" "${LIB_DIR}/"
        chmod +x "${LIB_DIR}/bstm_simple_run.sh"
    fi
}

require_libbstorm_core_for_cmake() {
    if [ ! -f "${LIB_DIR}/libbstorm_core.so" ]; then
        echo "ERROR: ${LIB_DIR}/libbstorm_core.so is required before ARM64 CMake (links bstorm_example_next)." >&2
        echo "  Build from BSTORM Core SDK (see README / Broadcom LiteRT Release Note), then either:" >&2
        echo "     export BSTORM_SDK_CORE_OUT=<path/to/core/out/core.<chip>.<...>.target>" >&2
        echo "     $0" >&2
        echo "  or copy libbstorm_core.so (and bstorm_simple_server, bstm.ko, bstm_simple_run.sh) into ${LIB_DIR}/" >&2
        exit 1
    fi
}

clean_build_outputs() {
    echo "[Broadcom] clean build: ${BUILD_DIR} ${HOST_TOOLS_DIR}"
    rm -rf "${BUILD_DIR}" "${HOST_TOOLS_DIR}"
}

clean_staged_libs() {
    rm -f "${LIB_DIR}/libLiteRtDispatch_bstorm.so" "${LIB_DIR}/libLiteRtCompilerPlugin_bstorm.so" "${LIB_DIR}/libLiteRt.so"
}

if [ "${1:-}" = "--clean" ]; then
    clean_build_outputs
    clean_staged_libs
    exit 0
fi

if [ -n "${1:-}" ]; then
    echo "Usage: $0 [--clean]" >&2
    exit 1
fi

echo "[Broadcom] Skipping compilation, searching dynamically for manual flatc..."

MANUAL_BUILD_DIR="$(cd "${SCRIPT_DIR}/../../../host_flatc_build" 2>/dev/null && pwd || echo "")"
FLATC=""

if [ -n "${MANUAL_BUILD_DIR}" ] && [ -d "${MANUAL_BUILD_DIR}" ]; then
    FLATC="$(find "${MANUAL_BUILD_DIR}" -name flatc -type f | head -1)"
fi

if [ -z "${FLATC}" ] || [ ! -f "${FLATC}" ]; then
    echo "ERROR: Could not dynamically find manual flatc binary inside host_flatc_build layout!" >&2
    exit 1
fi

echo "[Broadcom] Found valid flatc binary at: ${FLATC}"
FLATC_DIR="$(dirname "${FLATC}")"
# ==============================================================================

stage_bstorm_sdk_into_lib
require_libbstorm_core_for_cmake

echo "[Broadcom] cross-compile ARM64..."
mkdir -p "${BUILD_DIR}"
cmake -Wno-dev -DCMAKE_TOOLCHAIN_FILE="${SCRIPT_DIR}/cmake/toolchains.cmake" \
    -Dtoolchain="${BRCM_TOOLCHAIN_TOP}" \
    -DBRCM_ARCH=aarch64 \
    -DCMAKE_BUILD_TYPE=Release \
    -DTFLITE_HOST_TOOLS_DIR="${FLATC_DIR}" \
    -DFLATC_EXECUTABLE="${FLATC}" \
    -DFLATBUFFERS_BUILD_FLATC=OFF \
    -DBSTORM_PREBUILT_DIR="${LIB_DIR}" \
    "-DLITERT_TFLITE_EXTRA_CMAKE_ARGS=-DCMAKE_TOOLCHAIN_FILE=${SCRIPT_DIR}/cmake/toolchains.cmake;-Dtoolchain=${BRCM_TOOLCHAIN_TOP};-DFLATBUFFERS_BUILD_FLATC=OFF;-DTFLITE_HOST_TOOLS_DIR=${FLATC_DIR};-DBSTORM_PREBUILT_DIR=${LIB_DIR}" \
    -S "${SCRIPT_DIR}" \
    -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "[Broadcom] stage lib/"
mkdir -p "${LIB_DIR}"
cp "${BUILD_DIR}/dispatch/libLiteRtDispatch_bstorm.so" "${LIB_DIR}/"
cp "${BUILD_DIR}/compiler/libLiteRtCompilerPlugin_bstorm.so" "${LIB_DIR}/"
cp "${BUILD_DIR}/example/bstorm_example_next" "${LIB_DIR}/"
cp "${BUILD_DIR}/litert_build/c/libLiteRt.so" "${LIB_DIR}/"
if [ -f "${BUILD_DIR}/core_build/src/app/bstorm_simple_server" ]; then
    cp "${BUILD_DIR}/core_build/src/app/bstorm_simple_server" "${LIB_DIR}/"
fi

stage_bstorm_sdk_into_lib

echo "[Broadcom] Done. Ensure ${LIB_DIR}/ has libbstorm_core.so, bstm.ko, bstorm_simple_server, bstm_simple_run.sh, libLiteRtCompilerPlugin_bstorm.so; then ./run_inference.sh --target <ip>"
