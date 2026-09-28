#!/usr/bin/env bash
# Copyright 2026 The AI Edge LiteRT Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${LITERT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LITERT_DIR="${ROOT_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-/tmp/litert_binaries}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

if [[ ! -d "${LITERT_DIR}/litert" || ! -d "${LITERT_DIR}/tflite" ]]; then
  echo "ERROR: LiteRT source tree ('${LITERT_DIR}') is missing 'litert' or 'tflite' directory." >&2
  echo "Please run this script from the root of the LiteRT repository or specify LITERT_DIR." >&2
  exit 1
fi

run_build_and_stage() {
  local platform="${PLATFORM:-linux_arm64_oe_glibc2.35}"
  local preset="${PRESET:-linux-aarch64-oe-gcc11.2}"
  local bdir="${BDIR:-cmake_build_linux_aarch64_oe_gcc11_2}"
  local jobs="${JOBS:-$(nproc)}"

  trap 'if [[ -n "${HOST_UID:-}" && -n "${HOST_GID:-}" ]]; then chown -R "${HOST_UID}:${HOST_GID}" "${OUTPUT_DIR}" "${LITERT_DIR}/litert/${bdir}" 2>/dev/null || true; fi' EXIT

  pushd "${LITERT_DIR}/litert" > /dev/null
  echo "=== Building LiteRT for ${platform} (${preset}) ==="
  cmake --preset "${preset}"
  cmake --build "${bdir}" \
    --target litert_runtime_c_api_shared_lib dispatch_api_qualcomm_so qnn_compiler_plugin run_model apply_plugin_main \
    -j "${jobs}"

  # Stage release binaries
  local plat_out="${OUTPUT_DIR}/${platform}"
  mkdir -p "${plat_out}"
  cp -f "${bdir}/c/libLiteRt.so" "${plat_out}/"
  cp -f "${bdir}/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so" "${plat_out}/"
  cp -f "${bdir}/vendors/qualcomm/compiler/libLiteRtCompilerPlugin_Qualcomm.so" "${plat_out}/"
  cp -f "${bdir}/tools/run_model" "${plat_out}/"
  cp -f "${bdir}/tools/apply_plugin_main" "${plat_out}/"

  # Ensure RPATH is exactly $ORIGIN for dynamic shared plugins
  if command -v chrpath >/dev/null 2>&1; then
    chrpath -r '$ORIGIN' "${plat_out}/libLiteRtDispatch_Qualcomm.so"
    chrpath -r '$ORIGIN' "${plat_out}/libLiteRtCompilerPlugin_Qualcomm.so"
  else
    echo "WARNING: chrpath not found; relying on CMake link-time INSTALL_RPATH (\$ORIGIN)" >&2
  fi

  # Strip release binaries using cross-strip
  local strip_bin=""
  if [[ -n "${STRIP:-}" ]] && command -v "${STRIP}" >/dev/null 2>&1; then
    strip_bin="${STRIP}"
  elif command -v aarch64-qcom-linux-strip >/dev/null 2>&1; then
    strip_bin="aarch64-qcom-linux-strip"
  elif [[ -n "${LINUX_AARCH64_ESDK:-}" && -x "${LINUX_AARCH64_ESDK}/tmp/sysroots/x86_64/usr/bin/aarch64-qcom-linux/aarch64-qcom-linux-strip" ]]; then
    strip_bin="${LINUX_AARCH64_ESDK}/tmp/sysroots/x86_64/usr/bin/aarch64-qcom-linux/aarch64-qcom-linux-strip"
  elif command -v aarch64-linux-gnu-strip >/dev/null 2>&1; then
    strip_bin="aarch64-linux-gnu-strip"
  else
    echo "ERROR: No aarch64 strip binary found (aarch64-qcom-linux-strip or aarch64-linux-gnu-strip)" >&2
    exit 1
  fi
  "${strip_bin}" --strip-unneeded \
    "${plat_out}/"*.so \
    "${plat_out}/run_model" \
    "${plat_out}/apply_plugin_main"

  # Generate SHA256SUMS
  pushd "${plat_out}" > /dev/null
  sha256sum libLiteRt.so libLiteRtDispatch_Qualcomm.so libLiteRtCompilerPlugin_Qualcomm.so run_model apply_plugin_main > SHA256SUMS
  popd > /dev/null

  popd > /dev/null
}

if [[ "${RUN_DIRECTLY:-false}" == "true" ]]; then
  run_build_and_stage
  echo "Direct build completed. Release packages available at: ${OUTPUT_DIR}"
  exit 0
fi

DOCKER_IMAGE="${DOCKER_IMAGE:-litert-iot-linux:latest}"
DOCKERFILE="${SCRIPT_DIR}/litert-iot-linux.Dockerfile"

if [[ "${FORCE_LOCAL_DOCKER_BUILD:-false}" == "true" ]]; then
  echo "Building Docker container locally..."
  docker build "${SCRIPT_DIR}" -t "${DOCKER_IMAGE}" -f "${DOCKERFILE}"
elif ! docker image inspect "${DOCKER_IMAGE}" >/dev/null 2>&1; then
  if [[ "${DOCKER_IMAGE}" != "litert-iot-linux:latest" ]] && docker pull "${DOCKER_IMAGE}"; then
    echo "Successfully pulled pre-built container: ${DOCKER_IMAGE}"
  else
    echo "Building LiteRT Linux IoT build container (${DOCKER_IMAGE}) from ${DOCKERFILE}..."
    docker build "${SCRIPT_DIR}" -t "${DOCKER_IMAGE}" -f "${DOCKERFILE}"
  fi
fi

echo "Running LiteRT Linux IoT build inside Docker container..."
local_script_copy="/tmp/litert_run_build_${$}.sh"
cp -f "${SCRIPT_DIR}/build_iot_linux_with_docker.sh" "${local_script_copy}"
chmod +x "${local_script_copy}"

docker run --rm \
  -e RUN_DIRECTLY=true \
  -e LITERT_DIR=/litert_src \
  -e OUTPUT_DIR=/output \
  -e PLATFORM="${PLATFORM:-linux_arm64_oe_glibc2.35}" \
  -e PRESET="${PRESET:-linux-aarch64-oe-gcc11.2}" \
  -e BDIR="${BDIR:-cmake_build_linux_aarch64_oe_gcc11_2}" \
  -e JOBS="${JOBS:-$(nproc)}" \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  -v "${LITERT_DIR}:/litert_src" \
  -v "${OUTPUT_DIR}:/output" \
  -v "${local_script_copy}:/tmp/build_iot_linux_with_docker.sh:ro" \
  -w /litert_src \
  "${DOCKER_IMAGE}" \
  bash /tmp/build_iot_linux_with_docker.sh

rm -f "${local_script_copy}" 2>/dev/null || true

echo "Docker build completed. Release packages available at: ${OUTPUT_DIR}"
