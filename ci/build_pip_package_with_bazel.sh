#!/usr/bin/env bash
# Copyright 2024 The AI Edge LiteRT Authors.
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
set -ex

# Run this script under the root directory.
export TF_LOCAL_SOURCE_PATH=${TF_LOCAL_SOURCE_PATH:-"$(pwd)/third_party/tensorflow"}

ARCH="$(uname -m)"
OS_NAME="$(uname -s)"
TENSORFLOW_TARGET=${TENSORFLOW_TARGET:-$1}
if [ "${TENSORFLOW_TARGET}" = "rpi" ]; then
  export TENSORFLOW_TARGET="armhf"
fi

# Build python interpreter_wrapper.
case "${TENSORFLOW_TARGET}" in
  armhf)
    BAZEL_FLAGS="--config=elinux_armhf
      --copt=-march=armv7-a --copt=-mfpu=neon-vfpv4
      --copt=-O3 --copt=-fno-tree-pre --copt=-fpermissive
      --define tensorflow_mkldnn_contraction_kernel=0
      --define=raspberry_pi_with_neon=true
      --repo_env=USE_PYWRAP_RULES=True"
    ;;
  rpi0)
    BAZEL_FLAGS="--config=elinux_armhf
      --copt=-march=armv6 -mfpu=vfp -mfloat-abi=hard
      --copt=-O3 --copt=-fno-tree-pre --copt=-fpermissivec
      --define tensorflow_mkldnn_contraction_kernel=0
      --define=raspberry_pi_with_neon=true
      --repo_env=USE_PYWRAP_RULES=True"
    ;;
  aarch64)
    BAZEL_FLAGS="--config=release_arm64_linux
      --define tensorflow_mkldnn_contraction_kernel=0
      --copt=-O3
      --repo_env=USE_PYWRAP_RULES=True"
    ;;
  native)
    BAZEL_FLAGS="--copt=-O3
      --copt=-march=native
      --repo_env=USE_PYWRAP_RULES=True"
    ;;
  *)
    BAZEL_FLAGS="--copt=-O3
      --repo_env=USE_PYWRAP_RULES=True"
    ;;
esac

if [[ "${OS_NAME}" == "Darwin" ]]; then
  # Ensure LiteRT runtime dylib is built and linked for macOS wheels.
  BAZEL_FLAGS="${BAZEL_FLAGS} --config=macos_wheel"
fi

if [[ -n "${BAZEL_CONFIG_FLAGS}" ]]; then
  BAZEL_FLAGS="${BAZEL_FLAGS} ${BAZEL_CONFIG_FLAGS}"
fi

if [ ! -z "${NIGHTLY_RELEASE_DATE}" ]; then
  BAZEL_FLAGS="${BAZEL_FLAGS} --//ci/tools/python/wheel:nightly_iso_date=${NIGHTLY_RELEASE_DATE}"
fi

# Conditionally use local submodules vs http_archve tf
if [[ "${USE_LOCAL_TF}" == "true" ]]; then
  BUILD_FLAGS+=("--config=use_local_tf")
fi

# Set linkopt for arm64 architecture, and remote_cache for x86_64.
case "${ARCH}" in
  x86_64)
    ;;
  arm64)
    BAZEL_FLAGS="${BAZEL_FLAGS} --linkopt="-ld_classic""
    ;;
  aarch64)
    ;;
  *)
    echo "Unsupported architecture: ${ARCH}"
    exit 1
    ;;
esac

bazel ${BAZEL_STARTUP_OPTIONS} build -c opt --cxxopt=-std=gnu++17 \
  ${BAZEL_FLAGS} ${CUSTOM_BAZEL_FLAGS} //ci/tools/python/wheel:litert_wheel

# Move the wheel file to the root directory since it is not accessible from the
# bazel output directory to anyone other than the root user.
rm -fr ./dist
mkdir -p dist/
mv bazel-bin/ci/tools/python/wheel/dist/*.whl dist/

echo "Output can be found here:"
find "./dist/"

if [ "${TEST_MANYLINUX_COMPLIANCE}" = "true" ]; then
  echo "Testing manylinux compliance..."
  bazel ${BAZEL_STARTUP_OPTIONS} test -c opt \
    ${BAZEL_FLAGS} ${CUSTOM_BAZEL_FLAGS} //ci/tools/python/wheel:manylinux_compliance_test
fi

# Vendor SDKs

## Qualcomm SDK
bazel ${BAZEL_STARTUP_OPTIONS} build -c opt \
  ${BAZEL_FLAGS} ${CUSTOM_BAZEL_FLAGS} //ci/tools/python/vendor_sdk/qualcomm:ai_edge_litert_sdk_qualcomm_sdist

mv bazel-bin/ci/tools/python/vendor_sdk/qualcomm/ai_edge_litert_sdk_qualcomm*.tar.gz dist/

## Mediatek SDK
bazel ${BAZEL_STARTUP_OPTIONS} build -c opt \
  ${BAZEL_FLAGS} ${CUSTOM_BAZEL_FLAGS} //ci/tools/python/vendor_sdk/mediatek:ai_edge_litert_sdk_mediatek_sdist

mv bazel-bin/ci/tools/python/vendor_sdk/mediatek/ai_edge_litert_sdk_mediatek*.tar.gz dist/

## Google Tensor SDK
if [[ -d "ci/tools/python/vendor_sdk/google_tensor" ]]; then
  bazel ${BAZEL_STARTUP_OPTIONS} build -c opt \
    ${BAZEL_FLAGS} ${CUSTOM_BAZEL_FLAGS} //ci/tools/python/vendor_sdk/google_tensor:ai_edge_litert_sdk_google_tensor_sdist

  mv bazel-bin/ci/tools/python/vendor_sdk/google_tensor/ai_edge_litert_sdk_google_tensor*.tar.gz dist/
fi

echo "Output can be found here:"
find "./dist/"
