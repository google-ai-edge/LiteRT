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

#PYTHON="${CI_BUILD_PYTHON:-python3}"
#VERSION_SUFFIX=${VERSION_SUFFIX:-}
#export TENSORFLOW_DIR="./third_party/tensorflow"
#TENSORFLOW_LITE_DIR="./tflite"
ARCH="$(uname -m)"

#export PACKAGE_VERSION="${RELEASE_VERSION:-1.1.2}"
#export PROJECT_NAME=${WHEEL_PROJECT_NAME:-ai_edge_litert}

#BUILD_DIR="${TENSORFLOW_LITE_DIR}/gen/litert_pip/python3"
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
      --define=raspberry_pi_with_neon=true"
    ;;
  rpi0)
    BAZEL_FLAGS="--config=elinux_armhf
      --copt=-march=armv6 -mfpu=vfp -mfloat-abi=hard
      --copt=-O3 --copt=-fno-tree-pre --copt=-fpermissive
      --define tensorflow_mkldnn_contraction_kernel=0
      --define=raspberry_pi_with_neon=true"
    ;;
  aarch64)
    BAZEL_FLAGS="--config=release_arm64_linux
      --define tensorflow_mkldnn_contraction_kernel=0
      --copt=-O3"
    ;;
  native)
    BAZEL_FLAGS="--copt=-O3 --copt=-march=native"
    ;;
  *)
    BAZEL_FLAGS="--copt=-O3"
    ;;
esac

if [[ -n "${BAZEL_CONFIG_FLAGS}" ]]; then
  BAZEL_FLAGS="${BAZEL_FLAGS} ${BAZEL_CONFIG_FLAGS}"
fi

if [ ! -z "${NIGHTLY_RELEASE_DATE}" ]; then
  BAZEL_FLAGS="${BAZEL_FLAGS} --//tflite/tools/pip_package:nightly_iso_date=${NIGHTLY_RELEASE_DATE}"
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

bazel ${BAZEL_STARTUP_OPTIONS} build -c opt -s --config=monolithic --config=nogcp --config=nonccl \
  ${BAZEL_FLAGS} ${CUSTOM_BAZEL_FLAGS} //tflite/tools/pip_package:litert_wheel

echo "Output can be found here:"
find "bazel-bin/tflite/tools/pip_package/dist/"

# # Build debian package.
# if [[ "${BUILD_DEB}" != "y" ]]; then
#   exit 0
# fi

# PYTHON_VERSION=$(${PYTHON} -c "import sys;print(sys.version_info.major)")
# if [[ ${PYTHON_VERSION} != 3 ]]; then
#   echo "Debian package can only be generated for python3." >&2
#   exit 1
# fi

# DEB_VERSION=$(dpkg-parsechangelog --show-field Version | cut -d- -f1)
# if [[ "${DEB_VERSION}" != "${PACKAGE_VERSION}" ]]; then
#   cat << EOF > "${BUILD_DIR}/debian/changelog"
# ai_edge_litert (${PACKAGE_VERSION}-1) unstable; urgency=low

#   * Bump version to ${PACKAGE_VERSION}.

#  -- TensorFlow team <packages@tensorflow.org>  $(date -R)

# $(<"${BUILD_DIR}/debian/changelog")
# EOF
# fi

# case "${TENSORFLOW_TARGET}" in
#   armhf)
#     dpkg-buildpackage -b -rfakeroot -us -uc -tc -d -a armhf
#     ;;
#   rpi0)
#     dpkg-buildpackage -b -rfakeroot -us -uc -tc -d -a armel
#     ;;
#   aarch64)
#     dpkg-buildpackage -b -rfakeroot -us -uc -tc -d -a arm64
#     ;;
#   *)
#     dpkg-buildpackage -b -rfakeroot -us -uc -tc -d
#     ;;
# esac

# cat "${BUILD_DIR}/debian/changelog"
