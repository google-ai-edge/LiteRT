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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${CI_BUILD_PYTHON:-python3}"
VERSION_SUFFIX=${VERSION_SUFFIX:-}
export TENSORFLOW_DIR="./third_party/tensorflow"
TENSORFLOW_LITE_DIR="./tflite"
ARCH="$(uname -m)"

export PACKAGE_VERSION="${RELEASE_VERSION:-1.1.1}"
export PROJECT_NAME=${WHEEL_PROJECT_NAME:-ai_edge_litert}

if [ ! -z "${NIGHTLY_RELEASE_DATE}" ]; then
  export PACKAGE_VERSION="${PACKAGE_VERSION}.dev${NIGHTLY_RELEASE_DATE}"
  export PROJECT_NAME="${PROJECT_NAME}_nightly"
fi

BUILD_DIR="${TENSORFLOW_LITE_DIR}/gen/litert_pip/python3"
TENSORFLOW_TARGET=${TENSORFLOW_TARGET:-$1}
if [ "${TENSORFLOW_TARGET}" = "rpi" ]; then
  export TENSORFLOW_TARGET="armhf"
fi
export CROSSTOOL_PYTHON_INCLUDE_PATH=$(${PYTHON} -c "from sysconfig import get_paths as gp; print(gp()['include'])")

# Fix container image for cross build.
if [ ! -z "${CI_BUILD_HOME}" ] && [ `pwd` = "/workspace" ]; then
  # Fix for curl build problem in 32-bit, see https://stackoverflow.com/questions/35181744/size-of-array-curl-rule-01-is-negative
  if [ "${TENSORFLOW_TARGET}" = "armhf" ] && [ -f /usr/include/curl/curlbuild.h ]; then
    sudo sed -i 's/define CURL_SIZEOF_LONG 8/define CURL_SIZEOF_LONG 4/g' /usr/include/curl/curlbuild.h
    sudo sed -i 's/define CURL_SIZEOF_CURL_OFF_T 8/define CURL_SIZEOF_CURL_OFF_T 4/g' /usr/include/curl/curlbuild.h
  fi

  # The system-installed OpenSSL headers get pulled in by the latest BoringSSL
  # release on this configuration, so move them before we build:
  if [ -d /usr/include/openssl ]; then
    sudo mv /usr/include/openssl /usr/include/openssl.original
  fi
fi

# Build source tree.
rm -rf "${BUILD_DIR}" && mkdir -p "${BUILD_DIR}/ai_edge_litert"
cp -r "${TENSORFLOW_LITE_DIR}/tools/pip_package/debian" \
      "${TENSORFLOW_LITE_DIR}/tools/pip_package/MANIFEST.in" \
      "${TENSORFLOW_LITE_DIR}/python/interpreter_wrapper" \
      "${BUILD_DIR}"
cp  "${SCRIPT_DIR}/setup_with_binary.py" "${BUILD_DIR}/setup.py"
cp "${TENSORFLOW_LITE_DIR}/python/interpreter.py" \
   "${TENSORFLOW_LITE_DIR}/python/metrics/metrics_interface.py" \
   "${TENSORFLOW_LITE_DIR}/python/metrics/metrics_portable.py" \
   "${BUILD_DIR}/ai_edge_litert"

# Replace package name.
sed -i -e 's/tflite_runtime/ai_edge_litert/g' "${BUILD_DIR}/ai_edge_litert/interpreter.py"
sed -i -e 's/tflite_runtime/ai_edge_litert/g' "${BUILD_DIR}/ai_edge_litert/metrics_portable.py"

echo "__version__ = '${PACKAGE_VERSION}'" >> "${BUILD_DIR}/ai_edge_litert/__init__.py"
echo "__git_version__ = '$(git -C "${TENSORFLOW_DIR}" describe)'" >> "${BUILD_DIR}/ai_edge_litert/__init__.py"

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

# We need to pass down the environment variable with a possible alternate Python
# include path for Python 3.x builds to work.
export CROSSTOOL_PYTHON_INCLUDE_PATH

case "${TENSORFLOW_TARGET}" in
  windows)
    LIBRARY_EXTENSION=".pyd"
    ;;
  *)
    LIBRARY_EXTENSION=".so"
    ;;
esac

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
  ${BAZEL_FLAGS} ${CUSTOM_BAZEL_FLAGS} //tflite/python/interpreter_wrapper:_pywrap_tensorflow_interpreter_wrapper

cp "bazel-bin/tflite/python/interpreter_wrapper/_pywrap_tensorflow_interpreter_wrapper${LIBRARY_EXTENSION}" \
   "${BUILD_DIR}/ai_edge_litert"

# Build and add GenAI Ops library into the package.
bazel ${BAZEL_STARTUP_OPTIONS} build -c opt -s --config=monolithic --config=nogcp --config=nonccl \
  ${BAZEL_FLAGS} ${CUSTOM_BAZEL_FLAGS} //tflite/experimental/genai:pywrap_genai_ops

cp "bazel-bin/tflite/experimental/genai/pywrap_genai_ops${LIBRARY_EXTENSION}" \
   "${BUILD_DIR}/ai_edge_litert"

bazel ${BAZEL_STARTUP_OPTIONS} build -c opt -s --config=monolithic --config=nogcp --config=nonccl \
  ${BAZEL_FLAGS} ${CUSTOM_BAZEL_FLAGS} //tflite/python:schema_py

cp "bazel-bin/tflite/python/schema_py_generated.py" \
   "${BUILD_DIR}/ai_edge_litert"

# Build and add profiling protos to the package.
bazel ${BAZEL_STARTUP_OPTIONS} build -c opt -s --config=monolithic --config=nogcp --config=nonccl \
  ${BAZEL_FLAGS} ${CUSTOM_BAZEL_FLAGS} //tflite/profiling/proto:profiling_info_py
cp "bazel-bin/tflite/profiling/proto/profiling_info_pb2.py" \
   "${BUILD_DIR}/ai_edge_litert"

bazel ${BAZEL_STARTUP_OPTIONS} build -c opt -s --config=monolithic --config=nogcp --config=nonccl \
  ${BAZEL_FLAGS} ${CUSTOM_BAZEL_FLAGS} //tflite/profiling/proto:model_runtime_info_py
cp "bazel-bin/tflite/profiling/proto/model_runtime_info_pb2.py" \
   "${BUILD_DIR}/ai_edge_litert"

# Rename the namespace in the generated proto files to ai_edge_litert.
# This is required to maintain dependency between the two protos.
sed -i -e 's/tflite\.profiling\.proto/ai_edge_litert/g' "${BUILD_DIR}/ai_edge_litert/model_runtime_info_pb2.py"

# Bazel generates the wrapper library with r-x permissions for user.
# At least on Windows, we need write permissions to delete the file.
# Without this, setuptools fails to clean the build directory.
chmod u+w "${BUILD_DIR}/ai_edge_litert/_pywrap_tensorflow_interpreter_wrapper${LIBRARY_EXTENSION}"
chmod u+w "${BUILD_DIR}/ai_edge_litert/pywrap_genai_ops${LIBRARY_EXTENSION}"

# Build python wheel.
pushd "${BUILD_DIR}"
case "${TENSORFLOW_TARGET}" in
  armhf)
    WHEEL_PLATFORM_NAME="${WHEEL_PLATFORM_NAME:-linux-armv7l}"
    ${PYTHON} setup.py bdist --plat-name=${WHEEL_PLATFORM_NAME} \
                       bdist_wheel --plat-name=${WHEEL_PLATFORM_NAME}
    ;;
  rpi0)
    WHEEL_PLATFORM_NAME="${WHEEL_PLATFORM_NAME:-linux-armv6l}"
    ${PYTHON} setup.py bdist --plat-name=${WHEEL_PLATFORM_NAME} \
                       bdist_wheel --plat-name=${WHEEL_PLATFORM_NAME}
    ;;
  aarch64)
    WHEEL_PLATFORM_NAME="${WHEEL_PLATFORM_NAME:-linux-aarch64}"
    ${PYTHON} setup.py bdist --plat-name=${WHEEL_PLATFORM_NAME} \
                       bdist_wheel --plat-name=${WHEEL_PLATFORM_NAME}
    ;;
  *)
    # Assign the wheel name based on the platform and architecture. Naming follows
    # TF released wheel package.
    if test -e "/System/Library/CoreServices/SystemVersion.plist"; then
      if [[ "${ARCH}" == "arm64" ]]; then
        # MacOS Silicon
        WHEEL_PLATFORM_NAME="macosx_12_0_arm64"
      else
        # MacOS Intel
        WHEEL_PLATFORM_NAME="macosx_10_15_x86_64"
      fi
    elif test -e "/etc/lsb-release"; then
      # Linux
      if [[ "${ARCH}" == "aarch64" ]]; then
        WHEEL_PLATFORM_NAME="manylinux_2_17_aarch64"
      else
        WHEEL_PLATFORM_NAME="manylinux_2_17_x86_64"
      fi
    fi

    if [[ -n "${WHEEL_PLATFORM_NAME}" ]]; then
      ${PYTHON} setup.py bdist --plat-name=${WHEEL_PLATFORM_NAME} \
                         bdist_wheel --plat-name=${WHEEL_PLATFORM_NAME}
    else
      ${PYTHON} setup.py bdist bdist_wheel
    fi
    ;;
esac

echo "Output can be found here:"
popd
find "${BUILD_DIR}/dist"

# Build debian package.
if [[ "${BUILD_DEB}" != "y" ]]; then
  exit 0
fi

PYTHON_VERSION=$(${PYTHON} -c "import sys;print(sys.version_info.major)")
if [[ ${PYTHON_VERSION} != 3 ]]; then
  echo "Debian package can only be generated for python3." >&2
  exit 1
fi

DEB_VERSION=$(dpkg-parsechangelog --show-field Version | cut -d- -f1)
if [[ "${DEB_VERSION}" != "${PACKAGE_VERSION}" ]]; then
  cat << EOF > "${BUILD_DIR}/debian/changelog"
ai_edge_litert (${PACKAGE_VERSION}-1) unstable; urgency=low

  * Bump version to ${PACKAGE_VERSION}.

 -- TensorFlow team <packages@tensorflow.org>  $(date -R)

$(<"${BUILD_DIR}/debian/changelog")
EOF
fi

case "${TENSORFLOW_TARGET}" in
  armhf)
    dpkg-buildpackage -b -rfakeroot -us -uc -tc -d -a armhf
    ;;
  rpi0)
    dpkg-buildpackage -b -rfakeroot -us -uc -tc -d -a armel
    ;;
  aarch64)
    dpkg-buildpackage -b -rfakeroot -us -uc -tc -d -a arm64
    ;;
  *)
    dpkg-buildpackage -b -rfakeroot -us -uc -tc -d
    ;;
esac

cat "${BUILD_DIR}/debian/changelog"
