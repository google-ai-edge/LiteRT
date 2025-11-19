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

DOCKER_PYTHON_VERSION="${DOCKER_PYTHON_VERSION:-3.11}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."

if [ ! -d /root_dir ]; then
  # Running on host.
  cd ${SCRIPT_DIR}

  if [[ "$(uname -m)" == "aarch64" ]]; then
    DOCKER_FILE="tflite-py3-arm64.Dockerfile"
  else
    DOCKER_FILE="tflite-py3.Dockerfile"
  fi

  docker build . -t tflite-builder -f ${DOCKER_FILE}

  docker run \
    -v ${SCRIPT_DIR}/../third_party/tensorflow:/third_party_tensorflow \
    -v ${ROOT_DIR}:/root_dir \
    -v ${SCRIPT_DIR}:/script_dir \
    -e NIGHTLY_RELEASE_DATE=${NIGHTLY_RELEASE_DATE} \
    -e DOCKER_PYTHON_VERSION=${DOCKER_PYTHON_VERSION} \
    -e BAZEL_CONFIG_FLAGS="${BAZEL_CONFIG_FLAGS}" \
    -e CUSTOM_BAZEL_FLAGS=${CUSTOM_BAZEL_FLAGS} \
    -e TEST_MANYLINUX_COMPLIANCE="${TEST_MANYLINUX_COMPLIANCE}" \
    -e RELEASE_VERSION=${RELEASE_VERSION} \
    -e TEST_WHEEL=${TEST_WHEEL:-false} \
    -e USE_LOCAL_TF=${USE_LOCAL_TF:-false} \
    --entrypoint /script_dir/build_pip_package_with_docker.sh \
    tflite-builder
  exit 0
else
  # Running inside docker container
  cd /root_dir

  export CI_BUILD_PYTHON="python${DOCKER_PYTHON_VERSION}"
  export HERMETIC_PYTHON_VERSION="${DOCKER_PYTHON_VERSION}"
  export TF_LOCAL_SOURCE_PATH="/root_dir/third_party/tensorflow"

    # Run configure
    configs=(
      '/usr/bin/python3'
      '/usr/lib/python3/dist-packages'
      'N'
      'N'
      'Y'
      '/usr/lib/llvm-18/bin/clang'
      '-Wno-sign-compare -Wno-c++20-designator -Wno-gnu-inline-cpp-without-extern'
      'N'
    )
    printf '%s\n' "${configs[@]}" | ./configure

  ${CI_BUILD_PYTHON} -m pip install pip setuptools wheel

  if [[ "${TEST_WHEEL}" == "true" ]]; then
    # Source test_pip_package.sh to get environment variables.
    source /script_dir/test_pip_package.sh
    create_venv
    initialize_pip_wheel_environment
  fi

  bash /script_dir/build_pip_package_with_bazel.sh

  # Test build wheel
  if [[ "${TEST_WHEEL}" == "true" ]]; then
    install_sdk
    install_wheel
    test_import
    uninstall_pip
    deactivate  # deactivate virtualenv
  fi
fi
