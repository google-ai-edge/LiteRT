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
  docker build . -t tflite-builder -f tflite-py3.Dockerfile

  docker run \
    -v ${SCRIPT_DIR}/../third_party/tensorflow:/third_party_tensorflow \
    -v ${ROOT_DIR}:/root_dir \
    -v ${SCRIPT_DIR}:/script_dir \
    -e DOCKER_PYTHON_VERSION=${DOCKER_PYTHON_VERSION} \
    -e EXPERIMENTAL_TARGETS_ONLY=${EXPERIMENTAL_TARGETS_ONLY:-false} \
    -e BAZEL_CONFIG_FLAGS=${BAZEL_CONFIG_FLAGS} \
    --entrypoint /script_dir/run_bazel_test_with_docker.sh \
    tflite-builder
  exit 0
else
  # Running inside docker container
  cd /third_party_tensorflow

  # Run configure.
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
  cp .tf_configure.bazelrc /root_dir

  export HERMETIC_PYTHON_VERSION=${DOCKER_PYTHON_VERSION}

  cd /root_dir
  bash /script_dir/run_bazel_test.sh
fi
