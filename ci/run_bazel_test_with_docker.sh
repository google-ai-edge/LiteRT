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
    -e LITERT_TARGETS_ONLY=${LITERT_TARGETS_ONLY:-false} \
    -e IS_PRESUBMIT_GITHUB=${IS_PRESUBMIT_GITHUB:-false} \
    -e BAZEL_CONFIG_FLAGS=${BAZEL_CONFIG_FLAGS} \
    --entrypoint /script_dir/run_bazel_test_with_docker.sh \
    tflite-builder
  exit 0
else
  # Running inside docker container
  if [[ "${IS_PRESUBMIT_GITHUB}" == "true" ]]; then
    cd /root_dir
    # Add safe directory to avoid git submodule update error.
    # Main repo
    git config --global --add safe.directory /root_dir
    # Submodule
    git config --global --add safe.directory /root_dir/third_party/tensorflow
    git submodule update --init --recursive
    git submodule update --remote
  fi

  cd /root_dir

  # Run configure.
  # LINT.IfChange(configure_tflite_build_flags)
  # Keep compiler and Python configuration consistent with LiteRT GitHub Actions CI
  # (tflite_bazel_cmake.yml) to ensure Bazel remote cache hits.
  ln -sf "$(which python3)" /usr/local/bin/python3
  mkdir -p /usr/local/lib/python3.11
  ln -sfn "$(python3 -c 'import site; print(site.getsitepackages()[0])')" /usr/local/lib/python3.11/site-packages
  export PYTHON_BIN_PATH="/usr/local/bin/python3"
  export PYTHON_LIB_PATH="/usr/local/lib/python3.11/site-packages"
  export TF_NEED_ROCM=0
  export TF_NEED_CLANG=0
  export TF_NEED_CUDA=0
  export TF_SET_ANDROID_WORKSPACE=0
  export CC_OPT_FLAGS='-Wno-sign-compare'
  python3 configure.py < <(yes "")
  # LINT.ThenChange(../workflows/tflite_bazel_cmake.yml:configure_tflite_build_flags)

  export HERMETIC_PYTHON_VERSION=${DOCKER_PYTHON_VERSION}
  export TF_LOCAL_SOURCE_PATH="/root_dir/third_party/tensorflow"

  cd /root_dir
  bash /script_dir/run_bazel_test.sh
fi
