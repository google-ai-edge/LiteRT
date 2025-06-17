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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_DIR="gen"
ROOT_DIR="${SCRIPT_DIR}/.."

if [ ! -d /litert_build ]; then
  # Running on host.
  cd ${SCRIPT_DIR}
  rm -fr ${GEN_DIR}

  docker build ${ROOT_DIR}/build -t tflite-builder -f hermetic_build.Dockerfile

  docker run -v ${SCRIPT_DIR}/../third_party/tensorflow:/third_party_tensorflow \
    -v ${ROOT_DIR}:/litert_build \
    -v ${SCRIPT_DIR}:/script_dir \
    -e RELEASE_VERSION="${RELEASE_VERSION:-0.0.0-nightly-SNAPSHOT}" \
    -e BAZEL_CONFIG_FLAGS="${BAZEL_CONFIG_FLAGS}" \
    -e BUILD_LITERT_KOTLIN_API="${BUILD_LITERT_KOTLIN_API}" \
    -w /litert_build \
    tflite-builder /script_dir/build_maven_with_docker.sh

  echo "Output can be found here:"
  ls -lR ${GEN_DIR}/*

  exit 0
else
  # Running inside docker container
  # hermetic_build.Dockerfile already has Android SDK/NDK configured
  # and generates .tf_configure.bazelrc automatically via entrypoint
  
  cd /litert_build
  bash /script_dir/build_android_package.sh

  # Bundle the Maven package
  VERSION=${RELEASE_VERSION:-0.0.0-nightly-SNAPSHOT}
  PACKAGE_PATH=com/google/ai/edge/litert
  DEBUG_VERSION="0.0.0-nightly-debug-SNAPSHOT"

  rm -fr ${PACKAGE_PATH}

  LITERT_DIR=${PACKAGE_PATH}/litert/$VERSION
  mkdir -p ${LITERT_DIR}
  cp ./ci/gen/litert-$VERSION/* ${LITERT_DIR}

  LITERT_API_DIR=${PACKAGE_PATH}/litert-api/$VERSION
  mkdir -p ${LITERT_API_DIR}
  cp ./ci/gen/litert-api-$VERSION/* ${LITERT_API_DIR}

  LITERT_GPU_DIR=${PACKAGE_PATH}/litert-gpu/$VERSION
  mkdir -p ${LITERT_GPU_DIR}
  cp ./ci/gen/litert-gpu-$VERSION/* ${LITERT_GPU_DIR}

  LITERT_GPU_API_DIR=${PACKAGE_PATH}/litert-gpu-api/$VERSION
  mkdir -p ${LITERT_GPU_API_DIR}
  cp ./ci/gen/litert-gpu-api-$VERSION/* ${LITERT_GPU_API_DIR}

  if [[ "$VERSION" == "0.0.0-nightly-SNAPSHOT" ]]; then
    # Package debug version of litert, litert-gpu
    LITERT_DEBUG_DIR=${PACKAGE_PATH}/litert/${DEBUG_VERSION}
    mkdir -p ${LITERT_DEBUG_DIR}
    cp ./ci/gen/litert-${DEBUG_VERSION}/* ${LITERT_DEBUG_DIR}

    LITERT_GPU_DEBUG_DIR=${PACKAGE_PATH}/litert-gpu/${DEBUG_VERSION}
    mkdir -p ${LITERT_GPU_DEBUG_DIR}
    cp ./ci/gen/litert-gpu-${DEBUG_VERSION}/* ${LITERT_GPU_DEBUG_DIR}
  fi

  # Install zip
  sudo apt-get install zip

  ARTIFACT=litert.zip
  zip -r $ARTIFACT ${PACKAGE_PATH}
fi
