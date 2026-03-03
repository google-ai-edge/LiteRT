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

# Expected env variables:
#  - DOCKER_PYTHON_VERSION (default=3.11)
#  - ANDROID_TARGET_ARCHS (default="arm64-v8a") - comma-separated list
#  - ANDROID_API_LEVEL (default=21)
#  - NIGHTLY_RELEASE_DATE (optional)
#  - TEST_WHEEL (default=false)
#  - USE_LOCAL_TF (default=false)
#

DOCKER_PYTHON_VERSION="${DOCKER_PYTHON_VERSION:-3.11}"
ANDROID_TARGET_ARCHS="${ANDROID_TARGET_ARCHS:-arm64-v8a}"
ANDROID_API_LEVEL="${ANDROID_API_LEVEL:-21}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."

if [ ! -d /root_dir ]; then
  # Running on host.
  cd ${SCRIPT_DIR}

  docker build . -t tflite-android-builder -f tflite-android-py3.Dockerfile

  docker run \
    -v ${SCRIPT_DIR}/../third_party/tensorflow:/third_party_tensorflow \
    -v ${ROOT_DIR}:/root_dir \
    -v ${SCRIPT_DIR}:/script_dir \
    -e NIGHTLY_RELEASE_DATE=${NIGHTLY_RELEASE_DATE} \
    -e DOCKER_PYTHON_VERSION=${DOCKER_PYTHON_VERSION} \
    -e ANDROID_TARGET_ARCHS=${ANDROID_TARGET_ARCHS} \
    -e ANDROID_API_LEVEL=${ANDROID_API_LEVEL} \
    -e CUSTOM_BAZEL_FLAGS="${CUSTOM_BAZEL_FLAGS}" \
    -e TEST_WHEEL=${TEST_WHEEL:-false} \
    -e USE_LOCAL_TF=${USE_LOCAL_TF:-false} \
    -e RELEASE_VERSION=${RELEASE_VERSION} \
    --entrypoint /script_dir/build_android_python_wheel_with_docker.sh \
    tflite-android-builder
  exit 0
else
  # Running inside docker container
  cd /root_dir

  export CI_BUILD_PYTHON="python${DOCKER_PYTHON_VERSION}"
  export HERMETIC_PYTHON_VERSION="${DOCKER_PYTHON_VERSION}"
  export TF_LOCAL_SOURCE_PATH="/root_dir/third_party/tensorflow"
  export ANDROID_API_LEVEL="${ANDROID_API_LEVEL}"

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

  # Create dist directory
  mkdir -p /root_dir/dist

  # Parse Android target architectures and build wheels for each
  IFS=',' read -ra ARCHS <<< "$ANDROID_TARGET_ARCHS"
  for arch in "${ARCHS[@]}"; do
    arch=$(echo "$arch" | xargs)  # trim whitespace
    
    echo "Building LiteRT Python wheel for Android ${arch}..."
    
    # Map architecture to Bazel config
    case $arch in
      x86_64)
        bazel_config="android_x86_64"
        ;;
      x86)

      *)
        echo "Unknown architecture: $arch"
        exit 1
        ;;
    esac
    
    # Build wheel for this architecture
    bazel build \
      --config=${bazel_config} \
      --action_env=PYTHON_BIN_PATH=${CI_BUILD_PYTHON} \
      ${CUSTOM_BAZEL_FLAGS} \
      //litert/python:litert_wheel
    
    # Copy built wheel to dist directory with architecture suffix
    # Note: Adjust path based on actual Bazel build output location
    if [ -f "bazel-bin/litert/python/litert_wheel.whl" ]; then
      wheel_name=$(basename bazel-bin/litert/python/litert_wheel.whl .whl)
      cp "bazel-bin/litert/python/litert_wheel.whl" \
         "/root_dir/dist/${wheel_name}-android_${arch}.whl"
    fi
  done

  echo "Android Python wheels built successfully!"
  ls -lah /root_dir/dist/

fi
