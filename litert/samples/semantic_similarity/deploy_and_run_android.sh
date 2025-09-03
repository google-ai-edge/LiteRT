#!/bin/bash
#
# Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Target to build
readonly TARGET="//third_party/odml/litert/litert/samples/semantic_similarity:tokenizer_test"
# The Android build config suffix
readonly ANDROID_CONFIG="android_arm64"
# Name of the binary produced by the build
readonly BINARY_NAME="tokenizer_test"
# Local path to the tokenizer model
readonly LOCAL_MODEL_PATH="third_party/odml/litert/litert/samples/semantic_similarity/models/262144.model"
# Where to push files on the Android device (a world-writable location)
readonly DEVICE_DIR="/data/local/tmp"
# Full path to the binary and model on the device
readonly DEVICE_BINARY_PATH="${DEVICE_DIR}/${BINARY_NAME}"
readonly DEVICE_MODEL_PATH="${DEVICE_DIR}/262144.model"

# --- Helper Functions for Colored Output ---
print_info() {
  echo -e "\033[1;34m[INFO]\033[0m $1"
}
print_success() {
  echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}
print_error() {
  echo -e "\033[1;31m[ERROR]\033[0m $1" >&2
}

# --- Core Functions ---

usage() {
  echo "Usage: $0 [ACTION]..."
  echo "A script to build, push, and run a LiteRT sample on an Android device."
  echo
  echo "Actions:"
  echo "  build    Builds the binary for Android."
  echo "  push     Pushes the binary and model to the device."
  echo "  run      Executes the binary on the device."
  echo "  all      Performs 'build', 'push', and 'run' in sequence."
  echo
  echo "Example: $0 all"
  echo "Example: $0 build push"
}

build() {
  print_info "Building target for ${ANDROID_CONFIG}..."
  bazel build --config=${ANDROID_CONFIG} ${TARGET}
  print_success "Build complete."
}

push() {
  print_info "Pushing files to device directory: ${DEVICE_DIR}"

  # Find the compiled binary in the bazel-bin directory. This is more robust
  # than hardcoding the full path. The path will be something like:
  # bazel-bin/third_party/odml/litert/litert/samples/semantic_similarity/android_arm64_stripped/tokenizer_test
  local local_binary_path
  local_binary_path=$(find bazel-bin/ -path "*/${TARGET//://}" -type f | head -n 1)

  if [[ -z "${local_binary_path}" ]]; then
    print_error "Could not find built binary. Please run the 'build' action first."
    exit 1
  fi

  print_info "Found binary at: ${local_binary_path}"

  # Push the binary and the model file using adb
  adb push "${local_binary_path}" "${DEVICE_BINARY_PATH}"
  adb push "${LOCAL_MODEL_PATH}" "${DEVICE_MODEL_PATH}"
  print_success "Files pushed to the device."
}

run() {
  print_info "Running the binary on the device..."

  # Use adb shell to execute commands on the device:
  # 1. Make the binary executable.
  # 2. Run the binary, passing the on-device path to the tokenizer model.
  adb shell "chmod +x ${DEVICE_BINARY_PATH} && ${DEVICE_BINARY_PATH} --tokenizer=${DEVICE_MODEL_PATH}"

  print_success "Execution finished."
}

# --- Main Logic ---

# If no arguments are provided, show the usage information.
if [ "$#" -eq 0 ]; then
  usage
  exit 1
fi

# Loop through all arguments and execute the corresponding action.
for action in "$@"; do
  case "$action" in
    build)
      build
      ;;
    push)
      push
      ;;
    run)
      run
      ;;
    all)
      build
      push
      run
      ;;
    *)
      print_error "Unknown action: '$action'"
      usage
      exit 1
      ;;
  esac
done

print_success "All requested actions completed."
