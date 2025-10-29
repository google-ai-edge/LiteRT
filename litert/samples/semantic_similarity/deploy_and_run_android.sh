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

# --- Local Configuration ---
# Assuming this script is run from the root of the repository.
readonly ROOT_DIR="."
# Built libraries location for Bazel
readonly C_LIBRARY_LOCATION="${ROOT_DIR}/bazel-bin/litert/c"
# Target to build
readonly TARGET="//litert/samples/semantic_similarity:semantic_similarity"
# The Android build config suffix
readonly ANDROID_CONFIG="android_arm64"
# Name of the binary produced by the build
readonly BINARY_NAME="semantic_similarity"
# Local path to the models directory
readonly LOCAL_MODEL_DIR="${ROOT_DIR}/litert/samples/semantic_similarity/models"
# tokenizer filename
TOKENIZER_MODEL="${LOCAL_MODEL_DIR}/262144.model"
EMBEDDER_MODEL="${LOCAL_MODEL_DIR}/embedding_gemma_256_input_seq.tflite"

# --- Device Configuration ---
# Where to push files on the Android device (a world-writable location)
readonly DEVICE_DIR="/data/local/tmp/semantic_similarity_demo"
# Full path to the binary and model on the device
readonly DEVICE_BINARY_PATH="${DEVICE_DIR}/${BINARY_NAME}"
# Sentences for similarity comparison - to be set by command-line flags
SENTENCE1=""
SENTENCE2=""
ACCELERATOR="cpu"
DEVICE_LD_LIBRARY_PATH="${DEVICE_DIR}"
SEQUENCE_LENGTH=0

# Flag to determine if GPU libraries need to be pushed.
PUSH_GPU_LIBS="false"

# Flag to determine which NPU libraries to push.
# Options: "", "Qualcomm", "MediaTek"
SOC_MAN=""

# -- Build flags for Bazel --
# Common flags for Android builds.
# Users might need to configure NDK/SDK paths in their environment or WORKSPACE.bazel.
BUILD_FLAGS="-c opt --config=${ANDROID_CONFIG} --copt=-DABSL_FLAGS_STRIP_NAMES=0 --copt=-DABSL_FLAGS_STRIP_HELP=0"

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

# Parses and validates the comma-delimited accelerator string.
validate_accelerators() {
  print_info "Validating accelerators: ${ACCELERATOR}"
  local original_ifs="${IFS}"
  IFS=','
  local -a accelerator_list
  read -r -a accelerator_list <<< "${ACCELERATOR}"
  IFS="${original_ifs}"

  for accelerator in "${accelerator_list[@]}"; do
    local trimmed_accelerator
    trimmed_accelerator="$(echo "${accelerator}" | xargs)"
    case "${trimmed_accelerator}" in
      cpu|npu)
        ;;
      gpu)
        print_info "GPU accelerator detected. Will push GPU delegate libraries."
        PUSH_GPU_LIBS="true"
        ;;
      *)
        print_error "Invalid accelerator specified: '${trimmed_accelerator}'"
        print_error "Valid options are: cpu, gpu, npu."
        exit 1
        ;;
    esac
  done
  print_success "Accelerators validated."
}

usage() {
  echo "Usage: $0 [OPTIONS]..."
  echo "A script to build, push, and run a LiteRT sample on an Android device."
  echo
  echo "Options:"
  echo "  --tokenizer         Path to the tokenizer model. (Default: ${TOKENIZER_MODEL})"
  echo "  --embedder          Path to the embedder model. (Default: ${EMBEDDER_MODEL})"
  echo "  --sentence1         (Required) First sentence for comparison."
  echo "  --sentence2         (Required) Second sentence for comparison."
  echo "  --accelerator       Backend to use (e.g. cpu, gpu). Can be multiple e.g. 'cpu,npu' (Default: ${ACCELERATOR})"
  echo "  --sequence_length   Sequence length of the embedder model. (Default: ${SEQUENCE_LENGTH}, will be inferred from the model path)"
  echo "  --soc_man           SoC Manufacturer for NPU libs (e.g., Qualcomm, MediaTek). (Default: ${SOC_MAN})"
  echo
  echo "Example: $0 --sentence1=\"Hello world\" --sentence2=\"Hi there\""
}

build() {
  print_info "Building the LiteRt C library with Bazel..."
  bazel build ${BUILD_FLAGS} \
      //litert/c:libLiteRtRuntimeCApi.so

  print_info "Build command: ${BUILD_FLAGS} ${TARGET}"
  print_info "Building target for ${ANDROID_CONFIG} with Bazel..."
  bazel build ${BUILD_FLAGS} ${TARGET}
  print_success "Build complete."
}

push() {
  print_info "Pushing files to device directory: ${DEVICE_DIR}"
  adb shell "mkdir -p ${DEVICE_DIR}"

  local local_binary_path="${ROOT_DIR}/bazel-bin/litert/samples/semantic_similarity/semantic_similarity"

  if [[ ! -f "${local_binary_path}" ]]; then
    print_error "Could not find built binary at ${local_binary_path}. Please run the build step first."
    exit 1
  fi

  print_info "Found binary at: ${local_binary_path}"

  adb push --sync "${local_binary_path}" "${DEVICE_BINARY_PATH}"
  adb push --sync "${TOKENIZER_MODEL}" "${DEVICE_DIR}/${TOKENIZER_MODEL##*/}"
  adb push --sync "${EMBEDDER_MODEL}" "${DEVICE_DIR}/${EMBEDDER_MODEL##*/}"
  adb push --sync "${C_LIBRARY_LOCATION}/libLiteRtRuntimeCApi.so" "${DEVICE_LD_LIBRARY_PATH}"

  if [[ "${PUSH_GPU_LIBS}" == "true" ]]; then
    print_info "Pushing GPU accelerator library..."
    # This path assumes the library is checked into the repo or built by Bazel
    local local_gpu_library_path="${ROOT_DIR}/litert/samples/semantic_similarity/libs/libLiteRtOpenClAccelerator.so"
    if [[ ! -f "${local_gpu_library_path}" ]]; then
      print_error "Could not find GPU library at: ${local_gpu_library_path}"
      exit 1
    fi
    adb push --sync "${local_gpu_library_path}" "${DEVICE_LD_LIBRARY_PATH}"
  fi

  if [[ "${SOC_MAN}" == "Qualcomm" ]]; then
    print_info "Building and Pushing QNN accelerator library..."
    bazel build ${BUILD_FLAGS} //litert/vendors/qualcomm/dispatch:dispatch_api_so
    adb push --sync "${ROOT_DIR}/bazel-bin/litert/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so" "${DEVICE_LD_LIBRARY_PATH}"

    if [[ -z "${QNN_SDK_ROOT}" ]]; then
      print_error "QNN_SDK_ROOT environment variable is not set. Cannot push QNN HTP libraries."
      exit 1
    fi
    if [[ ! -d "${QNN_SDK_ROOT}" ]]; then
      print_error "Could not find QNN SDK at: ${QNN_SDK_ROOT}"
      exit 1
    fi
    print_info "Pushing QNN HTP libraries from ${QNN_SDK_ROOT}..."
    adb push --sync ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp*Stub.so ${DEVICE_LD_LIBRARY_PATH}
    adb push --sync ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so ${DEVICE_LD_LIBRARY_PATH}
    adb push --sync ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so ${DEVICE_LD_LIBRARY_PATH}
    # Note: The glob below might need adjustment based on QNN SDK structure.
    adb push --sync ${QNN_SDK_ROOT}/lib/hexagon-*/unsigned/libQnnHtp*Skel.so ${DEVICE_LD_LIBRARY_PATH}
  elif [[ "${SOC_MAN}" == "MediaTek" ]]; then
    print_info "Building and Pushing MTK accelerator library..."
    bazel build ${BUILD_FLAGS} //litert/vendors/mediatek/dispatch:dispatch_api_so
    adb push --sync "${ROOT_DIR}/bazel-bin/litert/vendors/mediatek/dispatch/libLiteRtDispatch_MediaTek.so" "${DEVICE_LD_LIBRARY_PATH}"
  fi

  print_success "Files pushed to the device."
}

run() {
  print_info "Running the binary on the device..."

  local device_tokenizer_path="${DEVICE_DIR}/${TOKENIZER_MODEL##*/}"
  local device_embedder_path="${DEVICE_DIR}/${EMBEDDER_MODEL##*/}"

  print_info "Before running the binary, please make sure the following files exist on the device: ${device_tokenizer_path}, ${device_embedder_path}"

  print_info "running command: chmod +x ${DEVICE_BINARY_PATH} && \
    LD_LIBRARY_PATH=${DEVICE_LD_LIBRARY_PATH} \
    ADSP_LIBRARY_PATH=${DEVICE_LD_LIBRARY_PATH} \
    ${DEVICE_BINARY_PATH} \
      --tokenizer=${device_tokenizer_path} \
      --embedder=${device_embedder_path} \
      --sentence1=\"${SENTENCE1}\" \
      --sentence2=\"${SENTENCE2}\" \
      --accelerator=\"${ACCELERATOR}\" \
      --sequence_length=${SEQUENCE_LENGTH} \
      --dispatch_library_dir=${DEVICE_DIR}"

  adb shell "chmod +x ${DEVICE_BINARY_PATH} && \
    LD_LIBRARY_PATH=${DEVICE_LD_LIBRARY_PATH} \
    ADSP_LIBRARY_PATH=${DEVICE_LD_LIBRARY_PATH} \
    ${DEVICE_BINARY_PATH} \
      --tokenizer=${device_tokenizer_path} \
      --embedder=${device_embedder_path} \
      --sentence1=\"${SENTENCE1}\" \
      --sentence2=\"${SENTENCE2}\" \
      --accelerator=\"${ACCELERATOR}\" \
      --sequence_length=${SEQUENCE_LENGTH} \
      --dispatch_library_dir=${DEVICE_DIR}"

  print_success "Execution finished."
}

parse_args() {
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --tokenizer)
        if [[ -z "$2" || "$2" == -* ]]; then print_error "Missing value for $1"; usage; exit 1; fi
        TOKENIZER_MODEL="$2"
        shift 2
        ;;
      --embedder)
        if [[ -z "$2" || "$2" == -* ]]; then print_error "Missing value for $1"; usage; exit 1; fi
        EMBEDDER_MODEL="$2"
        shift 2
        ;;
      --sentence1)
        if [[ -z "$2" || "$2" == -* ]]; then print_error "Missing value for $1"; usage; exit 1; fi
        SENTENCE1="$2"
        shift 2
        ;;
      --sentence2)
        if [[ -z "$2" || "$2" == -* ]]; then print_error "Missing value for $1"; usage; exit 1; fi
        SENTENCE2="$2"
        shift 2
        ;;
      --accelerator)
        if [[ -z "$2" || "$2" == -* ]]; then print_error "Missing value for $1"; usage; exit 1; fi
        ACCELERATOR="$2"
        shift 2
        ;;
      --sequence_length)
        if [[ -z "$2" || "$2" == -* ]]; then print_error "Missing value for $1"; usage; exit 1; fi
        SEQUENCE_LENGTH="$2"
        shift 2
        ;;
      --soc_man)
        if [[ -z "$2" || "$2" == -* ]]; then print_error "Missing value for $1"; usage; exit 1; fi
        SOC_MAN="$2"
        shift 2
        ;;
      *)
        print_error "Unknown option: $1"
        usage
        exit 1
        ;;
    esac
  done
}

# --- Main Logic ---
main() {
  parse_args "$@"

  validate_accelerators

  if [[ -z "${SENTENCE1}" || -z "${SENTENCE2}" ]]; then
    print_error "Error: --sentence1 and --sentence2 are required."
    usage
    exit 1
  fi

  build
  push
  run

  print_success "All actions completed."
}

main "$@"