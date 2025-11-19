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

# Script to deploy and run the image merger on an Android device via ADB
set -e

# --- Default values ---
ACCELERATOR="gpu" # Default accelerator if not specified
PHONE="s25" # Default phone model
BINARY_BUILD_PATH=""

# --- Usage ---
usage() {
    echo "Usage: $0 --accelerator=[gpu|npu|cpu] --phone=[s24|s25] <binary_build_path>"
    echo "  --accelerator : Specify the accelerator to use (gpu, npu, or cpu). Defaults to cpu if not provided."
    echo "  --phone       : Specify the phone model (e.g., s24, s25) to select the correct NPU libraries. Defaults to s25."
    echo "  --jit         : Specify whether to use JIT compilation (true or false). Only used for NPU. Defaults to false."
    echo "  <binary_build_path> : The path to the binary build directory (e.g., bazel-bin/)."
    exit 1
}

# --- Argument Parsing ---
# Check if any arguments are provided at all.
if [ "$#" -eq 0 ]; then
    echo "Error: No arguments provided."
    usage
fi

# Parse options
TEMP=$(getopt -o '' --long accelerator:,phone:,use_gl_buffers,jit,host_npu_lib:,host_npu_dispatch_lib:,host_npu_compiler_lib: -- "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing options." >&2
    usage
fi

eval set -- "$TEMP"
unset TEMP

USE_GL_BUFFERS=false
USE_JIT=false
HOST_NPU_LIB=""
HOST_NPU_DISPATCH_LIB=""
HOST_NPU_COMPILER_LIB=""

while true; do
    case "$1" in
        '--accelerator')
            ACCELERATOR="$2"
            # Validate accelerator value
            if [[ "$ACCELERATOR" != "gpu" && "$ACCELERATOR" != "npu" && "$ACCELERATOR" != "cpu" ]]; then
                echo "Error: Invalid value for --accelerator. Must be 'gpu', 'npu', or 'cpu'." >&2
                usage
                exit 1
            fi
            shift 2
            ;;
        '--phone')
            PHONE="$2"
            shift 2
            ;;
        '--use_gl_buffers')
            USE_GL_BUFFERS=true
            shift
            ;;
        '--jit')
            USE_JIT=true
            shift
            ;;
        '--host_npu_lib')
            HOST_NPU_LIB="$2"
            shift 2
            ;;
        '--host_npu_dispatch_lib')
            HOST_NPU_DISPATCH_LIB="$2"
            shift 2
            ;;
        '--host_npu_compiler_lib')
            HOST_NPU_COMPILER_LIB="$2"
            shift 2
            ;;
        '--')
            shift
            break
            ;;
        *)
            # This case should ideally not be reached if getopt is working correctly
            # and all options are defined.
            # However, it can catch unexpected issues or be used for positional args after options.
            break
            ;;
    esac
done

echo "Selected Accelerator: $ACCELERATOR"
echo "Use GL Buffers: $USE_GL_BUFFERS"

# The remaining argument should be the binary_build_path
if [ "$#" -ne 1 ]; then
    echo "Error: Incorrect number of arguments or invalid option."
    usage
fi

BINARY_BUILD_PATH="$1"

# Check if the binary_build_path is a valid directory.
if [ ! -d "$BINARY_BUILD_PATH" ]; then
    echo "Error: The provided binary_build_path ($BINARY_BUILD_PATH) is not a valid directory."
    exit 1
fi

# --- Configuration ---
ROOT_DIR="litert/"

PACKAGE_LOCATION="${ROOT_DIR}samples/async_segmentation"
C_LIBRARY_LOCATION="${BINARY_BUILD_PATH}/${ROOT_DIR}c"
PACKAGE_NAME="async_segmentation_${ACCELERATOR}"
OUTPUT_PATH="${BINARY_BUILD_PATH}/${PACKAGE_LOCATION}/${PACKAGE_NAME}"

# Device paths
DEVICE_BASE_DIR="/data/local/tmp/async_segmentation_android"
DEVICE_EXEC_NAME="async_segmentation_executable"
DEVICE_SHADER_DIR="${DEVICE_BASE_DIR}/shaders"
DEVICE_TEST_IMAGE_DIR="${DEVICE_BASE_DIR}/test_images"
DEVICE_MODEL_DIR="${DEVICE_BASE_DIR}/models"
DEVICE_NPU_LIBRARY_DIR="${DEVICE_BASE_DIR}/npu"

# Host paths (relative to this script's location or project root)
HOST_SHADER_DIR="${PACKAGE_LOCATION}/shaders"
HOST_TEST_IMAGE_DIR="${PACKAGE_LOCATION}/test_images"
HOST_MODEL_DIR="${PACKAGE_LOCATION}/models"
HOST_GPU_LIBRARY_DIR="${BINARY_BUILD_PATH}/${PACKAGE_LOCATION}/${PACKAGE_NAME}.runfiles/litert_gpu/jni/arm64-v8a/"

# Set NPU library path based on the --npu_dispatch_lib_path flag
if [[ -z "$HOST_NPU_LIB" ]]; then
    echo "Defaulting to QNN libraries path."
    HOST_NPU_LIB="${BINARY_BUILD_PATH}/${PACKAGE_LOCATION}/${PACKAGE_NAME}.runfiles/qairt/lib/"
fi
if [[ -z "$HOST_NPU_DISPATCH_LIB" ]]; then
    echo "Defaulting to internal dispatch library path."
    HOST_NPU_DISPATCH_LIB="${BINARY_BUILD_PATH}/${PACKAGE_LOCATION}/${PACKAGE_NAME}.runfiles/litert/vendors/qualcomm/dispatch"
fi
if [[ "$USE_JIT" == "true" ]]; then
    echo "Using NPU JIT compilation."
    if [[ -z "$HOST_NPU_COMPILER_LIB" ]]; then
        HOST_NPU_COMPILER_LIB="${BINARY_BUILD_PATH}/${PACKAGE_LOCATION}/${PACKAGE_NAME}.runfiles/litert/vendors/qualcomm/compiler"
    fi
fi

# Qualcomm NPU library path
LD_LIBRARY_PATH="${DEVICE_NPU_LIBRARY_DIR}/"
ADSP_LIBRARY_PATH="${DEVICE_NPU_LIBRARY_DIR}/"

# --- NPU Configuration ---
QNN_STUB_LIB=""
QNN_SKEL_LIB=""
QNN_SKEL_PATH_ARCH=""
case "$PHONE" in
    's24')
        QNN_STUB_LIB="libQnnHtpV75Stub.so"
        QNN_SKEL_LIB="libQnnHtpV75Skel.so"
        QNN_SKEL_PATH_ARCH="hexagon-v75"
        ;;
    's25')
        QNN_STUB_LIB="libQnnHtpV79Stub.so"
        QNN_SKEL_LIB="libQnnHtpV79Skel.so"
        QNN_SKEL_PATH_ARCH="hexagon-v79"
        ;;
    *)
        echo "Error: Unsupported phone model '$PHONE'. Supported models are 's24', 's25'." >&2
        exit 1
        ;;
esac


# --- Model Selection ---
MODEL_FILENAME="selfie_multiclass_256x256.tflite"
if [[ "$ACCELERATOR" == "npu" && "$USE_JIT" == "false" ]]; then
    if [[ "$PHONE" == "s24" ]]; then
        MODEL_FILENAME="selfie_multiclass_256x256_SM8650.tflite"
    elif [[ "$PHONE" == "s25" ]]; then
        MODEL_FILENAME="selfie_multiclass_256x256_SM8750.tflite"
    fi
fi


# --- Script Logic ---
echo "Starting deployment to Android device..."

# Determine executable path
HOST_EXEC_PATH="${OUTPUT_PATH}"
echo "Using output path: ${HOST_EXEC_PATH}"

if [ ! -f "${HOST_EXEC_PATH}" ]; then
    echo "Error: Executable not found at ${HOST_EXEC_PATH}"
    echo "Please ensure the project has been built and the correct path is provided."
    exit 1
fi

echo "Target device directory: ${DEVICE_BASE_DIR}"

# Create directories on device
adb shell "mkdir -p ${DEVICE_BASE_DIR}"
adb shell "mkdir -p ${DEVICE_SHADER_DIR}"
adb shell "mkdir -p ${DEVICE_TEST_IMAGE_DIR}"
adb shell "mkdir -p ${DEVICE_MODEL_DIR}"
adb shell "mkdir -p ${DEVICE_NPU_LIBRARY_DIR}"
echo "Created directories on device."

# Push executable
adb push --sync "${HOST_EXEC_PATH}" "${DEVICE_BASE_DIR}/${DEVICE_EXEC_NAME}"
echo "Pushed executable."

# Push shaders
adb push --sync "${HOST_SHADER_DIR}/passthrough_shader.vert" "${DEVICE_SHADER_DIR}/"
adb push --sync "${HOST_SHADER_DIR}/mask_blend_compute.glsl" "${DEVICE_SHADER_DIR}/"
adb push --sync "${HOST_SHADER_DIR}/resize_compute.glsl" "${DEVICE_SHADER_DIR}/"
adb push --sync "${HOST_SHADER_DIR}/preprocess_compute.glsl" "${DEVICE_SHADER_DIR}/"
adb push --sync "${HOST_SHADER_DIR}/deinterleave_masks.glsl" "${DEVICE_SHADER_DIR}/"
echo "Pushed shaders."

# Push test images
adb push --sync "${HOST_TEST_IMAGE_DIR}/image.jpeg" "${DEVICE_TEST_IMAGE_DIR}/"
echo "Pushed test images."

# Push model files
adb push --sync "${HOST_MODEL_DIR}/${MODEL_FILENAME}" "${DEVICE_MODEL_DIR}/"
echo "Pushed segmentation models."

# Push c api shared library
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${DEVICE_BASE_DIR}/"
adb push --sync "${C_LIBRARY_LOCATION}/libLiteRt.so" "${DEVICE_BASE_DIR}/"
echo "Pushed c api shared library."

# Push gpu accelerator shared library
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${DEVICE_BASE_DIR}/"
if [[ "$ACCELERATOR" == "gpu" ]]; then
    adb push --sync "${HOST_GPU_LIBRARY_DIR}/libLiteRtOpenClAccelerator.so" "${DEVICE_BASE_DIR}/"
fi
echo "Pushed gpu accelerator shared library."

# Push NPU dispatch library
if [[ "$ACCELERATOR" == "npu" ]]; then
adb push --sync "${HOST_NPU_DISPATCH_LIB}/libLiteRtDispatch_Qualcomm.so" "${DEVICE_NPU_LIBRARY_DIR}/"
echo "Pushed NPU dispatch library."

# Push NPU libraries
adb push --sync "${HOST_NPU_LIB}/aarch64-android/libQnnHtp.so" "${DEVICE_NPU_LIBRARY_DIR}/"
adb push --sync "${HOST_NPU_LIB}/aarch64-android/${QNN_STUB_LIB}" "${DEVICE_NPU_LIBRARY_DIR}/"
adb push --sync "${HOST_NPU_LIB}/aarch64-android/libQnnSystem.so" "${DEVICE_NPU_LIBRARY_DIR}/"
adb push --sync "${HOST_NPU_LIB}/aarch64-android/libQnnHtpPrepare.so" "${DEVICE_NPU_LIBRARY_DIR}/"
adb push --sync "${HOST_NPU_LIB}/${QNN_SKEL_PATH_ARCH}/unsigned/${QNN_SKEL_LIB}" "${DEVICE_NPU_LIBRARY_DIR}/"
echo "Pushed NPU libraries."

# Push NPU compiler library
if [[ "$USE_JIT" == "true" ]]; then
    adb push --sync "${HOST_NPU_COMPILER_LIB}/libLiteRtCompilerPlugin_Qualcomm.so" "${DEVICE_NPU_LIBRARY_DIR}/"
    echo "Pushed NPU compiler library."
fi
fi

# Set execute permissions
adb shell "chmod +x ${DEVICE_BASE_DIR}/${DEVICE_EXEC_NAME}"
echo "Set execute permissions on device."

echo "Cleaning up previous run results"
adb shell "rm -f ${DEVICE_BASE_DIR}/output_segmented.png"

echo ""
echo "Deployment complete."
echo "To run the async segmentation on the device, use a command like this:"

MODEL_PATH="./models/${MODEL_FILENAME}"

RUN_COMMAND="./${DEVICE_EXEC_NAME} ${MODEL_PATH} ./test_images/image.jpeg ./output_segmented.png"
if [[ "$ACCELERATOR" == "gpu" ]] && $USE_GL_BUFFERS; then
    RUN_COMMAND="${RUN_COMMAND} true"
fi

if [[ "$ACCELERATOR" == "npu" ]]; then
    FULL_COMMAND="cd ${DEVICE_BASE_DIR} && LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\" ADSP_LIBRARY_PATH=\"${ADSP_LIBRARY_PATH}\" ${RUN_COMMAND}"
    if [[ "$USE_JIT" == "true" ]]; then
        FULL_COMMAND="${FULL_COMMAND} true"
    fi
else
    FULL_COMMAND="cd ${DEVICE_BASE_DIR} && LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\" ${RUN_COMMAND}"
fi

echo "  adb shell \"${FULL_COMMAND}\""
adb shell "${FULL_COMMAND}"

echo ""
echo "To pull the result:"
echo "  adb pull ${DEVICE_BASE_DIR}/output_segmented.png ."
adb pull ${DEVICE_BASE_DIR}/output_segmented.png .
