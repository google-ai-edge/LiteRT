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

#!/bin/bash

# Script to deploy and run the image merger on an Android device via ADB
# --- Usage ---
# Checks if any arguments (paths) are provided.
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <poutput_path>"
    echo "Please provide the binary path i.e. bazel-bin/"
    exit 1
fi

# Check if the provided argument is a valid path.
if [ ! -d "$1" ]; then
    echo "Error: The provided path ($1) is not a valid directory."
    exit 1
fi

OUTPUT_PATH_ROOT="$1"

# --- Configuration ---
PACKAGE_LOCATION="third_party/odml/litert/litert/samples/async_segmentation"
PACKAGE_NAME="async_segmentation"
OUTPUT_PATH="${OUTPUT_PATH_ROOT}/${PACKAGE_LOCATION}/${PACKAGE_NAME}"

# Device paths
DEVICE_BASE_DIR="/data/local/tmp/${PACKAGE_NAME}_android"
DEVICE_EXEC_NAME="${PACKAGE_NAME}"
DEVICE_SHADER_DIR="${DEVICE_BASE_DIR}/shaders"
DEVICE_TEST_IMAGE_DIR="${DEVICE_BASE_DIR}/test_images"
DEVICE_MODEL_DIR="${DEVICE_BASE_DIR}/models"

# Host paths (relative to this script's location or project root)
HOST_SHADER_DIR="${PACKAGE_LOCATION}/shaders"
HOST_TEST_IMAGE_DIR="${PACKAGE_LOCATION}/test_images"
HOST_MODEL_DIR="${PACKAGE_LOCATION}/models"

# --- Script Logic ---
echo "Starting deployment to Android device..."

# Determine executable path
HOST_EXEC_PATH="${OUTPUT_PATH}"
echo "Using output path: ${HOST_EXEC_PATH}"

if [ ! -f "${HOST_EXEC_PATH}" ]; then
    echo "Error: Executable not found at ${HOST_EXEC_PATH}"
    echo "Please build the project first."
    exit 1
fi

echo "Target device directory: ${DEVICE_BASE_DIR}"

# Create directories on device
adb shell "mkdir -p ${DEVICE_BASE_DIR}"
adb shell "mkdir -p ${DEVICE_SHADER_DIR}"
adb shell "mkdir -p ${DEVICE_TEST_IMAGE_DIR}"
adb shell "mkdir -p ${DEVICE_MODEL_DIR}"
echo "Created directories on device."

# Push executable
adb push "${HOST_EXEC_PATH}" "${DEVICE_BASE_DIR}/${DEVICE_EXEC_NAME}"
echo "Pushed executable."

# Push shaders
adb push "${HOST_SHADER_DIR}/passthrough_shader.vert" "${DEVICE_SHADER_DIR}/"
adb push "${HOST_SHADER_DIR}/mask_blend_compute.glsl" "${DEVICE_SHADER_DIR}/"
adb push "${HOST_SHADER_DIR}/resize_compute.glsl" "${DEVICE_SHADER_DIR}/"
adb push "${HOST_SHADER_DIR}/preprocess_compute.glsl" "${DEVICE_SHADER_DIR}/"
echo "Pushed shaders."

# Push test images
adb push "${HOST_TEST_IMAGE_DIR}/image.jpeg" "${DEVICE_TEST_IMAGE_DIR}/"
echo "Pushed test images."

# Push model files
adb push "${HOST_MODEL_DIR}/selfie_multiclass_256x256.tflite" "${DEVICE_MODEL_DIR}/"
echo "Pushed segmentation model."

# Set execute permissions
adb shell "chmod +x ${DEVICE_BASE_DIR}/${DEVICE_EXEC_NAME}"
echo "Set execute permissions on device."

echo ""
echo "Deployment complete."
echo "To run the segmentation app on the device, use a command like this:"
echo "adb shell "cd ${DEVICE_BASE_DIR} \&\& ./${DEVICE_EXEC_NAME} ./test_images/image.jpeg ./output_segmented.png""
echo ""
echo "To pull the result:"
echo "adb pull ${DEVICE_BASE_DIR}/output_segmented.png ."
echo ""

adb shell "cd ${DEVICE_BASE_DIR} && ./${DEVICE_EXEC_NAME} ./test_images/image.jpeg ./output_segmented.png"
adb pull ${DEVICE_BASE_DIR}/output_segmented.png .