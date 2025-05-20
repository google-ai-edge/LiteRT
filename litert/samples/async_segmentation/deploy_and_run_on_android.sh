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

# --- Default values ---
ACCELERATOR="gpu" # Default accelerator if not specified
BINARY_BUILD_PATH=""

# --- Usage ---
usage() {
    echo "Usage: $0 --accelerator=[gpu|npu|cpu] <binary_build_path>"
    echo "  --accelerator : Specify the accelerator to use (gpu, npu, or cpu). Defaults to cpu if not provided."
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
TEMP=$(getopt -o '' --long accelerator: -- "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing options." >&2
    usage
fi

eval set -- "$TEMP"
unset TEMP

while true; do
    case "$1" in
        '--accelerator')
            ACCELERATOR="$2"
            # Validate accelerator value
            if [[ "$ACCELERATOR" != "gpu" && "$ACCELERATOR" != "npu" && "$ACCELERATOR" != "cpu" ]]; then
                echo "Error: Invalid value for --accelerator. Must be 'gpu', 'npu', or 'cpu'."
                usage
            fi
            shift 2
            continue
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
NPU_LIBRARY_LOCATION="${BINARY_BUILD_PATH}/${ROOT_DIR}vendors/qualcomm/dispatch"
PACKAGE_NAME="async_segmentation"
OUTPUT_PATH="${BINARY_BUILD_PATH}/${PACKAGE_LOCATION}/${PACKAGE_NAME}"

# Device paths
DEVICE_BASE_DIR="/data/local/tmp/${PACKAGE_NAME}_android"
DEVICE_EXEC_NAME="${PACKAGE_NAME}"
DEVICE_SHADER_DIR="${DEVICE_BASE_DIR}/shaders"
DEVICE_TEST_IMAGE_DIR="${DEVICE_BASE_DIR}/test_images"
DEVICE_MODEL_DIR="${DEVICE_BASE_DIR}/models"
DEVICE_NPU_LIBRARY_DIR="${DEVICE_BASE_DIR}/npu"

# Host paths (relative to this script's location or project root)
HOST_SHADER_DIR="${PACKAGE_LOCATION}/shaders"
HOST_TEST_IMAGE_DIR="${PACKAGE_LOCATION}/test_images"
HOST_MODEL_DIR="${PACKAGE_LOCATION}/models"
HOST_NPU_LIBRARY_DIR="${BINARY_BUILD_PATH}/${PACKAGE_LOCATION}/async_segmentation.runfiles/qairt/lib/"
HOST_GPU_LIBRARY_DIR="${BINARY_BUILD_PATH}/${PACKAGE_LOCATION}/async_segmentation.runfiles/litert_gpu/jni/arm64-v8a/"

# Qualcomm NPU library path
LD_LIBRARY_PATH="${DEVICE_NPU_LIBRARY_DIR}/"
ADSP_LIBRARY_PATH="${DEVICE_NPU_LIBRARY_DIR}/"

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
adb shell "mkdir -p ${DEVICE_NPU_LIBRARY_DIR}"
echo "Created directories on device."

# Push executable
adb push "${HOST_EXEC_PATH}" "${DEVICE_BASE_DIR}/${DEVICE_EXEC_NAME}"
echo "Pushed executable."

# Push shaders
adb push "${HOST_SHADER_DIR}/passthrough_shader.vert" "${DEVICE_SHADER_DIR}/"
adb push "${HOST_SHADER_DIR}/mask_blend_compute.glsl" "${DEVICE_SHADER_DIR}/"
adb push "${HOST_SHADER_DIR}/resize_compute.glsl" "${DEVICE_SHADER_DIR}/"
adb push "${HOST_SHADER_DIR}/preprocess_compute.glsl" "${DEVICE_SHADER_DIR}/"
adb push "${HOST_SHADER_DIR}/deinterleave_masks.glsl" "${DEVICE_SHADER_DIR}/"
echo "Pushed shaders."

# Push test images
adb push "${HOST_TEST_IMAGE_DIR}/image.jpeg" "${DEVICE_TEST_IMAGE_DIR}/"
echo "Pushed test images."

# Push model files
adb push "${HOST_MODEL_DIR}/selfie_multiclass_256x256.tflite" "${DEVICE_MODEL_DIR}/"
adb push "${HOST_MODEL_DIR}/selfie_multiclass_256x256_SM8750.tflite" "${DEVICE_MODEL_DIR}/"
echo "Pushed segmentation models."

# Push c api shared library
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${DEVICE_BASE_DIR}/" 
adb push "${C_LIBRARY_LOCATION}/libLiteRtRuntimeCApi.so" "${DEVICE_BASE_DIR}/"
echo "Pushed c api shared library."

# Push gpu accelerator shared library
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${DEVICE_BASE_DIR}/" 
adb push "${HOST_GPU_LIBRARY_DIR}/libLiteRtGpuAccelerator.so" "${DEVICE_BASE_DIR}/"
echo "Pushed gpu accelerator shared library."

# Push NPU dispatch library
adb push "${NPU_LIBRARY_LOCATION}/libLiteRtDispatch_Qualcomm.so" "${DEVICE_NPU_LIBRARY_DIR}/"
echo "Pushed NPU dispatch library."

# Push NPU libraries
adb push "${HOST_NPU_LIBRARY_DIR}/aarch64-android/libQnnHtp.so" "${DEVICE_NPU_LIBRARY_DIR}/"
adb push "${HOST_NPU_LIBRARY_DIR}/aarch64-android/libQnnHtpV79Stub.so" "${DEVICE_NPU_LIBRARY_DIR}/"
adb push "${HOST_NPU_LIBRARY_DIR}/aarch64-android/libQnnSystem.so" "${DEVICE_NPU_LIBRARY_DIR}/"
adb push "${HOST_NPU_LIBRARY_DIR}/aarch64-android/libQnnHtpPrepare.so" "${DEVICE_NPU_LIBRARY_DIR}/"
adb push "${HOST_NPU_LIBRARY_DIR}/hexagon-v79/unsigned/libQnnHtpV79Skel.so" "${DEVICE_NPU_LIBRARY_DIR}/"
echo "Pushed NPU libraries."

# Set execute permissions
adb shell "chmod +x ${DEVICE_BASE_DIR}/${DEVICE_EXEC_NAME}"
echo "Set execute permissions on device."

echo ""
echo "Deployment complete."
echo "To run the async segmentation on the device, use commands like these:"
echo "  CPU (default):"
echo "    adb shell \"cd ${DEVICE_BASE_DIR} && ./${DEVICE_EXEC_NAME} ./test_images/image.jpeg ./output_segmented.png\""
echo "  GPU:"
echo "    adb shell \"cd ${DEVICE_BASE_DIR} && ./${DEVICE_EXEC_NAME} ./test_images/image.jpeg ./output_segmented.png gpu\""
echo "  NPU:"
echo "    adb shell \"LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\" ADSP_LIBRARY_PATH=\"${ADSP_LIBRARY_PATH}\" cd ${DEVICE_BASE_DIR} && ./${DEVICE_EXEC_NAME} ./test_images/image.jpeg ./output_segmented.png npu\""
echo ""
echo "To pull the results:"
echo "  adb pull ${DEVICE_BASE_DIR}/output_segmented.png ."
echo "  adb pull ${DEVICE_BASE_DIR}/segmentation_mask_0.png ."
echo "Set the dynamic library path:"
echo "  LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\" ADSP_LIBRARY_PATH=\"${ADSP_LIBRARY_PATH}\""

adb shell "cd ${DEVICE_BASE_DIR} && LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\" ADSP_LIBRARY_PATH=\"${ADSP_LIBRARY_PATH}\" ./${DEVICE_EXEC_NAME} ./test_images/image.jpeg ./output_segmented.png ${ACCELERATOR}"
adb pull ${DEVICE_BASE_DIR}/output_segmented.png .