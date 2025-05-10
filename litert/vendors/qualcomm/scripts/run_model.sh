#!/bin/bash

print_usage() {
    echo "Usage: $0 <LiteRT Root> <DeviceKey> <ModelPath>"
    echo "DeviceKey: select a key from \$LiteRT/litert/vendors/qualcomm/scripts/devices.json"
    echo "ModelPath: compiled model path"
    echo "  example:"
    echo "    $0 \$LiteRT/ example model_path"
}

if [ "$#" -eq 1 ]; then
    if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
        print_usage
        exit 0
    fi
fi

if [ "$#" -ne 3 ]; then
    print_usage
    exit 1
fi

# read arguments
litert_root="$1"
device_key="$2"
model_path="$3"


# build run_model
pushd "${litert_root}"
echo "build run_model..."
bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility --config=android_arm64 --copt=-DABSL_FLAGS_STRIP_NAMES=0 //litert/vendors/qualcomm/dispatch:dispatch_api_so
bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility --config=android_arm64 --copt=-DABSL_FLAGS_STRIP_NAMES=0 //litert/tools:run_model
popd


# push so, executable, model into devices
json_path="${litert_root}/litert/vendors/qualcomm/scripts/devices.json"
json_data=$(jq '.' "${json_path}")
hostname=$(echo ${json_data} | jq -r ".${device_key}.hostname")
serial=$(echo ${json_data} | jq -r ".${device_key}.serial")
soc_arch=$(echo ${json_data} | jq -r ".${device_key}.soc_arch")

device_folder="/data/local/tmp/${USER}"

adb_command="adb -H ${hostname} -s ${serial}"
$adb_command shell "rm -rf ${device_folder}"
$adb_command shell "mkdir ${device_folder}"

# import utility for pushing QNN into devices
source $(dirname "$(readlink -f "$0")")/utils.sh
push_qnn_to_device ${litert_root} ${device_folder} ${device_key}

$adb_command push "${litert_root}/bazel-bin/litert/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so" "${device_folder}"
$adb_command push "${litert_root}/bazel-bin/litert/c/libLiteRtRuntimeCApi.so" "${device_folder}"
$adb_command push "${litert_root}/bazel-bin/litert/tools/run_model.runfiles/litert/litert/tools/run_model" "${device_folder}"
$adb_command push "${model_path}" "${device_folder}"


# execute
model_name=$(basename "${model_path}")
$adb_command shell "export LD_LIBRARY_PATH=${device_folder}:\$LD_LIBRARY_PATH &&
             export ADSP_LIBRARY_PATH=\"${device_folder}\" &&
             ./${device_folder}/run_model --graph=${device_folder}/${model_name} --dispatch_library_dir=${device_folder} --signature_index=0"
