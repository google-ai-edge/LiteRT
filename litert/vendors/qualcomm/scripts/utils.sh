#!/bin/bash

push_qnn_to_device() {
    local litert_root="$1"
    local device_folder="$2"
    local device_key="$3"

    local json_path="${litert_root}/litert/vendors/qualcomm/scripts/devices.json"
    local json_data=$(jq '.' "${json_path}")
    local hostname=$(echo ${json_data} | jq -r ".${device_key}.hostname")
    local serial=$(echo ${json_data} | jq -r ".${device_key}.serial")
    local soc_arch=$(echo ${json_data} | jq -r ".${device_key}.soc_arch")

    local device_folder="/data/local/tmp/${USER}"

    local adb_command="adb -H ${hostname} -s ${serial}"

    local qairt_lib="${litert_root}/third_party/qairt/latest/lib"
    $adb_command push "${qairt_lib}/aarch64-android/libQnnSystem.so" "${device_folder}"
    $adb_command push "${qairt_lib}/aarch64-android/libQnnHtp.so" "${device_folder}"
    $adb_command push "${qairt_lib}/aarch64-android/libQnnHtpPrepare.so" "${device_folder}"
    if [ "${soc_arch}" == "V68" ]; then
        $adb_command push "${qairt_lib}/aarch64-android/libQnnHtpV68Stub.so" "${device_folder}"
        $adb_command push "${qairt_lib}/hexagon-v68/unsigned/libQnnHtpV68Skel.so" "${device_folder}"

    elif [ "${soc_arch}" == "V69" ]; then
        $adb_command push "${qairt_lib}/aarch64-android/libQnnHtpV69Stub.so" "${device_folder}"
        $adb_command push "${qairt_lib}/hexagon-v69/unsigned/libQnnHtpV69Skel.so" "${device_folder}"

    elif [ "${soc_arch}" == "V73" ]; then
        $adb_command push "${qairt_lib}/aarch64-android/libQnnHtpV73Stub.so" "${device_folder}"
        $adb_command push "${qairt_lib}/hexagon-v73/unsigned/libQnnHtpV73Skel.so" "${device_folder}"

    elif [ "${soc_arch}" == "V75" ]; then
        $adb_command push "${qairt_lib}/aarch64-android/libQnnHtpV75Stub.so" "${device_folder}"
        $adb_command push "${qairt_lib}/hexagon-v75/unsigned/libQnnHtpV75Skel.so" "${device_folder}"

    elif [ "${soc_arch}" == "V79" ]; then
        $adb_command push "${qairt_lib}/aarch64-android/libQnnHtpV79Stub.so" "${device_folder}"
        $adb_command push "${qairt_lib}/hexagon-v79/unsigned/libQnnHtpV79Skel.so" "${device_folder}"

    else
        echo "Error: Unsupported SoC Arch ${soc_arch}"

    fi
}
