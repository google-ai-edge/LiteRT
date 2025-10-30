# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

source ${0%.sh}_lib.sh || exit 1

check_host=""
check_device=""
bin_args=()

function setup_context() {
  for flag in "$@"; do
    echo "flag: $flag"
    case $flag in
      --check_host*)
        check_host="true"
        ;;
      --check_device*)
        check_device="true"
        ;;
      *)
        bin_args+=("$flag")
        ;;
    esac
  done
}

setup_context "$@"

echo "check_host: $check_host"
echo "check_device: $check_device"
echo "host_bin: $(host_bin)"
echo "device_bin: $(device_bin)"
echo "host_libs: $(host_libs)"
echo "device_libs: $(device_libs)"
echo "data_files: $(data_files)"
echo "find_host_plugin: $(find_host_plugin)"
echo "find_device_plugin: $(find_device_plugin)"
echo "find_host_dispatch: $(find_host_dispatch)"
echo "find_device_dispatch: $(find_device_dispatch)"
echo "find_host_runtime_lib: $(find_host_runtime_lib)"
echo "find_device_runtime_lib: $(find_device_runtime_lib)"

function check_file() {
  local file=$1
  local tag=$2
  local suffix=$3
  if [[ -f "$file" && "$file" == *"$suffix" ]]; then
    echo "${tag}: $(basename ${file}) OK"
  else
    echo "${tag}: $(basename ${file}) NOT OK"
    exit 1
  fi
}

function check_len() {
  local tag=$1
  declare -i len=${2}
  local array=(${@:3})
  if (( ${#array[@]} == ${len} )); then
    echo "${tag} array len OK"
  else
    echo "${tag}: array len NOT OK, expected ${len}, got ${#array[@]}"
    exit 1
  fi
}

provided_models=($(get_provided_models))
if [ $? -ne 0 ]; then
    echo "Failed to get provided models."
    exit 1
else
    check_len "provided_models" 2 ${provided_models[*]}
fi

check_len "data_files" 2 $(data_files)
for file in $(data_files); do
  check_file "$file" "data_files" ".tflite"
done

if [[ -n "$check_host" ]]; then
  check_file "$(host_bin)" "host_bin" "dummy_binary_for_host"
  check_len "host_libs" 3 $(host_libs)
  for file in $(host_libs); do
    check_file "$file" "host_lib" ".so"
  done
  check_file "$(find_host_plugin)" "find_host_plugin" "Example.so"
  check_file "$(find_host_dispatch)" "find_host_dispatch" "Example.so"
  check_file "$(find_host_runtime_lib)" "find_host_runtime_lib" "libLiteRtRuntimeCApi.so"
  echo "Running host bin..."
  $(host_bin) "${bin_args[@]}"
fi

if [[ -n "$check_device" ]]; then
  check_file "$(device_bin)" "device_bin" "dummy_binary"
  check_len "device_libs" 3 $(device_libs)
  for file in $(device_libs); do
    check_file "$file" "device_lib" ".so"
  done
  check_file "$(find_device_plugin)" "find_device_plugin" "Example.so"
  check_file "$(find_device_dispatch)" "find_device_dispatch" "Example.so"
  check_file "$(find_device_runtime_lib)" "find_device_runtime_lib" "libLiteRtRuntimeCApi.so"
fi


# echo "$(get_provided_models)"
