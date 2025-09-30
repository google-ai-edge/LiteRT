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

reset_color='\033[0m'
host_color='\033[34m'
hightlight_color='\033[36m'

readonly device_runfiles_root="/data/local/tmp/runfiles"

should_exec=1
host_bin=@@host_bin@@
if [[ "$host_bin" == "@@"*"@@" ]]; then
  host_bin=""
  should_exec=0
fi

host_runfiles=@@host_runfiles@@
if [[ "$host_runfiles" == "@@"*"@@" ]]; then
  host_runfiles=""
fi

exec_args=@@exec_args@@
if [[ "$exec_args" == "@@"*"@@" ]]; then
  exec_args=""
fi

exec_env_vars=@@exec_env_vars@@
if [[ "$exec_env_vars" == "@@"*"@@" ]]; then
  exec_env_vars=""
fi

device_paths=()
device_bin_path=""

function device_path() {
  echo "${device_runfiles_root}/${1}" | sed "s|\.\.|external|g"
}

function print_info_key() {
  echo -e "${host_color}${1}:${reset_color}"
}

function print() {
  echo -e "${host_color}${1}${reset_color}"
}

function print_file() {
  if [[ "$#" -ne 2 ]]; then
    echo -e "    ${1}"
  else
    echo -e "    ${1} => ${2}"
  fi
}

# Setup ########################################################################

function print_args() {
  print "LiteRt Mobile Install Scripts"
  
  print_info_key "files"
  for i in "${!host_runfiles[@]}"; do
    print_file "${host_runfiles[$i]}" "${device_paths[$i]}"
  done

  print_info_key "device_runfiles_root"
  print_file "${device_runfiles_root}"
  
  if [[ $should_exec ]]; then
    print_info_key "bin"
    print_file "${host_bin}" "${device_bin_path}"
    if [[ -n "${exec_args}" ]]; then
      print_info_key "exec_args"
      echo "    ${exec_args}"
    fi
    if [[ -n "${exec_env_vars}" ]]; then
      print_info_key "exec_env_vars"
      echo "    ${exec_env_vars}"
    fi
  fi
}

for i in "${host_runfiles[@]}"; do
  device_paths+=("$(device_path "${i}")")
done

device_bin_path="$(device_path "${host_bin}")"

print_args

# Push and execute #############################################################
print "Pushing data files to device..."
for i in "${!host_runfiles[@]}"; do
  adb push --sync ${host_runfiles[$i]} ${device_paths[$i]}
done

if [[ $should_exec ]]; then
  print "Pushing binary to device..."
  adb push --sync ${host_bin} ${device_bin_path}
  print "Running: \"${hightlight_color}adb shell ${exec_env_vars} ${device_bin_path} ${exec_args}${host_color}\""
  adb shell ${exec_env_vars} ${device_bin_path} ${exec_args} 
fi








