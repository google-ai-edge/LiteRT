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

source "${0%.sh}_lib.sh" || exit 1

extra_args=("${@:1}")
d_bin=$(device_bin)
d_libs=($(device_libs))
d_data=($(data_files))
d_env_vars=($(exec_env_vars))
d_args=()
provided_models=()

dry_run=""

function setup_context() {
  function handle_user_data() {
    local user_data=()
    for f in "$@"; do
      if [[ -f "$f" ]]; then
        user_data+=($(realpath "${f}"))
      elif [[ -d "$f" ]]; then
        for ff in "${f%/}/"*; do
          if [[ -f "$ff" ]]; then
            user_data+=($(realpath "${ff}"))
          fi
        done
      else
        fatal "${f} is not a file or directory"
      fi
    done
    echo "${user_data[@]}"
  }

  local in_flags=$1
  for a in ${in_flags[@]}; do
    if [[ $a == "--user_data="* ]]; then
      d_data+=($(handle_user_data "${a#*=}"))
    elif [[ $a == "--dry_run"* ]]; then
      dry_run="true"
    else
      d_args+=("${a}")
    fi
  done
  
  provided_models=($(get_provided_models))
  if [ $? -ne 0 ]; then
    echo "Failed to get provided models."
    exit 1
  fi

  for f in "${provided_models[@]}"; do
    d_data+=("${f}")
  done
}

function print_args() {
  print_hightlight "<<< LiteRt Mobile Install Scripts >>>"
  
  print "libraries"
  for f in "${d_libs[@]}"; do
    print_file "${f}" "$(device_path "${f}")"
  done

  print "data"
  for f in "${d_data[@]}"; do
    print_file "${f}" "$(device_path "${f}")"
  done

  print "device_runfiles_root"
  print_file "${device_runfiles_root}"

  if [[ -n $d_bin ]]; then
    print "bin"
    print_file "${d_bin}" "$(device_path "${d_bin}")"
    if [[ -n "${d_args[@]}" ]]; then
      print "exec_args"
      echo "    ${d_args[*]}"
    fi
    if [[ -n "${d_env_vars[@]}" ]]; then
      print "exec_env_vars"
      echo "    ${d_env_vars[@]}"
    fi
  fi

  if [[ -n "${dry_run}" ]]; then
    print "dry_run"
    echo "    yes"
  fi
}

# Push and execute #############################################################

setup_context "${extra_args[*]}"

print_args

function push_file() {
  local cmd="adb push --sync "$1" "$(device_path "$1")""
  if [[ -n "${dry_run}" ]]; then
    cmd="echo [dry run] ${cmd}"
  fi
  eval "${cmd}"
}

has_adb=$(adb devices | tail -n +2)
if [[ -z ${has_adb} && -z ${dry_run} ]]; then
  fatal "No usb device found."
fi

print "Pushing libraries to device..."
for f in "${d_libs[@]}"; do
  push_file "${f}"
done

print "Pushing data to device..."
for f in "${d_data[@]}"; do
  push_file "${f}"
done

if [[ -n $d_bin ]]; then
  print "Pushing binary to device..."
  push_file "${d_bin}"
  print "Running: ${hightlight_color}\"adb shell ${d_env_vars} $(device_path "${d_bin}") ${d_args[*]}${host_color}\""
  if [[ -z "${dry_run}" ]]; then
    adb shell ${d_env_vars[*]} $(device_path "${d_bin}") ${d_args[*]} 
  fi
fi
