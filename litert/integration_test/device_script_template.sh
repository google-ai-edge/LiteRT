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

# Shell library for working with data and executable files from bzl between host
# and device. Meant to be templated via litert_device_script.bzl.

source litert/integration_test/device_script_common.sh || exit 1

# Root of runfiles on the device.
device_runfiles_root="/data/local/tmp/runfiles"

# Path to the binary built for host that is packed with this library.
function host_bin() {
  local host_bin=@@host_bin@@
  if [[ "$host_bin" == "@@"*"@@" ]]; then
    echo ""
  else
    echo "$host_bin"
  fi
}

# Path to the binary built for device that is packed with this library.
function device_bin() {
  local device_bin=@@device_bin@@
  if [[ "$device_bin" == "@@"*"@@" ]]; then
    echo ""
  else
    echo "$device_bin"
  fi
}

# Paths to all the data files built for host that are packed with this library.
# E.g. .tflite files.
function data_files() {
  local data=@@data@@
  local the_data=()
  if [[ "$data" == "@@"*"@@" ]]; then
    echo "$the_data"
  else
    for file in $data; do
      the_data+=(${file})
    done
    echo "${the_data[@]}"
  fi
}

# Paths to all shared libraries built for host that are packed with this library.
function host_libs() {
  local host_libs=@@host_libs@@
  local the_host_libs=()
  if [[ "$host_libs" == "@@"*"@@" ]]; then
    echo "$the_host_libs"
  else
    for file in $host_libs; do
      if [[ "$file" == *"for_host"* ]]; then
        the_host_libs+=($(realpath $file))
      else
        the_host_libs+=(${file})
      fi
    done
    echo "${the_host_libs[@]}"
  fi
}

# Paths to all shared libraries built for device that are packed with this library.
function device_libs() {
  local device_libs=@@device_libs@@
  local the_device_libs=()
  if [[ "$device_libs" == "@@"*"@@" ]]; then
    echo "$the_device_libs"
  else
    for file in $device_libs; do
      the_device_libs+=(${file})
    done
    echo "${the_device_libs[@]}"
  fi
}

# Environment variables to set for the device binary.
function exec_env_vars() {
  local exec_env_vars=@@exec_env_vars@@
  if [[ "$exec_env_vars" == "@@"*"@@" ]]; then
    echo ""
  else
    echo "${exec_env_vars[@]}"
  fi
}

# Get the canonical device path of a host file
# need to double check how realpat hits this
function device_path() {
  if [[ "$1" == "/"* ]]; then
    echo "${device_runfiles_root}/user${1}"
  else
    echo "${device_runfiles_root}/${1}" | sed "s|\.\.|external|g"
  fi
}


# Locate the compiler plugin directory for host if one has been packaged with this library.
function find_host_plugin() {
  for file in $(host_libs); do
    if [[ "$file" == *"libLiteRtCompilerPlugin"*."so" ]]; then
      echo "$file"
    fi
  done
}

# Locate the compiler plugin directory for device if one has been packaged with this library.
function find_device_plugin() {
  for file in $(device_libs); do
    if [[ "$file" == *"libLiteRtCompilerPlugin"*."so" ]]; then
      echo "$file"
    fi
  done
}

# Locate the dispatch directory for host if one has been packaged with this library.
function find_host_dispatch() {
  for file in $(host_libs); do
    if [[ "$file" == *"libLiteRtDispatch"*."so" ]]; then
      echo "$file"
    fi
  done
}

# Locate the dispatch directory for device if one has been packaged with this library.
function find_device_dispatch() {
  for file in $(device_libs); do
    if [[ "$file" == *"libLiteRtDispatch"*."so" ]]; then
      echo "$file"
    fi
  done
}

# Locate the runtime so directory for host if one has been packaged with this library.
function find_host_runtime_lib() {
  for file in $(host_libs); do
    if [[ "$file" == *"libLiteRtRuntimeCApi.so" ]]; then
      echo "$file"
    fi
  done
}

# Locate the runtime so directory for device if one has been packaged with this library.
function find_device_runtime_lib() {
  for file in $(device_libs); do
    if [[ "$file" == *"libLiteRtRuntimeCApi.so" ]]; then
      echo "$file"
    fi
  done
}

# Call any/all model provider scripts built with this tool. The return code
# of this need to be checked by callers.
function get_provided_models() {
  local model_providers=@@model_providers@@
  if [[ "$model_providers" == "@@"*"@@" ]]; then
    echo ""
  else
    for provider in ${model_providers[@]}; do
      $provider
    done
  fi
}


