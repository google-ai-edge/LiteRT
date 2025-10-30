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

# TODO: Unify workdirs with other scripts.
readonly models_out="/tmp/litert_extras/ats"
readonly exec_args=("${@:1}")

compile_bin=""
compile_args=()
dry_run=""
link_path=""
plugin_dir=""
runtime_lib_dir=""
compiler_libs=()

function setup_context() {
  mkdir -p "$models_out"
  rm -rf "$models_out"/*

  local in_flags=$1
  for a in ${in_flags[@]}; do
    if [[ $a == "--dry_run"* ]]; then
      dry_run="true"
    else
      compile_args+=("${a}")
    fi
  done
  compile_args+=("--models_out=${models_out}")

  compile_bin=$(host_bin)
  if [[ -z "$compile_bin" ]]; then
    fatal "No binary provided, please provide a binary to execute."
  fi

  local plugin_lib=$(find_host_plugin)
  if [[ -z "$plugin_lib" ]]; then
    fatal "No plugin dir provided, please provide a directory containing the plugin library."
  fi
  plugin_dir=$(dirname ${plugin_lib})
  compile_args+=("--plugin_dir=${plugin_dir}")
  link_path="${plugin_dir}:${link_path}"

  local runtime_lib=$(find_host_runtime_lib)
  if [[ -z "$runtime_lib" ]]; then
    fatal "No runtime lib dir provided, please provide a directory containing the runtime library."
  fi
  runtime_lib_dir=$(dirname ${runtime_lib})
  link_path="${runtime_lib_dir}:${link_path}"

  local in_host_libs=($(host_libs))
  for lib in ${in_host_libs[@]}; do
    if [[ "$lib" != *"libLiteRt"* ]]; then
      compiler_libs+=("${lib}")
      link_path=$(dirname ${lib}):${link_path}
    fi
  done

  local input_models=($(get_provided_models))
  if [[ $? -ne 0 ]]; then
    fatal "Failed to get provided models."
  fi

  if [[ -n "${input_models[*]}" ]]; then
    compile_args+=("--extra_models=$(str_join "," ${input_models[@]})")
  fi
}

function print_args() {
  print_hightlight "<<< LiteRt ATS Aot >>>"
  print "compile_bin"
  print_file "${compile_bin}"
  print "plugin_dir"
  print_file "${plugin_dir}"
  print "runtime_lib_dir"
  print_file "${runtime_lib_dir}"
  print "compile_args"
  echo "    ${compile_args[*]}"
  print "link_path"
  echo "    ${link_path}"
  if [[ -n "${dry_run}" ]]; then
    print "dry_run"
    echo "    yes"
  fi
  print "compiler_libs"
  printf "\t%s\n" "${compiler_libs[*]}"
}


# Model Provider Callable ######################################################

setup_context "${exec_args[*]}"
print_args

function provide_models() {
  local cmd="LD_LIBRARY_PATH=${link_path} ${compile_bin} ${compile_args[*]}"
  print "Running: ${hightlight_color}\"${cmd}\""
  if [[ -n "${dry_run}" ]]; then
    print "[dry run]"
    return 0
  fi
  eval "${cmd}"
  status=$?
  if [[ $status -ne 0 ]]; then
    return $status
  fi
  declare -n provided="$1"
  for f in $(ls "${models_out}"); do
    provided+=("${models_out}/${f}")
  done
  return 0
}

# Execute the model provider if this script is being called directly.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  provided_models=()
  provide_models provided_models 
  if [[ -n "${dry_run}" ]]; then
    exit 1
  fi
  print "\nCompiled models:"
  printf "${hightlight_color}\ttotal compiled models: %s\n${reset_color}" "${#provided_models[@]}"
  printf "\t%s\n" "${provided_models[@]}"
  exit $?
fi


