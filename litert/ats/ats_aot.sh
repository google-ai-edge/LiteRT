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
remote_helper="$(remote_path_helper)"
readonly remote_helper
if [[ -n "${remote_helper}" && -f "${remote_helper}" ]]; then
  source "${remote_helper}" || exit 1
fi

models_out_is_ephemeral=""
if [[ -n "${TEST_TMPDIR:-}" ]]; then
  models_out="${TEST_TMPDIR}/litert_aot_out"
else
  models_out="$(mktemp -d -t litert_aot_out.XXXXXX)"
  models_out_is_ephemeral="true"
fi
readonly models_out
readonly provider_sharded_marker="${TEST_TMPDIR:-/tmp}/litert_extras/.provider_sharded"
readonly exec_args=("${@:1}")

function cleanup_ats_aot() {
  if [[ -n "${models_out_is_ephemeral}" && -d "${models_out}" ]]; then
    rm -rf "${models_out}"
  fi
  if type cleanup_remote_paths >/dev/null 2>&1; then
    cleanup_remote_paths
  fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  trap cleanup_ats_aot EXIT
fi

compile_bin=""
host_loader_bin=""
compile_args=()
upload_models_out=""
dry_run=""
has_user_models=""
provided_model_count=0
link_path=""
plugin_dir=""
runtime_lib_dir=""
compiler_libs=()

# Strips outer shell quotes (\'...\', '...', "...") from macro-generated regex
# flag values so the compiler binary can be executed directly as an argv array
# without eval.
function strip_flag_quotes() {
  local flag="$1"
  if [[ "${flag}" != *=* ]]; then
    printf '%s\n' "${flag}"
    return 0
  fi
  local key="${flag%%=*}"
  local val="${flag#*=}"
  local bs_sq="\\'"
  local sq="'"
  local dq='"'
  if [[ "${val}" == "${bs_sq}"*"${bs_sq}" ]]; then
    val="${val#"${bs_sq}"}"
    val="${val%"${bs_sq}"}"
  elif [[ "${val}" == "${sq}"*"${sq}" ]]; then
    val="${val#"${sq}"}"
    val="${val%"${sq}"}"
  elif [[ "${val}" == "${dq}"*"${dq}" ]]; then
    val="${val#"${dq}"}"
    val="${val%"${dq}"}"
  fi
  printf '%s=%s\n' "${key}" "${val}"
}

# Parses runner-specific flags (--dry_run, --upload_models_out, --models) and
# collects passthrough arguments for the host compiler binary.
function parse_flags() {
  local arg
  local m_path
  for arg in "$@"; do
    arg="$(strip_flag_quotes "${arg}")"
    if [[ "${arg}" == "--dry_run"* ]]; then
      dry_run="true"
    elif [[ "${arg}" == "--upload_models_out="* ]]; then
      upload_models_out="${arg#*=}"
    elif [[ "${arg}" == "--models="* ]]; then
      m_path="${arg#*=}"
      compile_args+=("--extra_models=${m_path}")
      compile_args+=("--do_register=ExtraModel")
      has_user_models="true"
    else
      compile_args+=("${arg}")
    fi
  done
  compile_args+=("--models_out=${models_out}")
}

# Locates the host compiler binary, compiler plugin, runtime library, and vendor
# shared libraries packaged with this target, populating link_path.
function resolve_host_libraries() {
  compile_bin="$(host_bin)"
  if [[ -z "${compile_bin}" ]]; then
    fatal "No binary provided, please provide a binary to execute."
  fi

  local plugin_lib
  plugin_lib="$(find_host_plugin)"
  if [[ -z "${plugin_lib}" ]]; then
    fatal "No plugin dir provided, please provide a directory containing the plugin library."
  fi
  plugin_dir="$(dirname "${plugin_lib}")"
  compile_args+=("--plugin_dir=${plugin_dir}")
  link_path="${plugin_dir}:${link_path}"

  local runtime_lib
  runtime_lib="$(find_host_runtime_lib)"
  if [[ -z "${runtime_lib}" ]]; then
    fatal "No runtime lib dir provided, please provide a directory containing the runtime library."
  fi
  runtime_lib_dir="$(dirname "${runtime_lib}")"
  link_path="${runtime_lib_dir}:${link_path}"

  local raw_host_libs
  raw_host_libs="$(host_libs)"
  local in_host_libs=()
  if [[ -n "${raw_host_libs}" ]]; then
    read -r -a in_host_libs <<< "${raw_host_libs}"
  fi
  local lib
  local lib_dir
  for lib in "${in_host_libs[@]}"; do
    if [[ "${lib}" != *"libLiteRt"* ]]; then
      compiler_libs+=("${lib}")
      lib_dir="$(dirname "${lib}")"
      link_path="${lib_dir}:${link_path}"
    fi
  done
  link_path="${link_path%:}"
}

# Prebuilt host compiler plugins (for example, vendor SDKs built against a
# newer glibc than the host OS) may bundle a dynamic loader
# (ld-linux-x86-64.so.2) and sysroot library directories.
function configure_host_sysroot_loader() {
  host_loader_bin="$(host_loader)"
  local loader_lib_dirs
  loader_lib_dirs="$(host_loader_lib_dirs)"
  if [[ -n "${loader_lib_dirs}" ]]; then
    link_path="${link_path:+${link_path}:}${loader_lib_dirs}"
  fi
}

# Invokes packaged model provider scripts when the caller did not pass --models,
# clearing any stale shard marker first so unsharded targets stay isolated.
function stage_provided_models() {
  if [[ -n "${has_user_models}" ]]; then
    return 0
  fi
  rm -f "${provider_sharded_marker}"
  local raw_models
  if ! raw_models="$(get_provided_models)"; then
    fatal "Failed to get provided models."
  fi
  local input_models=()
  if [[ -n "${raw_models}" ]]; then
    mapfile -t input_models <<< "${raw_models}"
  fi
  provided_model_count="${#input_models[@]}"
  if (( ${#input_models[@]} > 0 )); then
    local joined_models
    joined_models="$(str_join "," "${input_models[@]}")"
    compile_args+=("--extra_models=${joined_models}")
  fi
}

function setup_context() {
  mkdir -p "${models_out}"
  rm -rf "${models_out:?}"/*

  parse_flags "$@"
  resolve_host_libraries
  configure_host_sysroot_loader
  stage_provided_models
}

function print_args() {
  print_hightlight "<<< LiteRt ATS Aot >>>"
  print "compile_bin"
  print_file "${compile_bin}"
  if [[ -n "${host_loader_bin}" ]]; then
    print "host_loader_bin"
    print_file "${host_loader_bin}"
  fi
  print "plugin_dir"
  print_file "${plugin_dir}"
  print "runtime_lib_dir"
  print_file "${runtime_lib_dir}"
  print "compile_args"
  echo "    ${compile_args[*]}"
  print "link_path"
  echo "    ${link_path}"
  if [[ -n "${upload_models_out}" ]]; then
    print "upload_models_out"
    echo "    ${upload_models_out}"
  fi
  if [[ -n "${dry_run}" ]]; then
    print "dry_run"
    echo "    yes"
  fi
  print "compiler_libs"
  printf "\t%s\n" "${compiler_libs[*]}"
}

# Returns 0 when a model provider already partitioned input models for this test
# shard. In that mode TEST_SHARD_* must be unset before running the compiler
# binary to prevent GTest from double-sharding the already-sliced model list.
function is_provider_sharded() {
  [[ -z "${has_user_models}" && -f "${provider_sharded_marker}" ]]
}

# Runs the host compiler binary with the resolved library path, optional bundled
# dynamic loader, and shard env isolation when inputs were pre-sharded.
function run_host_compiler() {
  local cmd_prefix=()
  if is_provider_sharded; then
    cmd_prefix=(
      env -u TEST_SHARD_INDEX -u TEST_TOTAL_SHARDS -u TEST_SHARD_STATUS_FILE
    )
  fi

  local exec_cmd=("${compile_bin}" "${compile_args[@]}")
  if [[ -n "${host_loader_bin}" ]]; then
    exec_cmd=("${host_loader_bin}" "${exec_cmd[@]}")
  fi

  print "Running: ${hightlight_color}\"${cmd_prefix[*]} LD_LIBRARY_PATH=${link_path} ${exec_cmd[*]}\""
  if [[ -n "${dry_run}" ]]; then
    print "[dry run]"
    return 0
  fi

  if (( ${#cmd_prefix[@]} > 0 )); then
    "${cmd_prefix[@]}" LD_LIBRARY_PATH="${link_path}" "${exec_cmd[@]}"
  else
    LD_LIBRARY_PATH="${link_path}" "${exec_cmd[@]}"
  fi
}

# Uploads any compiled .tflite outputs when --upload_models_out is set. Runs
# even if individual models fail to compile so partial shard outputs remain
# available downstream, preserving the compiler's non-zero exit code on failure.
function maybe_upload_compiled_models() {
  local compile_status="$1"
  if [[ -z "${upload_models_out}" || -n "${dry_run}" ]]; then
    return "${compile_status}"
  fi
  if ! type upload_compiled_models >/dev/null 2>&1; then
    fatal "upload_compiled_models helper is unavailable for --upload_models_out=${upload_models_out}."
  fi
  print "Uploading compiled models to ${upload_models_out}"
  local upload_status=0
  upload_compiled_models "${models_out}" "${upload_models_out}" || upload_status=$?
  if (( upload_status != 0 && compile_status == 0 )); then
    return "${upload_status}"
  fi
  return "${compile_status}"
}

# Model Provider Callable ######################################################

setup_context "${exec_args[@]}"
print_args

function provide_models() {
  if is_provider_sharded && (( provided_model_count == 0 )); then
    print "Shard has 0 provided models; nothing to compile."
    return 0
  fi

  local compile_status=0
  run_host_compiler || compile_status=$?
  maybe_upload_compiled_models "${compile_status}" || return $?

  declare -n provided="$1"
  local f
  for f in "${models_out}"/*; do
    if [[ -f "${f}" ]]; then
      provided+=("${f}")
    fi
  done
  return 0
}

# Execute the model provider if this script is being called directly.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  provided_models=()
  status=0
  provide_models provided_models || status=$?
  if [[ -n "${dry_run}" ]]; then
    exit 1
  fi
  if (( status != 0 )); then
    exit "${status}"
  fi
  print "\nCompiled models:"
  printf "${hightlight_color}\ttotal compiled models: %s\n${reset_color}" "${#provided_models[@]}"
  printf "\t%s\n" "${provided_models[@]}"
  exit 0
fi
