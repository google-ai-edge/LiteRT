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

source litert/integration_test/device_script_common.sh || exit 1

# TODO: Unify workdirs with other scripts.
readonly work_dir="/tmp/litert_extras"
rm -rf "${work_dir}"
mkdir -p "${work_dir}"

readonly cns_paths="@@cns_paths@@"

if [[ "${cns_paths}" == "@@"*"@@" ]]; then
  fatal "No cns_paths templated into the script."
elif [[ -z "${cns_paths}" ]]; then
  fatal "cns_paths is empty."
fi

sources=()
for path in ${cns_paths}; do
  if [[ "${path}" == *.tflite ]]; then
    sources+=("${path}")
  elif fileutil test -d "${path}"; then
    # Path is a directory. Copy all files in the directory.
    sources+=("${path}/*.tflite")
  elif fileutil test -f "${path}"; then
    # Path is a file. Copy the file.
    sources+=("${path}")
  else
    fatal "The specified CNS path '${path}' is not a valid file or directory, or it does not exist."
  fi
done

fileutil cp -parallelism 16 "${sources[@]}" "${work_dir}/" || \
  fatal "Failed to copy models from CNS."

for model_file in ${work_dir}/*; do
  if [[ -f "${model_file}" ]]; then
    echo "${model_file}"
  fi
done