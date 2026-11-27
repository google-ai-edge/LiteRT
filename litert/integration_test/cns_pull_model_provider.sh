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

readonly cns_path=@@cns_path@@

if [[ "$cns_path" == "@@"*"@@" ]]; then
  fatal "No cns_path templated into the script."
elif [[ -z "${cns_path}" ]]; then
  fatal "cns_path is empty."
fi

if fileutil test -d "${cns_path}"; then
  # Path is a directory. Copy all files in the directory.
  fileutil cp "${cns_path}/*.tflite" "${work_dir}/"

elif fileutil test -f "${cns_path}"; then
  # Path is a file. Copy the file.
  fileutil cp "${cns_path}" "${work_dir}/"

else
  fatal "The specified cns_path '${cns_path}' is not a valid file or directory, or it does not exist."
fi

for model_file in ${work_dir}/*; do
  if [[ -f "${model_file}" ]]; then
    echo "${model_file}"
  fi
done