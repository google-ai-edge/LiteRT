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
mkdir -p "${work_dir}"

readonly url=@@url@@

if [[ "$url" == "@@"*"@@" ]]; then
  fatal "No url templated into the script."
elif [[ -z "${url}" ]]; then
  fatal "Url is empty."
fi

readonly target_file="${work_dir}/$(basename ${url})"

if [[ ${target_file} != *".tar.gz" ]]; then
  fatal "Target file is not a .tar.gz: ${target_file}"
fi

wget -p -O ${target_file} ${url}
if [[ $? -ne 0 ]]; then
  fatal "Failed to download model from ${url}."
fi

models=($(tar -xhvf ${target_file} -C ${work_dir}))
rm -f ${target_file}
for model in ${models[@]}; do
  echo "${work_dir}/${model}"
done
