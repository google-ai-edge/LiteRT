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

source gbash.sh || exit

DEFINE_string --required bin "" "The binary to execute on the device."
DEFINE_array data --type=string "" "The data files to install on the device."
DEFINE_bool do_exec false "Whether to execute the target on the device."
DEFINE_array exec_args --type=string "" --delim="~" "The arguments to pass to the executable on device."
DEFINE_array exec_env_vars --type=string "" "The environment variables to set for the executable on device."
DEFINE_string device_rlocation_root "/data/local/tmp/runfiles" "The root directory for device relative locations."

gbash::init_google "$@"

echo "BIN=${FLAGS_bin}"
echo "DATA=${FLAGS_data[@]}"
echo "DO_EXEC=${FLAGS_do_exec}"
echo "EXEC_ARGS=${FLAGS_exec_args[@]}"
echo "EXEC_ENV_VARS=${FLAGS_exec_env_vars[@]}"


DEVICEBIN="${FLAGS_device_rlocation_root}/${FLAGS_bin#"google3/"}"
HOSTBIN="${RUNFILES}/${FLAGS_bin}"

echo "DEVICEBIN=${DEVICEBIN}"
echo "HOSTBIN=${HOSTBIN}"

adb push --sync "${HOSTBIN}" "${DEVICEBIN}"

for HOSTFILE in "${FLAGS_data[@]}"; do
  DEVICEFILE="${FLAGS_device_rlocation_root}/${HOSTFILE#"google3/"}"
  HOSTFILE="${RUNFILES}/${HOSTFILE}"

  echo "HOSTFILE=${HOSTFILE}"
  echo "DEVICEFILE=${DEVICEFILE}"

  adb push --sync "${HOSTFILE}" "${DEVICEFILE}"
done

if (( ! FLAGS_do_exec )); then
  echo "Finished file transfer. Not executing on device."
  exit 0
fi

echo "Running: adb shell ${FLAGS_exec_env_vars[@]}  \"${DEVICEBIN}\" ${FLAGS_exec_args[@]}"
adb shell "${FLAGS_exec_env_vars[@]}"  "${DEVICEBIN}" "${FLAGS_exec_args[@]}" "${GBASH_ARGV[@]}"
